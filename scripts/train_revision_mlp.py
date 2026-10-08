"""Train source-only MLP revision models with source-validation checkpoints."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import train_revision_trees as source_helpers
from sepsis_cross_hospital.models import revision_mlp as mlp

def train_one(train_x, train_y, val_x, val_y, seed, config):
    mlp.configure_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    model = mlp.make_mlp(train_x.shape[1], config["hidden_sizes"]).to(train_x.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=tuple(config["adam_betas"]),
        eps=config["adam_eps"],
        weight_decay=config["weight_decay"],
        foreach=False,
        fused=False,
    )
    generator = torch.Generator(device=train_x.device).manual_seed(seed)
    batch_size = config["batch_size_hours"]
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    stale_epochs = 0
    history = []

    for epoch in range(1, config["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = torch.randperm(
            len(train_y), device=train_x.device, generator=generator
        )
        total_loss = 0.0
        for offset in range(0, len(train_y), batch_size):
            indices = order[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[indices]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, train_y[indices])
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)

        val_loss = mlp.validation_bce(model, val_x, val_y, batch_size)
        if not np.isfinite(val_loss):
            raise ValueError("Nonfinite source-validation loss.")
        history.append({
            "epoch": epoch,
            "train_bce": total_loss / len(train_y),
            "source_validation_bce": val_loss,
            "seconds": time.perf_counter() - epoch_start,
        })
        print(
            f"Epoch {epoch:02d}: train BCE={history[-1]['train_bce']:.6f}, "
            f"validation BCE={val_loss:.6f}, "
            f"time={history[-1]['seconds']:.1f}s", flush=True
        )

        if val_loss < best_loss - config["minimum_improvement"]:
            best_loss = val_loss
            best_state = mlp.pack_state(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                print("Source-validation early stopping.", flush=True)
                break

    if best_state is None:
        raise ValueError("No valid source-validation checkpoint.")
    torch.cuda.synchronize()
    return {
        "state_dict": best_state,
        "history": history,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "best_source_validation_bce": best_loss,
        "training_seconds": time.perf_counter() - start,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path,
                        default=REPO / "runs/revision_mlp_models")
    parser.add_argument("--protocol", type=Path,
                        default=REPO / "configs/revision_mlp_protocol.json")
    parser.add_argument("--direction", default="all",
                        choices=["all", "A_to_B", "B_to_A"])
    parser.add_argument("--representation", default="all",
                        choices=["all", "V0", "V1", "V2"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (str(torch.__version__) != config["torch_version"]
            or sklearn.__version__ != config["sklearn_version"]):
        raise ValueError("Use the pinned PyTorch and scikit-learn versions.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fixed MLP training protocol.")

    directions = config["directions"] if args.direction == "all" else [args.direction]
    representations = (
        config["representations"] if args.representation == "all"
        else [args.representation]
    )
    seeds = config["seeds"] if args.seeds is None else args.seeds
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(config["seeds"]):
        raise ValueError("Seeds must be a unique subset of the frozen seeds.")
    selected = [
        (direction, representation, seed)
        for direction in directions
        for representation in representations
        for seed in seeds
    ]
    if not args.resume:
        for direction, representation, seed in selected:
            path = args.model_root / f"{direction}_{representation}/seed_{seed}.joblib"
            if path.exists():
                raise FileExistsError(f"Artifact exists; use --resume: {path}")

    reference_path = REPO / "expected_outputs/frozen_data_hashes.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
    }
    code_hashes = {
        "trainer": source_helpers.sha256(Path(__file__).resolve()),
        "mlp_helpers": source_helpers.sha256(Path(mlp.__file__)),
        "source_helpers": source_helpers.sha256(Path(source_helpers.__file__)),
    }
    protocol_hash = source_helpers.sha256(args.protocol)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    invocation = args.model_root / "training-invocations" / stamp
    invocation.mkdir(parents=True, exist_ok=False)
    source_helpers.write_json(invocation / "frozen_training_protocol.json", {
        "configuration": config,
        "protocol_sha256": protocol_hash,
        "code_sha256": code_hashes,
        "data_reference_sha256": source_helpers.sha256(reference_path),
        "versions": versions,
        "selected_models": selected,
        "external_patient_data_read": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"GPU: {versions['gpu']}", flush=True)

    artifact_hashes = {}
    for direction in directions:
        schema_path = (
            args.data_root
            / f"data/metadata/preprocessing/{direction}/feature_schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        canonical = json.dumps(
            schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        schema_hash = hashlib.sha256(canonical).hexdigest()
        if schema_hash != config["feature_schema_canonical_sha256"]:
            raise ValueError("Feature schema differs from the frozen study.")
        source = direction.split("_to_")[0]

        for representation in representations:
            features = schema[representation]
            if len(features) != config["feature_counts"][representation]:
                raise ValueError("Unexpected feature count.")
            frames = {}
            for partition in ["train", "validation"]:
                frames[partition] = source_helpers.load_source(
                    args.data_root / f"data/processed/{direction}/{partition}.parquet",
                    features,
                    reference["processed"][direction][partition],
                    reference["split"][source][partition],
                )
            train_raw = frames["train"][features].to_numpy(dtype=np.float32)
            scaler = mlp.fit_scaler(train_raw, config["scaler_quantiles"])
            arrays = {
                "train_x": mlp.scale_features(train_raw, scaler, config["feature_clip"]),
                "train_y": frames["train"]["SepsisLabel"].to_numpy(dtype=np.float32),
                "val_x": mlp.scale_features(
                    frames["validation"][features].to_numpy(dtype=np.float32),
                    scaler, config["feature_clip"],
                ),
                "val_y": frames["validation"]["SepsisLabel"].to_numpy(dtype=np.float32),
            }
            del frames, train_raw
            gc.collect()
            gpu_arrays = None

            for seed in seeds:
                contract = {
                    "model_family": "mlp",
                    "direction": direction,
                    "representation": representation,
                    "features": features,
                    "seed": seed,
                    "configuration": config,
                    "train_sha256": reference["processed"][direction]["train"],
                    "validation_sha256": reference["processed"][direction]["validation"],
                    "feature_schema_canonical_sha256": schema_hash,
                    "protocol_sha256": protocol_hash,
                    "code_sha256": code_hashes,
                    "versions": versions,
                }
                key = f"{direction}_{representation}/seed_{seed}.joblib"
                artifact = args.model_root / key
                artifact.parent.mkdir(parents=True, exist_ok=True)

                if artifact.exists():
                    saved = joblib.load(artifact)
                    for field, value in contract.items():
                        if saved.get(field) != value:
                            raise ValueError(f"Existing MLP contract mismatch: {field}")
                    for field in ["center_", "scale_"]:
                        if not np.array_equal(
                            getattr(saved["scaler"], field), getattr(scaler, field)
                        ):
                            raise ValueError("Existing scaler differs from source-training fit.")
                    restored = mlp.restore_mlp(
                        saved["state_dict"], len(features), config["hidden_sizes"]
                    )
                    del restored, saved
                    print(f"Verified and reused: {key}", flush=True)
                else:
                    if gpu_arrays is None:
                        gpu_arrays = {
                            name: torch.from_numpy(values).to("cuda")
                            for name, values in arrays.items()
                        }
                    print(
                        f"\nTraining MLP {direction} {representation} seed={seed}: "
                        f"{len(arrays['train_y']):,} hours, {len(features)} features",
                        flush=True,
                    )
                    result = train_one(
                        gpu_arrays["train_x"], gpu_arrays["train_y"],
                        gpu_arrays["val_x"], gpu_arrays["val_y"], seed, config,
                    )
                    saved = dict(
                        contract, **result, scaler=scaler,
                        train_hours=len(arrays["train_y"]),
                        validation_hours=len(arrays["val_y"]),
                        completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    )
                    source_helpers.dump_new(artifact, saved)
                    print(
                        f"Saved: {artifact}\nBest epoch: {result['best_epoch']}, "
                        f"best validation BCE: {result['best_source_validation_bce']:.6f}\n"
                        f"Fit time: {result['training_seconds'] / 60:.2f} minutes",
                        flush=True,
                    )
                    del result, saved

                artifact_hashes[key] = source_helpers.sha256(artifact)
                gc.collect()
            del gpu_arrays, arrays, scaler
            torch.cuda.empty_cache()
            gc.collect()

    source_helpers.write_json(invocation / "completion.json", {
        "status": "complete",
        "completed_requested_models": len(artifact_hashes),
        "full_family_grid": len(artifact_hashes) == 30,
        "model_sha256": artifact_hashes,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"\nMLP SOURCE TRAINING COMPLETE: {len(artifact_hashes)} requested models",
          flush=True)
    print(f"Models: {args.model_root}\nInvocation: {invocation}", flush=True)

if __name__ == "__main__":
    main()
