"""Train source-only RF or ExtraTrees revision models."""
import argparse
import gc
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

REPO = Path(__file__).resolve().parents[1]
CLASSES = {"rf": RandomForestClassifier, "extra_trees": ExtraTreesClassifier}

def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

def dump_new(path, value):
    # Open exclusively before entering the cleanup handler.
    # Existing artifacts can never be overwritten or deleted here.
    handle = path.open("xb")
    try:
        with handle:
            joblib.dump(value, handle, compress=3)
    except BaseException:
        path.unlink(missing_ok=True)
        raise

def load_source(path, features, expected_hash, expected_patients):
    reserved = {"patient_id", "hour_index", "SepsisLabel"}
    if reserved.intersection(features) or len(set(features)) != len(features):
        raise ValueError("Invalid feature list: identifiers, labels, or duplicates.")
    if sha256(path) != expected_hash:
        raise ValueError("Source-training data hash mismatch.")

    frame = pd.read_parquet(
        path, columns=["patient_id", "hour_index", "SepsisLabel"] + features
    ).sort_values(["patient_id", "hour_index"]).reset_index(drop=True)

    if frame.empty or frame.duplicated(["patient_id", "hour_index"]).any():
        raise ValueError("Empty source data or duplicate patient-hours.")
    if frame["patient_id"].nunique() != expected_patients:
        raise ValueError("Source-training patient count mismatch.")
    sequence = frame.groupby("patient_id", sort=False).cumcount()
    if not np.array_equal(frame["hour_index"].to_numpy(), sequence.to_numpy()):
        raise ValueError("Nonsequential source patient hours.")

    y = frame["SepsisLabel"].to_numpy()
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Source labels must contain both binary classes.")
    if not np.isfinite(frame[features].to_numpy(dtype=np.float32)).all():
        raise ValueError("Nonfinite source features.")
    return frame

def validate_existing(saved, contract):
    for key, expected in contract.items():
        if saved.get(key) != expected:
            raise ValueError(f"Existing artifact metadata mismatch: {key}")
    model = saved["model"]
    parameters = contract["parameters"]
    actual = model.get_params()
    if (
        type(model) is not CLASSES[contract["model_family"]]
        or any(actual[key] != value for key, value in parameters.items())
        or model.n_features_in_ != len(contract["features"])
        or len(model.estimators_) != parameters["n_estimators"]
        or model.classes_.tolist() != [0, 1]
    ):
        raise ValueError("Existing estimator does not match the training contract.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=list(CLASSES))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--protocol", type=Path,
                        default=REPO / "configs/revision_tree_protocol.json")
    parser.add_argument("--direction", default="all",
                        choices=["all", "A_to_B", "B_to_A"])
    parser.add_argument("--representation", default="all",
                        choices=["all", "V0", "V1", "V2"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sklearn.__version__ != protocol["sklearn_version"]:
        raise ValueError("Use the pinned scikit-learn version.")

    directions = (
        protocol["directions"] if args.direction == "all"
        else [args.direction]
    )
    representations = (
        protocol["representations"] if args.representation == "all"
        else [args.representation]
    )
    seeds = protocol["seeds"] if args.seeds is None else args.seeds
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(protocol["seeds"]):
        raise ValueError("Seeds must be a unique subset of the frozen seeds.")

    model_root = (
        args.model_root if args.model_root is not None
        else REPO / "runs/revision_tree_models" / args.family
    )
    selected = [
        (direction, representation, seed)
        for direction in directions
        for representation in representations
        for seed in seeds
    ]
    if not args.resume:
        for direction, representation, seed in selected:
            path = model_root / f"{direction}_{representation}/seed_{seed}.joblib"
            if path.exists():
                raise FileExistsError(f"Artifact exists; use --resume to verify/reuse: {path}")

    reference_path = REPO / "expected_outputs/frozen_data_hashes.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    protocol_hash = sha256(args.protocol)
    script_hash = sha256(Path(__file__).resolve())
    reference_hash = sha256(reference_path)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    invocation = model_root / "training-invocations" / stamp
    invocation.mkdir(parents=True, exist_ok=False)
    write_json(invocation / "frozen_training_protocol.json", {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "training_script_sha256": script_hash,
        "data_reference_sha256": reference_hash,
        "family": args.family,
        "selected_models": selected,
        "versions": versions,
        "target_patient_data_read": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })

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
        if schema_hash != protocol["feature_schema_canonical_sha256"]:
            raise ValueError("Feature schema differs from the frozen study.")

        train_hash = reference["processed"][direction]["train"]
        source = direction.split("_to_")[0]
        patients = reference["split"][source]["train"]
        train_path = args.data_root / f"data/processed/{direction}/train.parquet"

        for representation in representations:
            features = schema[representation]
            if len(features) != protocol["feature_counts"][representation]:
                raise ValueError("Unexpected feature count.")
            frame = load_source(train_path, features, train_hash, patients)
            X = frame[features].to_numpy(dtype=np.float32)
            y = frame["SepsisLabel"].to_numpy(dtype=np.int8)
            del frame
            gc.collect()

            for seed in seeds:
                parameters = dict(
                    protocol["common_parameters"],
                    bootstrap=protocol["bootstrap"][args.family],
                    random_state=seed,
                )
                contract = {
                    "model_family": args.family,
                    "direction": direction,
                    "representation": representation,
                    "features": features,
                    "parameters": parameters,
                    "train_sha256": train_hash,
                    "feature_schema_canonical_sha256": schema_hash,
                    "protocol_sha256": protocol_hash,
                    "training_script_sha256": script_hash,
                    "data_reference_sha256": reference_hash,
                    "sklearn_version": sklearn.__version__,
                    "versions": versions,
                }
                key = f"{direction}_{representation}/seed_{seed}.joblib"
                artifact = model_root / key
                artifact.parent.mkdir(parents=True, exist_ok=True)

                if artifact.exists():
                    saved = joblib.load(artifact)
                    validate_existing(saved, contract)
                    del saved
                    print(f"Verified and reused: {key}", flush=True)
                else:
                    print(
                        f"Training {args.family} {direction} {representation} "
                        f"seed={seed}: {len(y):,} hours, {len(features)} features",
                        flush=True,
                    )
                    start = time.perf_counter()
                    model = CLASSES[args.family](**parameters)
                    model.fit(X, y)
                    seconds = time.perf_counter() - start
                    saved = dict(
                        contract,
                        model=model,
                        all_estimator_parameters=model.get_params(),
                        train_hours=len(y),
                        train_patients=patients,
                        training_seconds=seconds,
                        completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    )
                    validate_existing(saved, contract)
                    dump_new(artifact, saved)
                    del model, saved
                    print(f"Saved: {artifact}\nFit time: {seconds / 60:.2f} minutes",
                          flush=True)

                artifact_hashes[key] = sha256(artifact)
                gc.collect()
            del X, y
            gc.collect()

    write_json(invocation / "completion.json", {
        "status": "complete",
        "completed_requested_models": len(artifact_hashes),
        "full_family_grid": len(artifact_hashes) == 30,
        "model_sha256": artifact_hashes,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"\nSOURCE TRAINING COMPLETE: {len(artifact_hashes)} requested models",
          flush=True)
    print(f"Models: {model_root}\nInvocation: {invocation}", flush=True)

if __name__ == "__main__":
    main()
