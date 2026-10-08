"""Evaluate frozen ExtraTrees/MLP ensembles without target adaptation."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import gc
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import train_revision_trees as helpers
from sepsis_cross_hospital.models import revision_mlp as mlp
from sepsis_cross_hospital.evaluation import external, metrics, thresholding

DIRECTIONS = ["A_to_B", "B_to_A"]
REPRESENTATIONS = ["V0", "V1", "V2"]
SEEDS = [1729, 2718, 31415, 57721, 65537]
FAMILIES = ["extra_trees", "mlp"]
PREFIX = {"extra_trees": "ET", "mlp": "MLP"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def patient_sequences(frame, values):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != len(frame):
        raise ValueError("Patient-hour/value alignment mismatch.")
    ids = frame["patient_id"].to_numpy()
    boundaries = np.r_[
        0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)
    ]
    return [
        values[left:right]
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]


def mean_probabilities(seed_probabilities):
    total = None
    count = 0
    for probability in seed_probabilities:
        probability = np.asarray(probability, dtype=np.float64)
        if (
            probability.ndim != 1
            or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
        ):
            raise ValueError("Invalid seed probabilities.")
        if total is None:
            total = np.zeros_like(probability)
        elif probability.shape != total.shape:
            raise ValueError("Seed probability shapes differ.")
        total += probability
        count += 1
    if count != len(SEEDS):
        raise ValueError("Exactly five seed probabilities are required.")
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--extra-trees-completion", required=True, type=Path)
    parser.add_argument("--mlp-completion", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO / "runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen MLP protocol.")

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
    }
    reference_path = REPO / "expected_outputs/frozen_data_hashes.json"
    reference = read_json(reference_path)
    reference_hash = helpers.sha256(reference_path)
    configurations = {
        "extra_trees": REPO / "configs/revision_tree_protocol.json",
        "mlp": REPO / "configs/revision_mlp_protocol.json",
    }
    completions = {
        "extra_trees": args.extra_trees_completion.resolve(),
        "mlp": args.mlp_completion.resolve(),
    }
    expected_keys = {
        f"{direction}_{representation}/seed_{seed}.joblib"
        for direction in DIRECTIONS
        for representation in REPRESENTATIONS
        for seed in SEEDS
    }
    mlp_code_hashes = {
        "trainer": helpers.sha256(REPO / "scripts/train_revision_mlp.py"),
        "mlp_helpers": helpers.sha256(Path(mlp.__file__)),
        "source_helpers": helpers.sha256(Path(helpers.__file__)),
    }
    bundles = {}
    for family in FAMILIES:
        config_path = configurations[family]
        config = read_json(config_path)
        completion = read_json(completions[family])
        frozen = read_json(
            completions[family].parent / "frozen_training_protocol.json"
        )
        if (
            completion.get("status") != "complete"
            or completion.get("full_family_grid") is not True
            or completion.get("completed_requested_models") != 30
            or set(completion["model_sha256"]) != expected_keys
        ):
            raise ValueError(f"Incomplete training manifest: {family}")
        config_field = "protocol" if family == "extra_trees" else "configuration"
        if (
            frozen[config_field] != config
            or frozen["protocol_sha256"] != helpers.sha256(config_path)
            or frozen["data_reference_sha256"] != reference_hash
            or config["directions"] != DIRECTIONS
            or config["representations"] != REPRESENTATIONS
            or config["seeds"] != SEEDS
            or config["sklearn_version"] != sklearn.__version__
        ):
            raise ValueError(f"Training protocol mismatch: {family}")
        if any(
            versions.get(key) != value
            for key, value in frozen["versions"].items()
        ):
            raise ValueError(f"Training/evaluation environment mismatch: {family}")
        if family == "extra_trees":
            if (
                frozen["family"] != family
                or frozen["training_script_sha256"]
                != helpers.sha256(Path(helpers.__file__))
            ):
                raise ValueError("ExtraTrees trainer changed after training.")
        elif (
            frozen["code_sha256"] != mlp_code_hashes
            or config["torch_version"] != str(torch.__version__)
        ):
            raise ValueError("MLP training code or PyTorch version changed.")
        bundles[family] = {
            "config": config,
            "frozen": frozen,
            "completion": completion,
            "root": completions[family].parents[2],
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = args.output_root / f"revision-ensembles-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Results directory: {output}", flush=True)

    schemas = {}
    fingerprints = {}
    expected_scalers = {}
    thresholds = {}
    threshold_rows = []

    for direction in DIRECTIONS:
        schema_path = (
            args.data_root
            / f"data/metadata/preprocessing/{direction}/feature_schema.json"
        )
        schema = read_json(schema_path)
        canonical = json.dumps(
            schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        schema_hash = hashlib.sha256(canonical).hexdigest()
        for bundle in bundles.values():
            if (
                schema_hash
                != bundle["config"]["feature_schema_canonical_sha256"]
                or any(
                    len(schema[representation])
                    != bundle["config"]["feature_counts"][representation]
                    for representation in REPRESENTATIONS
                )
            ):
                raise ValueError(f"Frozen feature schema mismatch: {direction}")
        schemas[direction] = schema
        fingerprints[f"{direction}/feature_schema"] = helpers.sha256(schema_path)

    def load_frame(direction, partition):
        source, target = direction.split("_to_")
        patients = (
            reference["raw_data"][target]["patients"]
            if partition == "external"
            else reference["split"][source][partition]
        )
        expected_hash = reference["processed"][direction][partition]
        frame = helpers.load_source(
            args.data_root / f"data/processed/{direction}/{partition}.parquet",
            schemas[direction]["V2"], expected_hash, patients,
        )
        if (
            partition == "external"
            and len(frame) != reference["raw_data"][target]["hours"]
        ):
            raise ValueError("External patient-hour count mismatch.")
        fingerprints[f"{direction}/{partition}"] = expected_hash
        return frame

    def seed_probabilities(family, direction, representation, frame):
        bundle = bundles[family]
        config = bundle["config"]
        frozen = bundle["frozen"]
        features = schemas[direction][representation]
        raw = frame[features].to_numpy(dtype=np.float32)
        for seed in SEEDS:
            key = f"{direction}_{representation}/seed_{seed}.joblib"
            path = bundle["root"] / key
            if helpers.sha256(path) != bundle["completion"]["model_sha256"][key]:
                raise ValueError(f"Trained model fingerprint changed: {family}/{key}")
            saved = joblib.load(path)
            contract = {
                "model_family": family,
                "direction": direction,
                "representation": representation,
                "features": features,
                "train_sha256": reference["processed"][direction]["train"],
                "feature_schema_canonical_sha256":
                    config["feature_schema_canonical_sha256"],
                "protocol_sha256": frozen["protocol_sha256"],
                "versions": frozen["versions"],
            }
            if family == "extra_trees":
                contract.update({
                    "parameters": dict(
                        config["common_parameters"],
                        bootstrap=config["bootstrap"][family],
                        random_state=seed,
                    ),
                    "training_script_sha256": frozen["training_script_sha256"],
                    "data_reference_sha256": reference_hash,
                    "sklearn_version": sklearn.__version__,
                })
                helpers.validate_existing(saved, contract)
                probability = saved["model"].predict_proba(raw)[:, 1]
            else:
                contract.update({
                    "configuration": config,
                    "seed": seed,
                    "validation_sha256":
                        reference["processed"][direction]["validation"],
                    "code_sha256": mlp_code_hashes,
                })
                for field, value in contract.items():
                    if saved.get(field) != value:
                        raise ValueError(f"MLP metadata mismatch: {key}/{field}")
                expected = expected_scalers[(direction, representation)]
                scaler = saved["scaler"]
                if (
                    type(scaler) is not type(expected)
                    or scaler.get_params() != expected.get_params()
                    or scaler.n_features_in_ != len(features)
                    or any(
                        not np.array_equal(
                            getattr(scaler, field), getattr(expected, field)
                        )
                        for field in ["center_", "scale_"]
                    )
                ):
                    raise ValueError(f"MLP scaler differs from source-only fit: {key}")
                mlp.configure_seed(seed)
                network = mlp.restore_mlp(
                    saved["state_dict"], len(features), config["hidden_sizes"]
                ).to("cuda")
                tensor = torch.from_numpy(
                    mlp.scale_features(raw, scaler, config["feature_clip"])
                ).to("cuda")
                probability = np.empty(len(frame), dtype=np.float32)
                batch_size = config["batch_size_hours"]
                with torch.inference_mode():
                    for start in range(0, len(frame), batch_size):
                        stop = min(start + batch_size, len(frame))
                        logits = network(tensor[start:stop]).squeeze(-1)
                        probability[start:stop] = torch.sigmoid(logits).cpu().numpy()
                del network, tensor
            del saved
            yield probability

    def predict(family, direction, representation, frame):
        return mean_probabilities(
            seed_probabilities(family, direction, representation, frame)
        )

    # Finish ALL source-only work before reading either held-out partition.
    for direction in DIRECTIONS:
        train = load_frame(direction, "train")
        config = bundles["mlp"]["config"]
        for representation in REPRESENTATIONS:
            expected_scalers[(direction, representation)] = mlp.fit_scaler(
                train[schemas[direction][representation]].to_numpy(dtype=np.float32),
                config["scaler_quantiles"],
            )
        del train
        frame = load_frame(direction, "validation")
        labels = patient_sequences(
            frame, frame["SepsisLabel"].to_numpy(dtype=np.int8)
        )
        predictions = frame[["patient_id", "hour_index", "SepsisLabel"]].copy()
        thresholds[direction] = {}
        for family in FAMILIES:
            for representation in REPRESENTATIONS:
                name = f"{PREFIX[family]}_{representation}"
                print(f"Source validation: {direction} {name}", flush=True)
                probability = predict(family, direction, representation, frame)
                selected = thresholding.select_utility_threshold(
                    labels, patient_sequences(frame, probability)
                )
                threshold = float(selected.threshold)
                thresholds[direction][name] = threshold
                threshold_rows.append({
                    "direction": direction, "model": name,
                    "threshold": threshold,
                    "source_validation_utility":
                        float(selected.normalized_utility),
                })
                predictions[f"{name}_probability"] = probability
                print(f"Selected threshold: {threshold:.3f}", flush=True)
        predictions.to_parquet(
            output / f"{direction}_validation_predictions.parquet", index=False
        )
        del frame, predictions, labels, probability
        gc.collect()

    if len(threshold_rows) != 12:
        raise ValueError("All twelve thresholds must be frozen.")
    pd.DataFrame(threshold_rows).to_csv(
        output / "source_thresholds.csv", index=False
    )
    code_hashes = {
        "evaluation": helpers.sha256(Path(__file__).resolve()),
        "tree_training": helpers.sha256(Path(helpers.__file__)),
        "mlp_helpers": helpers.sha256(Path(mlp.__file__)),
        "metrics": helpers.sha256(Path(metrics.__file__)),
        "thresholding": helpers.sha256(Path(thresholding.__file__)),
        "external_statistics": helpers.sha256(Path(external.__file__)),
    }
    helpers.write_json(output / "frozen_protocol.json", {
        "analysis": "Additional exploratory ExtraTrees and MLP revision analyses",
        "source_thresholds": thresholds,
        "probability_combination": "Equal average of five seed probabilities",
        "threshold_grid": "0.001 through 0.999; step 0.001; higher on ties",
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 1729,
        "contrasts": "Within each family: V1 minus V0; V2 minus V0",
        "calibration_usage": "Diagnostics only; probabilities unchanged",
        "bootstrap_scope": "Patient sampling conditional on fixed trained ensembles",
        "source_input_sha256": fingerprints.copy(),
        "data_reference_sha256": reference_hash,
        "code_sha256": code_hashes,
        "versions": versions,
        "training": {
            family: {
                "frozen_training_protocol": bundles[family]["frozen"],
                "completion_sha256": helpers.sha256(completions[family]),
                "model_sha256": bundles[family]["completion"]["model_sha256"],
            }
            for family in FAMILIES
        },
        "held_out_data_read_at_freeze": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print("ALL 12 THRESHOLDS FROZEN. Starting held-out evaluation.", flush=True)

    metric_rows = []
    contrast_rows = []
    for direction in DIRECTIONS:
        external_predictions = {}
        external_utilities = {}
        for partition in ["internal_test", "external"]:
            frame = load_frame(direction, partition)
            y = frame["SepsisLabel"].to_numpy(dtype=np.int8)
            labels = patient_sequences(frame, y)
            predictions = frame[["patient_id", "hour_index", "SepsisLabel"]].copy()
            for family in FAMILIES:
                for representation in REPRESENTATIONS:
                    name = f"{PREFIX[family]}_{representation}"
                    print(f"Evaluating: {direction} {partition} {name}", flush=True)
                    probability = predict(family, direction, representation, frame)
                    threshold = thresholds[direction][name]
                    binary = (probability >= threshold).astype(np.int8)
                    sequences = patient_sequences(frame, binary)
                    utility = float(
                        metrics.compute_challenge_utility(labels, sequences).normalized
                    )
                    calibration = external.calibration_intercept_slope(y, probability)
                    metric_rows.append({
                        "direction": direction, "partition": partition,
                        "model": name, "patients": len(labels), "hours": len(frame),
                        "threshold": threshold, "utility": utility,
                        "auroc": float(roc_auc_score(y, probability)),
                        "auprc": float(average_precision_score(y, probability)),
                        "brier": float(brier_score_loss(y, probability)),
                        "calibration_intercept":
                            calibration.intercept if calibration.success else None,
                        "calibration_slope":
                            calibration.slope if calibration.success else None,
                        "calibration_success": calibration.success,
                        "calibration_message": calibration.message,
                    })
                    predictions[f"{name}_probability"] = probability
                    predictions[f"{name}_alert"] = binary
                    if partition == "external":
                        external_predictions[name] = sequences
                        external_utilities[name] = utility
            predictions.to_parquet(
                output / f"{direction}_{partition}_predictions.parquet", index=False
            )
            pd.DataFrame(metric_rows).to_csv(output / "metrics_all.csv", index=False)
            if partition == "external":
                external_labels = labels
            del frame, predictions, probability
            gc.collect()

        print(f"Paired patient bootstrap: {direction}, 1000 replicates", flush=True)
        draws = external.paired_patient_bootstrap(
            external_labels, external_predictions, n_bootstrap=1000, seed=1729
        )
        draw_frame = pd.DataFrame(draws)
        draw_frame.insert(0, "replicate", np.arange(1000))
        draw_frame.to_csv(output / f"{direction}_bootstrap_draws.csv", index=False)
        for row in metric_rows:
            if row["direction"] == direction and row["partition"] == "external":
                low, high = external.percentile_ci(draws[row["model"]])
                row.update(utility_ci_lower=low, utility_ci_upper=high)
        for family in FAMILIES:
            baseline = f"{PREFIX[family]}_V0"
            for representation in ["V1", "V2"]:
                name = f"{PREFIX[family]}_{representation}"
                low, high = external.percentile_ci(draws[name] - draws[baseline])
                contrast_rows.append({
                    "direction": direction,
                    "contrast": f"{name} minus {baseline}",
                    "utility_difference":
                        external_utilities[name] - external_utilities[baseline],
                    "ci_lower": low, "ci_upper": high,
                })
        pd.DataFrame(metric_rows).to_csv(output / "metrics_all.csv", index=False)
        pd.DataFrame(contrast_rows).to_csv(
            output / "paired_contrasts.csv", index=False
        )
        del external_labels, external_predictions, draws, draw_frame
        gc.collect()

    all_metrics = pd.DataFrame(metric_rows)
    external_metrics = all_metrics.loc[
        all_metrics["partition"] == "external"
    ].copy()
    external_metrics.to_csv(output / "external_metrics.csv", index=False)
    internal = all_metrics.loc[
        all_metrics["partition"] == "internal_test",
        ["direction", "model", "utility"],
    ].rename(columns={"utility": "internal_utility"})
    gaps = external_metrics[["direction", "model", "utility"]].merge(
        internal, on=["direction", "model"], validate="one_to_one"
    )
    gaps["internal_minus_external_utility"] = gaps["internal_utility"] - gaps["utility"]
    gaps.to_csv(output / "transfer_gaps.csv", index=False)
    helpers.write_json(output / "input_fingerprints.json", fingerprints)
    helpers.write_json(output / "completion.json", {
        "status": "complete",
        "external_comparisons": len(external_metrics),
        "paired_contrasts": len(contrast_rows),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print("\nEXTERNAL RESULTS", flush=True)
    print(external_metrics[[
        "direction", "model", "threshold", "utility",
        "utility_ci_lower", "utility_ci_upper", "auroc", "auprc", "brier",
        "calibration_success",
    ]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPAIRED UTILITY DIFFERENCES", flush=True)
    print(pd.DataFrame(contrast_rows).to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print(f"\nEVALUATION COMPLETE\nResults: {output}", flush=True)


if __name__ == "__main__":
    main()
