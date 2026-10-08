import os
for thread_variable in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[thread_variable] = "1"
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from contextlib import nullcontext

import argparse
REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Source-only RF ensemble evaluation.")
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--model-root", type=Path,
                    default=REPO / "runs/revision_tree_models/rf")
parser.add_argument("--output-root", type=Path, default=REPO / "runs")
args = parser.parse_args()
base = args.data_root
sys.path.insert(0, str(REPO / "src"))
reference_path = REPO / "expected_outputs/frozen_data_hashes.json"
reference = json.loads(reference_path.read_text(encoding="utf-8"))
tree_config = json.loads(
    (REPO / "configs/revision_tree_protocol.json").read_text(encoding="utf-8")
)
legacy_reference = json.loads(
    (REPO / "expected_outputs/revision/rf_frozen_protocol.json").read_text(
        encoding="utf-8"
    )
)
if sklearn.__version__ != tree_config["sklearn_version"]:
    raise ValueError("Use the pinned scikit-learn version.")

from sepsis_cross_hospital.evaluation.thresholding import select_utility_threshold
from sepsis_cross_hospital.evaluation.metrics import compute_challenge_utility
from sepsis_cross_hospital.evaluation.external import (
    calibration_intercept_slope,
    paired_patient_bootstrap,
    percentile_ci,
)

directions = ["A_to_B", "B_to_A"]
representations = ["V0", "V1", "V2"]
seeds = [1729, 2718, 31415, 57721, 65537]
settings = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    n_jobs=8,
)

if settings != dict(
    tree_config["common_parameters"], bootstrap=tree_config["bootstrap"]["rf"]
):
    raise ValueError("RF configuration differs from the public training protocol.")

expected_external = {
    "A_to_B": (
        20000, 761995,
        "eef0c3183e8d251e74d83a784c84d76b"
        "db307867d3ad65994cc65ab4e01daec6",
    ),
    "B_to_A": (
        20336, 790215,
        "38d0ec27765ae5b1393cb2d2caa414bb"
        "3e0499d102d482ef1fa25f97c467974b",
    ),
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
output = args.output_root / f"rf-ensembles-{stamp}"
output.mkdir(parents=True, exist_ok=False)

schemas = {}
thresholds = {}
train_hashes = {}
weight_hashes = {}
input_hashes = {}
pilot_without_train_hash = []

def load_frame(direction, partition):
    path = base / f"data/processed/{direction}/{partition}.parquet"
    digest = sha256(path)
    if digest != reference["processed"][direction][partition]:
        raise ValueError(f"Frozen processed-data hash mismatch: {direction}/{partition}")
    key = f"{direction}/{partition}"
    if key in input_hashes and input_hashes[key] != digest:
        raise ValueError(f"Input changed during evaluation: {key}")
    input_hashes[key] = digest

    columns = ["patient_id", "hour_index", "SepsisLabel"]
    columns += schemas[direction]["V2"]
    frame = pd.read_parquet(
        path, columns=list(dict.fromkeys(columns))
    ).sort_values(["patient_id", "hour_index"]).reset_index(drop=True)

    if frame.empty or frame.duplicated(["patient_id", "hour_index"]).any():
        raise ValueError(f"Empty or duplicate patient-hours: {key}")
    if not np.isin(frame["SepsisLabel"].to_numpy(), [0, 1]).all():
        raise ValueError(f"Invalid labels: {key}")

    expected_hours = frame.groupby("patient_id", sort=False).cumcount()
    if not np.array_equal(frame["hour_index"].to_numpy(), expected_hours.to_numpy()):
        raise ValueError(f"Nonsequential patient hours: {key}")

    source, target = direction.split("_to_")
    expected_patients = (
        reference["raw_data"][target]["patients"] if partition == "external"
        else reference["split"][source][partition]
    )
    if frame["patient_id"].nunique() != expected_patients:
        raise ValueError(f"Patient count mismatch: {key}")
    if partition == "external":
        patients, hours, expected_hash = expected_external[direction]
        if (
            digest != expected_hash
            or len(frame) != hours
            or frame["patient_id"].nunique() != patients
        ):
            raise ValueError(f"External cohort does not match the original: {direction}")
    return frame

def patient_sequences(frame, values):
    ids = frame["patient_id"].astype(str).to_numpy()
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    return [values[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]

def ensemble_probability(direction, representation, frame):
    features = schemas[direction][representation]
    X = frame[features].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite processed features.")

    probability = np.zeros(len(frame), dtype=np.float64)
    for seed in seeds:
        key = f"{direction}_{representation}/seed_{seed}.joblib"
        path = args.model_root / key
        digest = sha256(path)
        if key in weight_hashes and weight_hashes[key] != digest:
            raise ValueError(f"Model changed during evaluation: {key}")
        weight_hashes[key] = digest

        saved = joblib.load(path)
        parameters = dict(settings, random_state=seed)
        if (
            saved["features"] != features
            or saved["parameters"] != parameters
            or saved["direction"] != direction
            or saved["representation"] != representation
            or saved["sklearn_version"] != sklearn.__version__
        ):
            raise ValueError(f"Model metadata mismatch: {key}")

        recorded_hash = saved.get("train_sha256")
        if recorded_hash is None:
            if (
                key != "A_to_B_V0/seed_1729.joblib"
                or digest != legacy_reference["model_sha256"].get(key)
            ):
                raise ValueError(f"Missing training fingerprint: {key}")
            if key not in pilot_without_train_hash:
                pilot_without_train_hash.append(key)
        elif recorded_hash != train_hashes[direction]:
            raise ValueError(f"Training data fingerprint mismatch: {key}")

        model = saved["model"]
        actual = model.get_params()
        if (
            not isinstance(model, RandomForestClassifier)
            or any(actual[k] != v for k, v in parameters.items())
            or model.n_features_in_ != len(features)
            or len(model.estimators_) != 200
            or not np.array_equal(model.classes_, [0, 1])
        ):
            raise ValueError(f"Estimator mismatch: {key}")

        probability += model.predict_proba(X)[:, 1]
        del model, saved

    probability /= len(seeds)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid ensemble probabilities.")
    return probability

# Complete source-only threshold selection for BOTH directions first.
for direction in directions:
    schema_path = base / f"data/metadata/preprocessing/{direction}/feature_schema.json"
    schemas[direction] = json.loads(schema_path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        schemas[direction], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != tree_config["feature_schema_canonical_sha256"]:
        raise ValueError("Feature schema differs from the frozen study.")
    input_hashes[f"{direction}/feature_schema"] = sha256(schema_path)
    train_hashes[direction] = sha256(
        base / f"data/processed/{direction}/train.parquet"
    )
    if train_hashes[direction] != reference["processed"][direction]["train"]:
        raise ValueError("Source-training fingerprint differs from the frozen study.")
    frame = load_frame(direction, "validation")
    y = frame["SepsisLabel"].to_numpy(dtype=np.int8)
    labels = patient_sequences(frame, y)
    thresholds[direction] = {}

    predictions = frame[["patient_id", "hour_index", "SepsisLabel"]].copy()
    for representation in representations:
        print(f"Source validation: {direction} {representation}", flush=True)
        probability = ensemble_probability(direction, representation, frame)
        result = select_utility_threshold(
            labels, patient_sequences(frame, probability)
        )
        thresholds[direction][representation] = {
            "threshold": float(result.threshold),
            "source_validation_utility": float(result.normalized_utility),
        }
        predictions[f"RF_{representation}_probability"] = probability
        print(f"Selected threshold: {result.threshold:.3f}", flush=True)

    predictions.to_parquet(
        output / f"{direction}_validation_predictions.parquet", index=False
    )
    del frame, predictions, labels, probability
    gc.collect()

protocol = {
    "analysis": "Additional exploratory Random Forest revision analysis",
    "data_reference_sha256": sha256(reference_path),
    "settings": settings,
    "seeds": seeds,
    "probability_combination": "Equal average across all five forests",
    "source_thresholds": thresholds,
    "threshold_grid": "0.001 through 0.999, step 0.001; higher threshold on ties",
    "bootstrap_replicates": 1000,
    "bootstrap_seed": 1729,
    "calibration_usage": "Evaluation diagnostics only; predictions unchanged",
    "source_train_sha256": train_hashes,
    "source_input_sha256": input_hashes.copy(),
    "model_sha256": weight_hashes.copy(),
    "pilot_without_recorded_train_hash": pilot_without_train_hash,
    "evaluation_script_sha256": sha256(Path(__file__).resolve()),
    "sklearn_version": sklearn.__version__,
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
}
write_json(output / "frozen_protocol.json", protocol)
print("ALL SIX THRESHOLDS FROZEN. Starting held-out evaluation.", flush=True)

metric_rows = []
contrast_rows = []

for direction in directions:
    external_labels = None
    external_predictions = {}
    external_utilities = {}

    for partition in ["internal_test", "external"]:
        frame = load_frame(direction, partition)
        y = frame["SepsisLabel"].to_numpy(dtype=np.int8)
        labels = patient_sequences(frame, y)
        predictions = frame[["patient_id", "hour_index", "SepsisLabel"]].copy()

        for representation in representations:
            name = f"RF_{representation}"
            print(f"Evaluating: {direction} {partition} {name}", flush=True)
            probability = ensemble_probability(direction, representation, frame)
            threshold = thresholds[direction][representation]["threshold"]
            binary = (probability >= threshold).astype(np.int8)
            binary_sequences = patient_sequences(frame, binary)

            utility = float(
                compute_challenge_utility(labels, binary_sequences).normalized
            )
            with nullcontext():
                calibration = calibration_intercept_slope(y, probability)

            metric_rows.append({
                "direction": direction,
                "partition": partition,
                "model": name,
                "patients": len(labels),
                "hours": len(frame),
                "threshold": threshold,
                "utility": utility,
                "auroc": float(roc_auc_score(y, probability)),
                "auprc": float(average_precision_score(y, probability)),
                "brier": float(brier_score_loss(y, probability)),
                "calibration_intercept": (
                    calibration.intercept if calibration.success else None
                ),
                "calibration_slope": (
                    calibration.slope if calibration.success else None
                ),
                "calibration_success": calibration.success,
                "calibration_message": calibration.message,
            })
            predictions[f"{name}_probability"] = probability
            predictions[f"{name}_alert"] = binary

            if partition == "external":
                external_labels = labels
                external_predictions[name] = binary_sequences
                external_utilities[name] = utility

        predictions.to_parquet(
            output / f"{direction}_{partition}_predictions.parquet", index=False
        )
        pd.DataFrame(metric_rows).to_csv(output / "metrics_all.csv", index=False)
        del frame, predictions, probability
        gc.collect()

    print(f"Patient bootstrap: {direction}, 1000 replicates", flush=True)
    with nullcontext():
        draws = paired_patient_bootstrap(
            external_labels,
            external_predictions,
            n_bootstrap=1000,
            seed=1729,
        )

    bootstrap_frame = pd.DataFrame(draws)
    bootstrap_frame.insert(0, "replicate", np.arange(1000))
    bootstrap_frame.to_csv(
        output / f"{direction}_bootstrap_draws.csv", index=False
    )

    for row in metric_rows:
        if row["direction"] == direction and row["partition"] == "external":
            low, high = percentile_ci(draws[row["model"]])
            row["utility_ci_lower"] = low
            row["utility_ci_upper"] = high

    for name in ["RF_V1", "RF_V2"]:
        difference = draws[name] - draws["RF_V0"]
        low, high = percentile_ci(difference)
        contrast_rows.append({
            "direction": direction,
            "contrast": f"{name} minus RF_V0",
            "utility_difference": external_utilities[name] - external_utilities["RF_V0"],
            "ci_lower": low,
            "ci_upper": high,
        })

    pd.DataFrame(metric_rows).to_csv(output / "metrics_all.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(output / "paired_contrasts.csv", index=False)
    del external_labels, external_predictions, draws, bootstrap_frame
    gc.collect()

metrics = pd.DataFrame(metric_rows)
external = metrics.loc[metrics["partition"] == "external"].copy()
external.to_csv(output / "external_metrics.csv", index=False)

internal = metrics.loc[metrics["partition"] == "internal_test"].set_index(
    ["direction", "model"]
)["utility"]
gaps = external[["direction", "model", "utility"]].copy()
gaps["internal_utility"] = [
    float(internal.loc[(row.direction, row.model)])
    for row in gaps.itertuples()
]
gaps["internal_minus_external_utility"] = gaps["internal_utility"] - gaps["utility"]
gaps.to_csv(output / "transfer_gaps.csv", index=False)

write_json(output / "input_fingerprints.json", input_hashes)
write_json(output / "completion.json", {
    "status": "complete",
    "external_comparisons": len(external),
    "paired_contrasts": len(contrast_rows),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
})

print("\nEXTERNAL RESULTS", flush=True)
print(external[[
    "direction", "model", "threshold", "utility",
    "utility_ci_lower", "utility_ci_upper", "auroc", "auprc", "brier",
    "calibration_success",
]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nPAIRED UTILITY DIFFERENCES", flush=True)
print(pd.DataFrame(contrast_rows).to_string(
    index=False, float_format=lambda x: f"{x:.6f}"
))
print(f"\nEVALUATION COMPLETE\nResults: {output}", flush=True)
