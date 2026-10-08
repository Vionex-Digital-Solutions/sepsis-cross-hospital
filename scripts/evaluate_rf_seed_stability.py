"""Descriptive RF seed sensitivity on the original fixed patient split."""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sepsis_cross_hospital.evaluation.metrics import compute_challenge_utility
from sepsis_cross_hospital.evaluation.thresholding import select_utility_threshold

DIRECTIONS = ["A_to_B", "B_to_A"]
REPRESENTATIONS = ["V0", "V1", "V2"]
SEEDS = [1729, 2718, 31415, 57721, 65537]
SETTINGS = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    n_jobs=8,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, required=True,
                    help="Project root containing data/processed and data/metadata.")
parser.add_argument("--model-root", type=Path, required=True,
                    help="Directory containing the six RF model directories.")
parser.add_argument("--ensemble-protocol", type=Path, required=True,
                    help="Completed RF ensemble evaluation's frozen_protocol.json.")
parser.add_argument("--output-root", type=Path, default=REPO / "runs")
args = parser.parse_args()

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

reference_path = REPO / "expected_outputs/frozen_data_hashes.json"
expected = json.loads(reference_path.read_text(encoding="utf-8"))
ensemble = json.loads(args.ensemble_protocol.read_text(encoding="utf-8"))

if ensemble["settings"] != SETTINGS or ensemble["seeds"] != SEEDS:
    raise ValueError("RF ensemble protocol differs from the fixed configuration.")
if ensemble["sklearn_version"] != sklearn.__version__:
    raise ValueError("Use the same scikit-learn version as the saved RF models.")

stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
output = args.output_root / f"rf-seed-stability-{stamp}"
output.mkdir(parents=True, exist_ok=False)

schemas = {}
input_hashes = {}
model_hashes = {}
thresholds = {}
source_rows = []

def load_frame(direction, partition):
    path = args.data_root / f"data/processed/{direction}/{partition}.parquet"
    file_hash = digest(path)
    if file_hash != expected["processed"][direction][partition]:
        raise ValueError(f"Frozen data hash mismatch: {direction}/{partition}")
    input_hashes[f"{direction}/{partition}"] = file_hash

    columns = ["patient_id", "hour_index", "SepsisLabel"]
    columns += schemas[direction]["V2"]
    frame = pd.read_parquet(
        path, columns=list(dict.fromkeys(columns))
    ).sort_values(["patient_id", "hour_index"]).reset_index(drop=True)

    if frame.empty or frame.duplicated(["patient_id", "hour_index"]).any():
        raise ValueError("Empty data or duplicate patient-hours.")
    if not np.isin(frame["SepsisLabel"].to_numpy(), [0, 1]).all():
        raise ValueError("Invalid binary labels.")
    sequential = frame.groupby("patient_id", sort=False).cumcount()
    if not np.array_equal(frame["hour_index"].to_numpy(), sequential.to_numpy()):
        raise ValueError("Patient hours are not sequential from zero.")

    source, target = direction.split("_to_")
    if partition == "validation":
        patient_count = expected["split"][source]["validation"]
        if frame["patient_id"].nunique() != patient_count:
            raise ValueError("Source-validation patient count mismatch.")
    elif partition == "external":
        cohort = expected["raw_data"][target]
        if (len(frame) != cohort["hours"]
                or frame["patient_id"].nunique() != cohort["patients"]):
            raise ValueError("External cohort size mismatch.")
    return frame

def sequences(frame, values):
    ids = frame["patient_id"].astype(str).to_numpy()
    edges = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    return [values[a:b] for a, b in zip(edges[:-1], edges[1:])]

def predict(direction, representation, seed, frame):
    key = f"{direction}_{representation}/seed_{seed}.joblib"
    path = args.model_root / key
    file_hash = digest(path)
    if file_hash != ensemble["model_sha256"][key]:
        raise ValueError(f"Model differs from the ensemble evaluation: {key}")
    model_hashes[key] = file_hash

    saved = joblib.load(path)
    features = schemas[direction][representation]
    parameters = dict(SETTINGS, random_state=seed)
    if (saved["features"] != features
            or saved["parameters"] != parameters
            or saved["direction"] != direction
            or saved["representation"] != representation
            or saved["sklearn_version"] != sklearn.__version__):
        raise ValueError(f"Model metadata mismatch: {key}")

    train_hash = saved.get("train_sha256")
    if train_hash is None:
        allowed = ensemble["pilot_without_recorded_train_hash"]
        if key != "A_to_B_V0/seed_1729.joblib" or key not in allowed:
            raise ValueError(f"Missing training fingerprint: {key}")
    elif train_hash != expected["processed"][direction]["train"]:
        raise ValueError(f"Training fingerprint mismatch: {key}")

    model = saved["model"]
    actual = model.get_params()
    if (not isinstance(model, RandomForestClassifier)
            or any(actual[k] != v for k, v in parameters.items())
            or model.n_features_in_ != len(features)
            or len(model.estimators_) != SETTINGS["n_estimators"]
            or not np.array_equal(model.classes_, [0, 1])):
        raise ValueError(f"Estimator mismatch: {key}")

    matrix = frame[features].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite processed features.")
    probability = model.predict_proba(matrix)[:, 1]
    if (not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Invalid probabilities.")
    return probability

# Freeze every individual model's threshold before reading external data.
for direction in DIRECTIONS:
    schema_path = (
        args.data_root
        / f"data/metadata/preprocessing/{direction}/feature_schema.json"
    )
    schemas[direction] = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_hash = digest(schema_path)
    if schema_hash != ensemble["source_input_sha256"][f"{direction}/feature_schema"]:
        raise ValueError("Feature schema differs from the RF ensemble evaluation.")
    input_hashes[f"{direction}/feature_schema"] = schema_hash

    for representation, size in {"V0": 40, "V1": 74, "V2": 142}.items():
        if len(schemas[direction][representation]) != size:
            raise ValueError("Unexpected feature count.")

    train_path = args.data_root / f"data/processed/{direction}/train.parquet"
    train_hash = digest(train_path)
    if (train_hash != expected["processed"][direction]["train"]
            or train_hash != ensemble["source_train_sha256"][direction]):
        raise ValueError("Source-training data changed.")
    input_hashes[f"{direction}/train"] = train_hash

    frame = load_frame(direction, "validation")
    y = frame["SepsisLabel"].to_numpy(dtype=np.int8)
    labels = sequences(frame, y)
    thresholds[direction] = {}

    for representation in REPRESENTATIONS:
        thresholds[direction][representation] = {}
        for seed in SEEDS:
            probability = predict(direction, representation, seed, frame)
            result = select_utility_threshold(
                labels, sequences(frame, probability)
            )
            threshold = float(result.threshold)
            thresholds[direction][representation][str(seed)] = threshold
            source_rows.append({
                "direction": direction,
                "representation": representation,
                "seed": seed,
                "threshold": threshold,
                "source_validation_utility": float(result.normalized_utility),
            })
            print(
                f"Source: {direction} {representation} seed={seed}, "
                f"threshold={threshold:.3f}", flush=True
            )
    del frame, labels, probability
    gc.collect()

pd.DataFrame(source_rows).to_csv(output / "source_thresholds.csv", index=False)
code_paths = [
    Path(__file__).resolve(),
    REPO / "src/sepsis_cross_hospital/evaluation/metrics.py",
    REPO / "src/sepsis_cross_hospital/evaluation/thresholding.py",
]
write_json(output / "frozen_protocol.json", {
    "analysis": "Additional descriptive RF seed sensitivity analysis",
    "settings": SETTINGS,
    "seeds": SEEDS,
    "source_thresholds": thresholds,
    "threshold_selection": "Per model, source-validation utility; original grid and tie-break",
    "variation_scope": "Estimator randomness and source-selected thresholds; fixed patient split",
    "seed_summary": "Descriptive mean, sample SD, range, and sign counts; not confidence intervals",
    "source_input_sha256": input_hashes.copy(),
    "model_sha256": model_hashes.copy(),
    "ensemble_protocol_sha256": digest(args.ensemble_protocol),
    "frozen_data_reference_sha256": digest(reference_path),
    "pilot_without_recorded_train_hash": ensemble["pilot_without_recorded_train_hash"],
    "code_sha256": {
        str(path.relative_to(REPO)).replace("\\", "/"): digest(path)
        for path in code_paths
    },
    "versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    },
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
})
print("\nALL 30 SOURCE THRESHOLDS FROZEN. Starting external scoring.", flush=True)

metric_rows = []
for direction in DIRECTIONS:
    frame = load_frame(direction, "external")
    y = frame["SepsisLabel"].to_numpy(dtype=np.int8)
    labels = sequences(frame, y)

    for representation in REPRESENTATIONS:
        for seed in SEEDS:
            probability = predict(direction, representation, seed, frame)
            threshold = thresholds[direction][representation][str(seed)]
            alert = (probability >= threshold).astype(np.int8)
            utility = float(
                compute_challenge_utility(labels, sequences(frame, alert)).normalized
            )
            metric_rows.append({
                "direction": direction,
                "representation": representation,
                "seed": seed,
                "threshold": threshold,
                "utility": utility,
                "auroc": float(roc_auc_score(y, probability)),
                "auprc": float(average_precision_score(y, probability)),
                "brier": float(brier_score_loss(y, probability)),
            })
            print(
                f"External: {direction} {representation} seed={seed}, "
                f"utility={utility:.6f}", flush=True
            )
    pd.DataFrame(metric_rows).to_csv(output / "seed_metrics.csv", index=False)
    del frame, labels, probability, alert
    gc.collect()

metrics = pd.DataFrame(metric_rows)
utilities = metrics.set_index(["direction", "seed", "representation"])["utility"]
contrasts = []
for direction in DIRECTIONS:
    for representation in ["V1", "V2"]:
        for seed in SEEDS:
            contrasts.append({
                "direction": direction,
                "contrast": f"{representation} minus V0",
                "seed": seed,
                "utility_difference": float(
                    utilities.loc[(direction, seed, representation)]
                    - utilities.loc[(direction, seed, "V0")]
                ),
            })

contrasts = pd.DataFrame(contrasts)
contrasts.to_csv(output / "seed_contrasts.csv", index=False)
summary = contrasts.groupby(
    ["direction", "contrast"], sort=False
)["utility_difference"].agg(
    n_seeds="size",
    mean="mean",
    sample_sd="std",
    minimum="min",
    maximum="max",
    positive_seeds=lambda values: int((values > 0).sum()),
    negative_seeds=lambda values: int((values < 0).sum()),
).reset_index()
summary.to_csv(output / "seed_summary.csv", index=False)

if len(metrics) != 30 or len(contrasts) != 20:
    raise ValueError("Incomplete seed analysis.")
write_json(output / "input_fingerprints.json", input_hashes)
write_json(output / "completion.json", {
    "status": "complete",
    "external_seed_evaluations": len(metrics),
    "paired_seed_contrasts": len(contrasts),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
})

print("\nSEED STABILITY SUMMARY — descriptive, not confidence intervals", flush=True)
print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print(f"\nSEED CHECK COMPLETE\nResults: {output}", flush=True)
