"""Verify revision aggregates against immutable numerical references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

def compare_tables(expected_path, actual_path, keys, atol=1e-10):
    expected = pd.read_csv(expected_path)
    actual = pd.read_csv(actual_path)
    if set(expected.columns) != set(actual.columns):
        raise ValueError("Column inventory mismatch.")
    if expected.duplicated(keys).any() or actual.duplicated(keys).any():
        raise ValueError("Duplicate comparison keys.")
    expected = expected.set_index(keys)
    actual = actual.set_index(keys)
    if set(expected.index) != set(actual.index):
        raise ValueError("Result inventory mismatch.")
    actual = actual.loc[expected.index]
    for column in expected.columns:
        if column.endswith("_message"):
            continue
        if pd.api.types.is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(
                actual[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=0, atol=atol, equal_nan=True,
                err_msg=f"Numerical mismatch: {column}",
            )
        elif (
            actual[column].fillna("<missing>").astype(str).tolist()
            != expected[column].fillna("<missing>").astype(str).tolist()
        ):
            raise ValueError(f"Descriptor mismatch: {column}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-evaluation", required=True, type=Path)
    parser.add_argument("--extra-mlp-evaluation", required=True, type=Path)
    parser.add_argument("--rf-seed-evaluation", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    reference = REPO / "expected_outputs/revision"
    manifest = json.loads((reference / "reference_manifest.json").read_text())
    for name, expected_hash in manifest["files_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Invalid reference filename.")
        actual_hash = hashlib.sha256((reference / name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen reference modified: {name}")
    groups = [
        (args.rf_evaluation, {"external_comparisons": 6, "paired_contrasts": 4}),
        (args.extra_mlp_evaluation, {"external_comparisons": 12, "paired_contrasts": 8}),
        (args.rf_seed_evaluation, {
            "external_seed_evaluations": 30, "paired_seed_contrasts": 20}),
        (args.summary, {
            "external_results": 32, "feature_contrasts": 24,
            "exploratory_incremental_contrasts": 8}),
    ]
    for root, counts in groups:
        completion = json.loads((root / "completion.json").read_text())
        if completion.get("status") != "complete" or any(
            completion.get(key) != value for key, value in counts.items()
        ):
            raise ValueError(f"Incomplete result group: {root}")
    pairs = [
        ("external_comparison.csv", args.summary / "external_comparison.csv",
         ["direction", "model"]),
        ("feature_contrasts.csv", args.summary / "feature_contrasts.csv",
         ["direction", "classifier", "contrast"]),
    ]
    for prefix, root in [
        ("rf", args.rf_evaluation), ("et_mlp", args.extra_mlp_evaluation)
    ]:
        for filename, keys in [
            ("metrics_all.csv", ["direction", "partition", "model"]),
            ("paired_contrasts.csv", ["direction", "contrast"]),
            ("transfer_gaps.csv", ["direction", "model"]),
        ]:
            pairs.append((f"{prefix}_{filename}", root / filename, keys))
    pairs.append((
        "et_mlp_source_thresholds.csv",
        args.extra_mlp_evaluation / "source_thresholds.csv",
        ["direction", "model"],
    ))
    for filename, keys in [
        ("seed_metrics.csv", ["direction", "representation", "seed"]),
        ("seed_contrasts.csv", ["direction", "contrast", "seed"]),
        ("seed_summary.csv", ["direction", "contrast"]),
        ("source_thresholds.csv", ["direction", "representation", "seed"]),
    ]:
        pairs.append((f"rf_{filename}", args.rf_seed_evaluation / filename, keys))
    for name, actual, keys in pairs:
        compare_tables(reference / name, actual, keys)
    print("ALL REVISION RESULTS: PASS", flush=True)

if __name__ == "__main__":
    main()
