"""Summarize completed evaluations; calculate exploratory V2-minus-V1 contrasts."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DIRECTIONS = ["A_to_B", "B_to_A"]
METRICS = ["utility", "auroc", "auprc", "brier", "threshold"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-evaluation", required=True, type=Path)
    parser.add_argument("--rf-evaluation", required=True, type=Path)
    parser.add_argument("--extra-mlp-evaluation", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO / "runs")
    args = parser.parse_args()

    groups = [
        ("original", args.original_evaluation, ["XGB"], 14),
        ("rf", args.rf_evaluation, ["RF"], 6),
        ("extra_mlp", args.extra_mlp_evaluation, ["ET", "MLP"], 12),
    ]
    external_rows = []
    contrast_rows = []
    fingerprints = {}

    for group, root, prefixes, expected_rows in groups:
        if group == "original":
            metadata = read_json(root / "run_metadata.json")
            if (
                metadata.get("external_evaluated") is not True
                or metadata["bootstrap_replicates"] != 1000
                or metadata["bootstrap_seed"] != 1729
            ):
                raise ValueError("Original evaluation is incomplete or incompatible.")
            metric_path = root / "external_metrics_all.csv"
        else:
            completion = read_json(root / "completion.json")
            protocol = read_json(root / "frozen_protocol.json")
            if (
                completion.get("status") != "complete"
                or completion["external_comparisons"] != expected_rows
                or protocol["bootstrap_replicates"] != 1000
                or protocol["bootstrap_seed"] != 1729
            ):
                raise ValueError(f"Incomplete or incompatible evaluation: {group}")
            metric_path = root / "external_metrics.csv"

        scores = pd.read_csv(metric_path)
        if (
            len(scores) != expected_rows
            or scores.duplicated(["direction", "model"]).any()
            or set(scores["direction"]) != set(DIRECTIONS)
            or not np.isfinite(scores[METRICS].to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"Invalid metric table: {group}")
        scores = scores.set_index(["direction", "model"])
        fingerprints[f"{group}/external_metrics"] = sha256(metric_path)

        if group == "original":
            reference = pd.read_csv(
                REPO / "expected_outputs/external_metrics.csv"
            ).set_index(["direction", "model"])
            if set(scores.index) != set(reference.index):
                raise ValueError("Original model inventory differs from reference.")
            np.testing.assert_allclose(
                scores.loc[reference.index, METRICS].to_numpy(dtype=float),
                reference[METRICS].to_numpy(dtype=float),
                rtol=0, atol=1e-12,
            )

        for direction in DIRECTIONS:
            draw_path = (
                root / direction / "bootstrap_replicates.csv"
                if group == "original"
                else root / f"{direction}_bootstrap_draws.csv"
            )
            draws = pd.read_csv(draw_path)
            if (
                len(draws) != 1000
                or not np.array_equal(draws["replicate"].to_numpy(), np.arange(1000))
            ):
                raise ValueError(f"Invalid bootstrap replicate alignment: {draw_path}")
            fingerprints[f"{group}/{direction}/bootstrap_draws"] = sha256(draw_path)

            for model, score in scores.loc[direction].iterrows():
                values = draws[model].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite bootstrap draws: {model}")
                low, high = np.percentile(values, [2.5, 97.5])
                external_rows.append({
                    "direction": direction,
                    "model": model,
                    "analysis_group": group,
                    **{metric: float(score[metric]) for metric in METRICS},
                    "utility_ci_lower": float(low),
                    "utility_ci_upper": float(high),
                })

            for prefix in prefixes:
                for left_rep, right_rep in [
                    ("V1", "V0"), ("V2", "V0"), ("V2", "V1")
                ]:
                    left = f"{prefix}_{left_rep}"
                    right = f"{prefix}_{right_rep}"
                    differences = (
                        draws[left].to_numpy(dtype=float)
                        - draws[right].to_numpy(dtype=float)
                    )
                    if not np.isfinite(differences).all():
                        raise ValueError("Nonfinite paired bootstrap differences.")
                    low, high = np.percentile(differences, [2.5, 97.5])
                    contrast_rows.append({
                        "direction": direction,
                        "classifier": prefix,
                        "contrast": f"{left_rep} minus {right_rep}",
                        "utility_difference": float(
                            scores.loc[(direction, left), "utility"]
                            - scores.loc[(direction, right), "utility"]
                        ),
                        "ci_lower": float(low),
                        "ci_upper": float(high),
                        "exploratory_follow_up": right_rep == "V1",
                    })

    combined = pd.DataFrame(external_rows)
    contrasts = pd.DataFrame(contrast_rows)
    if (
        len(combined) != 32
        or combined.duplicated(["direction", "model"]).any()
        or len(contrasts) != 24
    ):
        raise ValueError("Unexpected combined result inventory.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = args.output_root / f"revision-summary-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output / "external_comparison.csv", index=False)
    contrasts.to_csv(output / "feature_contrasts.csv", index=False)
    (output / "completion.json").write_text(
        json.dumps({
            "status": "complete",
            "external_results": 32,
            "feature_contrasts": 24,
            "exploratory_incremental_contrasts": 8,
            "multiplicity_adjustment": "None",
            "uncertainty": "Patient bootstrap conditional on fixed trained models",
            "input_sha256": fingerprints,
            "summary_script_sha256": sha256(Path(__file__).resolve()),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nEXPLORATORY V2 MINUS V1", flush=True)
    print(contrasts.loc[contrasts["exploratory_follow_up"], [
        "direction", "classifier", "utility_difference", "ci_lower", "ci_upper"
    ]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSUMMARY COMPLETE\nResults: {output}", flush=True)


if __name__ == "__main__":
    main()
