from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SOURCE_RUNS = (
    "A_to_B__logistic_v0",
    "A_to_B__xgb_m0",
    "A_to_B__xgb_v0",
    "A_to_B__xgb_v1",
    "A_to_B__xgb_v2",
    "A_to_B__gru",
    "A_to_B__pae",
    "B_to_A__logistic_v0",
    "B_to_A__xgb_m0",
    "B_to_A__xgb_v0",
    "B_to_A__xgb_v1",
    "B_to_A__xgb_v2",
    "B_to_A__gru",
    "B_to_A__pae",
)

EXTERNAL_RUN = "reproduction__external"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete frozen "
            "cross-hospital sepsis reproduction."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/paths.local.yaml"
        ),
        help=(
            "Machine-local PhysioNet "
            "cohort path configuration."
        ),
    )

    return parser.parse_args()


def run(
    command: list[str],
) -> None:
    print()
    print("=" * 88)
    print(
        "RUNNING:"
    )
    print(
        " ".join(command)
    )
    print("=" * 88)
    print()

    subprocess.run(
        command,
        check=True,
    )


def require_fresh_runs() -> None:
    existing = []

    for run_id in (
        *SOURCE_RUNS,
        EXTERNAL_RUN,
    ):
        path = (
            Path("runs")
            / run_id
        )

        if path.exists():
            existing.append(
                str(path)
            )

    if existing:
        formatted = "\n".join(
            f"  - {path}"
            for path in existing
        )

        raise RuntimeError(
            "Frozen reproduction run "
            "directories already exist.\n\n"
            "This driver intentionally refuses "
            "to overwrite model/evaluation "
            "artifacts.\n\n"
            "Existing paths:\n"
            f"{formatted}\n\n"
            "Remove or archive these generated "
            "run directories before starting a "
            "fresh end-to-end reproduction."
        )


def main() -> int:
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(
            "Local path configuration not found: "
            f"{args.config}\n"
            "Copy configs/paths.example.yaml to "
            "configs/paths.local.yaml and edit "
            "the two PhysioNet cohort paths."
        )

    require_fresh_runs()

    python = sys.executable
    config = str(
        args.config
    )

    print("=" * 88)
    print(
        "CROSS-HOSPITAL SEPSIS "
        "FULL REPRODUCTION"
    )
    print("=" * 88)

    # Environment and software tests.
    run(
        [
            python,
            "scripts/check_environment.py",
        ]
    )

    run(
        [
            python,
            "-m",
            "pytest",
            "-q",
        ]
    )

    # Raw public dataset verification.
    run(
        [
            python,
            "scripts/verify_data.py",
            "--config",
            config,
        ]
    )

    # Deterministic patient-level split.
    run(
        [
            python,
            "scripts/create_splits.py",
            "--config",
            config,
            "--seed",
            "1729",
        ]
    )

    # Frozen causal preprocessing.
    run(
        [
            python,
            "scripts/preprocess.py",
            "--config",
            config,
            "--direction",
            "all",
            "--batch-patients",
            "250",
            "--overwrite",
        ]
    )

    run(
        [
            python,
            "scripts/validate_processed.py",
        ]
    )

    # Logistic baselines.
    for direction in (
        "A_to_B",
        "B_to_A",
    ):
        run(
            [
                python,
                "scripts/train_logistic.py",
                "--direction",
                direction,
                "--run-id",
                f"{direction}__logistic_v0",
            ]
        )

    # XGBoost representations.
    for direction in (
        "A_to_B",
        "B_to_A",
    ):
        for representation in (
            "M0",
            "V0",
            "V1",
            "V2",
        ):
            run(
                [
                    python,
                    "scripts/train_xgboost.py",
                    "--direction",
                    direction,
                    "--representation",
                    representation,
                    "--run-id",
                    (
                        f"{direction}__xgb_"
                        f"{representation.lower()}"
                    ),
                ]
            )

    # Causal GRU.
    for direction in (
        "A_to_B",
        "B_to_A",
    ):
        run(
            [
                python,
                "scripts/train_gru.py",
                "--direction",
                direction,
                "--run-id",
                f"{direction}__gru",
            ]
        )

    # Process-Attenuated Ensemble.
    for direction in (
        "A_to_B",
        "B_to_A",
    ):
        run(
            [
                python,
                "scripts/train_pae.py",
                "--direction",
                direction,
                "--run-id",
                f"{direction}__pae",
            ]
        )

    # Verify every frozen source artifact before
    # touching the external target cohorts.
    run(
        [
            python,
            "scripts/evaluate_external.py",
            "--preflight",
        ]
    )

    # Atomic evaluation of both transfer directions.
    run(
        [
            python,
            "scripts/evaluate_external.py",
            "--run-id",
            EXTERNAL_RUN,
        ]
    )

    # Machine-readable comparison with the frozen
    # internal, external, calibration, bootstrap,
    # and paired-contrast reference outputs.
    run(
        [
            python,
            "scripts/verify_results.py",
            "--external-run",
            (
                "runs/"
                f"{EXTERNAL_RUN}"
            ),
        ]
    )

    print()
    print("=" * 88)
    print(
        "FULL REPRODUCTION: PASS"
    )
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )