from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INTERNAL_EXPECTED = Path(
    "expected_outputs/internal_metrics.csv"
)

EXTERNAL_EXPECTED = Path(
    "expected_outputs/external_metrics.csv"
)

BOOTSTRAP_EXPECTED = Path(
    "expected_outputs/bootstrap_summary.csv"
)


DEFAULT_EXTERNAL_RUN = Path(
    "runs/reproduction__external"
)


INTERNAL_COLUMNS = (
    "utility",
    "auroc",
    "auprc",
    "brier",
    "threshold",
)


EXTERNAL_COLUMNS = (
    "utility",
    "auroc",
    "auprc",
    "brier",
    "threshold",
    "calibration_intercept",
    "calibration_slope",
)


BOOTSTRAP_COLUMNS = (
    "point_estimate",
    "ci_2_5",
    "ci_97_5",
)


ATOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify reproduced internal and "
            "external results against frozen "
            "reference outputs."
        )
    )

    parser.add_argument(
        "--external-run",
        type=Path,
        default=DEFAULT_EXTERNAL_RUN,
        help=(
            "Directory containing the completed "
            "atomic external evaluation."
        ),
    )

    return parser.parse_args()


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            path
        )


def compare_numeric(
    *,
    label: str,
    expected: float,
    actual: float,
) -> None:
    if not np.isclose(
        float(actual),
        float(expected),
        rtol=0.0,
        atol=ATOL,
        equal_nan=True,
    ):
        raise RuntimeError(
            f"{label}: expected {expected}, "
            f"found {actual}."
        )


def verify_internal() -> None:
    require_file(
        INTERNAL_EXPECTED
    )

    expected = pd.read_csv(
        INTERNAL_EXPECTED
    )

    print()
    print(
        "SOURCE INTERNAL RESULTS"
    )
    print("-" * 72)

    for row in expected.itertuples(
        index=False
    ):
        metadata_path = (
            Path("runs")
            / row.source_run
            / "run_metadata.json"
        )

        require_file(
            metadata_path
        )

        metadata = pd.read_json(
            metadata_path,
            typ="series",
        )

        metrics = metadata[
            "internal_test_metrics"
        ]

        for column in INTERNAL_COLUMNS:
            compare_numeric(
                label=(
                    f"{row.source_run} "
                    f"{column}"
                ),
                expected=getattr(
                    row,
                    column,
                ),
                actual=metrics[
                    column
                ],
            )

        print(
            f"  {row.source_run:<24} PASS"
        )


def verify_external(
    external_run: Path,
) -> None:
    require_file(
        EXTERNAL_EXPECTED
    )

    actual_path = (
        external_run
        / "external_metrics_all.csv"
    )

    require_file(
        actual_path
    )

    expected = pd.read_csv(
        EXTERNAL_EXPECTED
    )

    actual = pd.read_csv(
        actual_path
    )

    keys = [
        "direction",
        "model",
        "source_run",
    ]

    expected = expected.set_index(
        keys
    ).sort_index()

    actual = actual.set_index(
        keys
    ).sort_index()

    if not expected.index.equals(
        actual.index
    ):
        raise RuntimeError(
            "External result row identities "
            "do not match frozen references."
        )

    print()
    print(
        "EXTERNAL RESULTS"
    )
    print("-" * 72)

    for key in expected.index:
        for column in EXTERNAL_COLUMNS:
            compare_numeric(
                label=(
                    f"{key} {column}"
                ),
                expected=expected.loc[
                    key,
                    column,
                ],
                actual=actual.loc[
                    key,
                    column,
                ],
            )

        print(
            f"  {key[0]:<6} "
            f"{key[1]:<12} PASS"
        )


def verify_bootstrap(
    external_run: Path,
) -> None:
    require_file(
        BOOTSTRAP_EXPECTED
    )

    actual_path = (
        external_run
        / "bootstrap_summary_all.csv"
    )

    require_file(
        actual_path
    )

    expected = pd.read_csv(
        BOOTSTRAP_EXPECTED
    )

    actual = pd.read_csv(
        actual_path
    )

    keys = [
        "direction",
        "quantity",
        "type",
    ]

    expected = expected.set_index(
        keys
    ).sort_index()

    actual = actual.set_index(
        keys
    ).sort_index()

    if not expected.index.equals(
        actual.index
    ):
        raise RuntimeError(
            "Bootstrap summary row identities "
            "do not match frozen references."
        )

    print()
    print(
        "BOOTSTRAP RESULTS"
    )
    print("-" * 72)

    for key in expected.index:
        for column in BOOTSTRAP_COLUMNS:
            compare_numeric(
                label=(
                    f"{key} {column}"
                ),
                expected=expected.loc[
                    key,
                    column,
                ],
                actual=actual.loc[
                    key,
                    column,
                ],
            )

        print(
            f"  {key[0]:<6} "
            f"{key[1]:<24} PASS"
        )


def main() -> int:
    args = parse_args()

    print("=" * 72)
    print(
        "FROZEN RESULT VERIFICATION"
    )
    print("=" * 72)

    verify_internal()

    verify_external(
        args.external_run
    )

    verify_bootstrap(
        args.external_run
    )

    print()
    print("=" * 72)
    print(
        "ALL FROZEN RESULTS: PASS"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )