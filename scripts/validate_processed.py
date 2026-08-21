from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from sepsis_cross_hospital.data.preprocessing import (
    CONTEXT_COLUMNS,
    DYNAMIC_COLUMNS,
)


METADATA_COLUMNS = [
    "patient_id",
    "cohort",
    "hour_index",
    "SepsisLabel",
]

FROZEN_PROCESSED_REFERENCE = {
    ("A_to_B", "train"): {
        "cohort": "A",
        "patients": 14235,
        "hours": 552303,
        "septic": 1253,
        "positive_hours": 11993,
        "sha256": (
            "7b20e5ef21f9db0ecdb5129a82e4223c"
            "d673686116b947a0e1e71bf226597857"
        ),
    },
    ("A_to_B", "validation"): {
        "cohort": "A",
        "patients": 3049,
        "hours": 118144,
        "septic": 268,
        "positive_hours": 2583,
        "sha256": (
            "5130f764f9c1805d591ac755c592d269"
            "dd858ec24c323809ee3cf0a31ed4a6be"
        ),
    },
    ("A_to_B", "internal_test"): {
        "cohort": "A",
        "patients": 3052,
        "hours": 119768,
        "septic": 269,
        "positive_hours": 2560,
        "sha256": (
            "38657d1dab6ca43f9b8d3ea6c6e59f8"
            "05ca1770691e36adb1bee54908a45e6cc"
        ),
    },
    ("A_to_B", "external"): {
        "cohort": "B",
        "patients": 20000,
        "hours": 761995,
        "septic": 1142,
        "positive_hours": 10780,
        "sha256": (
            "eef0c3183e8d251e74d83a784c84d76b"
            "db307867d3ad65994cc65ab4e01daec6"
        ),
    },
    ("B_to_A", "train"): {
        "cohort": "B",
        "patients": 13999,
        "hours": 531851,
        "septic": 799,
        "positive_hours": 7530,
        "sha256": (
            "2124eb7b8337cdeab7c86e1e3749c825"
            "5b427f336c115134ccda4959a27c8cef"
        ),
    },
    ("B_to_A", "validation"): {
        "cohort": "B",
        "patients": 2999,
        "hours": 114508,
        "septic": 171,
        "positive_hours": 1628,
        "sha256": (
            "a68b667e1af519366ab284b1d7343ad6"
            "cb0e56879734ea7b6901616bef1fd9b5"
        ),
    },
    ("B_to_A", "internal_test"): {
        "cohort": "B",
        "patients": 3002,
        "hours": 115636,
        "septic": 172,
        "positive_hours": 1622,
        "sha256": (
            "d01da816c3c49aa8020e52326d7089dc"
            "28a6ca17f428feccde316aaeb07668e0"
        ),
    },
    ("B_to_A", "external"): {
        "cohort": "A",
        "patients": 20336,
        "hours": 790215,
        "septic": 1790,
        "positive_hours": 17136,
        "sha256": (
            "38d0ec27765ae5b1393cb2d2caa414bb"
            "3e0499d102d482ef1fa25f97c467974b"
        ),
    },
}

FROZEN_SOURCE_UNAVAILABLE = {
    "A_to_B": ["EtCO2"],
    "B_to_A": [],
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate processed model-ready Parquet "
            "datasets before model training."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
    )

    return parser.parse_args()

def sha256_file(
    path: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()

def expected_feature_columns() -> set[str]:
    columns = set(
        DYNAMIC_COLUMNS
        + CONTEXT_COLUMNS
    )

    for column in DYNAMIC_COLUMNS:
        columns.add(
            f"obs_{column}"
        )
        columns.add(
            f"count6_{column}"
        )
        columns.add(
            f"tsl_{column}"
        )

    return columns


def load_source_unavailable(
    direction: str,
) -> list[str]:
    path = (
        Path(
            "data/metadata/"
            "preprocessing"
        )
        / direction
        / "source_statistics.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        statistics = json.load(
            handle
        )

    availability = statistics[
        "source_available_dynamic"
    ]

    return sorted(
        column
        for column, available
        in availability.items()
        if not available
    )


def validate_file(
    *,
    path: Path,
    expected_cohort: str,
    expected_patients: int,
    expected_hours: int,
    expected_septic: int,
    expected_positive_hours: int,
    unavailable: list[str],
    batch_size: int,
) -> dict:
    parquet = pq.ParquetFile(
        path
    )

    schema_columns = (
        parquet.schema_arrow.names
    )

    expected_columns = (
        set(METADATA_COLUMNS)
        | expected_feature_columns()
    )

    observed_columns = set(
        schema_columns
    )

    if observed_columns != expected_columns:
        missing = (
            expected_columns
            - observed_columns
        )

        extra = (
            observed_columns
            - expected_columns
        )

        raise RuntimeError(
            f"{path}: schema mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )

    if len(schema_columns) != 146:
        raise RuntimeError(
            f"{path}: expected 146 columns, "
            f"found {len(schema_columns)}."
        )

    patient_ids: set[str] = set()
    septic_patient_ids: set[str] = set()

    total_rows = 0
    positive_hours = 0
    total_missing = 0

    for batch in parquet.iter_batches(
        batch_size=batch_size
    ):
        frame = batch.to_pandas()

        total_rows += len(frame)

        total_missing += int(
            frame.isna().sum().sum()
        )

        cohorts = set(
            frame[
                "cohort"
            ].astype(str).unique()
        )

        if cohorts != {
            expected_cohort
        }:
            raise RuntimeError(
                f"{path}: unexpected "
                f"cohort values: "
                f"{sorted(cohorts)}"
            )

        labels = frame[
            "SepsisLabel"
        ]

        unique_labels = set(
            labels.unique().tolist()
        )

        if not unique_labels.issubset(
            {0, 1}
        ):
            raise RuntimeError(
                f"{path}: invalid labels "
                f"{sorted(unique_labels)}"
            )

        positive_mask = (
            labels == 1
        )

        positive_hours += int(
            positive_mask.sum()
        )

        patient_ids.update(
            frame[
                "patient_id"
            ].astype(str)
        )

        septic_patient_ids.update(
            frame.loc[
                positive_mask,
                "patient_id",
            ].astype(str)
        )

        # Source-unavailable variables must
        # remain completely inaccessible.
        for column in unavailable:
            if not (
                frame[column] == 0.0
            ).all():
                raise RuntimeError(
                    f"{path}: source-unavailable "
                    f"{column} has non-zero "
                    f"value channel."
                )

            if not (
                frame[
                    f"obs_{column}"
                ] == 0
            ).all():
                raise RuntimeError(
                    f"{path}: source-unavailable "
                    f"{column} exposes "
                    f"observation masks."
                )

            if not (
                frame[
                    f"count6_{column}"
                ] == 0.0
            ).all():
                raise RuntimeError(
                    f"{path}: source-unavailable "
                    f"{column} exposes "
                    f"measurement counts."
                )

            if not (
                frame[
                    f"tsl_{column}"
                ]
                == frame["ICULOS"]
            ).all():
                raise RuntimeError(
                    f"{path}: invalid TSL "
                    f"representation for "
                    f"source-unavailable "
                    f"{column}."
                )

    if total_missing != 0:
        raise RuntimeError(
            f"{path}: contains "
            f"{total_missing:,} missing values."
        )

    if total_rows != expected_hours:
        raise RuntimeError(
            f"{path}: expected "
            f"{expected_hours:,} rows, "
            f"found {total_rows:,}."
        )

    if len(patient_ids) != expected_patients:
        raise RuntimeError(
            f"{path}: expected "
            f"{expected_patients:,} patients, "
            f"found {len(patient_ids):,}."
        )

    if (
        len(septic_patient_ids)
        != expected_septic
    ):
        raise RuntimeError(
            f"{path}: expected "
            f"{expected_septic:,} septic "
            f"patients, found "
            f"{len(septic_patient_ids):,}."
        )

    if (
        positive_hours
        != expected_positive_hours
    ):
        raise RuntimeError(
            f"{path}: expected "
            f"{expected_positive_hours:,} "
            f"positive hours, found "
            f"{positive_hours:,}."
        )

    return {
        "file": str(path),
        "rows": total_rows,
        "patients": len(
            patient_ids
        ),
        "septic_patients": len(
            septic_patient_ids
        ),
        "positive_hours": (
            positive_hours
        ),
        "missing_values": (
            total_missing
        ),
        "columns": len(
            schema_columns
        ),
    }


def main() -> int:
    args = parse_args()

    manifest_path = Path(
        "data/metadata/"
        "preprocessing/"
        "processed_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = pd.read_csv(
        manifest_path
    )

    required_manifest_columns = {
        "direction",
        "partition",
        "cohort",
        "patients",
        "patient_hours",
        "septic_patients",
        "positive_label_hours",
        "file",
        "sha256",
    }

    missing_manifest_columns = (
        required_manifest_columns
        - set(manifest.columns)
    )

    if missing_manifest_columns:
        raise RuntimeError(
            "Processed manifest missing columns: "
            + ", ".join(
                sorted(
                    missing_manifest_columns
                )
            )
        )

    if len(manifest) != 8:
        raise RuntimeError(
            "Expected exactly 8 processed "
            f"partitions, found {len(manifest)}."
        )

    observed_keys = {
        (
            str(row["direction"]),
            str(row["partition"]),
        )
        for _, row in manifest.iterrows()
    }

    expected_keys = set(
        FROZEN_PROCESSED_REFERENCE
    )

    if observed_keys != expected_keys:
        raise RuntimeError(
            "Processed manifest partition set "
            "does not match frozen study reference.\n"
            f"Expected: {sorted(expected_keys)}\n"
            f"Observed: {sorted(observed_keys)}"
        )

    manifest_lookup = {
        (
            str(row["direction"]),
            str(row["partition"]),
        ): row
        for _, row in manifest.iterrows()
    }

    print("=" * 72)
    print(
        "PROCESSED DATASET VALIDATION"
    )
    print("=" * 72)

    # --------------------------------------------------
    # Source-unavailable channel contract
    # --------------------------------------------------
    for direction in (
        "A_to_B",
        "B_to_A",
    ):
        observed_unavailable = (
            load_source_unavailable(
                direction
            )
        )

        expected_unavailable = (
            FROZEN_SOURCE_UNAVAILABLE[
                direction
            ]
        )

        if (
            observed_unavailable
            != expected_unavailable
        ):
            raise RuntimeError(
                f"{direction}: source-unavailable "
                "dynamic variables do not match "
                "the frozen reference.\n"
                f"Expected: {expected_unavailable}\n"
                f"Observed: {observed_unavailable}"
            )

    patient_sets: dict[
        tuple[str, str],
        set[str],
    ] = {}

    results = []

    for direction in (
        "A_to_B",
        "B_to_A",
    ):
        unavailable = (
            FROZEN_SOURCE_UNAVAILABLE[
                direction
            ]
        )

        for partition in (
            "train",
            "validation",
            "internal_test",
            "external",
        ):
            key = (
                direction,
                partition,
            )

            expected = (
                FROZEN_PROCESSED_REFERENCE[
                    key
                ]
            )

            manifest_row = (
                manifest_lookup[
                    key
                ]
            )

            canonical_path = (
                Path("data/processed")
                / direction
                / f"{partition}.parquet"
            )

            if not canonical_path.exists():
                raise FileNotFoundError(
                    canonical_path
                )

            # ------------------------------------------
            # Verify generated manifest metadata against
            # the independent frozen reference.
            # ------------------------------------------
            manifest_checks = {
                "cohort": str(
                    manifest_row[
                        "cohort"
                    ]
                ),
                "patients": int(
                    manifest_row[
                        "patients"
                    ]
                ),
                "hours": int(
                    manifest_row[
                        "patient_hours"
                    ]
                ),
                "septic": int(
                    manifest_row[
                        "septic_patients"
                    ]
                ),
                "positive_hours": int(
                    manifest_row[
                        "positive_label_hours"
                    ]
                ),
                "sha256": str(
                    manifest_row[
                        "sha256"
                    ]
                ),
            }

            for (
                name,
                actual,
            ) in manifest_checks.items():
                expected_value = (
                    expected[
                        name
                    ]
                )

                if actual != expected_value:
                    raise RuntimeError(
                        f"{direction}/{partition}: "
                        f"manifest {name} mismatch.\n"
                        f"Expected: {expected_value}\n"
                        f"Observed: {actual}"
                    )

            # ------------------------------------------
            # Verify the actual Parquet bytes.
            # ------------------------------------------
            observed_hash = (
                sha256_file(
                    canonical_path
                )
            )

            if (
                observed_hash
                != expected["sha256"]
            ):
                raise RuntimeError(
                    f"{canonical_path}: SHA256 "
                    "does not match frozen reference.\n"
                    f"Expected: "
                    f"{expected['sha256']}\n"
                    f"Observed: {observed_hash}"
                )

            print()
            print(
                f"{direction} / "
                f"{partition}"
            )

            print(
                "  Frozen SHA256: PASS"
            )

            result = validate_file(
                path=canonical_path,
                expected_cohort=str(
                    expected[
                        "cohort"
                    ]
                ),
                expected_patients=int(
                    expected[
                        "patients"
                    ]
                ),
                expected_hours=int(
                    expected[
                        "hours"
                    ]
                ),
                expected_septic=int(
                    expected[
                        "septic"
                    ]
                ),
                expected_positive_hours=int(
                    expected[
                        "positive_hours"
                    ]
                ),
                unavailable=unavailable,
                batch_size=args.batch_size,
            )

            # Patient IDs only kept in memory;
            # nothing identifying is written.
            patient_ids: set[str] = set()

            parquet = pq.ParquetFile(
                canonical_path
            )

            for batch in parquet.iter_batches(
                columns=[
                    "patient_id"
                ],
                batch_size=args.batch_size,
            ):
                patient_ids.update(
                    batch.column(0)
                    .to_pandas()
                    .astype(str)
                    .tolist()
                )

            patient_sets[
                key
            ] = patient_ids

            results.append(
                result
            )

            print(
                f"  Structural checks: PASS | "
                f"{result['patients']:,} patients | "
                f"{result['rows']:,} hours | "
                f"{result['columns']} columns | "
                f"NaN={result['missing_values']}"
            )

    # --------------------------------------------------
    # Source split overlap checks
    # --------------------------------------------------
    for direction in (
        "A_to_B",
        "B_to_A",
    ):
        train = patient_sets[
            (
                direction,
                "train",
            )
        ]

        validation = patient_sets[
            (
                direction,
                "validation",
            )
        ]

        internal_test = (
            patient_sets[
                (
                    direction,
                    "internal_test",
                )
            ]
        )

        if not train.isdisjoint(
            validation
        ):
            raise RuntimeError(
                f"{direction}: train/validation "
                "patient overlap."
            )

        if not train.isdisjoint(
            internal_test
        ):
            raise RuntimeError(
                f"{direction}: train/internal-test "
                "patient overlap."
            )

        if not validation.isdisjoint(
            internal_test
        ):
            raise RuntimeError(
                f"{direction}: validation/"
                "internal-test patient overlap."
            )

        print()
        print(
            f"{direction}: source split "
            "disjointness PASS"
        )

    print()
    print("=" * 72)
    print(
        "RESULT: ALL FROZEN PROCESSED DATASETS PASS"
    )
    print("=" * 72)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())