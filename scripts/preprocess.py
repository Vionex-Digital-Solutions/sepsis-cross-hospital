from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from sepsis_cross_hospital.data.preprocessing import (
    CONTEXT_COLUMNS,
    DYNAMIC_COLUMNS,
    SourceStatistics,
    fit_source_statistics,
    transform_patient_causally,
)
from sepsis_cross_hospital.data.validation import (
    prepare_hourly_target,
)


DIRECTIONS = {
    "A_to_B": {
        "source": "A",
        "target": "B",
    },
    "B_to_A": {
        "source": "B",
        "target": "A",
    },
}

SOURCE_SPLITS = (
    "train",
    "validation",
    "internal_test",
)

FROZEN_SPLIT_SHA256 = (
    "1883b58134c36ecba43c6228aa4712d3"
    "cdc94b47ca3ad86507834c826d1a4c08"
)

FROZEN_BATCH_PATIENTS = 250

FROZEN_PROCESSED_SHA256 = {
    "A_to_B": {
        "train": (
            "7b20e5ef21f9db0ecdb5129a82e4223c"
            "d673686116b947a0e1e71bf226597857"
        ),
        "validation": (
            "5130f764f9c1805d591ac755c592d269"
            "dd858ec24c323809ee3cf0a31ed4a6be"
        ),
        "internal_test": (
            "38657d1dab6ca43f9b8d3ea6c6e59f8"
            "05ca1770691e36adb1bee54908a45e6cc"
        ),
        "external": (
            "eef0c3183e8d251e74d83a784c84d76b"
            "db307867d3ad65994cc65ab4e01daec6"
        ),
    },
    "B_to_A": {
        "train": (
            "2124eb7b8337cdeab7c86e1e3749c825"
            "5b427f336c115134ccda4959a27c8cef"
        ),
        "validation": (
            "a68b667e1af519366ab284b1d7343ad6"
            "cb0e56879734ea7b6901616bef1fd9b5"
        ),
        "internal_test": (
            "d01da816c3c49aa8020e52326d7089dc"
            "28a6ca17f428feccde316aaeb07668e0"
        ),
        "external": (
            "38d0ec27765ae5b1393cb2d2caa414bb"
            "3e0499d102d482ef1fa25f97c467974b"
        ),
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build causal model-ready Parquet datasets "
            "for bidirectional cross-hospital evaluation."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/paths.local.yaml"
        ),
    )

    parser.add_argument(
        "--direction",
        choices=[
            "A_to_B",
            "B_to_A",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--batch-patients",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--smoke-patients",
        type=int,
        default=0,
        help=(
            "If >0, use only the first N patients "
            "per partition and write under data/interim."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_paths(
    config_path: Path,
) -> dict[str, Path]:
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        ) or {}

    paths = {
        "A": Path(
            config["training_set_a"]
        ).expanduser().resolve(),
        "B": Path(
            config["training_set_b"]
        ).expanduser().resolve(),
    }

    for cohort, path in paths.items():
        if not path.is_dir():
            raise FileNotFoundError(
                f"Cohort {cohort}: {path}"
            )

    return paths


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

            digest.update(chunk)

    return digest.hexdigest()


def load_split_manifest() -> pd.DataFrame:
    path = Path(
        "data/interim/splits/"
        "split_manifest.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Frozen split manifest not found: "
            f"{path}"
        )

    manifest = pd.read_csv(
        path,
        dtype={
            "cohort": str,
            "patient_id": str,
            "septic": int,
            "split": str,
        },
    )

    required = {
        "cohort",
        "patient_id",
        "septic",
        "split",
    }

    missing = (
        required
        - set(manifest.columns)
    )

    if missing:
        raise ValueError(
            "Split manifest missing columns: "
            + ", ".join(sorted(missing))
        )

    return manifest


def verify_split_hash() -> str:
    split_path = Path(
        "data/interim/splits/"
        "split_manifest.csv"
    )

    metadata_path = Path(
        "data/metadata/splits/"
        "split_metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            metadata_path
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    observed = sha256_file(
        split_path
    )

    expected = metadata[
        "manifest_sha256"
    ]

    if observed != expected:
        raise RuntimeError(
            "Split manifest does not match "
            "its generated metadata.\n"
            f"Metadata: {expected}\n"
            f"Observed: {observed}"
        )

    if observed != FROZEN_SPLIT_SHA256:
        raise RuntimeError(
            "Split manifest is not the frozen "
            "study split.\n"
            f"Frozen:   {FROZEN_SPLIT_SHA256}\n"
            f"Observed: {observed}"
        )

    return observed


def patient_ids_for_split(
    manifest: pd.DataFrame,
    *,
    cohort: str,
    split: str | None,
    limit: int = 0,
) -> list[str]:
    if split is None:
        selected = manifest.loc[
            manifest["cohort"]
            == cohort
        ]
    else:
        selected = manifest.loc[
            (
                manifest["cohort"]
                == cohort
            )
            & (
                manifest["split"]
                == split
            )
        ]

    patient_ids = sorted(
        selected[
            "patient_id"
        ].astype(str).tolist()
    )

    if limit > 0:
        patient_ids = (
            patient_ids[:limit]
        )

    return patient_ids


def read_patient(
    cohort_path: Path,
    patient_id: str,
) -> pd.DataFrame:
    path = (
        cohort_path
        / f"{patient_id}.psv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return pd.read_csv(
        path,
        sep="|",
    )


def iter_patients(
    cohort_path: Path,
    patient_ids: list[str],
) -> Iterator[pd.DataFrame]:
    for patient_id in patient_ids:
        yield read_patient(
            cohort_path,
            patient_id,
        )


def source_statistics_to_json(
    statistics: SourceStatistics,
) -> dict:
    return asdict(
        statistics
    )


def make_master_frame(
    *,
    raw_patient: pd.DataFrame,
    patient_id: str,
    cohort: str,
    statistics: SourceStatistics,
) -> pd.DataFrame:
    features = (
        transform_patient_causally(
            raw_patient,
            statistics,
        )
    ).reset_index(drop=True)

    labels = prepare_hourly_target(
        raw_patient[
            "SepsisLabel"
        ].to_numpy()
    )

    if len(features) != len(labels):
        raise RuntimeError(
            f"{patient_id}: feature/label "
            "length mismatch."
        )

    metadata = pd.DataFrame(
        {
            "patient_id": [
                patient_id
            ] * len(features),
            "cohort": [
                cohort
            ] * len(features),
            "hour_index": range(
                len(features)
            ),
            "SepsisLabel": labels,
        }
    )

    return pd.concat(
        [
            metadata,
            features,
        ],
        axis=1,
    )


def write_partition(
    *,
    cohort: str,
    cohort_path: Path,
    patient_ids: list[str],
    statistics: SourceStatistics,
    output_path: Path,
    batch_patients: int,
    overwrite: bool,
) -> dict:
    if batch_patients <= 0:
        raise ValueError(
            "batch_patients must be positive."
        )

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. "
                "Use --overwrite to replace it."
            )

        output_path.unlink()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".part"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    writer: (
        pq.ParquetWriter | None
    ) = None

    batch_frames: list[
        pd.DataFrame
    ] = []

    patient_count = 0
    row_count = 0
    septic_patient_count = 0
    positive_label_hours = 0

    try:
        for index, patient_id in enumerate(
            patient_ids,
            start=1,
        ):
            raw = read_patient(
                cohort_path,
                patient_id,
            )

            labels = prepare_hourly_target(
                raw[
                    "SepsisLabel"
                ].to_numpy()
            )

            septic_patient_count += int(
                labels.any()
            )

            positive_label_hours += int(
                labels.sum()
            )

            master = make_master_frame(
                raw_patient=raw,
                patient_id=patient_id,
                cohort=cohort,
                statistics=statistics,
            )

            batch_frames.append(
                master
            )

            patient_count += 1
            row_count += len(master)

            should_flush = (
                len(batch_frames)
                >= batch_patients
                or index
                == len(patient_ids)
            )

            if should_flush:
                combined = pd.concat(
                    batch_frames,
                    ignore_index=True,
                )

                table = (
                    pa.Table.from_pandas(
                        combined,
                        preserve_index=False,
                    )
                )

                if writer is None:
                    writer = (
                        pq.ParquetWriter(
                            temporary_path,
                            table.schema,
                            compression="zstd",
                            compression_level=3,
                            use_dictionary=True,
                        )
                    )

                writer.write_table(
                    table
                )

                batch_frames.clear()

            if (
                index == 1
                or index % 1000 == 0
                or index
                == len(patient_ids)
            ):
                print(
                    f"    processed "
                    f"{index:,}/"
                    f"{len(patient_ids):,} "
                    f"patients"
                )

        if writer is None:
            raise RuntimeError(
                "No patients were written."
            )

    finally:
        if writer is not None:
            writer.close()

    os.replace(
        temporary_path,
        output_path,
    )

    file_hash = sha256_file(
        output_path
    )

    return {
        "cohort": cohort,
        "patients": patient_count,
        "patient_hours": row_count,
        "septic_patients": (
            septic_patient_count
        ),
        "positive_label_hours": (
            positive_label_hours
        ),
        "file": str(
            output_path.as_posix()
        ),
        "sha256": file_hash,
        "size_bytes": (
            output_path.stat().st_size
        ),
    }


def feature_schema() -> dict:
    value_columns = (
        DYNAMIC_COLUMNS
        + CONTEXT_COLUMNS
    )

    mask_columns = [
        f"obs_{column}"
        for column in DYNAMIC_COLUMNS
    ]

    tsl_columns = [
        f"tsl_{column}"
        for column in DYNAMIC_COLUMNS
    ]

    count_columns = [
        f"count6_{column}"
        for column in DYNAMIC_COLUMNS
    ]

    return {
        "M0": (
            CONTEXT_COLUMNS
            + mask_columns
            + tsl_columns
            + count_columns
        ),
        "V0": value_columns,
        "V1": (
            value_columns
            + mask_columns
        ),
        "V2": (
            value_columns
            + mask_columns
            + tsl_columns
            + count_columns
        ),
    }


def build_direction(
    *,
    direction: str,
    paths: dict[str, Path],
    manifest: pd.DataFrame,
    split_hash: str,
    batch_patients: int,
    smoke_patients: int,
    overwrite: bool,
) -> list[dict]:
    definition = DIRECTIONS[
        direction
    ]

    source = definition[
        "source"
    ]

    target = definition[
        "target"
    ]

    print()
    print("=" * 72)
    print(
        f"DIRECTION {direction}"
    )
    print("=" * 72)

    train_ids = patient_ids_for_split(
        manifest,
        cohort=source,
        split="train",
        limit=smoke_patients,
    )

    print(
        f"Fitting source statistics "
        f"from {len(train_ids):,} "
        f"{source}-train patients..."
    )

    statistics = (
        fit_source_statistics(
            iter_patients(
                paths[source],
                train_ids,
            )
        )
    )

    unavailable = [
        column
        for column, available
        in (
            statistics
            .source_available_dynamic
            .items()
        )
        if not available
    ]

    print(
        "Source-unavailable dynamic "
        f"variables: "
        f"{unavailable or 'none'}"
    )

    if smoke_patients > 0:
        root = Path(
            "data/interim/"
            "preprocess_smoke"
        ) / direction
    else:
        root = (
            Path("data/processed")
            / direction
        )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if smoke_patients > 0:
        metadata_root = (
            Path(
                "data/interim/"
                "preprocess_smoke_metadata"
            )
            / direction
        )
    else:
        metadata_root = (
            Path(
                "data/metadata/"
                "preprocessing"
            )
            / direction
        )

    metadata_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        metadata_root
        / "source_statistics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            source_statistics_to_json(
                statistics
            ),
            handle,
            indent=2,
        )

    with (
        metadata_root
        / "feature_schema.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            feature_schema(),
            handle,
            indent=2,
        )

    records: list[dict] = []

    for split in SOURCE_SPLITS:
        ids = patient_ids_for_split(
            manifest,
            cohort=source,
            split=split,
            limit=smoke_patients,
        )

        print()
        print(
            f"  {source} {split}: "
            f"{len(ids):,} patients"
        )

        record = write_partition(
            cohort=source,
            cohort_path=paths[source],
            patient_ids=ids,
            statistics=statistics,
            output_path=(
                root
                / f"{split}.parquet"
            ),
            batch_patients=(
                batch_patients
            ),
            overwrite=overwrite,
        )

        record[
            "direction"
        ] = direction

        record[
            "partition"
        ] = split

        records.append(record)

    external_ids = (
        patient_ids_for_split(
            manifest,
            cohort=target,
            split=None,
            limit=smoke_patients,
        )
    )

    print()
    print(
        f"  {target} external: "
        f"{len(external_ids):,} patients"
    )

    record = write_partition(
        cohort=target,
        cohort_path=paths[target],
        patient_ids=external_ids,
        statistics=statistics,
        output_path=(
            root
            / "external.parquet"
        ),
        batch_patients=batch_patients,
        overwrite=overwrite,
    )

    record[
        "direction"
    ] = direction

    record[
        "partition"
    ] = "external"

    records.append(record)

    direction_metadata = {
        "direction": direction,
        "source_cohort": source,
        "target_cohort": target,
        "split_manifest_sha256": (
            split_hash
        ),
        "smoke_patients": (
            smoke_patients
        ),
        "source_train_patients_used_to_fit": (
            len(train_ids)
        ),
        "source_unavailable_dynamic": (
            unavailable
        ),
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with (
        metadata_root
        / "preprocessing_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            direction_metadata,
            handle,
            indent=2,
        )

    return records

def verify_frozen_processed(
    records: list[dict],
    directions: list[str],
) -> list[str]:
    """
    Verify exact processed Parquet bytes against
    the frozen study reference.
    """
    failures: list[str] = []

    print()
    print("=" * 72)
    print("FROZEN PROCESSED-DATA VERIFICATION")
    print("=" * 72)

    lookup = {
        (
            str(record["direction"]),
            str(record["partition"]),
        ): str(record["sha256"])
        for record in records
    }

    for direction in directions:
        print()
        print(direction)
        print("-" * 40)

        for partition in (
            "train",
            "validation",
            "internal_test",
            "external",
        ):
            expected = (
                FROZEN_PROCESSED_SHA256[
                    direction
                ][
                    partition
                ]
            )

            actual = lookup.get(
                (
                    direction,
                    partition,
                )
            )

            passed = (
                actual == expected
            )

            print(
                f"{partition:<22}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

            if not passed:
                failures.append(
                    f"{direction}/{partition}: "
                    f"expected {expected}, "
                    f"found {actual}."
                )

    print()
    print("=" * 72)

    if failures:
        print(
            "Frozen processed-data verification: FAILED"
        )
    else:
        print(
            "Frozen processed-data verification: PASS"
        )

    print("=" * 72)

    return failures

def main() -> int:
    args = parse_args()

    print("=" * 72)
    print(
        "CAUSAL MODEL-READY DATASET BUILDER"
    )
    print("=" * 72)

    if args.smoke_patients < 0:
        raise ValueError(
            "smoke-patients cannot be negative."
        )

    paths = load_paths(
        args.config
    )

    manifest = (
        load_split_manifest()
    )

    split_hash = (
        verify_split_hash()
    )

    print(
        "Verified split SHA256:"
    )
    print(
        f"  {split_hash}"
    )

    if args.direction == "all":
        directions = [
            "A_to_B",
            "B_to_A",
        ]
    else:
        directions = [
            args.direction
        ]

    all_records: list[
        dict
    ] = []

    for direction in directions:
        all_records.extend(
            build_direction(
                direction=direction,
                paths=paths,
                manifest=manifest,
                split_hash=split_hash,
                batch_patients=(
                    args.batch_patients
                ),
                smoke_patients=(
                    args.smoke_patients
                ),
                overwrite=args.overwrite,
            )
        )

    records_frame = pd.DataFrame(
        all_records
    )

    if args.smoke_patients > 0:
        manifest_output = Path(
            "data/interim/"
            "preprocess_smoke_metadata/"
            "processed_manifest.csv"
        )
    else:
        manifest_output = Path(
            "data/metadata/"
            "preprocessing/"
            "processed_manifest.csv"
        )

    if (
        args.smoke_patients == 0
        and args.batch_patients
        != FROZEN_BATCH_PATIENTS
    ):
        raise ValueError(
            "Exact frozen reproduction requires "
            f"--batch-patients {FROZEN_BATCH_PATIENTS}. "
            f"Found {args.batch_patients}."
        )

    manifest_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records_frame.to_csv(
        manifest_output,
        index=False,
    )

    print()
    print("=" * 72)
    print("PREPROCESSING COMPLETE")
    print("=" * 72)

    print(
        records_frame[
            [
                "direction",
                "partition",
                "cohort",
                "patients",
                "patient_hours",
                "septic_patients",
                "positive_label_hours",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Manifest: "
        f"{manifest_output}"
    )

    if args.smoke_patients > 0:
        print()
        print(
            "Smoke preprocessing complete. "
            "Frozen Parquet identity was not "
            "verified in smoke mode."
        )

        return 0

    failures = verify_frozen_processed(
        records=all_records,
        directions=directions,
    )

    if failures:
        print()

        for failure in failures:
            print(
                f"FAIL: {failure}"
            )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())