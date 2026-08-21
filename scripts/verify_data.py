from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


AUDIT_VERSION = "0.1.0"

EXPECTED_COUNTS = {
    "A": 20336,
    "B": 20000,
}

FROZEN_DATASET_REFERENCE = {
    "A": {
        "psv_files": 20336,
        "patient_hours": 790215,
        "septic_patients": 1790,
        "nonseptic_patients": 18546,
        "tree_sha256": (
            "ad23f06c4580a873871f08f635a8f02a"
            "52e538089b1dc352f576be41d949fc70"
        ),
    },
    "B": {
        "psv_files": 20000,
        "patient_hours": 761995,
        "septic_patients": 1142,
        "nonseptic_patients": 18858,
        "tree_sha256": (
            "6ae3634d90dd3003e5c3c761d24b75a"
            "fe11d3741e7803ffd1ebfe6b886869da1"
        ),
    },
}

EXPECTED_COLUMNS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
    "SepsisLabel",
]

LABEL_COLUMN = "SepsisLabel"

STATIC_COLUMNS = [
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
]


@dataclass
class ColumnAccumulator:
    total_rows: int = 0
    missing_count: int = 0
    nonmissing_count: int = 0
    numeric_count: int = 0
    nonnumeric_count: int = 0
    value_sum: float = 0.0
    value_sumsq: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(
        self,
        original: pd.Series,
        numeric: pd.Series,
    ) -> None:
        self.total_rows += len(original)

        original_nonmissing = int(original.notna().sum())
        numeric_nonmissing = int(numeric.notna().sum())

        self.missing_count += len(original) - original_nonmissing
        self.nonmissing_count += original_nonmissing
        self.numeric_count += numeric_nonmissing
        self.nonnumeric_count += original_nonmissing - numeric_nonmissing

        values = numeric.dropna().to_numpy(dtype=np.float64)

        if values.size == 0:
            return

        self.value_sum += float(values.sum())
        self.value_sumsq += float(np.square(values).sum())

        current_min = float(values.min())
        current_max = float(values.max())

        if self.minimum is None or current_min < self.minimum:
            self.minimum = current_min

        if self.maximum is None or current_max > self.maximum:
            self.maximum = current_max

    def missingness_record(
        self,
        cohort: str,
        column: str,
    ) -> dict[str, Any]:
        if self.total_rows == 0:
            missing_pct = math.nan
        else:
            missing_pct = 100.0 * self.missing_count / self.total_rows

        return {
            "cohort": cohort,
            "column": column,
            "total_rows": self.total_rows,
            "missing_count": self.missing_count,
            "missing_pct": missing_pct,
            "nonmissing_count": self.nonmissing_count,
        }

    def value_record(
        self,
        cohort: str,
        column: str,
    ) -> dict[str, Any]:
        mean = math.nan
        std = math.nan

        if self.numeric_count > 0:
            mean = self.value_sum / self.numeric_count

        if self.numeric_count > 1:
            numerator = (
                self.value_sumsq
                - (self.value_sum**2 / self.numeric_count)
            )
            variance = max(
                numerator / (self.numeric_count - 1),
                0.0,
            )
            std = math.sqrt(variance)

        return {
            "cohort": cohort,
            "column": column,
            "numeric_count": self.numeric_count,
            "nonnumeric_count": self.nonnumeric_count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": std,
        }


@dataclass
class CohortAccumulator:
    cohort: str
    expected_count: int

    psv_files: int = 0
    total_files: int = 0
    zero_byte_files: int = 0
    parse_error_files: int = 0
    schema_mismatch_files: int = 0
    invalid_label_files: int = 0
    label_reversal_files: int = 0
    iculos_problem_files: int = 0
    static_value_change_files: int = 0

    parsed_files: int = 0
    total_patient_hours: int = 0
    septic_patients: int = 0
    nonseptic_patients: int = 0
    positive_label_hours: int = 0

    duplicate_patient_ids: int = 0

    lengths: list[int] = field(default_factory=list)
    first_positive_iculos: list[float] = field(default_factory=list)

    schema_counter: Counter[str] = field(default_factory=Counter)
    extension_counter: Counter[str] = field(default_factory=Counter)
    anomaly_counter: Counter[str] = field(default_factory=Counter)

    column_stats: dict[str, ColumnAccumulator] = field(
        default_factory=lambda: {
            column: ColumnAccumulator()
            for column in EXPECTED_COLUMNS
        }
    )

    tree_hasher: Any = field(
        default_factory=hashlib.sha256
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible audit of PhysioNet/CinC 2019 "
            "training cohorts A and B."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paths.local.yaml"),
        help="Machine-local path configuration.",
    )

    parser.add_argument(
        "--smoke-files",
        type=int,
        default=0,
        help=(
            "Audit only the first N PSV files per cohort. "
            "Outputs remain under data/interim."
        ),
    )

    return parser.parse_args()


def load_paths(config_path: Path) -> dict[str, Path]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Path configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required = {
        "training_set_a",
        "training_set_b",
    }

    missing = required - set(config)

    if missing:
        raise KeyError(
            "Missing keys in paths.local.yaml: "
            + ", ".join(sorted(missing))
        )

    paths = {
        key: Path(config[key]).expanduser().resolve()
        for key in required
    }

    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{key} does not exist: {path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                f"{key} is not a directory: {path}"
            )

    return paths


def git_metadata() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return None

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")

    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": math.nan,
            "p01": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "median": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
            "mean": math.nan,
        }

    array = np.asarray(values, dtype=np.float64)

    return {
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def schema_signature(columns: list[str]) -> str:
    return "|".join(columns)


def audit_file(
    path: Path,
    cohort_root: Path,
    accumulator: CohortAccumulator,
) -> dict[str, Any]:
    relative_path = path.relative_to(cohort_root).as_posix()
    patient_id = path.stem

    record: dict[str, Any] = {
        "cohort": accumulator.cohort,
        "patient_id": patient_id,
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": None,
        "parse_ok": False,
        "schema_ok": False,
        "n_rows": None,
        "septic_patient": None,
        "positive_label_hours": None,
        "first_positive_row": None,
        "first_positive_iculos": None,
        "label_monotonic": None,
        "iculos_sequential": None,
        "static_columns_constant": None,
        "anomaly_flags": "",
    }

    anomalies: list[str] = []

    if path.stat().st_size == 0:
        accumulator.zero_byte_files += 1
        accumulator.anomaly_counter["zero_byte_file"] += 1
        anomalies.append("zero_byte_file")
        record["anomaly_flags"] = ";".join(anomalies)
        return record

    try:
        raw = path.read_bytes()
        file_hash = sha256_bytes(raw)
        record["sha256"] = file_hash

        accumulator.tree_hasher.update(
            relative_path.encode("utf-8")
        )
        accumulator.tree_hasher.update(b"\0")
        accumulator.tree_hasher.update(
            file_hash.encode("ascii")
        )
        accumulator.tree_hasher.update(b"\n")

        df = pd.read_csv(
            io.BytesIO(raw),
            sep="|",
        )

    except Exception as exc:
        accumulator.parse_error_files += 1
        accumulator.anomaly_counter["parse_error"] += 1
        anomalies.append(
            f"parse_error:{type(exc).__name__}"
        )
        record["anomaly_flags"] = ";".join(anomalies)
        return record

    record["parse_ok"] = True
    accumulator.parsed_files += 1

    columns = list(df.columns)
    signature = schema_signature(columns)
    accumulator.schema_counter[signature] += 1

    schema_ok = columns == EXPECTED_COLUMNS
    record["schema_ok"] = schema_ok

    if not schema_ok:
        accumulator.schema_mismatch_files += 1
        accumulator.anomaly_counter[
            "schema_mismatch"
        ] += 1
        anomalies.append("schema_mismatch")

        record["n_rows"] = len(df)
        record["anomaly_flags"] = ";".join(anomalies)

        # Do not aggregate values from files with an unexpected schema.
        return record

    n_rows = len(df)

    record["n_rows"] = n_rows
    accumulator.lengths.append(n_rows)
    accumulator.total_patient_hours += n_rows

    numeric_df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    for column in EXPECTED_COLUMNS:
        accumulator.column_stats[column].update(
            original=df[column],
            numeric=numeric_df[column],
        )

    # --------------------------------------------------------
    # Label audit
    # --------------------------------------------------------
    labels = numeric_df[LABEL_COLUMN]

    valid_label_mask = labels.isin([0, 1])

    label_valid = (
        labels.notna().all()
        and bool(valid_label_mask.all())
    )

    if not label_valid:
        accumulator.invalid_label_files += 1
        accumulator.anomaly_counter[
            "invalid_label"
        ] += 1
        anomalies.append("invalid_label")

    clean_labels = labels.dropna()

    positive_hours = int((clean_labels == 1).sum())
    record["positive_label_hours"] = positive_hours
    accumulator.positive_label_hours += positive_hours

    septic = bool((clean_labels == 1).any())
    record["septic_patient"] = septic

    if septic:
        accumulator.septic_patients += 1

        positive_positions = np.flatnonzero(
            clean_labels.to_numpy() == 1
        )

        if positive_positions.size > 0:
            first_position = int(positive_positions[0])

            # Row is reported 1-based for human readability.
            record["first_positive_row"] = (
                first_position + 1
            )

            if first_position < len(numeric_df):
                first_iculos = numeric_df.iloc[
                    first_position
                ]["ICULOS"]

                if pd.notna(first_iculos):
                    first_iculos_float = float(first_iculos)
                    record[
                        "first_positive_iculos"
                    ] = first_iculos_float
                    accumulator.first_positive_iculos.append(
                        first_iculos_float
                    )
    else:
        accumulator.nonseptic_patients += 1

    if len(clean_labels) > 1:
        label_diffs = np.diff(
            clean_labels.to_numpy(dtype=np.float64)
        )
        monotonic = bool(np.all(label_diffs >= 0))
    else:
        monotonic = True

    record["label_monotonic"] = monotonic

    if not monotonic:
        accumulator.label_reversal_files += 1
        accumulator.anomaly_counter[
            "label_reversal_1_to_0"
        ] += 1
        anomalies.append("label_reversal_1_to_0")

    # --------------------------------------------------------
    # ICULOS audit
    # --------------------------------------------------------
    iculos = numeric_df["ICULOS"]

    if (
        len(iculos) > 0
        and iculos.notna().all()
    ):
        values = iculos.to_numpy(dtype=np.float64)

        expected = np.arange(
            values[0],
            values[0] + len(values),
            dtype=np.float64,
        )

        iculos_sequential = bool(
            np.array_equal(values, expected)
        )
    else:
        iculos_sequential = False

    record["iculos_sequential"] = iculos_sequential

    if not iculos_sequential:
        accumulator.iculos_problem_files += 1
        accumulator.anomaly_counter[
            "iculos_not_sequential"
        ] += 1
        anomalies.append("iculos_not_sequential")

    # --------------------------------------------------------
    # Static patient-level fields
    # --------------------------------------------------------
    static_constant = True

    for column in STATIC_COLUMNS:
        unique_values = numeric_df[column].dropna().unique()

        if len(unique_values) > 1:
            static_constant = False
            break

    record["static_columns_constant"] = static_constant

    if not static_constant:
        accumulator.static_value_change_files += 1
        accumulator.anomaly_counter[
            "static_column_changed"
        ] += 1
        anomalies.append("static_column_changed")

    record["anomaly_flags"] = ";".join(anomalies)

    return record


def audit_cohort(
    cohort: str,
    root: Path,
    smoke_files: int,
) -> tuple[
    CohortAccumulator,
    pd.DataFrame,
]:
    accumulator = CohortAccumulator(
        cohort=cohort,
        expected_count=EXPECTED_COUNTS[cohort],
    )

    all_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
    )

    accumulator.total_files = len(all_files)

    for path in all_files:
        suffix = (
            path.suffix.lower()
            if path.suffix
            else "<no_extension>"
        )
        accumulator.extension_counter[suffix] += 1

    psv_files = [
        path
        for path in all_files
        if path.suffix.lower() == ".psv"
    ]

    accumulator.psv_files = len(psv_files)

    patient_ids = [path.stem for path in psv_files]

    duplicate_counter = Counter(patient_ids)
    accumulator.duplicate_patient_ids = sum(
        count - 1
        for count in duplicate_counter.values()
        if count > 1
    )

    if accumulator.duplicate_patient_ids:
        accumulator.anomaly_counter[
            "duplicate_patient_id"
        ] = accumulator.duplicate_patient_ids

    files_to_process = psv_files

    if smoke_files > 0:
        files_to_process = psv_files[:smoke_files]

    inventory_records: list[dict[str, Any]] = []

    total_to_process = len(files_to_process)

    print()
    print(
        f"Cohort {cohort}: "
        f"{total_to_process:,} PSV files to audit"
    )

    for index, path in enumerate(
        files_to_process,
        start=1,
    ):
        inventory_records.append(
            audit_file(
                path=path,
                cohort_root=root,
                accumulator=accumulator,
            )
        )

        if (
            index == 1
            or index % 1000 == 0
            or index == total_to_process
        ):
            print(
                f"  processed "
                f"{index:,}/{total_to_process:,}"
            )

    inventory = pd.DataFrame(inventory_records)

    return accumulator, inventory


def cohort_summary_record(
    accumulator: CohortAccumulator,
    smoke_files: int,
) -> dict[str, Any]:
    full_run = smoke_files == 0

    if full_run:
        observed_count = accumulator.psv_files
        difference = (
            observed_count - accumulator.expected_count
        )
    else:
        observed_count = accumulator.parsed_files
        difference = math.nan

    return {
        "cohort": accumulator.cohort,
        "audit_mode": (
            "full" if full_run else "smoke"
        ),
        "expected_subjects_official": (
            accumulator.expected_count
        ),
        "observed_psv_files": (
            accumulator.psv_files
            if full_run
            else observed_count
        ),
        "difference_observed_minus_expected": difference,
        "total_files_in_directory": (
            accumulator.total_files
        ),
        "parsed_files": accumulator.parsed_files,
        "zero_byte_files": accumulator.zero_byte_files,
        "parse_error_files": (
            accumulator.parse_error_files
        ),
        "schema_mismatch_files": (
            accumulator.schema_mismatch_files
        ),
        "duplicate_patient_ids": (
            accumulator.duplicate_patient_ids
        ),
        "total_patient_hours": (
            accumulator.total_patient_hours
        ),
        "septic_patients": accumulator.septic_patients,
        "nonseptic_patients": (
            accumulator.nonseptic_patients
        ),
        "positive_label_hours": (
            accumulator.positive_label_hours
        ),
        "invalid_label_files": (
            accumulator.invalid_label_files
        ),
        "label_reversal_files": (
            accumulator.label_reversal_files
        ),
        "iculos_problem_files": (
            accumulator.iculos_problem_files
        ),
        "static_value_change_files": (
            accumulator.static_value_change_files
        ),
        "dataset_tree_sha256": (
            accumulator.tree_hasher.hexdigest()
            if accumulator.parsed_files > 0
            else None
        ),
    }


def build_sequence_summary(
    accumulator: CohortAccumulator,
) -> dict[str, Any]:
    return {
        "cohort": accumulator.cohort,
        "n_parsed_patients": (
            len(accumulator.lengths)
        ),
        **safe_quantiles(
            [float(x) for x in accumulator.lengths]
        ),
    }


def build_label_summary(
    accumulator: CohortAccumulator,
) -> dict[str, Any]:
    patients = (
        accumulator.septic_patients
        + accumulator.nonseptic_patients
    )

    if patients:
        septic_pct = (
            100.0
            * accumulator.septic_patients
            / patients
        )
    else:
        septic_pct = math.nan

    if accumulator.total_patient_hours:
        positive_hour_pct = (
            100.0
            * accumulator.positive_label_hours
            / accumulator.total_patient_hours
        )
    else:
        positive_hour_pct = math.nan

    first_positive = safe_quantiles(
        accumulator.first_positive_iculos
    )

    return {
        "cohort": accumulator.cohort,
        "patients_with_valid_schema": patients,
        "septic_patients": (
            accumulator.septic_patients
        ),
        "nonseptic_patients": (
            accumulator.nonseptic_patients
        ),
        "septic_patient_pct": septic_pct,
        "patient_hours": (
            accumulator.total_patient_hours
        ),
        "positive_label_hours": (
            accumulator.positive_label_hours
        ),
        "positive_label_hour_pct": (
            positive_hour_pct
        ),
        "first_positive_iculos_min": (
            first_positive["min"]
        ),
        "first_positive_iculos_p25": (
            first_positive["p25"]
        ),
        "first_positive_iculos_median": (
            first_positive["median"]
        ),
        "first_positive_iculos_p75": (
            first_positive["p75"]
        ),
        "first_positive_iculos_max": (
            first_positive["max"]
        ),
    }


def save_outputs(
    accumulators: dict[str, CohortAccumulator],
    inventories: dict[str, pd.DataFrame],
    smoke_files: int,
    config_path: Path,
) -> None:
    full_run = smoke_files == 0

    if full_run:
        aggregate_dir = Path(
            "data/metadata/audit"
        )
        detail_dir = Path(
            "data/interim/audit"
        )
    else:
        aggregate_dir = Path(
            "data/interim/audit_smoke"
        )
        detail_dir = aggregate_dir

    aggregate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    detail_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Local patient/file-level inventory
    # --------------------------------------------------------
    combined_inventory = pd.concat(
        inventories.values(),
        ignore_index=True,
    )

    combined_inventory.to_csv(
        detail_dir / "file_inventory.csv",
        index=False,
    )

    anomaly_inventory = combined_inventory[
        combined_inventory["anomaly_flags"].fillna("")
        != ""
    ].copy()

    anomaly_inventory.to_csv(
        detail_dir / "anomalies.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Aggregate cohort summary
    # --------------------------------------------------------
    cohort_summary = pd.DataFrame(
        [
            cohort_summary_record(acc, smoke_files)
            for acc in accumulators.values()
        ]
    )

    cohort_summary.to_csv(
        aggregate_dir / "cohort_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------
    missingness_records = []

    for cohort, acc in accumulators.items():
        for column, stats in acc.column_stats.items():
            missingness_records.append(
                stats.missingness_record(
                    cohort=cohort,
                    column=column,
                )
            )

    pd.DataFrame(
        missingness_records
    ).to_csv(
        aggregate_dir / "missingness_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Observed numeric ranges
    # --------------------------------------------------------
    value_records = []

    for cohort, acc in accumulators.items():
        for column, stats in acc.column_stats.items():
            value_records.append(
                stats.value_record(
                    cohort=cohort,
                    column=column,
                )
            )

    pd.DataFrame(
        value_records
    ).to_csv(
        aggregate_dir / "value_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Sequence lengths
    # --------------------------------------------------------
    pd.DataFrame(
        [
            build_sequence_summary(acc)
            for acc in accumulators.values()
        ]
    ).to_csv(
        aggregate_dir
        / "sequence_length_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    pd.DataFrame(
        [
            build_label_summary(acc)
            for acc in accumulators.values()
        ]
    ).to_csv(
        aggregate_dir / "label_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Schema variants
    # --------------------------------------------------------
    schema_records = []

    for cohort, acc in accumulators.items():
        for signature, count in (
            acc.schema_counter.items()
        ):
            schema_records.append(
                {
                    "cohort": cohort,
                    "count": count,
                    "matches_expected": (
                        signature
                        == schema_signature(
                            EXPECTED_COLUMNS
                        )
                    ),
                    "schema": signature,
                }
            )

    pd.DataFrame(
        schema_records
    ).to_csv(
        aggregate_dir / "schema_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # File extensions
    # --------------------------------------------------------
    extension_records = []

    for cohort, acc in accumulators.items():
        for extension, count in (
            sorted(
                acc.extension_counter.items()
            )
        ):
            extension_records.append(
                {
                    "cohort": cohort,
                    "extension": extension,
                    "count": count,
                }
            )

    pd.DataFrame(
        extension_records
    ).to_csv(
        aggregate_dir / "extension_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Aggregate anomaly counts
    # --------------------------------------------------------
    anomaly_records = []

    for cohort, acc in accumulators.items():
        for anomaly, count in sorted(
            acc.anomaly_counter.items()
        ):
            anomaly_records.append(
                {
                    "cohort": cohort,
                    "anomaly": anomaly,
                    "count": count,
                }
            )

    pd.DataFrame(
        anomaly_records,
        columns=[
            "cohort",
            "anomaly",
            "count",
        ],
    ).to_csv(
        aggregate_dir / "anomaly_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Cross-cohort patient ID overlap
    # --------------------------------------------------------
    ids_a = set(
        inventories["A"]["patient_id"].astype(str)
    )
    ids_b = set(
        inventories["B"]["patient_id"].astype(str)
    )

    overlaps = sorted(ids_a & ids_b)

    pd.DataFrame(
        {
            "patient_id": overlaps,
        }
    ).to_csv(
        detail_dir
        / "cross_cohort_patient_id_overlap.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Reproducibility metadata
    # --------------------------------------------------------
    metadata = {
        "audit_version": AUDIT_VERSION,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "mode": (
            "full" if full_run else "smoke"
        ),
        "smoke_files_per_cohort": smoke_files,
        "config_file": str(config_path),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "git": git_metadata(),
    }

    with (
        aggregate_dir / "audit_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # Dataset manifest — only for a completed full audit
    # --------------------------------------------------------
    if full_run:
        manifest_records = []

        for cohort, acc in accumulators.items():
            manifest_records.append(
                {
                    "dataset": (
                        "PhysioNet/CinC Challenge 2019"
                    ),
                    "version": "1.0.0",
                    "cohort": cohort,
                    "source_key": (
                        f"training_set_{cohort.lower()}"
                    ),
                    "expected_subjects": (
                        acc.expected_count
                    ),
                    "observed_psv_files": (
                        acc.psv_files
                    ),
                    "parsed_files": (
                        acc.parsed_files
                    ),
                    "total_patient_hours": (
                        acc.total_patient_hours
                    ),
                    "dataset_tree_sha256": (
                        acc.tree_hasher.hexdigest()
                    ),
                }
            )

        pd.DataFrame(
            manifest_records
        ).to_csv(
            "data/metadata/dataset_manifest.csv",
            index=False,
        )


def print_final_summary(
    accumulators: dict[str, CohortAccumulator],
    smoke_files: int,
) -> None:
    print()
    print("=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)

    if smoke_files > 0:
        print(
            f"SMOKE MODE: first {smoke_files} "
            "PSV files per cohort only."
        )

    for cohort, acc in accumulators.items():
        print()
        print(f"Cohort {cohort}")
        print("-" * 40)

        print(
            f"PSV files found       : "
            f"{acc.psv_files:,}"
        )

        if smoke_files == 0:
            difference = (
                acc.psv_files
                - acc.expected_count
            )

            print(
                f"Official expected     : "
                f"{acc.expected_count:,}"
            )
            print(
                f"Observed - expected   : "
                f"{difference:+,}"
            )

        print(
            f"Parsed files          : "
            f"{acc.parsed_files:,}"
        )
        print(
            f"Patient-hours         : "
            f"{acc.total_patient_hours:,}"
        )
        print(
            f"Septic patients       : "
            f"{acc.septic_patients:,}"
        )
        print(
            f"Non-septic patients   : "
            f"{acc.nonseptic_patients:,}"
        )
        print(
            f"Schema mismatches     : "
            f"{acc.schema_mismatch_files:,}"
        )
        print(
            f"Parse errors          : "
            f"{acc.parse_error_files:,}"
        )
        print(
            f"Invalid labels        : "
            f"{acc.invalid_label_files:,}"
        )
        print(
            f"Label reversals       : "
            f"{acc.label_reversal_files:,}"
        )
        print(
            f"ICULOS problems       : "
            f"{acc.iculos_problem_files:,}"
        )
        print(
            f"Duplicate patient IDs : "
            f"{acc.duplicate_patient_ids:,}"
        )
        print(
            f"Tree SHA256           : "
            f"{acc.tree_hasher.hexdigest()}"
        )

    print()
    print("=" * 72)

    if smoke_files == 0:
        print(
            "Full audit outputs written to:"
        )
        print(
            "  aggregate: data/metadata/audit/"
        )
        print(
            "  local detail: data/interim/audit/"
        )
        print(
            "  manifest: data/metadata/"
            "dataset_manifest.csv"
        )
    else:
        print(
            "Smoke outputs written to:"
        )
        print(
            "  data/interim/audit_smoke/"
        )

    print("=" * 72)

def verify_frozen_reference(
    accumulators: dict[
        str,
        CohortAccumulator,
    ],
) -> list[str]:
    """
    Compare a full raw-data audit against the
    dataset snapshot used for the frozen study.

    The tree SHA-256 covers every .psv filename
    and every byte of every .psv file.
    """
    failures: list[str] = []

    print()
    print("=" * 72)
    print("FROZEN DATASET VERIFICATION")
    print("=" * 72)

    for cohort in (
        "A",
        "B",
    ):
        accumulator = (
            accumulators[
                cohort
            ]
        )

        expected = (
            FROZEN_DATASET_REFERENCE[
                cohort
            ]
        )

        actual_hash = (
            accumulator
            .tree_hasher
            .hexdigest()
        )

        checks = [
            (
                "PSV files",
                accumulator.psv_files,
                expected["psv_files"],
            ),
            (
                "Parsed files",
                accumulator.parsed_files,
                expected["psv_files"],
            ),
            (
                "Patient-hours",
                accumulator.total_patient_hours,
                expected["patient_hours"],
            ),
            (
                "Septic patients",
                accumulator.septic_patients,
                expected["septic_patients"],
            ),
            (
                "Non-septic patients",
                accumulator.nonseptic_patients,
                expected["nonseptic_patients"],
            ),
            (
                "Tree SHA256",
                actual_hash,
                expected["tree_sha256"],
            ),
        ]

        print()
        print(
            f"Cohort {cohort}"
        )
        print("-" * 40)

        for (
            name,
            actual,
            expected_value,
        ) in checks:
            passed = (
                actual
                == expected_value
            )

            print(
                f"{name:<22}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

            if not passed:
                failures.append(
                    f"Cohort {cohort} {name}: "
                    f"expected {expected_value}, "
                    f"found {actual}."
                )

        anomaly_checks = {
            "zero-byte files": (
                accumulator.zero_byte_files
            ),
            "parse errors": (
                accumulator.parse_error_files
            ),
            "schema mismatches": (
                accumulator.schema_mismatch_files
            ),
            "invalid labels": (
                accumulator.invalid_label_files
            ),
            "label reversals": (
                accumulator.label_reversal_files
            ),
            "ICULOS problems": (
                accumulator.iculos_problem_files
            ),
            "static-field changes": (
                accumulator.static_value_change_files
            ),
            "duplicate patient IDs": (
                accumulator.duplicate_patient_ids
            ),
        }

        for (
            name,
            count,
        ) in anomaly_checks.items():
            passed = (
                count == 0
            )

            print(
                f"{name:<22}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

            if not passed:
                failures.append(
                    f"Cohort {cohort} {name}: "
                    f"expected 0, found {count}."
                )

    print()
    print("=" * 72)

    if failures:
        print(
            "Frozen dataset verification: FAILED"
        )
    else:
        print(
            "Frozen dataset verification: PASS"
        )

    print("=" * 72)

    return failures


def main() -> int:
    args = parse_args()

    if args.smoke_files < 0:
        raise ValueError(
            "--smoke-files must be >= 0"
        )

    print("=" * 72)
    print(
        "PHYSIONET 2019 CROSS-HOSPITAL "
        "DATA AUDIT"
    )
    print("=" * 72)

    print(
        f"Audit version : {AUDIT_VERSION}"
    )
    print(
        f"Python        : {sys.version.split()[0]}"
    )
    print(
        f"Config        : {args.config}"
    )

    paths = load_paths(args.config)

    cohort_paths = {
        "A": paths["training_set_a"],
        "B": paths["training_set_b"],
    }

    accumulators: dict[
        str,
        CohortAccumulator,
    ] = {}

    inventories: dict[
        str,
        pd.DataFrame,
    ] = {}

    for cohort, path in cohort_paths.items():
        print()
        print(
            f"Cohort {cohort} path: {path}"
        )

        accumulator, inventory = audit_cohort(
            cohort=cohort,
            root=path,
            smoke_files=args.smoke_files,
        )

        accumulators[cohort] = accumulator
        inventories[cohort] = inventory

    save_outputs(
        accumulators=accumulators,
        inventories=inventories,
        smoke_files=args.smoke_files,
        config_path=args.config,
    )

    print_final_summary(
        accumulators=accumulators,
        smoke_files=args.smoke_files,
    )

    if args.smoke_files > 0:
        print()
        print(
            "Smoke audit complete. "
            "Frozen dataset identity was not "
            "verified in smoke mode."
        )

        return 0

    failures = verify_frozen_reference(
        accumulators
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