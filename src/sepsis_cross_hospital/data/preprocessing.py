from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


DYNAMIC_COLUMNS = [
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
]

CONTINUOUS_STATIC_COLUMNS = [
    "Age",
    "HospAdmTime",
]

CATEGORICAL_STATIC_COLUMNS = [
    "Gender",
    "Unit1",
    "Unit2",
]

CONTEXT_COLUMNS = [
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
]

REQUIRED_FEATURE_COLUMNS = (
    DYNAMIC_COLUMNS
    + CONTINUOUS_STATIC_COLUMNS
    + CATEGORICAL_STATIC_COLUMNS
    + ["ICULOS"]
)

Representation = Literal[
    "M0",
    "V0",
    "V1",
    "V2",
]


@dataclass(frozen=True)
class SourceStatistics:
    dynamic_medians: dict[str, float]
    continuous_static_medians: dict[str, float]
    categorical_static_modes: dict[str, float]
    source_available_dynamic: dict[str, bool]
    n_training_patients: int


def _validate_feature_columns(
    frame: pd.DataFrame,
) -> None:
    missing = (
        set(REQUIRED_FEATURE_COLUMNS)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(sorted(missing))
        )


def _first_nonmissing_numeric(
    series: pd.Series,
) -> float | None:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    observed = numeric.dropna()

    if observed.empty:
        return None

    return float(
        observed.iloc[0]
    )


def _flush_dynamic_buffers(
    buffers: dict[str, list[np.ndarray]],
    chunks: dict[str, list[np.ndarray]],
) -> None:
    for column in DYNAMIC_COLUMNS:
        arrays = buffers[column]

        if arrays:
            chunks[column].append(
                np.concatenate(arrays)
            )

            arrays.clear()


def fit_source_statistics(
    patients: Iterable[pd.DataFrame],
    *,
    batch_size: int = 500,
) -> SourceStatistics:
    """
    Fit preprocessing statistics from source-training
    patients only.

    Dynamic medians are calculated from observed
    source-training patient-hours.

    Static statistics are calculated at patient level.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    dynamic_buffers: dict[
        str,
        list[np.ndarray],
    ] = {
        column: []
        for column in DYNAMIC_COLUMNS
    }

    dynamic_chunks: dict[
        str,
        list[np.ndarray],
    ] = {
        column: []
        for column in DYNAMIC_COLUMNS
    }

    continuous_static_values: dict[
        str,
        list[float],
    ] = {
        column: []
        for column
        in CONTINUOUS_STATIC_COLUMNS
    }

    categorical_counts: dict[
        str,
        Counter[float],
    ] = {
        column: Counter()
        for column
        in CATEGORICAL_STATIC_COLUMNS
    }

    n_patients = 0

    for patient in patients:
        _validate_feature_columns(
            patient
        )

        n_patients += 1

        for column in DYNAMIC_COLUMNS:
            numeric = pd.to_numeric(
                patient[column],
                errors="coerce",
            )

            values = numeric.dropna().to_numpy(
                dtype=np.float64
            )

            if len(values):
                dynamic_buffers[
                    column
                ].append(values)

        for column in (
            CONTINUOUS_STATIC_COLUMNS
        ):
            value = _first_nonmissing_numeric(
                patient[column]
            )

            if value is not None:
                continuous_static_values[
                    column
                ].append(value)

        for column in (
            CATEGORICAL_STATIC_COLUMNS
        ):
            value = _first_nonmissing_numeric(
                patient[column]
            )

            if value is not None:
                categorical_counts[
                    column
                ][value] += 1

        if (
            n_patients % batch_size
            == 0
        ):
            _flush_dynamic_buffers(
                dynamic_buffers,
                dynamic_chunks,
            )

    if n_patients == 0:
        raise ValueError(
            "No source-training patients supplied."
        )

    _flush_dynamic_buffers(
        dynamic_buffers,
        dynamic_chunks,
    )

    dynamic_medians: dict[
        str,
        float,
    ] = {}

    source_available_dynamic: dict[
        str,
        bool,
    ] = {}

    for column in DYNAMIC_COLUMNS:
        chunks = dynamic_chunks[column]

        if not chunks:
            # Important cross-hospital rule:
            # variable never observed in source.
            dynamic_medians[column] = 0.0

            source_available_dynamic[
                column
            ] = False

            continue

        all_values = np.concatenate(
            chunks
        )

        dynamic_medians[column] = float(
            np.median(all_values)
        )

        source_available_dynamic[
            column
        ] = True

    continuous_static_medians: dict[
        str,
        float,
    ] = {}

    for column in (
        CONTINUOUS_STATIC_COLUMNS
    ):
        values = continuous_static_values[
            column
        ]

        if not values:
            raise ValueError(
                f"No observed source-training "
                f"values for static variable "
                f"{column}."
            )

        continuous_static_medians[
            column
        ] = float(
            np.median(
                np.asarray(
                    values,
                    dtype=np.float64,
                )
            )
        )

    categorical_static_modes: dict[
        str,
        float,
    ] = {}

    for column in (
        CATEGORICAL_STATIC_COLUMNS
    ):
        counts = categorical_counts[
            column
        ]

        if not counts:
            raise ValueError(
                f"No observed source-training "
                f"values for static variable "
                f"{column}."
            )

        # Deterministic mode:
        # highest count, then smallest value.
        mode_value = sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )[0][0]

        categorical_static_modes[
            column
        ] = float(mode_value)

    return SourceStatistics(
        dynamic_medians=dynamic_medians,
        continuous_static_medians=(
            continuous_static_medians
        ),
        categorical_static_modes=(
            categorical_static_modes
        ),
        source_available_dynamic=(
            source_available_dynamic
        ),
        n_training_patients=n_patients,
    )


def _measurement_features(
    original: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct strictly causal observation-process
    features.

    obs_*    : measured at current hour
    tsl_*    : hours since last measurement
    count6_* : number of measurements during current
               and previous five hours
    """
    iculos = pd.to_numeric(
        original["ICULOS"],
        errors="coerce",
    )

    if iculos.isna().any():
        raise ValueError(
            "ICULOS contains missing or "
            "non-numeric values."
        )

    feature_data: dict[
        str,
        np.ndarray,
    ] = {}

    iculos_values = iculos.to_numpy(
        dtype=np.float64
    )

    for column in DYNAMIC_COLUMNS:
        numeric = pd.to_numeric(
            original[column],
            errors="coerce",
        )

        observed = (
            numeric.notna()
            .to_numpy(dtype=np.int8)
        )

        feature_data[
            f"obs_{column}"
        ] = observed

        count6 = (
            pd.Series(
                observed,
                index=original.index,
            )
            .rolling(
                window=6,
                min_periods=1,
            )
            .sum()
            .to_numpy(dtype=np.float64)
        )

        feature_data[
            f"count6_{column}"
        ] = count6

        last_observed_iculos: (
            float | None
        ) = None

        time_since = np.empty(
            len(original),
            dtype=np.float64,
        )

        for position in range(
            len(original)
        ):
            current_iculos = float(
                iculos_values[position]
            )

            if observed[position]:
                last_observed_iculos = (
                    current_iculos
                )

                time_since[
                    position
                ] = 0.0

            elif (
                last_observed_iculos
                is None
            ):
                time_since[
                    position
                ] = current_iculos

            else:
                time_since[
                    position
                ] = max(
                    0.0,
                    current_iculos
                    - last_observed_iculos,
                )

        feature_data[
            f"tsl_{column}"
        ] = time_since

    return pd.DataFrame(
        feature_data,
        index=original.index,
    )

def transform_patient_causally(
    patient: pd.DataFrame,
    statistics: SourceStatistics,
) -> pd.DataFrame:
    """
    Produce causal values + context + measurement
    process features for one patient.

    This function does not use SepsisLabel.
    """
    _validate_feature_columns(
        patient
    )

    value_data: dict[
        str,
        np.ndarray,
    ] = {}

    # --------------------------------------------------
    # Dynamic physiological values
    # --------------------------------------------------
    for column in DYNAMIC_COLUMNS:
        available = (
            statistics
            .source_available_dynamic[
                column
            ]
        )

        if not available:
            value_data[column] = np.zeros(
                len(patient),
                dtype=np.float64,
            )
            continue

        numeric = pd.to_numeric(
            patient[column],
            errors="coerce",
        )

        causal_filled = (
            numeric.ffill()
            .fillna(
                statistics
                .dynamic_medians[column]
            )
        )

        value_data[column] = (
            causal_filled.to_numpy(
                dtype=np.float64
            )
        )

    # --------------------------------------------------
    # Continuous static/context variables
    # --------------------------------------------------
    for column in (
        CONTINUOUS_STATIC_COLUMNS
    ):
        numeric = pd.to_numeric(
            patient[column],
            errors="coerce",
        )

        value_data[column] = (
            numeric.fillna(
                statistics
                .continuous_static_medians[
                    column
                ]
            )
            .to_numpy(
                dtype=np.float64
            )
        )

    # --------------------------------------------------
    # Categorical static/context variables
    # --------------------------------------------------
    for column in (
        CATEGORICAL_STATIC_COLUMNS
    ):
        numeric = pd.to_numeric(
            patient[column],
            errors="coerce",
        )

        value_data[column] = (
            numeric.fillna(
                statistics
                .categorical_static_modes[
                    column
                ]
            )
            .to_numpy(
                dtype=np.float64
            )
        )

    # --------------------------------------------------
    # ICULOS
    # --------------------------------------------------
    iculos = pd.to_numeric(
        patient["ICULOS"],
        errors="coerce",
    )

    if iculos.isna().any():
        raise ValueError(
            "ICULOS contains missing or "
            "non-numeric values."
        )

    value_data["ICULOS"] = (
        iculos.to_numpy(
            dtype=np.float64
        )
    )

    values_frame = pd.DataFrame(
        value_data,
        index=patient.index,
    )

    measurement_frame = (
        _measurement_features(
            patient
        )
    )

    # --------------------------------------------------
    # Source-unavailable variables
    # --------------------------------------------------
    # If a variable was never observed in the source
    # training hospital, the source-trained model must
    # not gain access to either its values or its
    # measurement-process signal in the target hospital.
    for column in DYNAMIC_COLUMNS:
        if not (
            statistics
            .source_available_dynamic[
                column
            ]
        ):
            measurement_frame[
                f"obs_{column}"
            ] = 0

            measurement_frame[
                f"count6_{column}"
            ] = 0.0

            measurement_frame[
                f"tsl_{column}"
            ] = (
                iculos.to_numpy(
                    dtype=np.float64
                )
            )

    return pd.concat(
        [
            values_frame,
            measurement_frame,
        ],
        axis=1,
    )

def build_feature_representation(
    patient: pd.DataFrame,
    statistics: SourceStatistics,
    representation: Representation,
) -> pd.DataFrame:
    """
    Build one of the frozen feature ablations:

    M0 = measurement process + context
    V0 = values + context
    V1 = values + context + current masks
    V2 = values + context + masks + TSL + counts
    """
    engineered = (
        transform_patient_causally(
            patient,
            statistics,
        )
    )

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

    if representation == "M0":
        selected = (
            CONTEXT_COLUMNS
            + mask_columns
            + tsl_columns
            + count_columns
        )

    elif representation == "V0":
        selected = value_columns

    elif representation == "V1":
        selected = (
            value_columns
            + mask_columns
        )

    elif representation == "V2":
        selected = (
            value_columns
            + mask_columns
            + tsl_columns
            + count_columns
        )

    else:
        raise ValueError(
            f"Unknown representation: "
            f"{representation}"
        )

    result = engineered[
        selected
    ].copy()

    if "SepsisLabel" in result.columns:
        raise AssertionError(
            "SepsisLabel leaked into features."
        )

    return result