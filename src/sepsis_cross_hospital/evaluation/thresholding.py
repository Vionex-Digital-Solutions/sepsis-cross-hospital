from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sepsis_cross_hospital.evaluation.metrics import (
    compute_patient_hour_utilities,
    make_optimal_predictions,
)


@dataclass(frozen=True)
class UtilityThresholdResult:
    threshold: float
    normalized_utility: float
    observed_utility: float
    optimal_utility: float
    inactive_utility: float
    thresholds: NDArray[np.float64]
    utilities: NDArray[np.float64]


def _validate_probabilities(
    probabilities: ArrayLike,
) -> NDArray[np.float64]:
    array = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Probabilities contain non-finite values."
        )

    if np.any(
        (array < 0.0)
        | (array > 1.0)
    ):
        raise ValueError(
            "Probabilities must lie in [0, 1]."
        )

    return array


def select_utility_threshold(
    labels_by_patient: Sequence[ArrayLike],
    probabilities_by_patient: Sequence[ArrayLike],
    thresholds: ArrayLike | None = None,
) -> UtilityThresholdResult:
    if (
        len(labels_by_patient)
        != len(probabilities_by_patient)
    ):
        raise ValueError(
            "Patient sequence counts must match."
        )

    if not labels_by_patient:
        raise ValueError(
            "At least one patient is required."
        )

    if thresholds is None:
        threshold_values = np.arange(
            0.001,
            1.000,
            0.001,
            dtype=np.float64,
        )
    else:
        threshold_values = np.asarray(
            thresholds,
            dtype=np.float64,
        )

    if threshold_values.ndim != 1:
        raise ValueError(
            "Thresholds must be one-dimensional."
        )

    if len(threshold_values) == 0:
        raise ValueError(
            "At least one threshold is required."
        )

    if not np.isfinite(
        threshold_values
    ).all():
        raise ValueError(
            "Thresholds contain non-finite values."
        )

    all_probabilities = []
    all_gains = []

    inactive_total = 0.0
    optimal_total = 0.0

    for labels, probabilities in zip(
        labels_by_patient,
        probabilities_by_patient,
        strict=True,
    ):
        y_true = np.asarray(
            labels,
            dtype=np.int8,
        )

        probability = (
            _validate_probabilities(
                probabilities
            )
        )

        if (
            len(y_true)
            != len(probability)
        ):
            raise ValueError(
                "Label/probability lengths "
                "must match."
            )

        (
            utility_negative,
            utility_positive,
        ) = compute_patient_hour_utilities(
            y_true
        )

        gain = (
            utility_positive
            - utility_negative
        )

        optimal_predictions = (
            make_optimal_predictions(
                y_true
            )
        )

        inactive_total += float(
            utility_negative.sum()
        )

        optimal_total += float(
            np.where(
                optimal_predictions == 1,
                utility_positive,
                utility_negative,
            ).sum()
        )

        all_probabilities.append(
            probability
        )

        all_gains.append(
            gain
        )

    probabilities_flat = np.concatenate(
        all_probabilities
    )

    gains_flat = np.concatenate(
        all_gains
    )

    denominator = (
        optimal_total
        - inactive_total
    )

    if denominator <= 0:
        raise ValueError(
            "Utility normalization denominator "
            "must be positive."
        )

    order = np.argsort(
        probabilities_flat,
        kind="mergesort",
    )

    sorted_probabilities = (
        probabilities_flat[order]
    )

    sorted_gains = (
        gains_flat[order]
    )

    reverse_gain = np.cumsum(
        sorted_gains[::-1],
        dtype=np.float64,
    )[::-1]

    utilities = np.empty(
        len(threshold_values),
        dtype=np.float64,
    )

    observed_values = np.empty(
        len(threshold_values),
        dtype=np.float64,
    )

    for index, threshold in enumerate(
        threshold_values
    ):
        first_selected = np.searchsorted(
            sorted_probabilities,
            threshold,
            side="left",
        )

        if (
            first_selected
            == len(
                sorted_probabilities
            )
        ):
            selected_gain = 0.0
        else:
            selected_gain = float(
                reverse_gain[
                    first_selected
                ]
            )

        observed = (
            inactive_total
            + selected_gain
        )

        normalized = (
            observed
            - inactive_total
        ) / denominator

        observed_values[
            index
        ] = observed

        utilities[
            index
        ] = normalized

    maximum = float(
        np.max(utilities)
    )

    tied = np.flatnonzero(
        np.isclose(
            utilities,
            maximum,
            rtol=0.0,
            atol=1e-12,
        )
    )

    # Frozen protocol:
    # if utility ties, choose higher threshold.
    best_index = int(
        tied[-1]
    )

    return UtilityThresholdResult(
        threshold=float(
            threshold_values[
                best_index
            ]
        ),
        normalized_utility=float(
            utilities[
                best_index
            ]
        ),
        observed_utility=float(
            observed_values[
                best_index
            ]
        ),
        optimal_utility=float(
            optimal_total
        ),
        inactive_utility=float(
            inactive_total
        ),
        thresholds=threshold_values,
        utilities=utilities,
    )