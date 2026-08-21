from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ============================================================
# PhysioNet/CinC Challenge 2019 utility parameters
# ============================================================

DT_EARLY = -12
DT_OPTIMAL = -6
DT_LATE = 3

MAX_U_TP = 1.0
MIN_U_FN = -2.0
U_FP = -0.05
U_TN = 0.0


@dataclass(frozen=True)
class ChallengeUtilityResult:
    normalized: float
    observed: float
    optimal: float
    inactive: float


def _as_binary_vector(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.int8]:
    """
    Convert a one-dimensional binary array-like input
    into an int8 NumPy vector.

    Raises ValueError for:
    - non-1D inputs
    - empty inputs
    - values other than 0 or 1
    """
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(array) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.all(
        np.isin(array, [0, 1])
    ):
        raise ValueError(
            f"{name} must contain only 0 and 1."
        )

    return array.astype(
        np.int8,
        copy=False,
    )


def compute_patient_hour_utilities(
    labels: ArrayLike,
    *,
    dt_early: int = DT_EARLY,
    dt_optimal: int = DT_OPTIMAL,
    dt_late: int = DT_LATE,
    max_u_tp: float = MAX_U_TP,
    min_u_fn: float = MIN_U_FN,
    u_fp: float = U_FP,
    u_tn: float = U_TN,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Return the utility at every patient-hour for making
    a negative prediction and a positive prediction.

    Returns
    -------
    utility_negative
        Utility received if prediction == 0 at each hour.

    utility_positive
        Utility received if prediction == 1 at each hour.

    Notes
    -----
    This follows the PhysioNet/CinC Challenge 2019
    piecewise utility function.

    The distributed SepsisLabel is already shifted
    six hours before clinical sepsis onset.

    Therefore:

        t_sepsis = first_positive_label_index - dt_optimal

    With dt_optimal == -6, this becomes:

        t_sepsis = first_positive_label_index + 6
    """
    y_true = _as_binary_vector(
        labels,
        name="labels",
    )

    if dt_early >= dt_optimal:
        raise ValueError(
            "dt_early must be before dt_optimal."
        )

    if dt_optimal >= dt_late:
        raise ValueError(
            "dt_optimal must be before dt_late."
        )

    # --------------------------------------------------------
    # Determine whether the patient becomes septic and infer
    # the clinical sepsis time from the released early label.
    # --------------------------------------------------------
    if np.any(y_true):
        is_septic = True

        t_sepsis = (
            int(np.argmax(y_true))
            - dt_optimal
        )
    else:
        is_septic = False
        t_sepsis = float("inf")

    # --------------------------------------------------------
    # Piecewise utility coefficients
    # --------------------------------------------------------

    # Rising TP utility from dt_early to dt_optimal.
    m_1 = (
        float(max_u_tp)
        / float(
            dt_optimal - dt_early
        )
    )

    b_1 = (
        -m_1
        * dt_early
    )

    # Falling TP utility from dt_optimal to dt_late.
    m_2 = (
        float(-max_u_tp)
        / float(
            dt_late - dt_optimal
        )
    )

    b_2 = (
        -m_2
        * dt_late
    )

    # Increasing FN penalty from dt_optimal to dt_late.
    m_3 = (
        float(min_u_fn)
        / float(
            dt_late - dt_optimal
        )
    )

    b_3 = (
        -m_3
        * dt_optimal
    )

    utility_negative = np.zeros(
        len(y_true),
        dtype=np.float64,
    )

    utility_positive = np.zeros(
        len(y_true),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Compute per-hour utilities
    # --------------------------------------------------------
    for t in range(
        len(y_true)
    ):
        # Challenge scoring ignores predictions sufficiently
        # late after sepsis onset.
        if (
            t
            > t_sepsis
            + dt_late
        ):
            continue

        if is_septic:
            # ------------------------------------------------
            # Before / at the optimal prediction time
            # ------------------------------------------------
            if (
                t
                <= t_sepsis
                + dt_optimal
            ):
                utility_positive[
                    t
                ] = max(
                    (
                        m_1
                        * (
                            t
                            - t_sepsis
                        )
                        + b_1
                    ),
                    u_fp,
                )

                utility_negative[
                    t
                ] = 0.0

            # ------------------------------------------------
            # Between optimal time and the late boundary
            # ------------------------------------------------
            elif (
                t
                <= t_sepsis
                + dt_late
            ):
                utility_positive[
                    t
                ] = (
                    m_2
                    * (
                        t
                        - t_sepsis
                    )
                    + b_2
                )

                utility_negative[
                    t
                ] = (
                    m_3
                    * (
                        t
                        - t_sepsis
                    )
                    + b_3
                )

        else:
            # Non-septic patient:
            # positive prediction = false-positive penalty
            # negative prediction = true-negative utility
            utility_positive[
                t
            ] = u_fp

            utility_negative[
                t
            ] = u_tn

    return (
        utility_negative,
        utility_positive,
    )


def compute_patient_utility(
    labels: ArrayLike,
    predictions: ArrayLike,
    *,
    dt_early: int = DT_EARLY,
    dt_optimal: int = DT_OPTIMAL,
    dt_late: int = DT_LATE,
    max_u_tp: float = MAX_U_TP,
    min_u_fn: float = MIN_U_FN,
    u_fp: float = U_FP,
    u_tn: float = U_TN,
) -> float:
    """
    Compute the unnormalized Challenge utility for one
    patient's binary prediction sequence.
    """
    y_true = _as_binary_vector(
        labels,
        name="labels",
    )

    y_pred = _as_binary_vector(
        predictions,
        name="predictions",
    )

    if len(y_true) != len(
        y_pred
    ):
        raise ValueError(
            "Numbers of predictions and labels "
            "must be the same."
        )

    (
        utility_negative,
        utility_positive,
    ) = compute_patient_hour_utilities(
        y_true,
        dt_early=dt_early,
        dt_optimal=dt_optimal,
        dt_late=dt_late,
        max_u_tp=max_u_tp,
        min_u_fn=min_u_fn,
        u_fp=u_fp,
        u_tn=u_tn,
    )

    return float(
        np.sum(
            np.where(
                y_pred == 1,
                utility_positive,
                utility_negative,
            )
        )
    )


def make_optimal_predictions(
    labels: ArrayLike,
    *,
    dt_early: int = DT_EARLY,
    dt_optimal: int = DT_OPTIMAL,
    dt_late: int = DT_LATE,
) -> NDArray[np.int8]:
    """
    Construct the optimal binary prediction vector used
    in the official Challenge normalization.

    For septic patients, optimal predictions are positive
    throughout the utility-relevant interval.

    For non-septic patients, the optimal classifier remains
    negative throughout.
    """
    y_true = _as_binary_vector(
        labels,
        name="labels",
    )

    if dt_early >= dt_optimal:
        raise ValueError(
            "dt_early must be before dt_optimal."
        )

    if dt_optimal >= dt_late:
        raise ValueError(
            "dt_optimal must be before dt_late."
        )

    predictions = np.zeros(
        len(y_true),
        dtype=np.int8,
    )

    if not np.any(
        y_true
    ):
        return predictions

    t_sepsis = (
        int(np.argmax(y_true))
        - dt_optimal
    )

    start = max(
        0,
        int(
            t_sepsis
            + dt_early
        ),
    )

    stop = min(
        int(
            t_sepsis
            + dt_late
            + 1
        ),
        len(y_true),
    )

    if start < stop:
        predictions[
            start:stop
        ] = 1

    return predictions


def compute_challenge_utility(
    labels_by_patient: Sequence[
        ArrayLike
    ],
    predictions_by_patient: Sequence[
        ArrayLike
    ],
) -> ChallengeUtilityResult:
    """
    Compute normalized PhysioNet/CinC Challenge 2019
    utility for an entire cohort.

    Normalized utility:

        observed - inactive
        -------------------
         optimal - inactive

    The inactive classifier always predicts 0.

    The optimal classifier is generated separately for
    each patient using the Challenge utility definition.
    """
    if (
        len(
            labels_by_patient
        )
        != len(
            predictions_by_patient
        )
    ):
        raise ValueError(
            "Numbers of patient label sequences "
            "and prediction sequences must match."
        )

    if len(
        labels_by_patient
    ) == 0:
        raise ValueError(
            "At least one patient is required."
        )

    observed_total = 0.0
    optimal_total = 0.0
    inactive_total = 0.0

    for (
        labels,
        predictions,
    ) in zip(
        labels_by_patient,
        predictions_by_patient,
        strict=True,
    ):
        y_true = _as_binary_vector(
            labels,
            name="labels",
        )

        y_pred = _as_binary_vector(
            predictions,
            name="predictions",
        )

        if len(
            y_true
        ) != len(
            y_pred
        ):
            raise ValueError(
                "Numbers of predictions and labels "
                "must be the same for each patient."
            )

        optimal_predictions = (
            make_optimal_predictions(
                y_true
            )
        )

        inactive_predictions = np.zeros(
            len(y_true),
            dtype=np.int8,
        )

        observed_total += (
            compute_patient_utility(
                y_true,
                y_pred,
            )
        )

        optimal_total += (
            compute_patient_utility(
                y_true,
                optimal_predictions,
            )
        )

        inactive_total += (
            compute_patient_utility(
                y_true,
                inactive_predictions,
            )
        )

    denominator = (
        optimal_total
        - inactive_total
    )

    if denominator <= 0:
        raise ValueError(
            "Challenge utility cannot be normalized "
            "because optimal and inactive utilities "
            "are identical. The cohort may contain "
            "no septic patients."
        )

    normalized = (
        observed_total
        - inactive_total
    ) / denominator

    return ChallengeUtilityResult(
        normalized=float(
            normalized
        ),
        observed=float(
            observed_total
        ),
        optimal=float(
            optimal_total
        ),
        inactive=float(
            inactive_total
        ),
    )