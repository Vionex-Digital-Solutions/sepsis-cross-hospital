from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_sepsis_labels(
    labels: ArrayLike,
) -> NDArray[np.int8]:
    """
    Validate one patient's distributed SepsisLabel sequence.

    The PhysioNet/CinC 2019 SepsisLabel is already the
    early-warning target. This function validates it but
    NEVER shifts or otherwise transforms the labels.
    """
    array = np.asarray(labels)

    if array.ndim != 1:
        raise ValueError(
            "SepsisLabel must be one-dimensional."
        )

    if len(array) == 0:
        raise ValueError(
            "SepsisLabel must not be empty."
        )

    try:
        numeric = array.astype(
            np.float64,
            copy=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SepsisLabel must be numeric."
        ) from exc

    if np.isnan(numeric).any():
        raise ValueError(
            "SepsisLabel contains missing values."
        )

    if not np.all(
        np.isin(numeric, [0.0, 1.0])
    ):
        raise ValueError(
            "SepsisLabel must contain only 0 and 1."
        )

    labels_int = numeric.astype(
        np.int8,
        copy=True,
    )

    # Once the released Challenge label becomes 1,
    # it must not revert to 0.
    if np.any(
        np.diff(labels_int) < 0
    ):
        raise ValueError(
            "SepsisLabel contains a 1-to-0 reversal."
        )

    return labels_int


def prepare_hourly_target(
    labels: ArrayLike,
) -> NDArray[np.int8]:
    """
    Return the validated supervised target unchanged.

    IMPORTANT:
    No additional six-hour shift is performed here.
    The distributed Challenge SepsisLabel already
    represents the intended early-warning target.
    """
    return validate_sepsis_labels(
        labels
    )


def patient_is_septic(
    labels: ArrayLike,
) -> bool:
    validated = validate_sepsis_labels(
        labels
    )

    return bool(
        np.any(validated == 1)
    )


def first_positive_index(
    labels: ArrayLike,
) -> Optional[int]:
    """
    Return the zero-based index of the first positive
    distributed SepsisLabel, or None for a non-septic
    patient.
    """
    validated = validate_sepsis_labels(
        labels
    )

    positive = np.flatnonzero(
        validated == 1
    )

    if len(positive) == 0:
        return None

    return int(
        positive[0]
    )