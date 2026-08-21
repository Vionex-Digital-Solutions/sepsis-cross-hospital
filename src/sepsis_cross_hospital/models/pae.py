from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sepsis_cross_hospital.evaluation.thresholding import (
    select_utility_threshold,
)


FROZEN_ALPHAS = (
    0.25,
    0.50,
    0.75,
)


@dataclass(frozen=True)
class AlphaCandidate:
    alpha: float
    threshold: float
    validation_utility: float


@dataclass(frozen=True)
class PAESelection:
    alpha: float
    threshold: float
    validation_utility: float
    candidates: tuple[AlphaCandidate, ...]
    probabilities_by_patient: tuple[
        np.ndarray,
        ...,
    ]


def blend_probabilities(
    v0_probabilities: Sequence[np.ndarray],
    v2_probabilities: Sequence[np.ndarray],
    alpha: float,
) -> list[np.ndarray]:
    """
    PAE fusion:

        p = alpha * p_V0
            + (1 - alpha) * p_V2

    alpha therefore represents reliance on the
    physiology-oriented V0 branch.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must lie in [0, 1]."
        )

    if (
        len(v0_probabilities)
        != len(v2_probabilities)
    ):
        raise ValueError(
            "V0 and V2 patient counts differ."
        )

    blended: list[np.ndarray] = []

    for v0, v2 in zip(
        v0_probabilities,
        v2_probabilities,
        strict=True,
    ):
        v0_array = np.asarray(
            v0,
            dtype=np.float64,
        )

        v2_array = np.asarray(
            v2,
            dtype=np.float64,
        )

        if v0_array.shape != v2_array.shape:
            raise ValueError(
                "V0 and V2 patient sequence "
                "lengths differ."
            )

        probability = (
            alpha * v0_array
            + (1.0 - alpha)
            * v2_array
        )

        blended.append(
            probability
        )

    return blended


def select_pae_alpha(
    labels_by_patient: Sequence[np.ndarray],
    v0_probabilities: Sequence[np.ndarray],
    v2_probabilities: Sequence[np.ndarray],
    *,
    alphas: Sequence[float] = FROZEN_ALPHAS,
) -> PAESelection:
    """
    Select alpha and threshold using source validation
    Challenge utility only.

    Frozen tie-break:
    choose the larger alpha when utilities tie.
    """
    if len(alphas) == 0:
        raise ValueError(
            "At least one alpha is required."
        )

    candidates: list[
        AlphaCandidate
    ] = []

    probability_sets: dict[
        float,
        list[np.ndarray],
    ] = {}

    for alpha_value in alphas:
        alpha = float(
            alpha_value
        )

        probabilities = (
            blend_probabilities(
                v0_probabilities,
                v2_probabilities,
                alpha,
            )
        )

        threshold_result = (
            select_utility_threshold(
                labels_by_patient,
                probabilities,
            )
        )

        candidates.append(
            AlphaCandidate(
                alpha=alpha,
                threshold=float(
                    threshold_result.threshold
                ),
                validation_utility=float(
                    threshold_result
                    .normalized_utility
                ),
            )
        )

        probability_sets[
            alpha
        ] = probabilities

    maximum = max(
        candidate.validation_utility
        for candidate in candidates
    )

    tied = [
        candidate
        for candidate in candidates
        if np.isclose(
            candidate.validation_utility,
            maximum,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    # Frozen tie-break:
    # favor more physiology / less explicit
    # measurement-process dependence.
    selected = max(
        tied,
        key=lambda candidate: (
            candidate.alpha
        ),
    )

    return PAESelection(
        alpha=selected.alpha,
        threshold=selected.threshold,
        validation_utility=(
            selected.validation_utility
        ),
        candidates=tuple(
            candidates
        ),
        probabilities_by_patient=tuple(
            probability_sets[
                selected.alpha
            ]
        ),
    )