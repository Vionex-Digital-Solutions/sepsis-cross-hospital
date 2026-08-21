from __future__ import annotations

import numpy as np
import pytest

from sepsis_cross_hospital.evaluation.metrics import (
    compute_challenge_utility,
)
from sepsis_cross_hospital.evaluation.thresholding import (
    select_utility_threshold,
)


def make_data():
    septic = np.array(
        [0] * 8 + [1] * 5,
        dtype=np.int8,
    )

    nonseptic = np.zeros(
        13,
        dtype=np.int8,
    )

    septic_prob = np.array(
        [
            0.01,
            0.02,
            0.03,
            0.10,
            0.20,
            0.40,
            0.70,
            0.80,
            0.90,
            0.95,
            0.95,
            0.90,
            0.85,
        ]
    )

    nonseptic_prob = np.array(
        [
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.10,
            0.11,
            0.12,
            0.13,
        ]
    )

    return (
        [septic, nonseptic],
        [
            septic_prob,
            nonseptic_prob,
        ],
    )


def test_fast_threshold_matches_brute_force():
    labels, probabilities = (
        make_data()
    )

    thresholds = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.40,
            0.80,
        ]
    )

    result = select_utility_threshold(
        labels,
        probabilities,
        thresholds,
    )

    brute = []

    for threshold in thresholds:
        predictions = [
            (
                probabilities[index]
                >= threshold
            ).astype(np.int8)
            for index in range(
                len(labels)
            )
        ]

        utility = (
            compute_challenge_utility(
                labels,
                predictions,
            ).normalized
        )

        brute.append(
            utility
        )

    assert np.allclose(
        result.utilities,
        brute,
        rtol=0.0,
        atol=1e-12,
    )


def test_tied_threshold_chooses_higher():
    labels = [
        np.array(
            [0, 0, 1, 1],
            dtype=np.int8,
        )
    ]

    probabilities = [
        np.array(
            [
                0.1,
                0.1,
                0.9,
                0.9,
            ]
        )
    ]

    result = select_utility_threshold(
        labels,
        probabilities,
        thresholds=np.array(
            [
                0.2,
                0.3,
                0.4,
            ]
        ),
    )

    assert (
        result.threshold
        == pytest.approx(0.4)
    )


def test_invalid_probability_rejected():
    labels = [
        np.array(
            [0, 1],
            dtype=np.int8,
        )
    ]

    probabilities = [
        np.array(
            [0.1, 1.5]
        )
    ]

    with pytest.raises(
        ValueError,
        match="\\[0, 1\\]",
    ):
        select_utility_threshold(
            labels,
            probabilities,
        )