from __future__ import annotations

import numpy as np
import pytest

from sepsis_cross_hospital.evaluation.external import (
    calibration_intercept_slope,
    paired_patient_bootstrap,
    utility_components_by_patient,
)
from sepsis_cross_hospital.evaluation.metrics import (
    compute_challenge_utility,
)


def test_patient_components_reconstruct_utility():
    labels = [
        np.array(
            [0, 0, 1, 1, 1],
            dtype=np.int8,
        ),
        np.array(
            [0, 0, 0, 0],
            dtype=np.int8,
        ),
    ]

    predictions = [
        np.array(
            [0, 1, 1, 1, 1],
            dtype=np.int8,
        ),
        np.array(
            [0, 0, 0, 0],
            dtype=np.int8,
        ),
    ]

    full = compute_challenge_utility(
        labels,
        predictions,
    )

    components = (
        utility_components_by_patient(
            labels,
            predictions,
        )
    )

    observed = (
        components.observed.sum()
    )

    optimal = (
        components.optimal.sum()
    )

    inactive = (
        components.inactive.sum()
    )

    reconstructed = (
        observed - inactive
    ) / (
        optimal - inactive
    )

    assert reconstructed == pytest.approx(
        full.normalized
    )


def test_paired_bootstrap_identical_models_identical():
    labels = [
        np.array(
            [0, 0, 1, 1],
            dtype=np.int8,
        ),
        np.array(
            [0, 0, 0],
            dtype=np.int8,
        ),
        np.array(
            [0, 1, 1],
            dtype=np.int8,
        ),
    ]

    predictions = [
        np.array(
            [0, 0, 1, 1],
            dtype=np.int8,
        ),
        np.array(
            [0, 0, 0],
            dtype=np.int8,
        ),
        np.array(
            [0, 1, 1],
            dtype=np.int8,
        ),
    ]

    result = paired_patient_bootstrap(
        labels,
        {
            "A": predictions,
            "B": predictions,
        },
        n_bootstrap=50,
        seed=1729,
    )

    np.testing.assert_array_equal(
        result["A"],
        result["B"],
    )


def test_calibration_recovers_near_identity():
    probability_levels = np.array(
        [
            0.1,
            0.2,
            0.5,
            0.8,
            0.9,
        ],
        dtype=np.float64,
    )

    probabilities = []
    labels = []

    for probability in (
        probability_levels
    ):
        n = 100

        n_positive = int(
            round(
                probability * n
            )
        )

        probabilities.extend(
            [probability] * n
        )

        labels.extend(
            [1] * n_positive
        )

        labels.extend(
            [0]
            * (
                n - n_positive
            )
        )

    result = (
        calibration_intercept_slope(
            np.asarray(
                labels,
                dtype=np.int8,
            ),
            np.asarray(
                probabilities,
                dtype=np.float64,
            ),
        )
    )

    assert result.success

    assert result.intercept == pytest.approx(
        0.0,
        abs=0.1,
    )

    assert result.slope == pytest.approx(
        1.0,
        abs=0.1,
    )


def test_calibration_rejects_invalid_probability():
    with pytest.raises(
        ValueError,
        match="Probabilities",
    ):
        calibration_intercept_slope(
            np.array(
                [0, 1],
                dtype=np.int8,
            ),
            np.array(
                [-0.1, 0.9],
                dtype=np.float64,
            ),
        )