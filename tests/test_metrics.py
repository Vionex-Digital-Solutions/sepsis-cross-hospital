from __future__ import annotations

import numpy as np
import pytest

from sepsis_cross_hospital.evaluation.metrics import (
    compute_challenge_utility,
    compute_patient_utility,
    make_optimal_predictions,
)


def test_official_reference_patient_utility():
    """
    Reference example given in the official
    Challenge evaluation code.
    """
    labels = np.array(
        [0, 0, 0, 0, 1, 1]
    )

    predictions = np.array(
        [0, 0, 1, 1, 1, 1]
    )

    utility = compute_patient_utility(
        labels,
        predictions,
    )

    assert utility == pytest.approx(
        3.388888888888889
    )


def make_mixed_test_cohort():
    septic = np.array(
        [0] * 10 + [1] * 5,
        dtype=np.int8,
    )

    nonseptic = np.zeros(
        15,
        dtype=np.int8,
    )

    return septic, nonseptic


def test_inactive_classifier_normalizes_to_zero():
    septic, nonseptic = (
        make_mixed_test_cohort()
    )

    labels = [
        septic,
        nonseptic,
    ]

    predictions = [
        np.zeros_like(septic),
        np.zeros_like(nonseptic),
    ]

    result = compute_challenge_utility(
        labels,
        predictions,
    )

    assert result.normalized == pytest.approx(
        0.0
    )


def test_optimal_classifier_normalizes_to_one():
    septic, nonseptic = (
        make_mixed_test_cohort()
    )

    labels = [
        septic,
        nonseptic,
    ]

    predictions = [
        make_optimal_predictions(
            septic
        ),
        make_optimal_predictions(
            nonseptic
        ),
    ]

    result = compute_challenge_utility(
        labels,
        predictions,
    )

    assert result.normalized == pytest.approx(
        1.0
    )


def test_nonseptic_false_positives_are_penalized():
    labels = np.zeros(
        5,
        dtype=np.int8,
    )

    predictions = np.array(
        [1, 0, 1, 0, 1],
        dtype=np.int8,
    )

    utility = compute_patient_utility(
        labels,
        predictions,
    )

    assert utility == pytest.approx(
        -0.15
    )


def test_late_prediction_is_worse_than_optimal():
    septic, _ = (
        make_mixed_test_cohort()
    )

    optimal = (
        make_optimal_predictions(
            septic
        )
    )

    late = np.zeros_like(
        septic
    )

    late[13:] = 1

    optimal_utility = (
        compute_patient_utility(
            septic,
            optimal,
        )
    )

    late_utility = (
        compute_patient_utility(
            septic,
            late,
        )
    )

    assert (
        late_utility
        < optimal_utility
    )


def test_invalid_labels_are_rejected():
    labels = np.array(
        [0, 0, 2, 1]
    )

    predictions = np.array(
        [0, 0, 0, 1]
    )

    with pytest.raises(
        ValueError,
        match="only 0 and 1",
    ):
        compute_patient_utility(
            labels,
            predictions,
        )


def test_length_mismatch_is_rejected():
    labels = np.array(
        [0, 0, 1, 1]
    )

    predictions = np.array(
        [0, 1, 1]
    )

    with pytest.raises(
        ValueError,
        match="must be the same",
    ):
        compute_patient_utility(
            labels,
            predictions,
        )