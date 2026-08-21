from __future__ import annotations

import numpy as np
import pytest

from sepsis_cross_hospital.data.validation import (
    first_positive_index,
    patient_is_septic,
    prepare_hourly_target,
    validate_sepsis_labels,
)


def test_official_target_is_preserved_exactly():
    labels = np.array(
        [0, 0, 0, 1, 1, 1],
        dtype=np.int8,
    )

    target = prepare_hourly_target(
        labels
    )

    assert np.array_equal(
        target,
        labels,
    )


def test_target_is_not_shifted_again():
    labels = np.array(
        [0, 0, 0, 0, 1, 1, 1],
        dtype=np.int8,
    )

    target = prepare_hourly_target(
        labels
    )

    assert first_positive_index(
        target
    ) == 4


def test_nonseptic_patient_is_preserved():
    labels = np.zeros(
        12,
        dtype=np.int8,
    )

    target = prepare_hourly_target(
        labels
    )

    assert np.array_equal(
        target,
        labels,
    )

    assert (
        patient_is_septic(target)
        is False
    )


def test_septic_patient_is_detected():
    labels = np.array(
        [0, 0, 0, 1, 1],
        dtype=np.int8,
    )

    assert (
        patient_is_septic(labels)
        is True
    )


def test_first_positive_index():
    labels = np.array(
        [0, 0, 0, 1, 1],
        dtype=np.int8,
    )

    assert (
        first_positive_index(labels)
        == 3
    )


def test_nonseptic_first_positive_is_none():
    labels = np.zeros(
        8,
        dtype=np.int8,
    )

    assert (
        first_positive_index(labels)
        is None
    )


def test_label_reversal_is_rejected():
    labels = np.array(
        [0, 0, 1, 1, 0, 1],
        dtype=np.int8,
    )

    with pytest.raises(
        ValueError,
        match="1-to-0 reversal",
    ):
        validate_sepsis_labels(
            labels
        )


def test_nonbinary_label_is_rejected():
    labels = np.array(
        [0, 0, 2, 1]
    )

    with pytest.raises(
        ValueError,
        match="only 0 and 1",
    ):
        validate_sepsis_labels(
            labels
        )


def test_missing_label_is_rejected():
    labels = np.array(
        [0.0, 0.0, np.nan, 1.0]
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        validate_sepsis_labels(
            labels
        )


def test_empty_label_sequence_is_rejected():
    labels = np.array(
        [],
        dtype=np.int8,
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        validate_sepsis_labels(
            labels
        )


def test_two_dimensional_labels_are_rejected():
    labels = np.array(
        [
            [0, 0],
            [1, 1],
        ]
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_sepsis_labels(
            labels
        )


def test_target_function_returns_copy():
    labels = np.array(
        [0, 0, 1, 1],
        dtype=np.int8,
    )

    target = prepare_hourly_target(
        labels
    )

    target[0] = 1

    assert labels[0] == 0