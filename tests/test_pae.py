from __future__ import annotations

import numpy as np
import pytest

from sepsis_cross_hospital.models.pae import (
    blend_probabilities,
    select_pae_alpha,
)


def test_blend_formula():
    v0 = [
        np.array(
            [0.2, 0.8]
        )
    ]

    v2 = [
        np.array(
            [0.6, 0.4]
        )
    ]

    result = blend_probabilities(
        v0,
        v2,
        alpha=0.25,
    )

    expected = np.array(
        [
            0.25 * 0.2
            + 0.75 * 0.6,
            0.25 * 0.8
            + 0.75 * 0.4,
        ]
    )

    np.testing.assert_allclose(
        result[0],
        expected,
    )


def test_mismatched_patient_counts_rejected():
    with pytest.raises(
        ValueError,
        match="patient counts",
    ):
        blend_probabilities(
            [
                np.array(
                    [0.1]
                )
            ],
            [
                np.array(
                    [0.1]
                ),
                np.array(
                    [0.2]
                ),
            ],
            alpha=0.5,
        )


def test_mismatched_sequence_lengths_rejected():
    with pytest.raises(
        ValueError,
        match="sequence",
    ):
        blend_probabilities(
            [
                np.array(
                    [0.1, 0.2]
                )
            ],
            [
                np.array(
                    [0.1]
                )
            ],
            alpha=0.5,
        )


def test_alpha_tie_prefers_larger_alpha():
    """
    V0 and V2 are identical, so every alpha gives
    identical probabilities and identical utility.
    Frozen tie-break must choose 0.75.
    """
    labels = [
        np.array(
            [0, 0, 1, 1],
            dtype=np.int8,
        ),
        np.array(
            [0, 0, 0, 0],
            dtype=np.int8,
        ),
    ]

    probabilities = [
        np.array(
            [0.1, 0.2, 0.8, 0.9]
        ),
        np.array(
            [0.1, 0.1, 0.1, 0.1]
        ),
    ]

    result = select_pae_alpha(
        labels,
        probabilities,
        probabilities,
    )

    assert result.alpha == pytest.approx(
        0.75
    )