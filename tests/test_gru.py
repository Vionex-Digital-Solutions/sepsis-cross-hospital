from __future__ import annotations

import torch

from sepsis_cross_hospital.models.gru import (
    CausalGRU,
)


def test_gru_output_shape():
    torch.manual_seed(1729)

    model = CausalGRU(
        input_size=5,
        hidden_size=8,
    )

    features = torch.randn(
        3,
        7,
        5,
    )

    lengths = torch.tensor(
        [7, 5, 3],
        dtype=torch.long,
    )

    logits = model(
        features,
        lengths,
    )

    assert logits.shape == (
        3,
        7,
    )


def test_gru_is_prefix_causal():
    """
    Appending future observations must not change
    logits for earlier patient-hours.
    """
    torch.manual_seed(1729)

    model = CausalGRU(
        input_size=4,
        hidden_size=8,
    )

    model.eval()

    full_sequence = torch.randn(
        1,
        8,
        4,
    )

    prefix_sequence = (
        full_sequence[
            :,
            :5,
            :,
        ].clone()
    )

    with torch.no_grad():
        full_logits = model(
            full_sequence,
            torch.tensor(
                [8],
                dtype=torch.long,
            ),
        )

        prefix_logits = model(
            prefix_sequence,
            torch.tensor(
                [5],
                dtype=torch.long,
            ),
        )

    torch.testing.assert_close(
        full_logits[:, :5],
        prefix_logits,
        rtol=1e-5,
        atol=1e-6,
    )