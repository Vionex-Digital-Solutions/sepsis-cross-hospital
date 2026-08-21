from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class CausalGRU(nn.Module):
    """
    Frozen sequential baseline.

    Architecture:
    - one unidirectional GRU layer
    - hidden size 64
    - linear output at every patient-hour
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.output = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        features: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        features
            Shape:
            [batch, time, features]

        lengths
            True sequence length for each patient.

        Returns
        -------
        logits
            Shape:
            [batch, time]
        """
        if features.ndim != 3:
            raise ValueError(
                "features must have shape "
                "[batch, time, features]."
            )

        if lengths.ndim != 1:
            raise ValueError(
                "lengths must be one-dimensional."
            )

        if features.shape[0] != lengths.shape[0]:
            raise ValueError(
                "Batch size and number of lengths "
                "must match."
            )

        packed = pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.gru(
            packed
        )

        padded_output, _ = (
            pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=features.shape[1],
            )
        )

        logits = self.output(
            padded_output
        ).squeeze(-1)

        return logits