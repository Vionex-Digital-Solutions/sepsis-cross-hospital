import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_revision_ensembles import mean_probabilities, patient_sequences


def test_probability_averaging_is_not_majority_voting():
    seeds = [np.array([value]) for value in [0.4, 0.4, 0.4, 0.9, 0.9]]
    probability = mean_probabilities(seeds)
    np.testing.assert_allclose(probability, [0.6], rtol=0, atol=1e-15)
    assert bool(probability[0] >= 0.5)
    assert sum(bool(values[0] >= 0.5) for values in seeds) == 2


@pytest.mark.parametrize("invalid", [np.array([np.nan]), np.array([1.1])])
def test_invalid_probabilities_are_rejected(invalid):
    with pytest.raises(ValueError, match="Invalid seed probabilities"):
        mean_probabilities([invalid] * 5)


def test_patient_sequences_preserve_hour_alignment():
    frame = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2", "p2", "p2"],
        "hour_index": [0, 1, 0, 1, 2],
    })
    sequences = patient_sequences(frame, np.array([10, 11, 20, 21, 22]))
    assert [values.tolist() for values in sequences] == [[10, 11], [20, 21, 22]]


def test_patient_hour_length_mismatch_is_rejected():
    frame = pd.DataFrame({"patient_id": ["p1", "p1"]})
    with pytest.raises(ValueError, match="alignment mismatch"):
        patient_sequences(frame, np.array([0.1]))
