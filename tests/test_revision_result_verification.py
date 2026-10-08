import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_revision_results import compare_tables

def files(tmp_path, expected, actual):
    first, second = tmp_path / "expected.csv", tmp_path / "actual.csv"
    expected.to_csv(first, index=False)
    actual.to_csv(second, index=False)
    return first, second

def test_comparison_aligns_model_keys_and_column_order(tmp_path):
    expected = pd.DataFrame({"model": ["a", "b"], "utility": [0.1, 0.3]})
    actual = expected.iloc[::-1][["utility", "model"]]
    compare_tables(*files(tmp_path, expected, actual), ["model"])

def test_comparison_rejects_changed_result(tmp_path):
    expected = pd.DataFrame({"model": ["a"], "utility": [0.1]})
    actual = pd.DataFrame({"model": ["a"], "utility": [0.101]})
    with pytest.raises(AssertionError, match="Numerical mismatch"):
        compare_tables(*files(tmp_path, expected, actual), ["model"])

def test_comparison_rejects_duplicate_model_keys(tmp_path):
    expected = pd.DataFrame({"model": ["a"], "utility": [0.1]})
    actual = pd.concat([expected, expected], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_tables(*files(tmp_path, expected, actual), ["model"])

def test_comparison_rejects_missing_model(tmp_path):
    expected = pd.DataFrame({"model": ["a", "b"], "utility": [0.1, 0.3]})
    with pytest.raises(ValueError, match="inventory"):
        compare_tables(*files(tmp_path, expected, expected.iloc[:1]), ["model"])

def test_comparison_preserves_matching_missing_diagnostics(tmp_path):
    expected = pd.DataFrame({"model": ["a"], "calibration_slope": [np.nan]})
    compare_tables(*files(tmp_path, expected, expected.copy()), ["model"])
