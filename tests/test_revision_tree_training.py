import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_revision_trees.py"
spec = importlib.util.spec_from_file_location("revision_tree_training", SCRIPT)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)

@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "train.parquet"
    pd.DataFrame({
        "patient_id": ["b", "a", "b", "a"],
        "hour_index": [1, 0, 0, 1],
        "SepsisLabel": [1, 0, 0, 1],
        "HR": [40.0, 10.0, 30.0, 20.0],
    }).to_parquet(path, index=False)
    return path, training.sha256(path)

def test_sorting_preserves_feature_label_alignment(source_file):
    path, digest = source_file
    frame = training.load_source(path, ["HR"], digest, 2)
    assert frame["HR"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert frame["SepsisLabel"].tolist() == [0, 1, 0, 1]
    assert frame["hour_index"].tolist() == [0, 1, 0, 1]

def test_modified_data_rejected_before_training(source_file):
    path, digest = source_file
    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        training.load_source(path, ["HR"], digest, 2)

def test_label_cannot_be_used_as_feature(source_file):
    path, digest = source_file
    with pytest.raises(ValueError, match="Invalid feature"):
        training.load_source(path, ["HR", "SepsisLabel"], digest, 2)

def test_existing_artifact_is_preserved(tmp_path):
    path = tmp_path / "seed.joblib"
    original = b"existing artifact"
    path.write_bytes(original)
    with pytest.raises(FileExistsError):
        training.dump_new(path, {"replacement": True})
    assert path.read_bytes() == original
