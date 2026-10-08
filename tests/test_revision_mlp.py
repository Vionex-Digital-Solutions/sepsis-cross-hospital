import numpy as np
import pytest
import torch
from torch import nn

from sepsis_cross_hospital.models import revision_mlp as mlp

def test_validation_values_do_not_change_source_scaler():
    scaler = mlp.fit_scaler(np.array([[0.0], [2.0], [4.0]], dtype=np.float32))
    transformed = mlp.scale_features(
        np.array([[100.0]], dtype=np.float32), scaler, 10.0
    )
    assert scaler.center_.tolist() == [2.0]
    assert scaler.scale_.tolist() == [2.0]
    assert transformed.tolist() == [[10.0]]

def test_checkpoint_restores_identical_predictions():
    torch.manual_seed(17)
    original = mlp.make_mlp(2, [4, 3]).eval()
    restored = mlp.restore_mlp(mlp.pack_state(original), 2, [4, 3])
    values = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    with torch.inference_mode():
        assert torch.equal(original(values), restored(values))

def test_checkpoint_rejects_wrong_feature_width():
    state = mlp.pack_state(mlp.make_mlp(2, [4, 3]))
    with pytest.raises(RuntimeError, match="size mismatch"):
        mlp.restore_mlp(state, 3, [4, 3])

def test_validation_loss_weights_final_partial_batch_correctly():
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.5)
    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0], [3.0]])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    logits = features.numpy().ravel().astype(np.float64) + 0.5
    expected = np.mean(np.logaddexp(0.0, logits) - labels.numpy() * logits)
    actual = mlp.validation_bce(model, features, labels, batch_size=2)
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
