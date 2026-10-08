"""Shared tabular MLP construction, scaling, and checkpoint helpers."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from torch import nn

def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def make_mlp(feature_count, hidden_sizes):
    widths = [feature_count, *hidden_sizes, 1]
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right, dtype=torch.float32))
        if index < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def fit_scaler(source_training_features, quantiles=(25.0, 75.0)):
    raw = np.asarray(source_training_features, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Invalid source-training matrix.")
    return RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=tuple(quantiles),
    ).fit(raw)

def scale_features(raw, scaler, clip=10.0):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Invalid feature matrix.")
    values = scaler.transform(raw).astype(np.float32, copy=False)
    np.clip(values, -clip, clip, out=values)
    return np.ascontiguousarray(values)

def pack_state(model):
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in model.state_dict().items()
    }

def restore_mlp(state, feature_count, hidden_sizes):
    model = make_mlp(feature_count, hidden_sizes)
    model.load_state_dict({
        name: torch.from_numpy(value)
        for name, value in state.items()
    }, strict=True)
    return model.eval()

@torch.inference_mode()
def validation_bce(model, features, labels, batch_size):
    model.eval()
    total_loss = 0.0
    for start in range(0, len(labels), batch_size):
        logits = model(features[start:start + batch_size]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels[start:start + batch_size], reduction="sum"
        )
        total_loss += float(loss.item())
    return total_loss / len(labels)
