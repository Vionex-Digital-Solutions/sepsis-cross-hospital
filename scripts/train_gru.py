from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Sequence

# PyTorch documents deterministic CUDA/RNN behavior
# as potentially requiring a cuBLAS workspace setting.
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from sepsis_cross_hospital.evaluation.metrics import (
    compute_challenge_utility,
)
from sepsis_cross_hospital.evaluation.thresholding import (
    select_utility_threshold,
)
from sepsis_cross_hospital.models.gru import (
    CausalGRU,
)


# ============================================================
# Frozen protocol
# ============================================================

HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 60
PATIENCE = 8

# Frozen study configuration.
BATCH_SIZE = 128

CLIP_LIMIT = 10.0

SEEDS = (
    1729,
    2718,
    31415,
    57721,
    65537,
)

DEVICE_NAME = "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen causal GRU baseline."
        )
    )

    parser.add_argument(
        "--direction",
        required=True,
        choices=[
            "A_to_B",
            "B_to_A",
        ],
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    return parser.parse_args()


def git_commit() -> str:
    """
    Return the current Git commit when available.

    During clean-room reproduction testing, the
    public repository may intentionally have no
    commit yet.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return "UNCOMMITTED"


def git_dirty() -> bool:
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return bool(
        result.stdout.strip()
    )


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True
    )


def load_feature_columns(
    direction: str,
) -> list[str]:
    path = (
        Path(
            "data/metadata/"
            "preprocessing"
        )
        / direction
        / "feature_schema.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        schema = json.load(
            handle
        )

    # Frozen GRU input:
    #
    # values/context
    # + current masks
    # + time since last measurement
    #
    # 6-hour count features are excluded.
    columns = [
        column
        for column in schema["V2"]
        if not column.startswith(
            "count6_"
        )
    ]

    if len(columns) != 108:
        raise RuntimeError(
            "Expected exactly 108 GRU "
            f"features, found {len(columns)}."
        )

    return columns


def load_partition(
    direction: str,
    partition: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    if partition == "external":
        raise RuntimeError(
            "External evaluation is locked."
        )

    path = (
        Path("data/processed")
        / direction
        / f"{partition}.parquet"
    )

    frame = pd.read_parquet(
        path,
        columns=(
            [
                "patient_id",
                "hour_index",
                "SepsisLabel",
            ]
            + feature_columns
        ),
    )

    frame = (
        frame.sort_values(
            [
                "patient_id",
                "hour_index",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    validate_patient_hours(
        frame
    )

    return frame


def validate_patient_hours(
    frame: pd.DataFrame,
) -> None:
    duplicated = frame.duplicated(
        subset=[
            "patient_id",
            "hour_index",
        ]
    )

    if duplicated.any():
        raise RuntimeError(
            "Duplicate patient/hour rows detected."
        )

    for _, group in frame.groupby(
        "patient_id",
        sort=False,
    ):
        hours = group[
            "hour_index"
        ].to_numpy()

        expected = np.arange(
            len(group)
        )

        if not np.array_equal(
            hours,
            expected,
        ):
            raise RuntimeError(
                "Patient hour_index is not "
                "sequential from zero."
            )


class PatientSequenceDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            str,
        ]
    ]
):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        patient_ids: np.ndarray,
    ) -> None:
        if len(features) != len(labels):
            raise ValueError(
                "Feature/label lengths differ."
            )

        if len(features) != len(
            patient_ids
        ):
            raise ValueError(
                "Feature/patient lengths differ."
            )

        if len(features) == 0:
            raise ValueError(
                "Dataset must not be empty."
            )

        patient_ids = np.asarray(
            patient_ids,
            dtype=str,
        )

        changes = (
            np.flatnonzero(
                patient_ids[1:]
                != patient_ids[:-1]
            )
            + 1
        )

        starts = np.concatenate(
            [
                np.array(
                    [0],
                    dtype=np.int64,
                ),
                changes.astype(
                    np.int64
                ),
            ]
        )

        stops = np.concatenate(
            [
                changes.astype(
                    np.int64
                ),
                np.array(
                    [len(patient_ids)],
                    dtype=np.int64,
                ),
            ]
        )

        unique_patients = np.unique(
            patient_ids
        )

        if len(unique_patients) != len(
            starts
        ):
            raise RuntimeError(
                "Patient rows are not contiguous."
            )

        self.features = (
            np.ascontiguousarray(
                features,
                dtype=np.float32,
            )
        )

        self.labels = (
            np.ascontiguousarray(
                labels,
                dtype=np.float32,
            )
        )

        self.starts = starts
        self.stops = stops

        self.patient_ids = [
            str(
                patient_ids[start]
            )
            for start in starts
        ]

    def __len__(self) -> int:
        return len(
            self.starts
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        str,
    ]:
        start = int(
            self.starts[index]
        )

        stop = int(
            self.stops[index]
        )

        features = torch.from_numpy(
            self.features[
                start:stop
            ]
        )

        labels = torch.from_numpy(
            self.labels[
                start:stop
            ]
        )

        return (
            features,
            labels,
            self.patient_ids[index],
        )


def collate_patient_batch(
    batch: Sequence[
        tuple[
            torch.Tensor,
            torch.Tensor,
            str,
        ]
    ],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
]:
    features = [
        item[0]
        for item in batch
    ]

    labels = [
        item[1]
        for item in batch
    ]

    patient_ids = [
        item[2]
        for item in batch
    ]

    lengths = torch.tensor(
        [
            item.shape[0]
            for item in features
        ],
        dtype=torch.long,
    )

    padded_features = pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0,
    )

    padded_labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=0.0,
    )

    return (
        padded_features,
        padded_labels,
        lengths,
        patient_ids,
    )


def make_loader(
    dataset: PatientSequenceDataset,
    *,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None

    if shuffle:
        if seed is None:
            raise ValueError(
                "A seed is required "
                "for shuffled loading."
            )

        generator = torch.Generator()

        generator.manual_seed(
            seed
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=(
            collate_patient_batch
        ),
        generator=generator,
    )


def valid_mask(
    lengths: torch.Tensor,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(
        max_length,
        device=device,
    ).unsqueeze(0)

    return (
        positions
        < lengths.to(
            device=device
        ).unsqueeze(1)
    )


def train_epoch(
    model: CausalGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_hours = 0

    for (
        features,
        labels,
        lengths,
        _,
    ) in loader:
        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            features,
            lengths,
        )

        mask = valid_mask(
            lengths,
            logits.shape[1],
            device,
        )

        losses = (
            F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
        )

        loss_sum = losses.masked_select(
            mask
        ).sum()

        n_hours = int(
            mask.sum().item()
        )

        loss = (
            loss_sum
            / n_hours
        )

        loss.backward()

        optimizer.step()

        total_loss += float(
            loss_sum.detach().item()
        )

        total_hours += n_hours

    return (
        total_loss
        / total_hours
    )


@torch.inference_mode()
def validation_loss(
    model: CausalGRU,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_hours = 0

    for (
        features,
        labels,
        lengths,
        _,
    ) in loader:
        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(
            features,
            lengths,
        )

        mask = valid_mask(
            lengths,
            logits.shape[1],
            device,
        )

        losses = (
            F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
        )

        loss_sum = losses.masked_select(
            mask
        ).sum()

        total_loss += float(
            loss_sum.item()
        )

        total_hours += int(
            mask.sum().item()
        )

    return (
        total_loss
        / total_hours
    )


@torch.inference_mode()
def predict_dataset(
    model: CausalGRU,
    dataset: PatientSequenceDataset,
    device: torch.device,
) -> tuple[
    list[str],
    list[np.ndarray],
    list[np.ndarray],
]:
    loader = make_loader(
        dataset,
        shuffle=False,
    )

    model.eval()

    patient_ids: list[str] = []
    labels_by_patient: list[
        np.ndarray
    ] = []

    probabilities_by_patient: list[
        np.ndarray
    ] = []

    for (
        features,
        labels,
        lengths,
        batch_patient_ids,
    ) in loader:
        features = features.to(
            device,
            non_blocking=True,
        )

        logits = model(
            features,
            lengths,
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu()

        for batch_index, patient_id in enumerate(
            batch_patient_ids
        ):
            length = int(
                lengths[
                    batch_index
                ].item()
            )

            patient_ids.append(
                patient_id
            )

            labels_by_patient.append(
                labels[
                    batch_index,
                    :length,
                ]
                .numpy()
                .astype(
                    np.int8,
                    copy=False,
                )
            )

            probabilities_by_patient.append(
                probabilities[
                    batch_index,
                    :length,
                ]
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )

    return (
        patient_ids,
        labels_by_patient,
        probabilities_by_patient,
    )


def ensemble_probabilities(
    seed_probabilities: Sequence[
        Sequence[np.ndarray]
    ],
) -> list[np.ndarray]:
    if not seed_probabilities:
        raise ValueError(
            "No seed probabilities supplied."
        )

    n_patients = len(
        seed_probabilities[0]
    )

    for probabilities in (
        seed_probabilities
    ):
        if len(probabilities) != n_patients:
            raise ValueError(
                "Seed patient counts differ."
            )

    ensemble = []

    for patient_index in range(
        n_patients
    ):
        stacked = np.stack(
            [
                probabilities[
                    patient_index
                ]
                for probabilities
                in seed_probabilities
            ],
            axis=0,
        )

        ensemble.append(
            np.mean(
                stacked,
                axis=0,
            )
        )

    return ensemble


def evaluate_sequences(
    labels_by_patient: list[
        np.ndarray
    ],
    probabilities_by_patient: list[
        np.ndarray
    ],
    threshold: float,
) -> dict:
    predictions_by_patient = [
        (
            probabilities
            >= threshold
        ).astype(
            np.int8
        )
        for probabilities
        in probabilities_by_patient
    ]

    utility = (
        compute_challenge_utility(
            labels_by_patient,
            predictions_by_patient,
        ).normalized
    )

    labels = np.concatenate(
        labels_by_patient
    ).astype(
        np.int8,
        copy=False,
    )

    probabilities = np.concatenate(
        probabilities_by_patient
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int8
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[
                0,
                1,
            ],
        ).ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else float("nan")
    )

    ppv = (
        tp / (tp + fp)
        if (tp + fp)
        else float("nan")
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn)
        else float("nan")
    )

    return {
        "utility": float(
            utility
        ),
        "auroc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "auprc": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "brier": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        "threshold": float(
            threshold
        ),
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "ppv": float(
            ppv
        ),
        "npv": float(
            npv
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def save_predictions(
    path: Path,
    patient_ids: list[str],
    labels_by_patient: list[
        np.ndarray
    ],
    probabilities_by_patient: list[
        np.ndarray
    ],
) -> None:
    frames = []

    for (
        patient_id,
        labels,
        probabilities,
    ) in zip(
        patient_ids,
        labels_by_patient,
        probabilities_by_patient,
        strict=True,
    ):
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": (
                        patient_id
                    ),
                    "hour_index": (
                        np.arange(
                            len(labels)
                        )
                    ),
                    "label": labels,
                    "probability": (
                        probabilities
                    ),
                }
            )
        )

    pd.concat(
        frames,
        ignore_index=True,
    ).to_parquet(
        path,
        index=False,
    )


def prepare_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    scaler: RobustScaler,
) -> PatientSequenceDataset:
    raw = frame[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    transformed = scaler.transform(
        raw
    ).astype(
        np.float32,
        copy=False,
    )

    np.clip(
        transformed,
        -CLIP_LIMIT,
        CLIP_LIMIT,
        out=transformed,
    )

    labels = frame[
        "SepsisLabel"
    ].to_numpy(
        dtype=np.float32
    )

    patient_ids = frame[
        "patient_id"
    ].astype(str).to_numpy()

    return PatientSequenceDataset(
        transformed,
        labels,
        patient_ids,
    )


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this "
            "configured GRU run."
        )

    device = torch.device(
        DEVICE_NAME
    )

    run_dir = (
        Path("runs")
        / args.run_id
    )

    if run_dir.exists():
        raise FileExistsError(
            f"Run already exists: "
            f"{run_dir}"
        )

    run_dir.mkdir(
        parents=True
    )

    checkpoints_dir = (
        run_dir
        / "checkpoints"
    )

    checkpoints_dir.mkdir()

    print("=" * 72)
    print(
        "CAUSAL GRU BASELINE"
    )
    print("=" * 72)

    print(
        f"Run ID      : "
        f"{args.run_id}"
    )

    print(
        f"Direction   : "
        f"{args.direction}"
    )

    print(
        f"PyTorch     : "
        f"{torch.__version__}"
    )

    print(
        f"Device      : "
        f"{torch.cuda.get_device_name(0)}"
    )

    feature_columns = (
        load_feature_columns(
            args.direction
        )
    )

    print(
        f"Features    : "
        f"{len(feature_columns)}"
    )

    print(
        f"Hidden size : "
        f"{HIDDEN_SIZE}"
    )

    print(
        f"Batch size  : "
        f"{BATCH_SIZE} patients"
    )

    # ========================================================
    # Source TRAIN only: fit robust scaling.
    # ========================================================

    print()
    print(
        "Loading source train..."
    )

    train_frame = load_partition(
        args.direction,
        "train",
        feature_columns,
    )

    print(
        f"Train hours : "
        f"{len(train_frame):,}"
    )

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(
            25.0,
            75.0,
        ),
    )

    train_raw = train_frame[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    print(
        "Fitting source-training "
        "RobustScaler..."
    )

    scaler.fit(
        train_raw
    )

    del train_raw

    train_dataset = prepare_dataset(
        train_frame,
        feature_columns,
        scaler,
    )

    del train_frame

    gc.collect()

    # ========================================================
    # Source validation.
    # ========================================================

    print(
        "Loading source validation..."
    )

    validation_frame = load_partition(
        args.direction,
        "validation",
        feature_columns,
    )

    print(
        f"Validation hours : "
        f"{len(validation_frame):,}"
    )

    validation_dataset = (
        prepare_dataset(
            validation_frame,
            feature_columns,
            scaler,
        )
    )

    del validation_frame

    gc.collect()

    validation_loader = make_loader(
        validation_dataset,
        shuffle=False,
    )

    seed_validation_probabilities = []

    training_history_rows = []

    seed_metadata = []

    validation_patient_ids: (
        list[str] | None
    ) = None

    validation_labels: (
        list[np.ndarray] | None
    ) = None

    print()
    print("=" * 72)
    print(
        "FIVE-SEED GRU TRAINING"
    )
    print("=" * 72)

    for seed in SEEDS:
        print()
        print(
            "-" * 72
        )

        print(
            f"Seed {seed}"
        )

        set_seed(
            seed
        )

        train_loader = make_loader(
            train_dataset,
            shuffle=True,
            seed=seed,
        )

        model = CausalGRU(
            input_size=len(
                feature_columns
            ),
            hidden_size=HIDDEN_SIZE,
        ).to(
            device
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        checkpoint_path = (
            checkpoints_dir
            / f"seed_{seed}.pt"
        )

        best_validation_loss = (
            float("inf")
        )

        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(
            1,
            MAX_EPOCHS + 1,
        ):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
            )

            val_loss = validation_loss(
                model,
                validation_loader,
                device,
            )

            training_history_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "train_bce": (
                        train_loss
                    ),
                    "validation_bce": (
                        val_loss
                    ),
                }
            )

            print(
                f"  epoch {epoch:02d} | "
                f"train BCE "
                f"{train_loss:.6f} | "
                f"val BCE "
                f"{val_loss:.6f}"
            )

            if (
                val_loss
                < best_validation_loss
                - 1e-8
            ):
                best_validation_loss = (
                    val_loss
                )

                best_epoch = epoch

                epochs_without_improvement = 0

                torch.save(
                    model.state_dict(),
                    checkpoint_path,
                )

            else:
                epochs_without_improvement += 1

                if (
                    epochs_without_improvement
                    >= PATIENCE
                ):
                    print(
                        "  early stopping"
                    )

                    break

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
            )
        )

        (
            patient_ids,
            labels_by_patient,
            probabilities_by_patient,
        ) = predict_dataset(
            model,
            validation_dataset,
            device,
        )

        if validation_patient_ids is None:
            validation_patient_ids = (
                patient_ids
            )

            validation_labels = (
                labels_by_patient
            )

        else:
            if (
                patient_ids
                != validation_patient_ids
            ):
                raise RuntimeError(
                    "Validation patient order "
                    "changed across seeds."
                )

        seed_validation_probabilities.append(
            probabilities_by_patient
        )

        seed_metadata.append(
            {
                "seed": seed,
                "best_epoch": (
                    best_epoch
                ),
                "best_validation_bce": (
                    best_validation_loss
                ),
            }
        )

        print(
            f"  best epoch : "
            f"{best_epoch}"
        )

        print(
            f"  best BCE   : "
            f"{best_validation_loss:.6f}"
        )

        del model
        del optimizer
        del train_loader

        torch.cuda.empty_cache()

        gc.collect()

    if (
        validation_patient_ids
        is None
        or validation_labels
        is None
    ):
        raise RuntimeError(
            "No validation predictions "
            "were generated."
        )

    validation_probabilities = (
        ensemble_probabilities(
            seed_validation_probabilities
        )
    )

    threshold_result = (
        select_utility_threshold(
            validation_labels,
            validation_probabilities,
        )
    )

    threshold = float(
        threshold_result.threshold
    )

    validation_metrics = (
        evaluate_sequences(
            validation_labels,
            validation_probabilities,
            threshold,
        )
    )

    print()
    print("=" * 72)
    print(
        "SOURCE VALIDATION ENSEMBLE"
    )
    print("=" * 72)

    print(
        f"Threshold : "
        f"{threshold:.3f}"
    )

    print(
        f"Utility   : "
        f"{validation_metrics['utility']:.6f}"
    )

    print(
        f"AUROC     : "
        f"{validation_metrics['auroc']:.6f}"
    )

    print(
        f"AUPRC     : "
        f"{validation_metrics['auprc']:.6f}"
    )

    print(
        f"Brier     : "
        f"{validation_metrics['brier']:.6f}"
    )

    # ========================================================
    # Source internal test only.
    # External remains locked.
    # ========================================================

    print()
    print(
        "Loading source internal test..."
    )

    internal_frame = load_partition(
        args.direction,
        "internal_test",
        feature_columns,
    )

    print(
        f"Internal hours : "
        f"{len(internal_frame):,}"
    )

    internal_dataset = (
        prepare_dataset(
            internal_frame,
            feature_columns,
            scaler,
        )
    )

    del internal_frame

    gc.collect()

    seed_internal_probabilities = []

    internal_patient_ids: (
        list[str] | None
    ) = None

    internal_labels: (
        list[np.ndarray] | None
    ) = None

    for seed in SEEDS:
        print(
            f"Predicting internal "
            f"seed {seed}..."
        )

        model = CausalGRU(
            input_size=len(
                feature_columns
            ),
            hidden_size=HIDDEN_SIZE,
        ).to(
            device
        )

        checkpoint_path = (
            checkpoints_dir
            / f"seed_{seed}.pt"
        )

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
            )
        )

        (
            patient_ids,
            labels_by_patient,
            probabilities_by_patient,
        ) = predict_dataset(
            model,
            internal_dataset,
            device,
        )

        if internal_patient_ids is None:
            internal_patient_ids = (
                patient_ids
            )

            internal_labels = (
                labels_by_patient
            )

        else:
            if (
                patient_ids
                != internal_patient_ids
            ):
                raise RuntimeError(
                    "Internal patient order "
                    "changed across seeds."
                )

        seed_internal_probabilities.append(
            probabilities_by_patient
        )

        del model

        torch.cuda.empty_cache()

        gc.collect()

    if (
        internal_patient_ids is None
        or internal_labels is None
    ):
        raise RuntimeError(
            "No internal predictions generated."
        )

    internal_probabilities = (
        ensemble_probabilities(
            seed_internal_probabilities
        )
    )

    internal_metrics = (
        evaluate_sequences(
            internal_labels,
            internal_probabilities,
            threshold,
        )
    )

    print()
    print("=" * 72)
    print(
        "SOURCE INTERNAL TEST"
    )
    print("=" * 72)

    print(
        f"Utility : "
        f"{internal_metrics['utility']:.6f}"
    )

    print(
        f"AUROC   : "
        f"{internal_metrics['auroc']:.6f}"
    )

    print(
        f"AUPRC   : "
        f"{internal_metrics['auprc']:.6f}"
    )

    print(
        f"Brier   : "
        f"{internal_metrics['brier']:.6f}"
    )

    print()
    print(
        "External hospital: NOT EVALUATED"
    )

    # ========================================================
    # Artifacts
    # ========================================================

    joblib.dump(
        scaler,
        run_dir
        / "scaler.joblib",
    )

    pd.DataFrame(
        training_history_rows
    ).to_csv(
        run_dir
        / "training_history.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "threshold": (
                threshold_result
                .thresholds
            ),
            "normalized_utility": (
                threshold_result
                .utilities
            ),
        }
    ).to_csv(
        run_dir
        / "threshold_curve.csv",
        index=False,
    )

    save_predictions(
        run_dir
        / "validation_predictions.parquet",
        validation_patient_ids,
        validation_labels,
        validation_probabilities,
    )

    save_predictions(
        run_dir
        / "internal_predictions.parquet",
        internal_patient_ids,
        internal_labels,
        internal_probabilities,
    )

    metadata = {
        "run_id": args.run_id,
        "model": "GRU",
        "direction": (
            args.direction
        ),
        "external_evaluated": False,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "torch_version": (
            torch.__version__
        ),
        "cuda_version": (
            torch.version.cuda
        ),
        "device": (
            torch.cuda
            .get_device_name(0)
        ),
        "feature_definition": (
            "values/context + current masks "
            "+ time-since-last-measurement"
        ),
        "feature_count": len(
            feature_columns
        ),
        "hidden_size": (
            HIDDEN_SIZE
        ),
        "num_layers": 1,
        "bidirectional": False,
        "learning_rate": (
            LEARNING_RATE
        ),
        "optimizer": "Adam",
        "loss": (
            "unweighted BCEWithLogitsLoss"
        ),
        "batch_size_patients": (
            BATCH_SIZE
        ),
        "max_epochs": (
            MAX_EPOCHS
        ),
        "early_stopping_patience": (
            PATIENCE
        ),
        "early_stopping_metric": (
            "source validation BCE"
        ),
        "scaling": (
            "source-train RobustScaler "
            "median/IQR, clipped [-10,10]"
        ),
        "seeds": list(
            SEEDS
        ),
        "seed_metadata": (
            seed_metadata
        ),
        "selected_threshold": (
            threshold
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "internal_test_metrics": (
            internal_metrics
        ),
    }

    with (
        run_dir
        / "run_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print()
    print(
        f"Artifacts: "
        f"{run_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )