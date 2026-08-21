from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from torch.nn.utils.rnn import (
    pad_sequence,
)

from sepsis_cross_hospital.evaluation.external import (
    calibration_intercept_slope,
    paired_patient_bootstrap,
    percentile_ci,
)
from sepsis_cross_hospital.evaluation.metrics import (
    compute_challenge_utility,
)
from sepsis_cross_hospital.models.gru import (
    CausalGRU,
)
from sepsis_cross_hospital.models.pae import (
    blend_probabilities,
)


BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 1729

MODEL_ORDER = (
    "Logistic_V0",
    "XGB_M0",
    "XGB_V0",
    "XGB_V1",
    "XGB_V2",
    "GRU",
    "PAE",
)

FINAL_RUNS = {
    "A_to_B": {
        "Logistic_V0": "A_to_B__logistic_v0",
        "XGB_M0": "A_to_B__xgb_m0",
        "XGB_V0": "A_to_B__xgb_v0",
        "XGB_V1": "A_to_B__xgb_v1",
        "XGB_V2": "A_to_B__xgb_v2",
        "GRU": "A_to_B__gru",
        "PAE": "A_to_B__pae",
    },
    "B_to_A": {
        "Logistic_V0": "B_to_A__logistic_v0",
        "XGB_M0": "B_to_A__xgb_m0",
        "XGB_V0": "B_to_A__xgb_v0",
        "XGB_V1": "B_to_A__xgb_v1",
        "XGB_V2": "B_to_A__xgb_v2",
        "GRU": "B_to_A__gru",
        "PAE": "B_to_A__pae",
    },
}

EXPECTED_EXTERNAL = {
    "A_to_B": {
        "sha256": (
            "eef0c3183e8d251e74d83a784c84d76b"
            "db307867d3ad65994cc65ab4e01daec6"
        ),
        "patients": 20000,
        "hours": 761995,
    },
    "B_to_A": {
        "sha256": (
            "38d0ec27765ae5b1393cb2d2caa414bb"
            "3e0499d102d482ef1fa25f97c467974b"
        ),
        "patients": 20336,
        "hours": 790215,
    },
}

FROZEN_SEEDS = [
    1729,
    2718,
    31415,
    57721,
    65537,
]


FROZEN_SOURCE_RUNS = {
    "A_to_B": {
        "Logistic_V0": {
            "selected_C": 10.0,
            "selected_threshold": 0.027,
            "internal_utility": 0.32267397302268586,
        },
        "XGB_M0": {
            "representation": "M0",
            "max_depth": 3,
            "learning_rate": 0.10,
            "selected_threshold": 0.024,
            "internal_utility": 0.416402,
        },
        "XGB_V0": {
            "representation": "V0",
            "max_depth": 5,
            "learning_rate": 0.10,
            "selected_threshold": 0.024,
            "internal_utility": 0.424778,
        },
        "XGB_V1": {
            "representation": "V1",
            "max_depth": 3,
            "learning_rate": 0.03,
            "selected_threshold": 0.024,
            "internal_utility": 0.428781,
        },
        "XGB_V2": {
            "representation": "V2",
            "max_depth": 5,
            "learning_rate": 0.03,
            "selected_threshold": 0.022,
            "internal_utility": 0.44308450848150405,
        },
        "GRU": {
            "selected_threshold": 0.030,
            "internal_utility": 0.4008328223993462,
        },
        "PAE": {
            "selected_alpha": 0.25,
            "selected_threshold": 0.022,
            "v0_max_depth": 5,
            "v0_learning_rate": 0.10,
            "v2_max_depth": 5,
            "v2_learning_rate": 0.03,
            "internal_utility": 0.4514791538933168,
        },
    },
    "B_to_A": {
        "Logistic_V0": {
            "selected_C": 10.0,
            "selected_threshold": 0.023,
            "internal_utility": 0.301128,
        },
        "XGB_M0": {
            "representation": "M0",
            "max_depth": 5,
            "learning_rate": 0.10,
            "selected_threshold": 0.024,
            "internal_utility": 0.419070,
        },
        "XGB_V0": {
            "representation": "V0",
            "max_depth": 5,
            "learning_rate": 0.03,
            "selected_threshold": 0.020,
            "internal_utility": 0.377259,
        },
        "XGB_V1": {
            "representation": "V1",
            "max_depth": 5,
            "learning_rate": 0.10,
            "selected_threshold": 0.022,
            "internal_utility": 0.390961,
        },
        "XGB_V2": {
            "representation": "V2",
            "max_depth": 5,
            "learning_rate": 0.03,
            "selected_threshold": 0.019,
            "internal_utility": 0.419892,
        },
        "GRU": {
            "selected_threshold": 0.026,
            "internal_utility": 0.412500,
        },
        "PAE": {
            "selected_alpha": 0.25,
            "selected_threshold": 0.018,
            "v0_max_depth": 5,
            "v0_learning_rate": 0.03,
            "v2_max_depth": 5,
            "v2_learning_rate": 0.03,
            "internal_utility": 0.416334,
        },
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen final cross-hospital external "
            "evaluation."
        )
    )

    parser.add_argument(
        "--run-id",
        help=(
            "Run ID for the atomic final "
            "external evaluation."
        ),
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate frozen artifacts without "
            "calculating external model performance."
        ),
    )

    args = parser.parse_args()

    if (
        not args.preflight
        and not args.run_id
    ):
        parser.error(
            "--run-id is required unless "
            "--preflight is used."
        )

    return args


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


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def run_dir(
    experiment_id: str,
) -> Path:
    return (
        Path("runs")
        / experiment_id
    )


def source_metadata(
    experiment_id: str,
) -> dict[str, Any]:
    path = (
        run_dir(
            experiment_id
        )
        / "run_metadata.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return load_json(
        path
    )


def schema_for(
    direction: str,
) -> dict[str, list[str]]:
    path = (
        Path(
            "data/metadata/"
            "preprocessing"
        )
        / direction
        / "feature_schema.json"
    )

    return load_json(
        path
    )


def external_path(
    direction: str,
) -> Path:
    return (
        Path("data/processed")
        / direction
        / "external.parquet"
    )


def candidate_name(
    depth: int,
    learning_rate: float,
) -> str:
    eta = str(
        float(
            learning_rate
        )
    ).replace(
        ".",
        "p",
    )

    return (
        f"depth_{depth}"
        f"__eta_{eta}"
    )


def nested_strings(
    value: Any,
):
    if isinstance(
        value,
        dict,
    ):
        for nested in (
            value.values()
        ):
            yield from nested_strings(
                nested
            )

    elif isinstance(
        value,
        list,
    ):
        for nested in value:
            yield from nested_strings(
                nested
            )

    elif isinstance(
        value,
        str,
    ):
        yield value


def resolve_saved_path(
    text: str,
    base: Path,
) -> Path | None:
    candidate = Path(
        text
    )

    possibilities = [
        candidate,
        base / candidate,
    ]

    for possibility in (
        possibilities
    ):
        if possibility.exists():
            return possibility

    return None


def find_scaler(
    directory: Path,
    metadata: dict[str, Any],
) -> Path:
    for text in nested_strings(
        metadata
    ):
        if (
            "scaler" in text.lower()
            and (
                text.endswith(
                    ".joblib"
                )
                or text.endswith(
                    ".pkl"
                )
            )
        ):
            resolved = resolve_saved_path(
                text,
                directory,
            )

            if resolved:
                return resolved

    exact = [
        directory
        / "scaler.joblib",
        directory
        / "robust_scaler.joblib",
        directory
        / "scaler.pkl",
    ]

    for path in exact:
        if path.exists():
            return path

    candidates = [
        path
        for path in directory.rglob(
            "*"
        )
        if path.is_file()
        and "scaler" in path.name.lower()
        and path.suffix.lower()
        in {
            ".joblib",
            ".pkl",
        }
    ]

    if len(
        candidates
    ) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Could not uniquely identify scaler "
        f"in {directory}. Candidates={candidates}"
    )


def find_logistic_model(
    directory: Path,
    metadata: dict[str, Any],
) -> Path:
    for text in nested_strings(
        metadata
    ):
        lower = text.lower()

        if (
            "model" in lower
            and "scaler" not in lower
            and (
                lower.endswith(
                    ".joblib"
                )
                or lower.endswith(
                    ".pkl"
                )
            )
        ):
            resolved = resolve_saved_path(
                text,
                directory,
            )

            if resolved:
                return resolved

    exact = [
        directory
        / "model.joblib",
        directory
        / "selected_model.joblib",
        directory
        / "logistic_model.joblib",
        directory
        / "classifier.joblib",
        directory
        / "model.pkl",
    ]

    for path in exact:
        if path.exists():
            return path

    candidates = [
        path
        for path in directory.rglob(
            "*"
        )
        if path.is_file()
        and path.suffix.lower()
        in {
            ".joblib",
            ".pkl",
        }
        and "scaler"
        not in path.name.lower()
    ]

    if len(
        candidates
    ) == 1:
        return candidates[0]

    raise RuntimeError(
        "Could not uniquely identify logistic "
        f"model in {directory}. "
        f"Candidates={candidates}"
    )


def patient_slices(
    frame: pd.DataFrame,
) -> tuple[
    list[str],
    list[tuple[int, int]],
]:
    ids = (
        frame[
            "patient_id"
        ]
        .astype(str)
        .to_numpy()
    )

    if len(ids) == 0:
        raise ValueError(
            "Empty external cohort."
        )

    changes = (
        np.flatnonzero(
            ids[1:]
            != ids[:-1]
        )
        + 1
    )

    starts = np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64,
            ),
            changes,
        ]
    )

    stops = np.concatenate(
        [
            changes,
            np.array(
                [len(ids)],
                dtype=np.int64,
            ),
        ]
    )

    patient_ids = [
        str(
            ids[start]
        )
        for start in starts
    ]

    slices = [
        (
            int(start),
            int(stop),
        )
        for start, stop in zip(
            starts,
            stops,
            strict=True,
        )
    ]

    return (
        patient_ids,
        slices,
    )


def sequences_from_flat(
    values: np.ndarray,
    slices: list[
        tuple[int, int]
    ],
    *,
    dtype,
) -> list[np.ndarray]:
    array = np.asarray(
        values
    )

    return [
        array[
            start:stop
        ].astype(
            dtype,
            copy=False,
        )
        for start, stop in slices
    ]


def validate_external_frame(
    frame: pd.DataFrame,
    direction: str,
) -> None:
    expected = (
        EXPECTED_EXTERNAL[
            direction
        ]
    )

    if len(
        frame
    ) != expected[
        "hours"
    ]:
        raise RuntimeError(
            f"{direction}: expected "
            f"{expected['hours']:,} hours, "
            f"found {len(frame):,}."
        )

    if frame[
        "SepsisLabel"
    ].isna().any():
        raise RuntimeError(
            "Missing external labels."
        )

    labels = frame[
        "SepsisLabel"
    ].to_numpy()

    if not np.isin(
        labels,
        [0, 1],
    ).all():
        raise RuntimeError(
            "External labels are not binary."
        )

    if frame.duplicated(
        [
            "patient_id",
            "hour_index",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate external patient-hour."
        )

    patient_ids, slices = (
        patient_slices(
            frame
        )
    )

    if len(
        patient_ids
    ) != expected[
        "patients"
    ]:
        raise RuntimeError(
            f"{direction}: expected "
            f"{expected['patients']:,} patients, "
            f"found {len(patient_ids):,}."
        )

    hours = frame[
        "hour_index"
    ].to_numpy()

    for start, stop in slices:
        patient_hours = (
            hours[
                start:stop
            ]
        )

        expected_hours = np.arange(
            stop - start
        )

        if not np.array_equal(
            patient_hours,
            expected_hours,
        ):
            raise RuntimeError(
                "Non-sequential hour_index "
                "detected."
            )


def load_external_frame(
    direction: str,
    schema: dict[str, list[str]],
) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            [
                "patient_id",
                "hour_index",
                "SepsisLabel",
            ]
            + list(
                schema["V2"]
            )
        )
    )

    frame = pd.read_parquet(
        external_path(
            direction
        ),
        columns=columns,
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

    validate_external_frame(
        frame,
        direction,
    )

    return frame


def predict_logistic(
    experiment_id: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    np.ndarray,
    float,
]:
    directory = run_dir(
        experiment_id
    )

    metadata = source_metadata(
        experiment_id
    )

    scaler_path = find_scaler(
        directory,
        metadata,
    )

    model_path = (
        find_logistic_model(
            directory,
            metadata,
        )
    )

    scaler = joblib.load(
        scaler_path
    )

    model = joblib.load(
        model_path
    )

    features = (
        frame[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    transformed = scaler.transform(
        features
    ).astype(
        np.float32,
        copy=False,
    )

    np.clip(
        transformed,
        -10.0,
        10.0,
        out=transformed,
    )

    probabilities = (
        model.predict_proba(
            transformed
        )[:, 1]
        .astype(
            np.float64,
            copy=False,
        )
    )

    threshold = float(
        metadata[
            "selected_threshold"
        ]
    )

    return (
        probabilities,
        threshold,
    )


def predict_xgb_models(
    model_directory: Path,
    seed_results_path: Path,
    frame: pd.DataFrame,
    feature_columns: list[str],
    seeds: list[int],
) -> np.ndarray:
    features = (
        frame[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    matrix = xgb.DMatrix(
        features
    )

    del features

    gc.collect()

    seed_results = pd.read_csv(
        seed_results_path
    )

    accumulator = np.zeros(
        len(frame),
        dtype=np.float64,
    )

    for seed in seeds:
        rows = seed_results.loc[
            seed_results[
                "seed"
            ] == seed
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"Missing metadata for "
                f"XGBoost seed {seed}."
            )

        best_iteration = int(
            rows.iloc[0][
                "best_iteration"
            ]
        )

        model_path = (
            model_directory
            / f"seed_{seed}.json"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                model_path
            )

        booster = xgb.Booster()

        booster.load_model(
            model_path
        )

        booster.set_param(
            {
                "device": "cuda"
            }
        )

        probability = booster.predict(
            matrix,
            iteration_range=(
                0,
                best_iteration + 1,
            ),
        )

        accumulator += (
            probability.astype(
                np.float64,
                copy=False,
            )
        )

        del booster

    accumulator /= len(
        seeds
    )

    return accumulator


def predict_xgboost_run(
    experiment_id: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    np.ndarray,
    float,
]:
    directory = run_dir(
        experiment_id
    )

    metadata = source_metadata(
        experiment_id
    )

    depth = int(
        metadata[
            "selected_max_depth"
        ]
    )

    learning_rate = float(
        metadata[
            "selected_learning_rate"
        ]
    )

    selected_directory = (
        directory
        / "candidate_models"
        / candidate_name(
            depth,
            learning_rate,
        )
    )

    probabilities = (
        predict_xgb_models(
            selected_directory,
            selected_directory
            / "seed_results.csv",
            frame,
            feature_columns,
            [
                int(seed)
                for seed in metadata[
                    "seeds"
                ]
            ],
        )
    )

    threshold = float(
        metadata[
            "selected_threshold"
        ]
    )

    return (
        probabilities,
        threshold,
    )


@torch.inference_mode()
def predict_gru(
    experiment_id: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    slices: list[
        tuple[int, int]
    ],
) -> tuple[
    np.ndarray,
    float,
]:
    directory = run_dir(
        experiment_id
    )

    metadata = source_metadata(
        experiment_id
    )

    scaler_path = find_scaler(
        directory,
        metadata,
    )

    scaler = joblib.load(
        scaler_path
    )

    raw = (
        frame[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    transformed = scaler.transform(
        raw
    ).astype(
        np.float32,
        copy=False,
    )

    del raw

    np.clip(
        transformed,
        -10.0,
        10.0,
        out=transformed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable for frozen GRU "
            "external evaluation."
        )

    device = torch.device(
        "cuda"
    )

    hidden_size = int(
        metadata[
            "hidden_size"
        ]
    )

    batch_size = int(
        metadata[
            "batch_size_patients"
        ]
    )

    seeds = [
        int(seed)
        for seed in metadata[
            "seeds"
        ]
    ]

    accumulator = np.zeros(
        len(frame),
        dtype=np.float64,
    )

    for seed in seeds:
        checkpoint = (
            directory
            / "checkpoints"
            / f"seed_{seed}.pt"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                checkpoint
            )

        model = CausalGRU(
            input_size=len(
                feature_columns
            ),
            hidden_size=hidden_size,
        ).to(
            device
        )

        state_dict = torch.load(
            checkpoint,
            map_location=device,
        )

        model.load_state_dict(
            state_dict
        )

        model.eval()

        for batch_start in range(
            0,
            len(slices),
            batch_size,
        ):
            batch_slices = slices[
                batch_start:
                batch_start
                + batch_size
            ]

            sequences = [
                torch.from_numpy(
                    transformed[
                        start:stop
                    ]
                )
                for start, stop
                in batch_slices
            ]

            lengths = torch.tensor(
                [
                    stop - start
                    for start, stop
                    in batch_slices
                ],
                dtype=torch.long,
            )

            padded = pad_sequence(
                sequences,
                batch_first=True,
                padding_value=0.0,
            ).to(
                device,
                non_blocking=True,
            )

            logits = model(
                padded,
                lengths,
            )

            probability = torch.sigmoid(
                logits
            ).cpu()

            for index, (
                start,
                stop,
            ) in enumerate(
                batch_slices
            ):
                length = (
                    stop - start
                )

                accumulator[
                    start:stop
                ] += probability[
                    index,
                    :length,
                ].numpy()

            del padded
            del logits
            del probability

        del model
        del state_dict

        torch.cuda.empty_cache()

        gc.collect()

    accumulator /= len(
        seeds
    )

    threshold = float(
        metadata[
            "selected_threshold"
        ]
    )

    return (
        accumulator,
        threshold,
    )


def predict_pae_branch(
    pae_directory: Path,
    branch_name: str,
    branch_metadata: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    depth = int(
        branch_metadata[
            "max_depth"
        ]
    )

    learning_rate = float(
        branch_metadata[
            "learning_rate"
        ]
    )

    selected_directory = (
        pae_directory
        / f"{branch_name}_branch"
        / candidate_name(
            depth,
            learning_rate,
        )
    )

    pae_metadata = load_json(
        pae_directory
        / "run_metadata.json"
    )

    seeds = [
        int(seed)
        for seed in pae_metadata.get(
            "seeds",
            [
                1729,
                2718,
                31415,
                57721,
                65537,
            ],
        )
    ]

    return predict_xgb_models(
        selected_directory,
        selected_directory
        / "seed_results.csv",
        frame,
        feature_columns,
        seeds,
    )


def predict_pae(
    experiment_id: str,
    frame: pd.DataFrame,
    v0_columns: list[str],
    v2_columns: list[str],
) -> tuple[
    np.ndarray,
    float,
]:
    directory = run_dir(
        experiment_id
    )

    metadata = source_metadata(
        experiment_id
    )

    v0_probability = (
        predict_pae_branch(
            directory,
            "V0",
            metadata["V0"],
            frame,
            v0_columns,
        )
    )

    v2_probability = (
        predict_pae_branch(
            directory,
            "V2",
            metadata["V2"],
            frame,
            v2_columns,
        )
    )

    alpha = float(
        metadata[
            "selected_alpha"
        ]
    )

    probability = (
        alpha * v0_probability
        + (
            1.0 - alpha
        ) * v2_probability
    )

    threshold = float(
        metadata[
            "selected_threshold"
        ]
    )

    return (
        probability,
        threshold,
    )


def evaluate_probabilities(
    labels_flat: np.ndarray,
    labels_by_patient: list[
        np.ndarray
    ],
    probabilities: np.ndarray,
    slices: list[
        tuple[int, int]
    ],
    threshold: float,
) -> tuple[
    dict[str, Any],
    list[np.ndarray],
]:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int8
    )

    probabilities_by_patient = (
        sequences_from_flat(
            probabilities,
            slices,
            dtype=np.float64,
        )
    )

    predictions_by_patient = (
        sequences_from_flat(
            predictions,
            slices,
            dtype=np.int8,
        )
    )

    utility = (
        compute_challenge_utility(
            labels_by_patient,
            predictions_by_patient,
        )
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            labels_flat,
            predictions,
            labels=[
                0,
                1,
            ],
        ).ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else float("nan")
    )

    ppv = (
        tp / (tp + fp)
        if tp + fp
        else float("nan")
    )

    npv = (
        tn / (tn + fn)
        if tn + fn
        else float("nan")
    )

    calibration = (
        calibration_intercept_slope(
            labels_flat,
            probabilities,
        )
    )

    metrics = {
        "utility": float(
            utility.normalized
        ),
        "auroc": float(
            roc_auc_score(
                labels_flat,
                probabilities,
            )
        ),
        "auprc": float(
            average_precision_score(
                labels_flat,
                probabilities,
            )
        ),
        "brier": float(
            brier_score_loss(
                labels_flat,
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
        "calibration_intercept": (
            calibration.intercept
        ),
        "calibration_slope": (
            calibration.slope
        ),
        "calibration_fit_success": (
            calibration.success
        ),
        "calibration_fit_message": (
            calibration.message
        ),
    }

    return (
        metrics,
        predictions_by_patient,
    )


def validate_source_run(
    direction: str,
    model_name: str,
    experiment_id: str,
) -> None:
    metadata = source_metadata(
        experiment_id
    )

    expected = (
        FROZEN_SOURCE_RUNS[
            direction
        ][
            model_name
        ]
    )

    def require_close(
        name: str,
        actual: Any,
        frozen: float,
        *,
        atol: float = 1e-12,
    ) -> None:
        if actual is None:
            raise RuntimeError(
                f"{experiment_id}: missing {name}."
            )

        if not np.isclose(
            float(actual),
            float(frozen),
            rtol=0.0,
            atol=atol,
        ):
            raise RuntimeError(
                f"{experiment_id}: {name} "
                f"mismatch; expected {frozen}, "
                f"found {actual}."
            )

    if metadata.get(
        "run_id"
    ) != experiment_id:
        raise RuntimeError(
            f"{experiment_id}: run_id mismatch."
        )

    if metadata.get(
        "direction"
    ) != direction:
        raise RuntimeError(
            f"{experiment_id}: expected "
            f"{direction}, found "
            f"{metadata.get('direction')}."
        )

    if metadata.get(
        "external_evaluated"
    ) is not False:
        raise RuntimeError(
            f"{experiment_id} already reports "
            "external evaluation."
        )

    if (
        "internal_test_metrics"
        not in metadata
    ):
        raise RuntimeError(
            f"{experiment_id} has no locked "
            "internal-test metrics."
        )

    require_close(
        "internal utility",
        metadata[
            "internal_test_metrics"
        ].get(
            "utility"
        ),
        expected[
            "internal_utility"
        ],
        atol=1e-6,
    )

    if model_name == "Logistic_V0":
        if metadata.get(
            "model"
        ) != "B1_logistic_regression":
            raise RuntimeError(
                f"{experiment_id}: unexpected "
                "logistic model identifier."
            )

        if metadata.get(
            "feature_representation"
        ) != "V0":
            raise RuntimeError(
                f"{experiment_id}: expected "
                "V0 representation."
            )

        if metadata.get(
            "feature_count"
        ) != 40:
            raise RuntimeError(
                f"{experiment_id}: expected "
                "40 logistic features."
            )

        if metadata.get(
            "C_grid"
        ) != [
            0.1,
            1.0,
            10.0,
        ]:
            raise RuntimeError(
                f"{experiment_id}: frozen "
                "C grid mismatch."
            )

        if metadata.get(
            "split_seed"
        ) != 1729:
            raise RuntimeError(
                f"{experiment_id}: split "
                "seed mismatch."
            )

        require_close(
            "selected_C",
            metadata.get(
                "selected_C"
            ),
            expected[
                "selected_C"
            ],
        )

        require_close(
            "selected_threshold",
            metadata.get(
                "selected_threshold"
            ),
            expected[
                "selected_threshold"
            ],
        )

        return

    if model_name.startswith(
        "XGB_"
    ):
        if metadata.get(
            "model"
        ) != "xgboost":
            raise RuntimeError(
                f"{experiment_id}: unexpected "
                "XGBoost model identifier."
            )

        if metadata.get(
            "feature_representation"
        ) != expected[
            "representation"
        ]:
            raise RuntimeError(
                f"{experiment_id}: frozen "
                "representation mismatch."
            )

        if metadata.get(
            "xgboost_version"
        ) != "3.2.0":
            raise RuntimeError(
                f"{experiment_id}: XGBoost "
                "version mismatch."
            )

        if metadata.get(
            "device"
        ) != "cuda":
            raise RuntimeError(
                f"{experiment_id}: expected "
                "CUDA XGBoost."
            )

        if metadata.get(
            "tree_method"
        ) != "hist":
            raise RuntimeError(
                f"{experiment_id}: tree "
                "method mismatch."
            )

        if metadata.get(
            "objective"
        ) != "binary:logistic":
            raise RuntimeError(
                f"{experiment_id}: objective "
                "mismatch."
            )

        if metadata.get(
            "eval_metric"
        ) != "logloss":
            raise RuntimeError(
                f"{experiment_id}: evaluation "
                "metric mismatch."
            )

        if metadata.get(
            "max_boost_rounds"
        ) != 1000:
            raise RuntimeError(
                f"{experiment_id}: maximum "
                "boost rounds mismatch."
            )

        if metadata.get(
            "early_stopping_rounds"
        ) != 50:
            raise RuntimeError(
                f"{experiment_id}: early "
                "stopping mismatch."
            )

        require_close(
            "subsample",
            metadata.get(
                "subsample"
            ),
            0.8,
        )

        require_close(
            "colsample_bytree",
            metadata.get(
                "colsample_bytree"
            ),
            0.8,
        )

        require_close(
            "min_child_weight",
            metadata.get(
                "min_child_weight"
            ),
            1.0,
        )

        require_close(
            "reg_lambda",
            metadata.get(
                "reg_lambda"
            ),
            1.0,
        )

        if metadata.get(
            "max_bin"
        ) != 256:
            raise RuntimeError(
                f"{experiment_id}: max_bin "
                "mismatch."
            )

        if metadata.get(
            "seeds"
        ) != FROZEN_SEEDS:
            raise RuntimeError(
                f"{experiment_id}: frozen "
                "seed list mismatch."
            )

        if metadata.get(
            "candidate_tie_break"
        ) != (
            "shallower max_depth, then "
            "lower learning_rate"
        ):
            raise RuntimeError(
                f"{experiment_id}: candidate "
                "tie-break mismatch."
            )

        if metadata.get(
            "selected_max_depth"
        ) != expected[
            "max_depth"
        ]:
            raise RuntimeError(
                f"{experiment_id}: selected "
                "max_depth mismatch."
            )

        require_close(
            "selected_learning_rate",
            metadata.get(
                "selected_learning_rate"
            ),
            expected[
                "learning_rate"
            ],
        )

        require_close(
            "selected_threshold",
            metadata.get(
                "selected_threshold"
            ),
            expected[
                "selected_threshold"
            ],
        )

        return

    if model_name == "GRU":
        if metadata.get(
            "model"
        ) != "GRU":
            raise RuntimeError(
                f"{experiment_id}: unexpected "
                "GRU model identifier."
            )

        if metadata.get(
            "feature_count"
        ) != 108:
            raise RuntimeError(
                f"{experiment_id}: expected "
                "108 GRU features."
            )

        if metadata.get(
            "hidden_size"
        ) != 64:
            raise RuntimeError(
                f"{experiment_id}: hidden "
                "size mismatch."
            )

        if metadata.get(
            "num_layers"
        ) != 1:
            raise RuntimeError(
                f"{experiment_id}: GRU layer "
                "count mismatch."
            )

        if metadata.get(
            "bidirectional"
        ) is not False:
            raise RuntimeError(
                f"{experiment_id}: GRU must "
                "be unidirectional."
            )

        require_close(
            "learning_rate",
            metadata.get(
                "learning_rate"
            ),
            0.001,
        )

        if metadata.get(
            "optimizer"
        ) != "Adam":
            raise RuntimeError(
                f"{experiment_id}: optimizer "
                "mismatch."
            )

        if metadata.get(
            "loss"
        ) != "unweighted BCEWithLogitsLoss":
            raise RuntimeError(
                f"{experiment_id}: loss "
                "configuration mismatch."
            )

        if metadata.get(
            "batch_size_patients"
        ) != 128:
            raise RuntimeError(
                f"{experiment_id}: batch "
                "size mismatch."
            )

        if metadata.get(
            "max_epochs"
        ) != 60:
            raise RuntimeError(
                f"{experiment_id}: max "
                "epochs mismatch."
            )

        if metadata.get(
            "early_stopping_patience"
        ) != 8:
            raise RuntimeError(
                f"{experiment_id}: GRU "
                "patience mismatch."
            )

        if metadata.get(
            "early_stopping_metric"
        ) != "source validation BCE":
            raise RuntimeError(
                f"{experiment_id}: GRU "
                "early-stopping metric mismatch."
            )

        if metadata.get(
            "seeds"
        ) != FROZEN_SEEDS:
            raise RuntimeError(
                f"{experiment_id}: frozen "
                "GRU seed list mismatch."
            )

        require_close(
            "selected_threshold",
            metadata.get(
                "selected_threshold"
            ),
            expected[
                "selected_threshold"
            ],
        )

        return

    if model_name == "PAE":
        if metadata.get(
            "abbreviation"
        ) != "PAE":
            raise RuntimeError(
                f"{experiment_id}: unexpected "
                "PAE model identifier."
            )

        if metadata.get(
            "alphas"
        ) != [
            0.25,
            0.5,
            0.75,
        ]:
            raise RuntimeError(
                f"{experiment_id}: frozen "
                "alpha grid mismatch."
            )

        if metadata.get(
            "alpha_tie_break"
        ) != "larger alpha":
            raise RuntimeError(
                f"{experiment_id}: alpha "
                "tie-break mismatch."
            )

        require_close(
            "selected_alpha",
            metadata.get(
                "selected_alpha"
            ),
            expected[
                "selected_alpha"
            ],
        )

        require_close(
            "selected_threshold",
            metadata.get(
                "selected_threshold"
            ),
            expected[
                "selected_threshold"
            ],
        )

        v0 = metadata.get(
            "V0",
            {},
        )

        v2 = metadata.get(
            "V2",
            {},
        )

        if v0.get(
            "max_depth"
        ) != expected[
            "v0_max_depth"
        ]:
            raise RuntimeError(
                f"{experiment_id}: PAE V0 "
                "depth mismatch."
            )

        require_close(
            "PAE V0 learning_rate",
            v0.get(
                "learning_rate"
            ),
            expected[
                "v0_learning_rate"
            ],
        )

        if v2.get(
            "max_depth"
        ) != expected[
            "v2_max_depth"
        ]:
            raise RuntimeError(
                f"{experiment_id}: PAE V2 "
                "depth mismatch."
            )

        require_close(
            "PAE V2 learning_rate",
            v2.get(
                "learning_rate"
            ),
            expected[
                "v2_learning_rate"
            ],
        )

        return

    raise RuntimeError(
        f"Unsupported frozen model: "
        f"{model_name}."
    )

def preflight_direction(
    direction: str,
) -> None:
    print()
    print(
        f"[{direction}]"
    )

    schema = schema_for(
        direction
    )

    path = external_path(
        direction
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    actual_hash = sha256_file(
        path
    )

    expected_hash = (
        EXPECTED_EXTERNAL[
            direction
        ][
            "sha256"
        ]
    )

    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{direction}: external parquet "
            "hash mismatch."
        )

    print(
        "  external parquet hash: PASS"
    )

    mapping = FINAL_RUNS[
        direction
    ]

    for model_name in MODEL_ORDER:
        experiment_id = (
            mapping[
                model_name
            ]
        )

        validate_source_run(
            direction,
            model_name,
            experiment_id,
        )

        directory = run_dir(
            experiment_id
        )

        metadata = source_metadata(
            experiment_id
        )

        if model_name == "Logistic_V0":
            scaler = find_scaler(
                directory,
                metadata,
            )

            model = find_logistic_model(
                directory,
                metadata,
            )

            joblib.load(
                scaler
            )

            joblib.load(
                model
            )

        elif model_name.startswith(
            "XGB_"
        ):
            depth = int(
                metadata[
                    "selected_max_depth"
                ]
            )

            eta = float(
                metadata[
                    "selected_learning_rate"
                ]
            )

            selected = (
                directory
                / "candidate_models"
                / candidate_name(
                    depth,
                    eta,
                )
            )

            if not (
                selected
                / "seed_results.csv"
            ).exists():
                raise FileNotFoundError(
                    selected
                    / "seed_results.csv"
                )

            for seed in metadata[
                "seeds"
            ]:
                model_path = (
                    selected
                    / f"seed_{seed}.json"
                )

                if not model_path.exists():
                    raise FileNotFoundError(
                        model_path
                    )

        elif model_name == "GRU":
            find_scaler(
                directory,
                metadata,
            )

            for seed in metadata[
                "seeds"
            ]:
                checkpoint = (
                    directory
                    / "checkpoints"
                    / f"seed_{seed}.pt"
                )

                if not checkpoint.exists():
                    raise FileNotFoundError(
                        checkpoint
                    )

        elif model_name == "PAE":
            for branch in (
                "V0",
                "V2",
            ):
                branch_metadata = (
                    metadata[
                        branch
                    ]
                )

                selected = (
                    directory
                    / f"{branch}_branch"
                    / candidate_name(
                        int(
                            branch_metadata[
                                "max_depth"
                            ]
                        ),
                        float(
                            branch_metadata[
                                "learning_rate"
                            ]
                        ),
                    )
                )

                if not (
                    selected
                    / "seed_results.csv"
                ).exists():
                    raise FileNotFoundError(
                        selected
                        / "seed_results.csv"
                    )

        print(
            f"  {model_name:<12} "
            f"{experiment_id}: PASS"
        )

    gru_columns = [
        column
        for column in schema[
            "V2"
        ]
        if not column.startswith(
            "count6_"
        )
    ]

    if len(
        gru_columns
    ) != 108:
        raise RuntimeError(
            f"{direction}: expected 108 "
            "GRU columns."
        )


def run_preflight() -> None:
    print("=" * 72)
    print(
        "FROZEN EXTERNAL EVALUATION PREFLIGHT"
    )
    print("=" * 72)

    for direction in (
        "A_to_B",
        "B_to_A",
    ):
        preflight_direction(
            direction
        )

    print()
    print(
        "PREFLIGHT: PASS"
    )

    print(
        "No external model performance "
        "was calculated."
    )


def evaluate_direction(
    direction: str,
    destination: Path,
) -> dict[str, Any]:
    destination.mkdir(
        parents=True
    )

    mapping = FINAL_RUNS[
        direction
    ]

    schema = schema_for(
        direction
    )

    print(
        f"{direction}: loading frozen "
        "external cohort..."
    )

    frame = load_external_frame(
        direction,
        schema,
    )

    patient_ids, slices = (
        patient_slices(
            frame
        )
    )

    labels_flat = (
        frame[
            "SepsisLabel"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    labels_by_patient = (
        sequences_from_flat(
            labels_flat,
            slices,
            dtype=np.int8,
        )
    )

    probabilities: dict[
        str,
        np.ndarray
    ] = {}

    thresholds: dict[
        str,
        float
    ] = {}

    print(
        f"{direction}: Logistic..."
    )

    (
        probabilities[
            "Logistic_V0"
        ],
        thresholds[
            "Logistic_V0"
        ],
    ) = predict_logistic(
        mapping[
            "Logistic_V0"
        ],
        frame,
        list(
            schema[
                "V0"
            ]
        ),
    )

    for representation in (
        "M0",
        "V0",
        "V1",
        "V2",
    ):
        model_name = (
            f"XGB_{representation}"
        )

        print(
            f"{direction}: "
            f"{model_name}..."
        )

        (
            probabilities[
                model_name
            ],
            thresholds[
                model_name
            ],
        ) = predict_xgboost_run(
            mapping[
                model_name
            ],
            frame,
            list(
                schema[
                    representation
                ]
            ),
        )

    gru_columns = [
        column
        for column in schema[
            "V2"
        ]
        if not column.startswith(
            "count6_"
        )
    ]

    print(
        f"{direction}: GRU..."
    )

    (
        probabilities[
            "GRU"
        ],
        thresholds[
            "GRU"
        ],
    ) = predict_gru(
        mapping[
            "GRU"
        ],
        frame,
        gru_columns,
        slices,
    )

    print(
        f"{direction}: PAE..."
    )

    (
        probabilities[
            "PAE"
        ],
        thresholds[
            "PAE"
        ],
    ) = predict_pae(
        mapping[
            "PAE"
        ],
        frame,
        list(
            schema[
                "V0"
            ]
        ),
        list(
            schema[
                "V2"
            ]
        ),
    )

    metrics_rows = []

    predictions_binary: dict[
        str,
        list[np.ndarray]
    ] = {}

    for model_name in MODEL_ORDER:
        metrics, binary = (
            evaluate_probabilities(
                labels_flat,
                labels_by_patient,
                probabilities[
                    model_name
                ],
                slices,
                thresholds[
                    model_name
                ],
            )
        )

        metrics_rows.append(
            {
                "direction": (
                    direction
                ),
                "model": (
                    model_name
                ),
                "source_run": (
                    mapping[
                        model_name
                    ]
                ),
                **metrics,
            }
        )

        predictions_binary[
            model_name
        ] = binary

    metrics_frame = pd.DataFrame(
        metrics_rows
    )

    metrics_frame.to_csv(
        destination
        / "external_metrics.csv",
        index=False,
    )

    prediction_frame = pd.DataFrame(
        {
            "patient_id": (
                frame[
                    "patient_id"
                ].astype(str)
            ),
            "hour_index": (
                frame[
                    "hour_index"
                ]
            ),
            "SepsisLabel": (
                labels_flat
            ),
        }
    )

    for model_name in MODEL_ORDER:
        prediction_frame[
            model_name
        ] = probabilities[
            model_name
        ]

    prediction_frame.to_parquet(
        destination
        / "external_predictions.parquet",
        index=False,
    )

    transfer_rows = []

    for model_name in MODEL_ORDER:
        metadata = source_metadata(
            mapping[
                model_name
            ]
        )

        internal_utility = float(
            metadata[
                "internal_test_metrics"
            ][
                "utility"
            ]
        )

        external_utility = float(
            metrics_frame.loc[
                metrics_frame[
                    "model"
                ] == model_name,
                "utility",
            ].iloc[0]
        )

        transfer_rows.append(
            {
                "direction": (
                    direction
                ),
                "model": (
                    model_name
                ),
                "source_internal_utility": (
                    internal_utility
                ),
                "external_utility": (
                    external_utility
                ),
                "transfer_gap": (
                    internal_utility
                    - external_utility
                ),
            }
        )

    transfer_frame = pd.DataFrame(
        transfer_rows
    )

    transfer_frame.to_csv(
        destination
        / "transfer_gaps.csv",
        index=False,
    )

    print(
        f"{direction}: paired "
        f"{BOOTSTRAP_REPLICATES}-patient-bootstrap "
        "replicates..."
    )

    bootstrap = (
        paired_patient_bootstrap(
            labels_by_patient,
            predictions_binary,
            n_bootstrap=(
                BOOTSTRAP_REPLICATES
            ),
            seed=(
                BOOTSTRAP_SEED
            ),
        )
    )

    bootstrap_frame = pd.DataFrame(
        {
            "replicate": np.arange(
                BOOTSTRAP_REPLICATES
            )
        }
    )

    for model_name in MODEL_ORDER:
        bootstrap_frame[
            model_name
        ] = bootstrap[
            model_name
        ]

    bootstrap_frame[
        "PAE_minus_XGB_V2"
    ] = (
        bootstrap[
            "PAE"
        ]
        - bootstrap[
            "XGB_V2"
        ]
    )

    bootstrap_frame[
        "XGB_V1_minus_XGB_V0"
    ] = (
        bootstrap[
            "XGB_V1"
        ]
        - bootstrap[
            "XGB_V0"
        ]
    )

    bootstrap_frame[
        "XGB_V2_minus_XGB_V0"
    ] = (
        bootstrap[
            "XGB_V2"
        ]
        - bootstrap[
            "XGB_V0"
        ]
    )

    bootstrap_frame.to_csv(
        destination
        / "bootstrap_replicates.csv",
        index=False,
    )

    summary_rows = []

    for model_name in MODEL_ORDER:
        low, high = percentile_ci(
            bootstrap[
                model_name
            ]
        )

        point = float(
            metrics_frame.loc[
                metrics_frame[
                    "model"
                ] == model_name,
                "utility",
            ].iloc[0]
        )

        summary_rows.append(
            {
                "direction": (
                    direction
                ),
                "quantity": (
                    model_name
                ),
                "type": (
                    "model_utility"
                ),
                "point_estimate": (
                    point
                ),
                "ci_2_5": low,
                "ci_97_5": high,
            }
        )

    contrast_specs = {
        "PAE_minus_XGB_V2": (
            "PAE",
            "XGB_V2",
        ),
        "XGB_V1_minus_XGB_V0": (
            "XGB_V1",
            "XGB_V0",
        ),
        "XGB_V2_minus_XGB_V0": (
            "XGB_V2",
            "XGB_V0",
        ),
    }

    metric_lookup = (
        metrics_frame.set_index(
            "model"
        )[
            "utility"
        ].to_dict()
    )

    for contrast_name, (
        first,
        second,
    ) in contrast_specs.items():
        values = (
            bootstrap_frame[
                contrast_name
            ].to_numpy()
        )

        low, high = percentile_ci(
            values
        )

        point = float(
            metric_lookup[
                first
            ]
            - metric_lookup[
                second
            ]
        )

        summary_rows.append(
            {
                "direction": (
                    direction
                ),
                "quantity": (
                    contrast_name
                ),
                "type": (
                    "paired_contrast"
                ),
                "point_estimate": (
                    point
                ),
                "ci_2_5": low,
                "ci_97_5": high,
            }
        )

    bootstrap_summary = (
        pd.DataFrame(
            summary_rows
        )
    )

    bootstrap_summary.to_csv(
        destination
        / "bootstrap_summary.csv",
        index=False,
    )

    source_metadata_hashes = {
        model_name: sha256_file(
            run_dir(
                mapping[
                    model_name
                ]
            )
            / "run_metadata.json"
        )
        for model_name
        in MODEL_ORDER
    }

    direction_metadata = {
        "direction": direction,
        "external_evaluated": True,
        "external_parquet": str(
            external_path(
                direction
            )
        ),
        "external_parquet_sha256": (
            sha256_file(
                external_path(
                    direction
                )
            )
        ),
        "external_patients": len(
            patient_ids
        ),
        "external_hours": len(
            frame
        ),
        "source_runs": mapping,
        "source_run_metadata_sha256": (
            source_metadata_hashes
        ),
        "bootstrap_replicates": (
            BOOTSTRAP_REPLICATES
        ),
        "bootstrap_seed": (
            BOOTSTRAP_SEED
        ),
        "calibration_note": (
            "Calibration intercept/slope are "
            "evaluation statistics only; "
            "predictions were not recalibrated."
        ),
    }

    with (
        destination
        / "direction_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            direction_metadata,
            handle,
            indent=2,
        )

    del frame
    del prediction_frame
    del probabilities

    gc.collect()

    torch.cuda.empty_cache()

    return {
        "metrics": metrics_frame,
        "transfer": transfer_frame,
        "bootstrap": bootstrap_summary,
    }


def main() -> int:
    args = parse_args()

    if args.preflight:
        run_preflight()

        return 0

    run_preflight()

    destination = (
        Path("runs")
        / args.run_id
    )

    if destination.exists():
        raise FileExistsError(
            f"Final evaluation run already "
            f"exists: {destination}"
        )

    destination.mkdir(
        parents=True
    )

    root_metadata = {
        "run_id": args.run_id,
        "purpose": (
            "atomic final external evaluation "
            "of both directions"
        ),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "external_evaluated": True,
        "directions": [
            "A_to_B",
            "B_to_A",
        ],
        "source_runs": (
            FINAL_RUNS
        ),
        "bootstrap_replicates": (
            BOOTSTRAP_REPLICATES
        ),
        "bootstrap_seed": (
            BOOTSTRAP_SEED
        ),
    }

    try:
        results = {}

        for direction in (
            "A_to_B",
            "B_to_A",
        ):
            results[
                direction
            ] = evaluate_direction(
                direction,
                destination
                / direction,
            )

        with (
            destination
            / "run_metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                root_metadata,
                handle,
                indent=2,
            )

    except Exception:
        failure = {
            **root_metadata,
            "status": "FAILED",
            "traceback": (
                traceback.format_exc()
            ),
        }

        with (
            destination
            / "failure.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                failure,
                handle,
                indent=2,
            )

        raise

    # Do not reveal one direction before the other
    # has completed.
    print()
    print("=" * 88)
    print(
        "FINAL EXTERNAL RESULTS — "
        "BOTH DIRECTIONS COMPLETE"
    )
    print("=" * 88)

    combined_metrics = []

    for direction in (
        "A_to_B",
        "B_to_A",
    ):
        frame = results[
            direction
        ][
            "metrics"
        ].copy()

        combined_metrics.append(
            frame
        )

    combined = pd.concat(
        combined_metrics,
        ignore_index=True,
    )

    display = combined[
        [
            "direction",
            "model",
            "utility",
            "auroc",
            "auprc",
            "brier",
        ]
    ].copy()

    print(
        display.to_string(
            index=False
        )
    )

    combined.to_csv(
        destination
        / "external_metrics_all.csv",
        index=False,
    )

    combined_transfer = pd.concat(
        [
            results[
                direction
            ][
                "transfer"
            ]
            for direction in (
                "A_to_B",
                "B_to_A",
            )
        ],
        ignore_index=True,
    )

    combined_transfer.to_csv(
        destination
        / "transfer_gaps_all.csv",
        index=False,
    )

    combined_bootstrap = pd.concat(
        [
            results[
                direction
            ][
                "bootstrap"
            ]
            for direction in (
                "A_to_B",
                "B_to_A",
            )
        ],
        ignore_index=True,
    )

    combined_bootstrap.to_csv(
        destination
        / "bootstrap_summary_all.csv",
        index=False,
    )

    print()
    print(
        "Final external evaluation: COMPLETE"
    )

    print(
        f"Artifacts: {destination}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )