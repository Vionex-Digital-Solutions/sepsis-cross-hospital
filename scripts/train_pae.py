from __future__ import annotations

import argparse
import gc
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from sepsis_cross_hospital.evaluation.metrics import (
    compute_challenge_utility,
)
from sepsis_cross_hospital.evaluation.thresholding import (
    select_utility_threshold,
)
from sepsis_cross_hospital.models.pae import (
    FROZEN_ALPHAS,
    blend_probabilities,
    select_pae_alpha,
)


DEPTHS = (
    3,
    5,
)

LEARNING_RATES = (
    0.03,
    0.10,
)

SEEDS = (
    1729,
    2718,
    31415,
    57721,
    65537,
)

MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50

SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
MIN_CHILD_WEIGHT = 1.0
REG_LAMBDA = 1.0

MAX_BIN = 256

DEVICE = "cuda"
TREE_METHOD = "hist"


@dataclass
class SelectedBranch:
    representation: str
    max_depth: int
    learning_rate: float
    threshold: float

    validation_patient_ids: list[str]
    validation_labels: list[
        np.ndarray
    ]

    validation_probabilities: list[
        np.ndarray
    ]

    candidate_directory: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train frozen Process-Attenuated "
            "Ensemble proposed model."
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


def load_schema(
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

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


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

    return pd.read_parquet(
        path,
        columns=(
            [
                "patient_id",
                "SepsisLabel",
            ]
            + feature_columns
        ),
    )


def patient_sequences(
    frame: pd.DataFrame,
    values: np.ndarray,
) -> tuple[
    list[str],
    list[np.ndarray],
    list[np.ndarray],
]:
    if len(frame) != len(values):
        raise ValueError(
            "Frame/value length mismatch."
        )

    helper = pd.DataFrame(
        {
            "patient_id": (
                frame[
                    "patient_id"
                ].astype(str)
            ),
            "label": (
                frame[
                    "SepsisLabel"
                ].to_numpy(
                    dtype=np.int8
                )
            ),
            "value": np.asarray(
                values,
                dtype=np.float64,
            ),
        }
    )

    patient_ids: list[str] = []
    labels: list[np.ndarray] = []
    probabilities: list[
        np.ndarray
    ] = []

    for patient_id, group in helper.groupby(
        "patient_id",
        sort=False,
    ):
        patient_ids.append(
            str(patient_id)
        )

        labels.append(
            group[
                "label"
            ].to_numpy(
                dtype=np.int8
            )
        )

        probabilities.append(
            group[
                "value"
            ].to_numpy(
                dtype=np.float64
            )
        )

    return (
        patient_ids,
        labels,
        probabilities,
    )


def evaluate_sequences(
    labels_by_patient: list[np.ndarray],
    probabilities_by_patient: list[np.ndarray],
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


def candidate_name(
    depth: int,
    learning_rate: float,
) -> str:
    eta = str(
        learning_rate
    ).replace(
        ".",
        "p",
    )

    return (
        f"depth_{depth}"
        f"__eta_{eta}"
    )


def train_branch(
    *,
    direction: str,
    representation: str,
    feature_columns: list[str],
    branch_root: Path,
) -> SelectedBranch:
    print()
    print("=" * 72)
    print(
        f"{representation} BRANCH"
    )
    print("=" * 72)

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    train = load_partition(
        direction,
        "train",
        feature_columns,
    )

    validation = load_partition(
        direction,
        "validation",
        feature_columns,
    )

    X_train = (
        train[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_train = (
        train[
            "SepsisLabel"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    X_validation = (
        validation[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_validation = (
        validation[
            "SepsisLabel"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    print(
        f"Train hours      : "
        f"{len(train):,}"
    )

    print(
        f"Validation hours : "
        f"{len(validation):,}"
    )

    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        max_bin=MAX_BIN,
    )

    dvalidation = (
        xgb.QuantileDMatrix(
            X_validation,
            label=y_validation,
            ref=dtrain,
            max_bin=MAX_BIN,
        )
    )

    del X_train
    del X_validation

    gc.collect()

    zero_values = np.zeros(
        len(validation),
        dtype=np.float64,
    )

    (
        validation_patient_ids,
        validation_labels,
        _,
    ) = patient_sequences(
        validation,
        zero_values,
    )

    candidate_rows = []

    candidate_probability_sets: dict[
        str,
        list[np.ndarray],
    ] = {}

    candidate_directories: dict[
        str,
        Path,
    ] = {}

    for depth in DEPTHS:
        for learning_rate in (
            LEARNING_RATES
        ):
            name = candidate_name(
                depth,
                learning_rate,
            )

            directory = (
                branch_root
                / name
            )

            directory.mkdir(
                parents=True
            )

            candidate_directories[
                name
            ] = directory

            print()
            print(
                "-" * 72
            )

            print(
                f"depth={depth}, "
                f"eta={learning_rate}"
            )

            seed_probabilities = []
            seed_rows = []

            for seed in SEEDS:
                print(
                    f"  seed {seed} ..."
                )

                params = {
                    "objective": (
                        "binary:logistic"
                    ),
                    "eval_metric": (
                        "logloss"
                    ),
                    "tree_method": (
                        TREE_METHOD
                    ),
                    "device": DEVICE,
                    "max_depth": depth,
                    "eta": (
                        learning_rate
                    ),
                    "subsample": (
                        SUBSAMPLE
                    ),
                    "colsample_bytree": (
                        COLSAMPLE_BYTREE
                    ),
                    "min_child_weight": (
                        MIN_CHILD_WEIGHT
                    ),
                    "lambda": (
                        REG_LAMBDA
                    ),
                    "max_bin": (
                        MAX_BIN
                    ),
                    "seed": seed,
                    "verbosity": 0,
                }

                booster = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=(
                        MAX_BOOST_ROUNDS
                    ),
                    evals=[
                        (
                            dvalidation,
                            "validation",
                        )
                    ],
                    early_stopping_rounds=(
                        EARLY_STOPPING_ROUNDS
                    ),
                    verbose_eval=False,
                )

                best_iteration = int(
                    booster.best_iteration
                )

                best_logloss = float(
                    booster.best_score
                )

                probabilities = (
                    booster.predict(
                        dvalidation,
                        iteration_range=(
                            0,
                            best_iteration
                            + 1,
                        ),
                    )
                )

                seed_probabilities.append(
                    probabilities.astype(
                        np.float64
                    )
                )

                model_path = (
                    directory
                    / f"seed_{seed}.json"
                )

                booster.save_model(
                    model_path
                )

                seed_rows.append(
                    {
                        "seed": seed,
                        "best_iteration": (
                            best_iteration
                        ),
                        "best_logloss": (
                            best_logloss
                        ),
                    }
                )

                print(
                    f"    iteration "
                    f"{best_iteration} | "
                    f"logloss "
                    f"{best_logloss:.6f}"
                )

                del booster

                gc.collect()

            ensemble_flat = np.mean(
                np.stack(
                    seed_probabilities,
                    axis=0,
                ),
                axis=0,
            )

            (
                candidate_patient_ids,
                candidate_labels,
                ensemble_sequences,
            ) = patient_sequences(
                validation,
                ensemble_flat,
            )

            if (
                candidate_patient_ids
                != validation_patient_ids
            ):
                raise RuntimeError(
                    "Validation patient order changed."
                )

            for first, second in zip(
                candidate_labels,
                validation_labels,
                strict=True,
            ):
                if not np.array_equal(
                    first,
                    second,
                ):
                    raise RuntimeError(
                        "Validation labels changed."
                    )

            threshold_result = (
                select_utility_threshold(
                    validation_labels,
                    ensemble_sequences,
                )
            )

            metrics = evaluate_sequences(
                validation_labels,
                ensemble_sequences,
                threshold_result.threshold,
            )

            print(
                f"  threshold : "
                f"{threshold_result.threshold:.3f}"
            )

            print(
                f"  utility   : "
                f"{metrics['utility']:.6f}"
            )

            print(
                f"  AUROC     : "
                f"{metrics['auroc']:.6f}"
            )

            print(
                f"  AUPRC     : "
                f"{metrics['auprc']:.6f}"
            )

            pd.DataFrame(
                seed_rows
            ).to_csv(
                directory
                / "seed_results.csv",
                index=False,
            )

            candidate_probability_sets[
                name
            ] = ensemble_sequences

            candidate_rows.append(
                {
                    "candidate_name": (
                        name
                    ),
                    "max_depth": depth,
                    "learning_rate": (
                        learning_rate
                    ),
                    "threshold": float(
                        threshold_result
                        .threshold
                    ),
                    "validation_utility": float(
                        metrics[
                            "utility"
                        ]
                    ),
                    "validation_auroc": float(
                        metrics[
                            "auroc"
                        ]
                    ),
                    "validation_auprc": float(
                        metrics[
                            "auprc"
                        ]
                    ),
                    "validation_brier": float(
                        metrics[
                            "brier"
                        ]
                    ),
                }
            )

    candidate_frame = pd.DataFrame(
        candidate_rows
    )

    candidate_frame.to_csv(
        branch_root
        / "candidate_results.csv",
        index=False,
    )

    maximum = float(
        candidate_frame[
            "validation_utility"
        ].max()
    )

    tied = candidate_frame.loc[
        np.isclose(
            candidate_frame[
                "validation_utility"
            ],
            maximum,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    selected = (
        tied.sort_values(
            [
                "max_depth",
                "learning_rate",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .iloc[0]
    )

    selected_name = str(
        selected[
            "candidate_name"
        ]
    )

    print()
    print(
        f"Selected {representation}: "
        f"depth="
        f"{int(selected['max_depth'])}, "
        f"eta="
        f"{float(selected['learning_rate'])}"
    )

    print(
        f"Validation utility: "
        f"{float(selected['validation_utility']):.6f}"
    )

    return SelectedBranch(
        representation=(
            representation
        ),
        max_depth=int(
            selected[
                "max_depth"
            ]
        ),
        learning_rate=float(
            selected[
                "learning_rate"
            ]
        ),
        threshold=float(
            selected[
                "threshold"
            ]
        ),
        validation_patient_ids=(
            validation_patient_ids
        ),
        validation_labels=(
            validation_labels
        ),
        validation_probabilities=(
            candidate_probability_sets[
                selected_name
            ]
        ),
        candidate_directory=(
            candidate_directories[
                selected_name
            ]
        ),
    )


def predict_selected_branch(
    *,
    direction: str,
    feature_columns: list[str],
    branch: SelectedBranch,
) -> tuple[
    list[str],
    list[np.ndarray],
    list[np.ndarray],
]:
    frame = load_partition(
        direction,
        "internal_test",
        feature_columns,
    )

    X = (
        frame[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    matrix = xgb.DMatrix(
        X
    )

    del X

    gc.collect()

    seed_results = pd.read_csv(
        branch.candidate_directory
        / "seed_results.csv"
    )

    seed_predictions = []

    for seed in SEEDS:
        row = seed_results.loc[
            seed_results[
                "seed"
            ] == seed
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Missing metadata "
                f"for seed {seed}."
            )

        best_iteration = int(
            row.iloc[0][
                "best_iteration"
            ]
        )

        booster = xgb.Booster()

        booster.load_model(
            branch.candidate_directory
            / f"seed_{seed}.json"
        )

        probability = booster.predict(
            matrix,
            iteration_range=(
                0,
                best_iteration + 1,
            ),
        )

        seed_predictions.append(
            probability.astype(
                np.float64
            )
        )

        del booster

    ensemble_flat = np.mean(
        np.stack(
            seed_predictions,
            axis=0,
        ),
        axis=0,
    )

    return patient_sequences(
        frame,
        ensemble_flat,
    )


def save_predictions(
    path: Path,
    patient_ids: list[str],
    labels: list[np.ndarray],
    probabilities: list[np.ndarray],
) -> None:
    frames = []

    for (
        patient_id,
        patient_labels,
        patient_probabilities,
    ) in zip(
        patient_ids,
        labels,
        probabilities,
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
                            len(
                                patient_labels
                            )
                        )
                    ),
                    "label": (
                        patient_labels
                    ),
                    "probability": (
                        patient_probabilities
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


def main() -> int:
    args = parse_args()

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

    print("=" * 72)
    print(
        "PROCESS-ATTENUATED ENSEMBLE (PAE)"
    )
    print("=" * 72)

    print(
        f"Run ID    : {args.run_id}"
    )

    print(
        f"Direction : {args.direction}"
    )

    print(
        f"XGBoost   : "
        f"{xgb.__version__}"
    )

    print(
        f"Device    : {DEVICE}"
    )

    print(
        f"Alphas    : "
        f"{FROZEN_ALPHAS}"
    )

    schema = load_schema(
        args.direction
    )

    v0_columns = list(
        schema["V0"]
    )

    v2_columns = list(
        schema["V2"]
    )

    v0_branch = train_branch(
        direction=args.direction,
        representation="V0",
        feature_columns=v0_columns,
        branch_root=(
            run_dir
            / "V0_branch"
        ),
    )

    gc.collect()

    v2_branch = train_branch(
        direction=args.direction,
        representation="V2",
        feature_columns=v2_columns,
        branch_root=(
            run_dir
            / "V2_branch"
        ),
    )

    if (
        v0_branch.validation_patient_ids
        != v2_branch.validation_patient_ids
    ):
        raise RuntimeError(
            "V0/V2 validation patient "
            "orders differ."
        )

    for first, second in zip(
        v0_branch.validation_labels,
        v2_branch.validation_labels,
        strict=True,
    ):
        if not np.array_equal(
            first,
            second,
        ):
            raise RuntimeError(
                "V0/V2 validation labels differ."
            )

    print()
    print("=" * 72)
    print(
        "PAE SOURCE VALIDATION SELECTION"
    )
    print("=" * 72)

    pae_selection = (
        select_pae_alpha(
            v0_branch.validation_labels,
            v0_branch
            .validation_probabilities,
            v2_branch
            .validation_probabilities,
        )
    )

    alpha_rows = []

    for candidate in (
        pae_selection.candidates
    ):
        print(
            f"alpha={candidate.alpha:.2f} | "
            f"threshold="
            f"{candidate.threshold:.3f} | "
            f"utility="
            f"{candidate.validation_utility:.6f}"
        )

        alpha_rows.append(
            {
                "alpha": (
                    candidate.alpha
                ),
                "threshold": (
                    candidate.threshold
                ),
                "validation_utility": (
                    candidate
                    .validation_utility
                ),
            }
        )

    selected_alpha = float(
        pae_selection.alpha
    )

    selected_threshold = float(
        pae_selection.threshold
    )

    validation_probabilities = list(
        pae_selection
        .probabilities_by_patient
    )

    validation_metrics = (
        evaluate_sequences(
            v0_branch.validation_labels,
            validation_probabilities,
            selected_threshold,
        )
    )

    print()
    print(
        f"Selected alpha     : "
        f"{selected_alpha:.2f}"
    )

    print(
        f"Selected threshold : "
        f"{selected_threshold:.3f}"
    )

    print(
        f"Validation utility : "
        f"{validation_metrics['utility']:.6f}"
    )

    pd.DataFrame(
        alpha_rows
    ).to_csv(
        run_dir
        / "alpha_results.csv",
        index=False,
    )

    # ========================================================
    # INTERNAL TEST ONLY
    # ========================================================

    print()
    print(
        "Predicting V0 internal test..."
    )

    (
        v0_internal_ids,
        v0_internal_labels,
        v0_internal_probabilities,
    ) = predict_selected_branch(
        direction=args.direction,
        feature_columns=v0_columns,
        branch=v0_branch,
    )

    print(
        "Predicting V2 internal test..."
    )

    (
        v2_internal_ids,
        v2_internal_labels,
        v2_internal_probabilities,
    ) = predict_selected_branch(
        direction=args.direction,
        feature_columns=v2_columns,
        branch=v2_branch,
    )

    if (
        v0_internal_ids
        != v2_internal_ids
    ):
        raise RuntimeError(
            "Internal V0/V2 patient "
            "orders differ."
        )

    for first, second in zip(
        v0_internal_labels,
        v2_internal_labels,
        strict=True,
    ):
        if not np.array_equal(
            first,
            second,
        ):
            raise RuntimeError(
                "Internal labels differ "
                "between branches."
            )

    internal_probabilities = (
        blend_probabilities(
            v0_internal_probabilities,
            v2_internal_probabilities,
            selected_alpha,
        )
    )

    internal_metrics = (
        evaluate_sequences(
            v0_internal_labels,
            internal_probabilities,
            selected_threshold,
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

    save_predictions(
        run_dir
        / "validation_predictions.parquet",
        v0_branch.validation_patient_ids,
        v0_branch.validation_labels,
        validation_probabilities,
    )

    save_predictions(
        run_dir
        / "internal_predictions.parquet",
        v0_internal_ids,
        v0_internal_labels,
        internal_probabilities,
    )

    metadata = {
        "run_id": args.run_id,
        "model": (
            "Process-Attenuated "
            "Ensemble"
        ),
        "abbreviation": "PAE",
        "direction": args.direction,
        "external_evaluated": False,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "xgboost_version": (
            xgb.__version__
        ),
        "device": DEVICE,
        "alphas": list(
            FROZEN_ALPHAS
        ),
        "alpha_tie_break": (
            "larger alpha"
        ),
        "selected_alpha": (
            selected_alpha
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "V0": {
            "max_depth": (
                v0_branch.max_depth
            ),
            "learning_rate": (
                v0_branch
                .learning_rate
            ),
            "validation_threshold": (
                v0_branch.threshold
            ),
        },
        "V2": {
            "max_depth": (
                v2_branch.max_depth
            ),
            "learning_rate": (
                v2_branch
                .learning_rate
            ),
            "validation_threshold": (
                v2_branch.threshold
            ),
        },
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
        f"Artifacts: {run_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )