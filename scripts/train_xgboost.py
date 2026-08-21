from __future__ import annotations

import argparse
import gc
import json
import subprocess
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


# ============================================================
# Frozen protocol
# ============================================================

REPRESENTATIONS = (
    "M0",
    "V0",
    "V1",
    "V2",
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
OBJECTIVE = "binary:logistic"
EVAL_METRIC = "logloss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen GPU XGBoost "
            "cross-hospital baseline."
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
        "--representation",
        required=True,
        choices=REPRESENTATIONS,
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


def load_feature_columns(
    direction: str,
    representation: str,
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

    if representation not in schema:
        raise KeyError(
            f"Representation "
            f"{representation} missing "
            f"from {path}."
        )

    return list(
        schema[
            representation
        ]
    )


def load_partition(
    direction: str,
    partition: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    # IMPORTANT:
    # This function is intentionally only called
    # for source train / validation / internal_test.
    # The external file is never opened here.
    path = (
        Path("data/processed")
        / direction
        / f"{partition}.parquet"
    )

    if partition == "external":
        raise RuntimeError(
            "External evaluation is locked."
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


def make_patient_sequences(
    frame: pd.DataFrame,
    values: np.ndarray,
) -> tuple[
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

    labels_by_patient = []
    values_by_patient = []

    for _, group in helper.groupby(
        "patient_id",
        sort=False,
    ):
        labels_by_patient.append(
            group[
                "label"
            ].to_numpy(
                dtype=np.int8
            )
        )

        values_by_patient.append(
            group[
                "value"
            ].to_numpy(
                dtype=np.float64
            )
        )

    return (
        labels_by_patient,
        values_by_patient,
    )


def evaluate_probabilities(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y_true = frame[
        "SepsisLabel"
    ].to_numpy(
        dtype=np.int8
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(np.int8)

    (
        labels_by_patient,
        probabilities_by_patient,
    ) = make_patient_sequences(
        frame,
        probabilities,
    )

    predictions_by_patient = [
        (
            probability
            >= threshold
        ).astype(np.int8)
        for probability
        in probabilities_by_patient
    ]

    utility = (
        compute_challenge_utility(
            labels_by_patient,
            predictions_by_patient,
        ).normalized
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
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
                y_true,
                probabilities,
            )
        ),
        "auprc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "brier": float(
            brier_score_loss(
                y_true,
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


def predict_best(
    booster: xgb.Booster,
    matrix: xgb.DMatrix,
) -> np.ndarray:
    if not hasattr(
        booster,
        "best_iteration",
    ):
        raise RuntimeError(
            "Booster has no best_iteration."
        )

    return booster.predict(
        matrix,
        iteration_range=(
            0,
            booster.best_iteration
            + 1,
        ),
    )


def candidate_directory_name(
    *,
    depth: int,
    learning_rate: float,
) -> str:
    learning_rate_text = (
        str(
            learning_rate
        )
        .replace(
            ".",
            "p",
        )
    )

    return (
        f"depth_{depth}"
        f"__eta_{learning_rate_text}"
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

    candidates_dir = (
        run_dir
        / "candidate_models"
    )

    candidates_dir.mkdir()

    print("=" * 72)
    print(
        "GPU XGBOOST BASELINE"
    )
    print("=" * 72)

    print(
        f"Run ID         : "
        f"{args.run_id}"
    )

    print(
        f"Direction      : "
        f"{args.direction}"
    )

    print(
        f"Representation : "
        f"{args.representation}"
    )

    print(
        f"XGBoost        : "
        f"{xgb.__version__}"
    )

    print(
        f"Device         : "
        f"{DEVICE}"
    )

    feature_columns = (
        load_feature_columns(
            args.direction,
            args.representation,
        )
    )

    print(
        f"Feature count  : "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # Load SOURCE data only.
    # ========================================================

    train = load_partition(
        args.direction,
        "train",
        feature_columns,
    )

    validation = load_partition(
        args.direction,
        "validation",
        feature_columns,
    )

    internal_test = (
        load_partition(
            args.direction,
            "internal_test",
            feature_columns,
        )
    )

    print()
    print(
        f"Train hours      : "
        f"{len(train):,}"
    )

    print(
        f"Validation hours : "
        f"{len(validation):,}"
    )

    print(
        f"Internal hours   : "
        f"{len(internal_test):,}"
    )

    # XGBoost tree baseline uses raw causal values.
    # No scaling / clipping is applied.

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

    X_internal = (
        internal_test[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_internal = (
        internal_test[
            "SepsisLabel"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    print()
    print(
        "Building QuantileDMatrix..."
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

    dinternal = (
        xgb.QuantileDMatrix(
            X_internal,
            label=y_internal,
            ref=dtrain,
            max_bin=MAX_BIN,
        )
    )

    # Arrays are no longer needed after QDM creation.
    del X_train
    del X_validation
    del X_internal

    gc.collect()

    (
        validation_labels_by_patient,
        _,
    ) = make_patient_sequences(
        validation,
        np.zeros(
            len(validation),
            dtype=np.float64,
        ),
    )

    candidate_results = []

    print()
    print("=" * 72)
    print(
        "SOURCE VALIDATION MODEL SELECTION"
    )
    print("=" * 72)

    # ========================================================
    # Four frozen depth / learning-rate candidates.
    # ========================================================

    for depth in DEPTHS:
        for learning_rate in (
            LEARNING_RATES
        ):
            print()
            print(
                "-" * 72
            )

            print(
                f"Candidate: "
                f"depth={depth}, "
                f"eta={learning_rate}"
            )

            candidate_name = (
                candidate_directory_name(
                    depth=depth,
                    learning_rate=(
                        learning_rate
                    ),
                )
            )

            candidate_dir = (
                candidates_dir
                / candidate_name
            )

            candidate_dir.mkdir()

            seed_validation_probabilities = []

            seed_rows = []

            for seed in SEEDS:
                print(
                    f"  seed {seed} ..."
                )

                params = {
                    "objective": (
                        OBJECTIVE
                    ),
                    "eval_metric": (
                        EVAL_METRIC
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

                evals_result: dict = {}

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
                    evals_result=(
                        evals_result
                    ),
                    early_stopping_rounds=(
                        EARLY_STOPPING_ROUNDS
                    ),
                    verbose_eval=False,
                )

                probability = (
                    predict_best(
                        booster,
                        dvalidation,
                    )
                )

                seed_validation_probabilities.append(
                    probability.astype(
                        np.float64
                    )
                )

                model_path = (
                    candidate_dir
                    / (
                        f"seed_{seed}.json"
                    )
                )

                booster.save_model(
                    model_path
                )

                best_iteration = int(
                    booster.best_iteration
                )

                best_score = float(
                    booster.best_score
                )

                seed_rows.append(
                    {
                        "seed": seed,
                        "best_iteration": (
                            best_iteration
                        ),
                        "best_logloss": (
                            best_score
                        ),
                        "model_file": (
                            str(
                                model_path
                            )
                        ),
                    }
                )

                print(
                    f"    best iteration : "
                    f"{best_iteration}"
                )

                print(
                    f"    best logloss   : "
                    f"{best_score:.6f}"
                )

                del booster

                gc.collect()

            # =================================================
            # Five-seed ensemble for this candidate.
            # =================================================

            validation_probabilities = (
                np.mean(
                    np.stack(
                        seed_validation_probabilities,
                        axis=0,
                    ),
                    axis=0,
                )
            )

            (
                _,
                validation_probabilities_by_patient,
            ) = make_patient_sequences(
                validation,
                validation_probabilities,
            )

            threshold_result = (
                select_utility_threshold(
                    validation_labels_by_patient,
                    validation_probabilities_by_patient,
                )
            )

            validation_metrics = (
                evaluate_probabilities(
                    validation,
                    validation_probabilities,
                    threshold_result.threshold,
                )
            )

            print()
            print(
                f"  ensemble threshold : "
                f"{threshold_result.threshold:.3f}"
            )

            print(
                f"  ensemble utility   : "
                f"{validation_metrics['utility']:.6f}"
            )

            print(
                f"  ensemble AUROC     : "
                f"{validation_metrics['auroc']:.6f}"
            )

            print(
                f"  ensemble AUPRC     : "
                f"{validation_metrics['auprc']:.6f}"
            )

            print(
                f"  ensemble Brier     : "
                f"{validation_metrics['brier']:.6f}"
            )

            pd.DataFrame(
                seed_rows
            ).to_csv(
                candidate_dir
                / "seed_results.csv",
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
                candidate_dir
                / "threshold_curve.csv",
                index=False,
            )

            candidate_results.append(
                {
                    "max_depth": depth,
                    "learning_rate": (
                        learning_rate
                    ),
                    "threshold": float(
                        threshold_result.threshold
                    ),
                    "validation_utility": float(
                        validation_metrics[
                            "utility"
                        ]
                    ),
                    "validation_auroc": float(
                        validation_metrics[
                            "auroc"
                        ]
                    ),
                    "validation_auprc": float(
                        validation_metrics[
                            "auprc"
                        ]
                    ),
                    "validation_brier": float(
                        validation_metrics[
                            "brier"
                        ]
                    ),
                    "candidate_name": (
                        candidate_name
                    ),
                }
            )

            del seed_validation_probabilities
            del validation_probabilities

            gc.collect()

    # ========================================================
    # Candidate selection.
    # ========================================================

    candidate_frame = pd.DataFrame(
        candidate_results
    )

    maximum_utility = float(
        candidate_frame[
            "validation_utility"
        ].max()
    )

    tied = candidate_frame.loc[
        np.isclose(
            candidate_frame[
                "validation_utility"
            ],
            maximum_utility,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()

    # Frozen deterministic tie-break:
    # 1. shallower tree
    # 2. lower learning rate
    selected_row = (
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

    selected_depth = int(
        selected_row[
            "max_depth"
        ]
    )

    selected_learning_rate = float(
        selected_row[
            "learning_rate"
        ]
    )

    selected_threshold = float(
        selected_row[
            "threshold"
        ]
    )

    selected_candidate_name = str(
        selected_row[
            "candidate_name"
        ]
    )

    print()
    print("=" * 72)
    print(
        "SELECTED VALIDATION MODEL"
    )
    print("=" * 72)

    print(
        f"max_depth    : "
        f"{selected_depth}"
    )

    print(
        f"learning_rate: "
        f"{selected_learning_rate}"
    )

    print(
        f"threshold    : "
        f"{selected_threshold:.3f}"
    )

    print(
        f"utility      : "
        f"{float(selected_row['validation_utility']):.6f}"
    )

    # ========================================================
    # Reload only selected five seed models and evaluate
    # SOURCE INTERNAL TEST.
    # ========================================================

    selected_candidate_dir = (
        candidates_dir
        / selected_candidate_name
    )

    seed_internal_probabilities = []

    selected_seed_metadata = []

    seed_results = pd.read_csv(
        selected_candidate_dir
        / "seed_results.csv"
    )

    for seed in SEEDS:
        model_path = (
            selected_candidate_dir
            / f"seed_{seed}.json"
        )

        booster = xgb.Booster()

        booster.load_model(
            model_path
        )

        row = seed_results.loc[
            seed_results[
                "seed"
            ] == seed
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Missing seed metadata "
                f"for {seed}."
            )

        best_iteration = int(
            row.iloc[0][
                "best_iteration"
            ]
        )

        probability = booster.predict(
            dinternal,
            iteration_range=(
                0,
                best_iteration + 1,
            ),
        )

        seed_internal_probabilities.append(
            probability.astype(
                np.float64
            )
        )

        selected_seed_metadata.append(
            {
                "seed": seed,
                "best_iteration": (
                    best_iteration
                ),
                "best_logloss": float(
                    row.iloc[0][
                        "best_logloss"
                    ]
                ),
            }
        )

        del booster

        gc.collect()

    internal_probabilities = np.mean(
        np.stack(
            seed_internal_probabilities,
            axis=0,
        ),
        axis=0,
    )

    internal_metrics = (
        evaluate_probabilities(
            internal_test,
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

    # ========================================================
    # Run artifacts
    # ========================================================

    candidate_frame.to_csv(
        run_dir
        / "candidate_results.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "patient_id": (
                internal_test[
                    "patient_id"
                ]
            ),
            "label": (
                internal_test[
                    "SepsisLabel"
                ]
            ),
            "probability": (
                internal_probabilities
            ),
        }
    ).to_parquet(
        run_dir
        / "internal_predictions.parquet",
        index=False,
    )

    metadata = {
        "run_id": args.run_id,
        "model": "xgboost",
        "direction": args.direction,
        "feature_representation": (
            args.representation
        ),
        "feature_count": len(
            feature_columns
        ),
        "external_evaluated": False,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "xgboost_version": (
            xgb.__version__
        ),
        "device": DEVICE,
        "tree_method": TREE_METHOD,
        "objective": OBJECTIVE,
        "eval_metric": EVAL_METRIC,
        "max_boost_rounds": (
            MAX_BOOST_ROUNDS
        ),
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "subsample": SUBSAMPLE,
        "colsample_bytree": (
            COLSAMPLE_BYTREE
        ),
        "min_child_weight": (
            MIN_CHILD_WEIGHT
        ),
        "reg_lambda": REG_LAMBDA,
        "max_bin": MAX_BIN,
        "seeds": list(SEEDS),
        "candidate_grid": [
            {
                "max_depth": depth,
                "learning_rate": lr,
            }
            for depth in DEPTHS
            for lr in LEARNING_RATES
        ],
        "candidate_tie_break": (
            "shallower max_depth, then "
            "lower learning_rate"
        ),
        "selected_max_depth": (
            selected_depth
        ),
        "selected_learning_rate": (
            selected_learning_rate
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "selected_seed_metadata": (
            selected_seed_metadata
        ),
        "validation_metrics": {
            "utility": float(
                selected_row[
                    "validation_utility"
                ]
            ),
            "auroc": float(
                selected_row[
                    "validation_auroc"
                ]
            ),
            "auprc": float(
                selected_row[
                    "validation_auprc"
                ]
            ),
            "brier": float(
                selected_row[
                    "validation_brier"
                ]
            ),
        },
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