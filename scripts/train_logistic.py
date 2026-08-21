from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

from sepsis_cross_hospital.evaluation.metrics import (
    compute_challenge_utility,
)
from sepsis_cross_hospital.evaluation.thresholding import (
    select_utility_threshold,
)


C_GRID = [
    0.1,
    1.0,
    10.0,
]

CLIP_LIMIT = 10.0

SEED = 1729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen "
            "logistic-regression baseline."
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

    return list(
        schema["V0"]
    )


def load_partition(
    direction: str,
    partition: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    path = (
        Path("data/processed")
        / direction
        / f"{partition}.parquet"
    )

    columns = (
        [
            "patient_id",
            "SepsisLabel",
        ]
        + feature_columns
    )

    return pd.read_parquet(
        path,
        columns=columns,
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
            "value": values,
        }
    )

    labels = []
    predictions = []

    for _, group in helper.groupby(
        "patient_id",
        sort=False,
    ):
        labels.append(
            group[
                "label"
            ].to_numpy(
                dtype=np.int8
            )
        )

        predictions.append(
            group[
                "value"
            ].to_numpy(
                dtype=np.float64
            )
        )

    return (
        labels,
        predictions,
    )


def transform(
    scaler: RobustScaler,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    matrix = frame[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    transformed = scaler.transform(
        matrix
    )

    np.clip(
        transformed,
        -CLIP_LIMIT,
        CLIP_LIMIT,
        out=transformed,
    )

    return transformed


def evaluate(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y_true = frame[
        "SepsisLabel"
    ].to_numpy(
        dtype=np.int8
    )

    binary = (
        probabilities
        >= threshold
    ).astype(np.int8)

    (
        labels_by_patient,
        probability_by_patient,
    ) = make_patient_sequences(
        frame,
        probabilities,
    )

    binary_by_patient = [
        (
            probability
            >= threshold
        ).astype(np.int8)
        for probability
        in probability_by_patient
    ]

    utility = (
        compute_challenge_utility(
            labels_by_patient,
            binary_by_patient,
        ).normalized
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            binary,
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
        "LOGISTIC REGRESSION BASELINE"
    )
    print("=" * 72)

    print(
        f"Run ID    : {args.run_id}"
    )

    print(
        f"Direction : {args.direction}"
    )

    feature_columns = (
        load_feature_columns(
            args.direction
        )
    )

    print(
        f"Features  : "
        f"{len(feature_columns)} (V0)"
    )

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

    print()
    print(
        "Fitting source-training "
        "RobustScaler..."
    )

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(
            25.0,
            75.0,
        ),
    )

    X_train_raw = train[
        feature_columns
    ].to_numpy(
        dtype=np.float64
    )

    scaler.fit(
        X_train_raw
    )

    X_train = scaler.transform(
        X_train_raw
    )

    np.clip(
        X_train,
        -CLIP_LIMIT,
        CLIP_LIMIT,
        out=X_train,
    )

    del X_train_raw

    X_validation = transform(
        scaler,
        validation,
        feature_columns,
    )

    y_train = train[
        "SepsisLabel"
    ].to_numpy(
        dtype=np.int8
    )

    candidate_rows = []
    candidate_models = {}

    print()
    print("=" * 72)
    print("SOURCE VALIDATION MODEL SELECTION")
    print("=" * 72)

    for C in C_GRID:
        print()
        print(
            f"Training C={C} ..."
        )

        model = LogisticRegression(
            C=C,
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
            fit_intercept=True,
            tol=1e-4,
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        (
            labels_by_patient,
            probabilities_by_patient,
        ) = make_patient_sequences(
            validation,
            probabilities,
        )

        threshold_result = (
            select_utility_threshold(
                labels_by_patient,
                probabilities_by_patient,
            )
        )

        metrics = evaluate(
            validation,
            probabilities,
            threshold_result.threshold,
        )

        row = {
            "C": C,
            "threshold": (
                threshold_result.threshold
            ),
            "validation_utility": (
                metrics["utility"]
            ),
            "validation_auroc": (
                metrics["auroc"]
            ),
            "validation_auprc": (
                metrics["auprc"]
            ),
            "validation_brier": (
                metrics["brier"]
            ),
            "iterations": int(
                model.n_iter_[0]
            ),
        }

        candidate_rows.append(
            row
        )

        candidate_models[
            C
        ] = (
            model,
            probabilities,
            threshold_result,
        )

        print(
            f"  threshold = "
            f"{row['threshold']:.3f}"
        )

        print(
            f"  utility   = "
            f"{row['validation_utility']:.6f}"
        )

        print(
            f"  AUROC     = "
            f"{row['validation_auroc']:.6f}"
        )

        print(
            f"  AUPRC     = "
            f"{row['validation_auprc']:.6f}"
        )

    candidate_frame = pd.DataFrame(
        candidate_rows
    )

    maximum_utility = (
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
    ]

    # Logistic tie-break:
    # stronger regularization / smaller C.
    selected_row = (
        tied.sort_values(
            "C",
            ascending=True,
        ).iloc[0]
    )

    selected_C = float(
        selected_row["C"]
    )

    (
        selected_model,
        validation_probabilities,
        selected_threshold_result,
    ) = candidate_models[
        selected_C
    ]

    threshold = float(
        selected_threshold_result.threshold
    )

    print()
    print("=" * 72)
    print("SELECTED MODEL")
    print("=" * 72)

    print(
        f"C         : {selected_C}"
    )

    print(
        f"Threshold : {threshold:.3f}"
    )

    validation_metrics = evaluate(
        validation,
        validation_probabilities,
        threshold,
    )

    # --------------------------------------------------
    # INTERNAL TEST ONLY.
    # EXTERNAL HOSPITAL REMAINS LOCKED.
    # --------------------------------------------------
    X_internal = transform(
        scaler,
        internal_test,
        feature_columns,
    )

    internal_probabilities = (
        selected_model.predict_proba(
            X_internal
        )[:, 1]
    )

    internal_metrics = evaluate(
        internal_test,
        internal_probabilities,
        threshold,
    )

    print()
    print("=" * 72)
    print("SOURCE INTERNAL TEST")
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

    # --------------------------------------------------
    # Artifacts
    # --------------------------------------------------
    joblib.dump(
        selected_model,
        run_dir
        / "model.joblib",
    )

    joblib.dump(
        scaler,
        run_dir
        / "scaler.joblib",
    )

    candidate_frame.to_csv(
        run_dir
        / "candidate_results.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "threshold": (
                selected_threshold_result
                .thresholds
            ),
            "normalized_utility": (
                selected_threshold_result
                .utilities
            ),
        }
    ).to_csv(
        run_dir
        / "threshold_curve.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "patient_id": validation[
                "patient_id"
            ],
            "label": validation[
                "SepsisLabel"
            ],
            "probability": (
                validation_probabilities
            ),
        }
    ).to_parquet(
        run_dir
        / "validation_predictions.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "patient_id": internal_test[
                "patient_id"
            ],
            "label": internal_test[
                "SepsisLabel"
            ],
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
        "model": "B1_logistic_regression",
        "direction": args.direction,
        "feature_representation": "V0",
        "feature_count": len(
            feature_columns
        ),
        "C_grid": C_GRID,
        "selected_C": selected_C,
        "selected_threshold": threshold,
        "clip_limit": CLIP_LIMIT,
        "split_seed": SEED,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "sklearn_version": (
            sklearn.__version__
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "internal_test_metrics": (
            internal_metrics
        ),
        "external_evaluated": False,
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
    raise SystemExit(main())