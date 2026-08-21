from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from sepsis_cross_hospital.evaluation.metrics import (
    compute_challenge_utility,
    compute_patient_utility,
    make_optimal_predictions,
)


@dataclass(frozen=True)
class CalibrationResult:
    intercept: float
    slope: float
    success: bool
    message: str


@dataclass(frozen=True)
class UtilityComponents:
    observed: np.ndarray
    optimal: np.ndarray
    inactive: np.ndarray


def calibration_intercept_slope(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> CalibrationResult:
    """
    Estimate calibration intercept and slope as
    evaluation statistics only.

    This does NOT recalibrate or modify predictions.
    """
    y = np.asarray(
        labels,
        dtype=np.float64,
    )

    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError(
            "labels and probabilities must be 1D."
        )

    if len(y) != len(p):
        raise ValueError(
            "labels/probability length mismatch."
        )

    if len(y) == 0:
        raise ValueError(
            "Empty calibration input."
        )

    if not np.isfinite(p).all():
        raise ValueError(
            "Non-finite probabilities detected."
        )

    if np.any(
        (p < 0.0)
        | (p > 1.0)
    ):
        raise ValueError(
            "Probabilities must lie in [0, 1]."
        )

    if not np.isin(
        y,
        [0.0, 1.0],
    ).all():
        raise ValueError(
            "Labels must be binary."
        )

    epsilon = 1e-6

    clipped = np.clip(
        p,
        epsilon,
        1.0 - epsilon,
    )

    logit_probability = (
        np.log(clipped)
        - np.log1p(-clipped)
    )

    def objective(
        parameters: np.ndarray,
    ) -> float:
        intercept = parameters[0]
        slope = parameters[1]

        linear = (
            intercept
            + slope
            * logit_probability
        )

        loss = (
            np.logaddexp(
                0.0,
                linear,
            )
            - y * linear
        )

        return float(
            np.mean(loss)
        )

    def gradient(
        parameters: np.ndarray,
    ) -> np.ndarray:
        intercept = parameters[0]
        slope = parameters[1]

        linear = (
            intercept
            + slope
            * logit_probability
        )

        fitted = expit(
            linear
        )

        residual = (
            fitted - y
        )

        return np.array(
            [
                np.mean(
                    residual
                ),
                np.mean(
                    residual
                    * logit_probability
                ),
            ],
            dtype=np.float64,
        )

    result = minimize(
        objective,
        x0=np.array(
            [0.0, 1.0],
            dtype=np.float64,
        ),
        jac=gradient,
        method="L-BFGS-B",
    )

    if not np.isfinite(
        result.x
    ).all():
        return CalibrationResult(
            intercept=float("nan"),
            slope=float("nan"),
            success=False,
            message=(
                "Non-finite calibration "
                "solution."
            ),
        )

    return CalibrationResult(
        intercept=float(
            result.x[0]
        ),
        slope=float(
            result.x[1]
        ),
        success=bool(
            result.success
        ),
        message=str(
            result.message
        ),
    )


def utility_components_by_patient(
    labels_by_patient: Sequence[
        np.ndarray
    ],
    predictions_by_patient: Sequence[
        np.ndarray
    ],
) -> UtilityComponents:
    """
    Compute raw Challenge-utility contributions
    separately for each patient.

    Important:
    Challenge utility is normalized only at the
    cohort level. Individual non-septic patients
    can have optimal == inactive utility, so
    compute_challenge_utility() must NOT be called
    patient-by-patient.

    The raw observed, optimal, and inactive
    contributions are additive across patients and
    can therefore be safely recombined during the
    patient-level bootstrap.
    """
    if (
        len(labels_by_patient)
        != len(predictions_by_patient)
    ):
        raise ValueError(
            "Patient counts differ."
        )

    if len(labels_by_patient) == 0:
        raise ValueError(
            "At least one patient is required."
        )

    n_patients = len(
        labels_by_patient
    )

    observed = np.empty(
        n_patients,
        dtype=np.float64,
    )

    optimal = np.empty(
        n_patients,
        dtype=np.float64,
    )

    inactive = np.empty(
        n_patients,
        dtype=np.float64,
    )

    for index, (
        labels,
        predictions,
    ) in enumerate(
        zip(
            labels_by_patient,
            predictions_by_patient,
            strict=True,
        )
    ):
        y_true = np.asarray(
            labels,
            dtype=np.int8,
        )

        y_pred = np.asarray(
            predictions,
            dtype=np.int8,
        )

        if y_true.ndim != 1:
            raise ValueError(
                "Patient labels must be 1D."
            )

        if y_pred.ndim != 1:
            raise ValueError(
                "Patient predictions must be 1D."
            )

        if len(y_true) != len(y_pred):
            raise ValueError(
                "Patient label/prediction "
                "length mismatch."
            )

        if len(y_true) == 0:
            raise ValueError(
                "Patient sequence must not "
                "be empty."
            )

        if not np.isin(
            y_true,
            [0, 1],
        ).all():
            raise ValueError(
                "Patient labels must be binary."
            )

        if not np.isin(
            y_pred,
            [0, 1],
        ).all():
            raise ValueError(
                "Patient predictions must "
                "be binary."
            )

        optimal_predictions = (
            make_optimal_predictions(
                y_true
            )
        )

        inactive_predictions = np.zeros(
            len(y_true),
            dtype=np.int8,
        )

        observed[index] = (
            compute_patient_utility(
                y_true,
                y_pred,
            )
        )

        optimal[index] = (
            compute_patient_utility(
                y_true,
                optimal_predictions,
            )
        )

        inactive[index] = (
            compute_patient_utility(
                y_true,
                inactive_predictions,
            )
        )

    return UtilityComponents(
        observed=observed,
        optimal=optimal,
        inactive=inactive,
    )

def normalized_from_components(
    observed: float,
    optimal: float,
    inactive: float,
) -> float:
    denominator = (
        optimal - inactive
    )

    if np.isclose(
        denominator,
        0.0,
    ):
        return 0.0

    return float(
        (
            observed
            - inactive
        )
        / denominator
    )


def paired_patient_bootstrap(
    labels_by_patient: Sequence[
        np.ndarray
    ],
    predictions_by_model: Mapping[
        str,
        Sequence[np.ndarray],
    ],
    *,
    n_bootstrap: int = 1000,
    seed: int = 1729,
) -> dict[str, np.ndarray]:
    """
    Patient-level paired bootstrap.

    The same sampled patient indices are used for
    every model within each replicate.
    """
    if n_bootstrap <= 0:
        raise ValueError(
            "n_bootstrap must be positive."
        )

    n_patients = len(
        labels_by_patient
    )

    if n_patients == 0:
        raise ValueError(
            "No patients supplied."
        )

    if not predictions_by_model:
        raise ValueError(
            "No models supplied."
        )

    model_names = list(
        predictions_by_model
    )

    observed_rows = []

    reference_optimal = None
    reference_inactive = None

    for model_name in model_names:
        predictions = (
            predictions_by_model[
                model_name
            ]
        )

        if len(
            predictions
        ) != n_patients:
            raise ValueError(
                f"{model_name}: patient "
                "count mismatch."
            )

        components = (
            utility_components_by_patient(
                labels_by_patient,
                predictions,
            )
        )

        if (
            reference_optimal is None
            or reference_inactive is None
        ):
            reference_optimal = (
                components.optimal
            )

            reference_inactive = (
                components.inactive
            )

        else:
            assert reference_optimal is not None
            assert reference_inactive is not None

            np.testing.assert_allclose(
                components.optimal,
                reference_optimal,
                rtol=0.0,
                atol=1e-12,
            )

            np.testing.assert_allclose(
                components.inactive,
                reference_inactive,
                rtol=0.0,
                atol=1e-12,
            )

        observed_rows.append(
            components.observed
        )

    assert (
        reference_optimal
        is not None
    )

    assert (
        reference_inactive
        is not None
    )

    observed_matrix = np.stack(
        observed_rows,
        axis=0,
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap = {
        model_name: np.empty(
            n_bootstrap,
            dtype=np.float64,
        )
        for model_name
        in model_names
    }

    for replicate in range(
        n_bootstrap
    ):
        sampled_indices = rng.integers(
            0,
            n_patients,
            size=n_patients,
        )

        counts = np.bincount(
            sampled_indices,
            minlength=n_patients,
        ).astype(
            np.float64,
            copy=False,
        )

        optimal = float(
            np.dot(
                reference_optimal,
                counts,
            )
        )

        inactive = float(
            np.dot(
                reference_inactive,
                counts,
            )
        )

        observed = (
            observed_matrix
            @ counts
        )

        denominator = (
            optimal - inactive
        )

        if np.isclose(
            denominator,
            0.0,
        ):
            normalized = np.zeros(
                len(model_names),
                dtype=np.float64,
            )
        else:
            normalized = (
                observed - inactive
            ) / denominator

        for model_index, model_name in enumerate(
            model_names
        ):
            bootstrap[
                model_name
            ][
                replicate
            ] = normalized[
                model_index
            ]

    return bootstrap


def percentile_ci(
    values: np.ndarray,
) -> tuple[
    float,
    float,
]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    low, high = np.percentile(
        values,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(low),
        float(high),
    )