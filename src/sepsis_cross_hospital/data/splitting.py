from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


SPLIT_VERSION = "split-v1"
DEFAULT_SEED = 1729

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


@dataclass(frozen=True)
class SplitCounts:
    train: int
    validation: int
    internal_test: int


def stable_patient_key(
    patient_id: str,
    cohort: str,
    seed: int,
) -> str:
    """
    Stable deterministic ordering key.

    Does not depend on Python's hash randomization,
    pandas version, NumPy RNG, or scikit-learn.
    """
    value = f"{SPLIT_VERSION}|{seed}|{cohort}|{patient_id}"

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def allocate_counts(n: int) -> SplitCounts:
    """
    Allocate approximately 70/15/15 while ensuring
    every patient belongs to exactly one partition.
    """
    n_train = int(n * TRAIN_FRACTION)
    n_validation = int(n * VALIDATION_FRACTION)

    n_internal_test = (
        n - n_train - n_validation
    )

    return SplitCounts(
        train=n_train,
        validation=n_validation,
        internal_test=n_internal_test,
    )


def assign_stratified_splits(
    patients: pd.DataFrame,
    cohort: str,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Deterministically split patients separately inside
    each sepsis class.

    Required columns:
        patient_id
        septic
    """
    required = {
        "patient_id",
        "septic",
    }

    missing = required - set(patients.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame = patients[
        ["patient_id", "septic"]
    ].copy()

    if frame["patient_id"].isna().any():
        raise ValueError(
            "patient_id contains missing values."
        )

    frame["patient_id"] = (
        frame["patient_id"].astype(str)
    )

    if frame["patient_id"].duplicated().any():
        duplicates = frame.loc[
            frame["patient_id"].duplicated(),
            "patient_id",
        ].tolist()

        raise ValueError(
            f"Duplicate patient IDs: {duplicates[:10]}"
        )

    if not frame["septic"].isin([0, 1]).all():
        raise ValueError(
            "Column 'septic' must contain only 0/1."
        )

    output_frames: list[pd.DataFrame] = []

    for septic_value in (0, 1):
        group = frame.loc[
            frame["septic"] == septic_value
        ].copy()

        group["_stable_key"] = group[
            "patient_id"
        ].map(
            lambda patient_id: stable_patient_key(
                patient_id=str(patient_id),
                cohort=cohort,
                seed=seed,
            )
        )

        group = group.sort_values(
            ["_stable_key", "patient_id"]
        ).reset_index(drop=True)

        counts = allocate_counts(
            len(group)
        )

        split_values = (
            ["train"] * counts.train
            + ["validation"]
            * counts.validation
            + ["internal_test"]
            * counts.internal_test
        )

        group["split"] = split_values

        output_frames.append(
            group.drop(
                columns=["_stable_key"]
            )
        )

    result = pd.concat(
        output_frames,
        ignore_index=True,
    )

    result["cohort"] = cohort

    result = result[
        [
            "cohort",
            "patient_id",
            "septic",
            "split",
        ]
    ].sort_values(
        "patient_id"
    ).reset_index(drop=True)

    validate_split_manifest(result)

    return result


def validate_split_manifest(
    manifest: pd.DataFrame,
) -> None:
    allowed_splits = {
        "train",
        "validation",
        "internal_test",
    }

    if manifest.empty:
        raise ValueError(
            "Split manifest is empty."
        )

    if manifest["patient_id"].duplicated().any():
        raise ValueError(
            "A patient appears more than once."
        )

    observed_splits = set(
        manifest["split"].unique()
    )

    if not observed_splits.issubset(
        allowed_splits
    ):
        raise ValueError(
            f"Unexpected split values: "
            f"{observed_splits - allowed_splits}"
        )

    if observed_splits != allowed_splits:
        raise ValueError(
            "All three split partitions must exist."
        )

    if not manifest["septic"].isin([0, 1]).all():
        raise ValueError(
            "Invalid septic status."
        )


def canonical_manifest_bytes(
    manifest: pd.DataFrame,
) -> bytes:
    """
    Stable bytes for reproducible SHA256 calculation.
    """
    ordered = manifest.sort_values(
        [
            "cohort",
            "patient_id",
        ]
    ).reset_index(drop=True)

    csv_text = ordered.to_csv(
        index=False,
        lineterminator="\n",
    )

    return csv_text.encode("utf-8")


def manifest_sha256(
    manifest: pd.DataFrame,
) -> str:
    return hashlib.sha256(
        canonical_manifest_bytes(manifest)
    ).hexdigest()