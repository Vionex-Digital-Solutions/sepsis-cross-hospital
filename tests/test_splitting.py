from __future__ import annotations

import pandas as pd

from sepsis_cross_hospital.data.splitting import (
    allocate_counts,
    assign_stratified_splits,
    manifest_sha256,
)


def make_patients(
    n_nonseptic: int = 80,
    n_septic: int = 20,
) -> pd.DataFrame:
    records = []

    for index in range(n_nonseptic):
        records.append(
            {
                "patient_id": f"n{index:04d}",
                "septic": 0,
            }
        )

    for index in range(n_septic):
        records.append(
            {
                "patient_id": f"s{index:04d}",
                "septic": 1,
            }
        )

    return pd.DataFrame(records)


def test_allocate_counts_preserves_total():
    for n in range(1, 500):
        counts = allocate_counts(n)

        assert (
            counts.train
            + counts.validation
            + counts.internal_test
            == n
        )


def test_each_patient_appears_once():
    patients = make_patients()

    manifest = assign_stratified_splits(
        patients,
        cohort="A",
        seed=1729,
    )

    assert len(manifest) == len(patients)

    assert (
        manifest["patient_id"]
        .duplicated()
        .sum()
        == 0
    )


def test_split_is_deterministic():
    patients = make_patients()

    first = assign_stratified_splits(
        patients,
        cohort="A",
        seed=1729,
    )

    second = assign_stratified_splits(
        patients.sample(
            frac=1.0,
            random_state=999,
        ),
        cohort="A",
        seed=1729,
    )

    assert (
        manifest_sha256(first)
        == manifest_sha256(second)
    )


def test_different_seed_changes_assignment():
    patients = make_patients()

    first = assign_stratified_splits(
        patients,
        cohort="A",
        seed=1729,
    )

    second = assign_stratified_splits(
        patients,
        cohort="A",
        seed=2718,
    )

    merged = first.merge(
        second,
        on=[
            "cohort",
            "patient_id",
            "septic",
        ],
        suffixes=(
            "_first",
            "_second",
        ),
    )

    assert (
        merged["split_first"]
        != merged["split_second"]
    ).any()


def test_septic_patients_present_in_all_splits():
    patients = make_patients(
        n_nonseptic=80,
        n_septic=20,
    )

    manifest = assign_stratified_splits(
        patients,
        cohort="A",
        seed=1729,
    )

    septic = manifest.loc[
        manifest["septic"] == 1
    ]

    assert set(
        septic["split"]
    ) == {
        "train",
        "validation",
        "internal_test",
    }


def test_no_patient_overlap_between_partitions():
    patients = make_patients()

    manifest = assign_stratified_splits(
        patients,
        cohort="A",
        seed=1729,
    )

    train = set(
        manifest.loc[
            manifest["split"] == "train",
            "patient_id",
        ]
    )

    validation = set(
        manifest.loc[
            manifest["split"] == "validation",
            "patient_id",
        ]
    )

    internal_test = set(
        manifest.loc[
            manifest["split"]
            == "internal_test",
            "patient_id",
        ]
    )

    assert train.isdisjoint(validation)
    assert train.isdisjoint(internal_test)
    assert validation.isdisjoint(
        internal_test
    )


def test_stratification_is_preserved():
    patients = make_patients(
        n_nonseptic=800,
        n_septic=200,
    )

    manifest = assign_stratified_splits(
        patients,
        cohort="A",
        seed=1729,
    )

    overall_prevalence = (
        manifest["septic"].mean()
    )

    for split in (
        "train",
        "validation",
        "internal_test",
    ):
        split_prevalence = (
            manifest.loc[
                manifest["split"] == split,
                "septic",
            ].mean()
        )

        assert abs(
            split_prevalence
            - overall_prevalence
        ) < 0.01