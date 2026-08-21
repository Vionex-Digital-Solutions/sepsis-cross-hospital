from __future__ import annotations

import numpy as np
import pandas as pd

from sepsis_cross_hospital.data.preprocessing import (
    DYNAMIC_COLUMNS,
    build_feature_representation,
    fit_source_statistics,
)


def make_patient(
    hours: int,
) -> pd.DataFrame:
    data: dict[str, object] = {}

    for column in DYNAMIC_COLUMNS:
        data[column] = [
            1.0
        ] * hours

    data["HR"] = [
        80.0
    ] * hours

    data["Age"] = [
        60.0
    ] * hours

    data["Gender"] = [
        1.0
    ] * hours

    data["Unit1"] = [
        1.0
    ] * hours

    data["Unit2"] = [
        0.0
    ] * hours

    data["HospAdmTime"] = [
        -4.0
    ] * hours

    data["ICULOS"] = list(
        range(1, hours + 1)
    )

    data["SepsisLabel"] = [
        0
    ] * hours

    return pd.DataFrame(data)


def test_future_rows_cannot_change_past_features():
    source = make_patient(
        hours=8
    )

    stats = fit_source_statistics(
        [source]
    )

    patient = make_patient(
        hours=10
    )

    patient.loc[
        1,
        "HR",
    ] = np.nan

    patient.loc[
        2,
        "HR",
    ] = np.nan

    # Future values are intentionally extreme.
    patient.loc[
        6:,
        "HR",
    ] = 9999.0

    cutoff = 5

    truncated = patient.iloc[
        :cutoff
    ].copy()

    for representation in (
        "M0",
        "V0",
        "V1",
        "V2",
    ):
        features_truncated = (
            build_feature_representation(
                truncated,
                stats,
                representation,
            )
            .reset_index(drop=True)
        )

        features_full = (
            build_feature_representation(
                patient,
                stats,
                representation,
            )
            .iloc[:cutoff]
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(
            features_truncated,
            features_full,
            check_exact=True,
        )


def test_target_values_cannot_change_features():
    source = make_patient(
        hours=5
    )

    stats = fit_source_statistics(
        [source]
    )

    first = make_patient(
        hours=5
    )

    second = first.copy()

    first["SepsisLabel"] = [
        0,
        0,
        0,
        0,
        0,
    ]

    second["SepsisLabel"] = [
        0,
        1,
        1,
        1,
        1,
    ]

    for representation in (
        "M0",
        "V0",
        "V1",
        "V2",
    ):
        a = build_feature_representation(
            first,
            stats,
            representation,
        )

        b = build_feature_representation(
            second,
            stats,
            representation,
        )

        pd.testing.assert_frame_equal(
            a,
            b,
            check_exact=True,
        )


def test_target_patient_does_not_change_source_statistics():
    source = make_patient(
        hours=3
    )

    source["HR"] = [
        10.0,
        20.0,
        30.0,
    ]

    stats_before = (
        fit_source_statistics(
            [source]
        )
    )

    target = make_patient(
        hours=3
    )

    target["HR"] = [
        10000.0,
        20000.0,
        30000.0,
    ]

    # Transforming target data must not refit
    # or mutate source statistics.
    build_feature_representation(
        target,
        stats_before,
        "V2",
    )

    assert (
        stats_before
        .dynamic_medians[
            "HR"
        ]
        == 20.0
    )