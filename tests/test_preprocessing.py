from __future__ import annotations

import numpy as np
import pandas as pd

from sepsis_cross_hospital.data.preprocessing import (
    CATEGORICAL_STATIC_COLUMNS,
    CONTINUOUS_STATIC_COLUMNS,
    DYNAMIC_COLUMNS,
    build_feature_representation,
    fit_source_statistics,
    transform_patient_causally,
)


def make_patient(
    *,
    hours: int = 5,
    hr: list[float] | None = None,
    etco2: list[float] | None = None,
) -> pd.DataFrame:
    data: dict[str, object] = {}

    for column in DYNAMIC_COLUMNS:
        data[column] = [
            np.nan
        ] * hours

    data["HR"] = (
        hr
        if hr is not None
        else [80.0] * hours
    )

    data["EtCO2"] = (
        etco2
        if etco2 is not None
        else [np.nan] * hours
    )

    # Give every other dynamic variable
    # a valid source observation so the
    # test fixture is simple.
    for column in DYNAMIC_COLUMNS:
        if column in {
            "HR",
            "EtCO2",
        }:
            continue

        data[column] = [
            1.0
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
        -5.0
    ] * hours

    data["ICULOS"] = list(
        range(1, hours + 1)
    )

    data["SepsisLabel"] = [
        0
    ] * hours

    return pd.DataFrame(data)


def test_source_median_used_before_first_measurement():
    source = make_patient(
        hours=3,
        hr=[50.0, 70.0, 90.0],
    )

    stats = fit_source_statistics(
        [source]
    )

    patient = make_patient(
        hours=3,
        hr=[np.nan, 100.0, np.nan],
    )

    transformed = (
        transform_patient_causally(
            patient,
            stats,
        )
    )

    assert transformed[
        "HR"
    ].tolist() == [
        70.0,
        100.0,
        100.0,
    ]


def test_forward_fill_is_causal():
    source = make_patient(
        hours=3,
        hr=[70.0, 80.0, 90.0],
    )

    stats = fit_source_statistics(
        [source]
    )

    patient = make_patient(
        hours=4,
        hr=[
            100.0,
            np.nan,
            np.nan,
            200.0,
        ],
    )

    transformed = (
        transform_patient_causally(
            patient,
            stats,
        )
    )

    assert transformed[
        "HR"
    ].tolist() == [
        100.0,
        100.0,
        100.0,
        200.0,
    ]


def test_source_unavailable_variable_is_disabled():
    source = make_patient(
        hours=4,
        etco2=[
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
    )

    stats = fit_source_statistics(
        [source]
    )

    assert (
        stats
        .source_available_dynamic[
            "EtCO2"
        ]
        is False
    )

    target = make_patient(
        hours=4,
        etco2=[
            30.0,
            35.0,
            40.0,
            45.0,
        ],
    )

    transformed = (
        transform_patient_causally(
            target,
            stats,
        )
    )

    # Value channel disabled.
    assert transformed[
        "EtCO2"
    ].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # Target-only observation information
    # must also remain unavailable.
    assert transformed[
        "obs_EtCO2"
    ].tolist() == [
        0,
        0,
        0,
        0,
    ]

    assert transformed[
        "count6_EtCO2"
    ].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    assert transformed[
        "tsl_EtCO2"
    ].tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

def test_current_observation_mask():
    source = make_patient()

    stats = fit_source_statistics(
        [source]
    )

    patient = make_patient(
        hours=4,
        hr=[
            np.nan,
            90.0,
            np.nan,
            95.0,
        ],
    )

    transformed = (
        transform_patient_causally(
            patient,
            stats,
        )
    )

    assert transformed[
        "obs_HR"
    ].tolist() == [
        0,
        1,
        0,
        1,
    ]


def test_time_since_last_measurement():
    source = make_patient()

    stats = fit_source_statistics(
        [source]
    )

    patient = make_patient(
        hours=5,
        hr=[
            np.nan,
            90.0,
            np.nan,
            np.nan,
            95.0,
        ],
    )

    transformed = (
        transform_patient_causally(
            patient,
            stats,
        )
    )

    assert transformed[
        "tsl_HR"
    ].tolist() == [
        1.0,
        0.0,
        1.0,
        2.0,
        0.0,
    ]


def test_six_hour_trailing_count():
    source = make_patient()

    stats = fit_source_statistics(
        [source]
    )

    patient = make_patient(
        hours=7,
        hr=[
            80.0,
            np.nan,
            82.0,
            np.nan,
            84.0,
            np.nan,
            86.0,
        ],
    )

    transformed = (
        transform_patient_causally(
            patient,
            stats,
        )
    )

    assert transformed[
        "count6_HR"
    ].tolist() == [
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
    ]


def test_target_never_enters_features():
    source = make_patient()

    stats = fit_source_statistics(
        [source]
    )

    patient = make_patient()

    patient["SepsisLabel"] = [
        0,
        0,
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
        features = (
            build_feature_representation(
                patient,
                stats,
                representation,
            )
        )

        assert (
            "SepsisLabel"
            not in features.columns
        )


def test_all_representations_have_no_nan():
    source = make_patient()

    stats = fit_source_statistics(
        [source]
    )

    patient = make_patient(
        hours=4,
        hr=[
            np.nan,
            90.0,
            np.nan,
            95.0,
        ],
    )

    for representation in (
        "M0",
        "V0",
        "V1",
        "V2",
    ):
        features = (
            build_feature_representation(
                patient,
                stats,
                representation,
            )
        )

        assert (
            features.isna().sum().sum()
            == 0
        )