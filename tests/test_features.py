import numpy as np
import pandas as pd

from credit_risk.features import (
    DAYS_EMPLOYED_ANOMALY,
    ENGINEERED_FEATURES,
    add_features,
    clean_feature_name,
    split_column_types,
)
from credit_risk.ingestion import synthetic_applications


def test_add_features_handles_employment_anomaly_and_zero_division():
    df = pd.DataFrame({
        "DAYS_EMPLOYED": [-1000.0, DAYS_EMPLOYED_ANOMALY], "DAYS_BIRTH": [-10000.0, -20000.0],
        "AMT_CREDIT": [100_000.0, 50_000.0], "AMT_INCOME_TOTAL": [50_000.0, 0.0],
        "AMT_ANNUITY": [10_000.0, 5_000.0], "EXT_SOURCE_1": [0.2, np.nan], "EXT_SOURCE_2": [0.4, 0.5],
    })
    out = add_features(df)
    assert out["DAYS_EMPLOYED_ANOMALY"].tolist() == [0.0, 1.0]
    assert np.isnan(out.loc[1, "DAYS_EMPLOYED"])
    assert out.loc[0, "CREDIT_TO_INCOME"] == 2.0
    assert np.isnan(out.loc[1, "CREDIT_TO_INCOME"])          # income 0 -> NaN, never inf
    assert out.loc[1, "EXT_SOURCE_NULLS"] == 1.0
    assert "AGE_YEARS" not in df                             # input not mutated


def test_add_features_tolerates_missing_columns():
    out = add_features(pd.DataFrame({"AMT_CREDIT": [1.0]}))
    assert list(out.columns) == ["AMT_CREDIT"]


def test_all_documented_features_are_created():
    out = add_features(synthetic_applications(50))
    missing = set(ENGINEERED_FEATURES) - set(out.columns)
    assert not missing, missing
    assert not np.isinf(out.select_dtypes("number")).any().any()


def test_add_features_is_row_wise():
    df = synthetic_applications(20)
    whole = add_features(df)
    one = add_features(df.iloc[[7]])
    pd.testing.assert_frame_equal(whole.iloc[[7]], one)


def test_split_column_types():
    num, cat = split_column_types(synthetic_applications(50).drop(columns=["TARGET"]))
    assert "NAME_CONTRACT_TYPE" in cat and "CREDIT_TO_INCOME" in num
    assert not set(num) & set(cat)


def test_clean_feature_name():
    assert clean_feature_name("WALLSMATERIAL_MODE", "Stone, brick") == "WALLSMATERIAL_MODE__Stone_brick"
