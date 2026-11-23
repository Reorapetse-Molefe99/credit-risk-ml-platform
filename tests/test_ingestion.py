import numpy as np
import pandas as pd
import pytest

from credit_risk import load_config
from credit_risk.ingestion import (
    feature_columns,
    load_raw,
    sample_splits,
    split_frame,
    standardise_types,
    synthetic_applications,
)
from credit_risk.validation import validate_applications

CFG = load_config()


@pytest.fixture(scope="module")
def apps():
    return standardise_types(synthetic_applications(3000, seed=1), CFG)


def test_synthetic_data_is_valid(apps):
    rep = validate_applications(apps, target="TARGET", id_col="SK_ID_CURR")
    assert rep.ok, rep.errors
    assert 0.02 < apps["TARGET"].mean() < 0.4


def test_split_is_stratified_disjoint_and_complete(apps):
    s = split_frame(apps, CFG)
    ids = [set(df["SK_ID_CURR"]) for df in (s.train, s.val, s.test)]
    assert sum(map(len, ids)) == len(apps)
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    rates = [df["TARGET"].mean() for df in (s.train, s.val, s.test)]
    assert max(rates) - min(rates) < 0.02
    assert len(s.test) == pytest.approx(0.2 * len(apps), abs=2)


def test_feature_columns_exclude_id_target_and_protected(apps):
    cols = feature_columns(apps, CFG)
    assert {"SK_ID_CURR", "TARGET", "CODE_GENDER"}.isdisjoint(cols)
    assert "AMT_CREDIT" in cols


def test_standardise_types_casts_integer_features_only():
    df = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1], "CNT_CHILDREN": [0, 2]})
    out = standardise_types(df, CFG)
    assert out["CNT_CHILDREN"].dtype == "float64"
    assert out["SK_ID_CURR"].dtype.kind == "i" and out["TARGET"].dtype.kind == "i"


def test_sample_splits_keeps_ratio(apps):
    s = sample_splits(split_frame(apps, CFG), 500, CFG)
    assert 450 <= len(s.train) <= 550


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.drop(columns=["AMT_CREDIT"]), "missing required columns"),
    (lambda d: d.assign(TARGET=2), "must be binary"),
    (lambda d: d.assign(SK_ID_CURR=1), "duplicate"),
    (lambda d: d.assign(EXT_SOURCE_2=1.5), "EXT_SOURCE_2"),
    (lambda d: d.assign(AMT_INCOME_TOTAL=-1.0), "AMT_INCOME_TOTAL"),
])
def test_validation_catches_bad_data(apps, mutate, message):
    rep = validate_applications(mutate(apps.head(50)), target="TARGET", id_col="SK_ID_CURR")
    assert not rep.ok
    assert any(message in e for e in rep.errors)
    with pytest.raises(ValueError):
        rep.raise_if_invalid()


def test_validation_warns_on_sparse_columns(apps):
    rep = validate_applications(apps.head(50).assign(EMPTY=np.nan))
    assert rep.ok and rep.warnings


def test_load_raw_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw"):
        load_raw(CFG, path=str(tmp_path / "nope.csv"))
