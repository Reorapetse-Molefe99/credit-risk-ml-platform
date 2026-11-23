import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk import load_config
from credit_risk.calibration import PrefitCalibratedClassifier, base_estimator
from credit_risk.evaluate import (
    capture_at,
    classification_metrics,
    decile_table,
    ks_statistic,
    performance_summary,
)
from credit_risk.ingestion import split_frame, standardise_types, synthetic_applications
from credit_risk.monitoring import (
    SCENARIOS,
    categorical_psi,
    drift_report,
    feature_stability,
    psi,
    simulate_drift,
)
from credit_risk.predict import assign_grade, contributions, decide, explain, score_frame
from credit_risk.train import ALGORITHMS, build_pipeline, train_and_log, trusted_types_for

CFG = load_config()
FAST = {"logreg": {}, "rf": {"n_estimators": 30, "max_depth": 6}, "hgb": {"max_iter": 50}, "xgboost": {"n_estimators": 50}, "lightgbm": {"n_estimators": 50}}


@pytest.fixture(scope="module")
def splits():
    return split_frame(standardise_types(synthetic_applications(3000, seed=2), CFG), CFG)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_pipelines_learn_signal(splits, algo):
    X, y = splits.xy("train", CFG)
    Xt, yt = splits.xy("test", CFG)
    pipe, params = build_pipeline(algo, X, FAST[algo], CFG)
    pipe.fit(X, y)
    p = pipe.predict_proba(Xt)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(yt, p) > 0.6
    assert trusted_types_for(pipe)  # model serialises with the allow-listed types only


def test_metrics_on_perfect_scores():
    y = np.array([0] * 90 + [1] * 10)
    p = np.r_[np.linspace(0, 0.4, 90), np.linspace(0.6, 1, 10)]
    m = classification_metrics(y, p, "test_")
    assert m["test_roc_auc"] == 1.0 and m["test_gini"] == 1.0
    assert ks_statistic(y, p) == 1.0
    t = decile_table(y, p)
    assert t.loc[0, "default_rate"] == 1.0 and t["cum_capture_rate"].iloc[-1] == 1.0
    assert set(performance_summary(y, p)) == {"metrics", "roc_curve", "calibration", "deciles"}


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.beta(2, 20, 20_000)
    assert psi(a, rng.beta(2, 20, 20_000)) < 0.01
    assert psi(a, rng.beta(5, 10, 20_000)) > 0.25
    s = pd.Series(["a"] * 50 + ["b"] * 50)
    assert categorical_psi(s, s) == pytest.approx(0)
    assert categorical_psi(s, pd.Series(["a"] * 95 + ["c"] * 5)) > 0.25


def test_grades_and_decisions():
    grades = assign_grade([0.01, 0.03, 0.07, 0.15, 0.6], CFG["risk_grades"])
    assert grades.tolist() == ["A", "B", "C", "D", "E"]
    assert decide([0.05, 0.15, 0.5], CFG["decision"]).tolist() == ["approve", "refer", "decline"]


def test_train_and_log_end_to_end(splits, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("ci", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("ci")
    res = train_and_log("lightgbm", splits, FAST["lightgbm"], cfg=CFG)
    assert res["test_roc_auc"] > 0.6 and res["val_test_psi"] >= 0

    model = mlflow.pyfunc.load_model(res["model_uri"])
    schema = model.metadata.get_input_schema().input_names()
    assert "CODE_GENDER" not in schema and "SK_ID_CURR" not in schema

    X_test, _ = splits.xy("test", CFG)
    scored = score_frame(model, X_test, CFG)
    assert list(scored.columns) == ["probability_of_default", "risk_grade", "decision"]
    reasons = explain(model, X_test.head(1), top=5)
    assert len(reasons) == 5 and {"feature", "contribution", "direction"} <= set(reasons.columns)

    # partial input: missing columns are imputed, not rejected
    assert len(score_frame(model, pd.DataFrame({"AMT_INCOME_TOTAL": [1e5], "AMT_CREDIT": [3e5]}), CFG)) == 1

    reference = splits.train.assign(probability_of_default=score_frame(model, splits.train, CFG)
                                    ["probability_of_default"])
    rep = drift_report(reference, splits.test, scored["probability_of_default"].to_numpy(), CFG)
    assert rep["score_status"] == "stable"
    assert not feature_stability(reference, splits.test, cfg=CFG).empty


def test_capture_at_top_fraction():
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(1, 0, 100)                      # the 10 defaulters have the highest scores
    assert capture_at(y, p, 0.10) == (1.0, 10.0)
    recall, lift = capture_at(y, np.zeros(100), 0.5)  # ties: a stable sort keeps the first half
    assert recall == 1.0 and lift == 2.0
    m = classification_metrics(y, p)
    assert m["recall_top10"] == 1.0 and m["lift_top10"] == 10.0


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_fixes_a_biased_model(splits, method):
    X, y = splits.xy("train", CFG)
    Xv, yv = splits.xy("val", CFG)
    Xt, yt = splits.xy("test", CFG)
    # A deliberately over-confident model: heavy positive class weight inflates every PD.
    pipe, _ = build_pipeline("logreg", X, {"class_weight": {0: 1, 1: 8}}, CFG)
    pipe.fit(X, y)
    p_raw = pipe.predict_proba(Xt)[:, 1]
    base = pipe.steps[-1][1]
    cal = PrefitCalibratedClassifier(base, method).fit(pipe[:-1].transform(Xv), yv)
    pipe.steps[-1] = ("model", cal)
    p_cal = pipe.predict_proba(Xt)[:, 1]
    assert ((p_cal >= 0) & (p_cal <= 1)).all()
    assert abs(p_cal.mean() - yt.mean()) < abs(p_raw.mean() - yt.mean())
    assert base_estimator(cal) is base and base_estimator(base) is base
    if method == "sigmoid":                         # monotonic: ranking (AUC) is preserved
        assert roc_auc_score(yt, p_cal) == pytest.approx(roc_auc_score(yt, p_raw))
    assert trusted_types_for(pipe)


def test_calibration_rejects_unknown_method(splits):
    X, y = splits.xy("train", CFG)
    pipe, _ = build_pipeline("logreg", X, {}, CFG)
    pipe.fit(X, y)
    with pytest.raises(ValueError, match="Unknown calibration method"):
        PrefitCalibratedClassifier(pipe.steps[-1][1], "beta").fit(pipe[:-1].transform(X), y)


def test_simulate_drift_shifts_inputs_and_drops_labels(splits):
    df = splits.test
    shifted = simulate_drift(df, **SCENARIOS["recession"], cfg=CFG)
    assert CFG["data"]["target"] not in shifted
    assert shifted["AMT_INCOME_TOTAL"].sum() == pytest.approx(df["AMT_INCOME_TOTAL"].sum() * 0.8)
    assert shifted["EXT_SOURCE_2"].isna().mean() > df["EXT_SOURCE_2"].isna().mean()
    assert shifted["EXT_SOURCE_2"].dropna().between(0, 1).all()
    assert simulate_drift(df, cfg=CFG).equals(df.drop(columns=[CFG["data"]["target"]]))  # no-op scenario


def test_missingness_jump_is_flagged_even_when_psi_is_stable(splits):
    ref = splits.train
    cur = simulate_drift(splits.test, thin_file=0.3, cfg=CFG)   # values unchanged, 30% more missing
    fs = feature_stability(ref, cur, features=["EXT_SOURCE_2", "AMT_CREDIT"], cfg=CFG).set_index("feature")
    assert fs.loc["EXT_SOURCE_2", "psi"] < CFG["monitoring"]["psi_warning"]
    assert fs.loc["EXT_SOURCE_2", "status"] == "alert"
    assert fs.loc["AMT_CREDIT", "status"] == "stable"


def test_calibrated_model_end_to_end(splits, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("ci-cal", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("ci-cal")
    res = train_and_log("lightgbm", splits, FAST["lightgbm"], cfg=CFG, calibrate="sigmoid")
    assert {"test_brier", "test_uncal_brier", "val_roc_auc", "test_recall_top10"} <= set(res)
    model = mlflow.pyfunc.load_model(res["model_uri"])
    X_test, _ = splits.xy("test", CFG)
    scored = score_frame(model, X_test, CFG)
    assert scored["probability_of_default"].between(0, 1).all()
    c = contributions(model, X_test.head(20))   # SHAP still available through the calibration layer
    assert c.shape[0] == 20 and "EXT_SOURCE_MEAN" in c.columns
    assert len(explain(model, X_test.head(1), top=3)) == 3
    with pytest.raises(ValueError, match="Unknown calibration method"):
        train_and_log("lightgbm", splits, FAST["lightgbm"], cfg=CFG, calibrate="platt", log_model=False)
