import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk import load_config
from credit_risk.evaluate import classification_metrics, decile_table, ks_statistic, performance_summary
from credit_risk.ingestion import split_frame, standardise_types, synthetic_applications
from credit_risk.monitoring import categorical_psi, drift_report, feature_stability, psi
from credit_risk.predict import assign_grade, decide, explain, score_frame
from credit_risk.train import ALGORITHMS, build_pipeline, train_and_log, trusted_types_for

CFG = load_config()
FAST = {"logreg": {}, "hgb": {"max_iter": 50}, "xgboost": {"n_estimators": 50}, "lightgbm": {"n_estimators": 50}}


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
