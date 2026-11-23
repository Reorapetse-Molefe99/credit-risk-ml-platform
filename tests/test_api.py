import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow.models import infer_signature

from credit_risk import load_config
from credit_risk.ingestion import split_frame, standardise_types, synthetic_applications
from credit_risk.train import build_pipeline, trusted_types_for

CFG = load_config()
APPLICATION = {
    "SK_ID_CURR": 123, "NAME_CONTRACT_TYPE": "Cash loans", "AMT_INCOME_TOTAL": 180000.0,
    "AMT_CREDIT": 500000.0, "AMT_ANNUITY": 27000.0, "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2000,
    "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.5, "CODE_GENDER": "F",
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    splits = split_frame(standardise_types(synthetic_applications(1500, seed=3), CFG), CFG)
    X, y = splits.xy("train", CFG)
    pipe, _ = build_pipeline("logreg", X, cfg=CFG)
    pipe.fit(X, y)
    path = tmp_path_factory.mktemp("model") / "model"
    mlflow.sklearn.save_model(pipe, path, signature=infer_signature(X.head(50), pipe.predict_proba(X.head(50))),
                              pyfunc_predict_fn="predict_proba", skops_trusted_types=trusted_types_for(pipe))
    return str(path)


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_URI", model_dir)
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["n_input_features"] > 20 and set(info["risk_grades"]) == {"A", "B", "C", "D", "E"}


def test_predict_single_with_reason_codes(client):
    r = client.post("/predict", json=APPLICATION)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["risk_grade"] in "ABCDE" and body["decision"] in {"approve", "refer", "decline"}
    assert body["application_id"] == 123
    assert len(body["reason_codes"]) == 5
    assert "CODE_GENDER" not in {rc["feature"] for rc in body["reason_codes"]}


def test_predict_minimal_payload(client):
    r = client.post("/predict?explain_top=0", json={"AMT_INCOME_TOTAL": 100000, "AMT_CREDIT": 200000})
    assert r.status_code == 200 and r.json()["reason_codes"] is None


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"applications": [APPLICATION, {**APPLICATION, "SK_ID_CURR": 9}]})
    assert r.status_code == 200
    assert [p["application_id"] for p in r.json()["predictions"]] == [123, 9]


@pytest.mark.parametrize("payload", [
    {"AMT_INCOME_TOTAL": -1, "AMT_CREDIT": 1000},
    {"AMT_INCOME_TOTAL": 1000},
    {**APPLICATION, "EXT_SOURCE_2": 1.7},
    {**APPLICATION, "NAME_CONTRACT_TYPE": "Mortgage"},
])
def test_invalid_input_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_degraded_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_URI", str(tmp_path / "does-not-exist"))
    from api.main import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=APPLICATION).status_code == 503
