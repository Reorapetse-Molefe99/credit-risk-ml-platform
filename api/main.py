"""Credit-risk scoring API.

    uv run uvicorn api.main:app --reload --port 8000       # docs at http://127.0.0.1:8000/docs

Loads the model once at start-up (MODEL_URI, else artifacts/model, else the @champion alias) and
returns PD, risk grade, decision and – for single applications – reason codes.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    LoanApplication,
    ModelInfo,
    Prediction,
    PredictionResponse,
    ReasonCode,
)
from credit_risk import __version__, load_config, project_path
from credit_risk.predict import explain, load_model, model_metadata, resolve_model_uri, score_frame

log = logging.getLogger("credit_risk.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.cfg = cfg
    app.state.model, app.state.error, app.state.meta, app.state.version = None, None, {}, None
    try:
        app.state.model_uri = resolve_model_uri(cfg=cfg)
        app.state.model = load_model(app.state.model_uri, cfg)
        # metrics.json describes the exported champion – only use it when that is what we serve.
        exported = str(project_path(cfg["artifacts"]["model_dir"]))
        meta = model_metadata(cfg) if app.state.model_uri == exported else {}
        app.state.meta = meta
        app.state.version = str(meta.get("version") or app.state.model.metadata.run_id or "unknown")
        log.info("Loaded model %s (version %s)", app.state.model_uri, app.state.version)
    except Exception as exc:  # noqa: BLE001 – keep serving /health so orchestration can see the failure
        app.state.error = f"{type(exc).__name__}: {exc}"
        log.exception("Model failed to load")
    yield


app = FastAPI(title="Credit Risk Scoring API", version=__version__, lifespan=lifespan,
              description="Probability-of-default scoring for loan applications.")


def _model(request: Request):
    if request.app.state.model is None:
        raise HTTPException(503, f"Model not loaded: {request.app.state.error}")
    return request.app.state.model


def _frame(apps: list[LoanApplication]) -> pd.DataFrame:
    return pd.DataFrame([a.model_dump() for a in apps])


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    ok = request.app.state.model is not None
    return HealthResponse(status="ok" if ok else "degraded", model_loaded=ok,
                          model_version=getattr(request.app.state, "version", None) if ok else None,
                          detail=request.app.state.error)


@app.get("/model/info", response_model=ModelInfo)
def model_info(request: Request):
    model, st = _model(request), request.app.state
    return ModelInfo(
        model_name=st.cfg["mlflow"]["registered_model_name"],
        model_version=st.version,
        algorithm=st.meta.get("algorithm"),
        model_uri=st.model_uri,
        n_input_features=len(model.metadata.get_input_schema().input_names()),
        test_metrics=st.meta.get("metrics", {}),
        risk_grades=st.cfg["risk_grades"],
        decision_thresholds=st.cfg["decision"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication, request: Request,
            explain_top: int = Query(5, ge=0, le=20, description="Number of reason codes (0 = none)")):
    model, st = _model(request), request.app.state
    df = _frame([application])
    row = score_frame(model, df, st.cfg).iloc[0]
    reasons = None
    if explain_top:
        contrib = explain(model, df, top=explain_top)
        if contrib is not None:
            reasons = [ReasonCode(feature=r.feature, contribution=float(r.contribution), direction=r.direction)
                       for r in contrib.itertuples()]
    return PredictionResponse(application_id=application.SK_ID_CURR, model_version=st.version,
                              probability_of_default=float(row.probability_of_default),
                              risk_grade=row.risk_grade, decision=row.decision, reason_codes=reasons)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    model, st = _model(request), request.app.state
    scored = score_frame(model, _frame(batch.applications), st.cfg)
    preds = [Prediction(application_id=a.SK_ID_CURR, probability_of_default=float(r.probability_of_default),
                        risk_grade=r.risk_grade, decision=r.decision)
             for a, r in zip(batch.applications, scored.itertuples(), strict=True)]
    return BatchResponse(model_version=st.version, n_applications=len(preds), predictions=preds)
