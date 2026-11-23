"""Credit-risk dashboard.

    uv run streamlit run dashboard/app.py        # http://localhost:8501

This file is both the home page and the shared helper module for pages/ (they `import app`);
page content only renders when it runs as the main script.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import credit_risk  # noqa: F401
except ModuleNotFoundError:  # Streamlit Community Cloud installs requirements.txt but not the project itself
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk import load_config  # noqa: E402
from credit_risk.ingestion import load_reference, synthetic_applications  # noqa: E402
from credit_risk.predict import (  # noqa: E402
    contributions,
    load_model,
    model_metadata,
    resolve_model_uri,
    score_frame,
    scored_portfolio,
)

cfg = load_config()
GRADE_COLORS = {"A": "#1a9850", "B": "#91cf60", "C": "#fee08b", "D": "#fc8d59", "E": "#d73027"}
DECISION_COLORS = {"approve": "#1a9850", "refer": "#fdae61", "decline": "#d73027"}
STATUS_ICON = {"stable": "🟢", "warning": "🟠", "alert": "🔴", "n/a": "⚪"}


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model(cfg=cfg)


@st.cache_data(show_spinner="Scoring portfolio…")
def get_portfolio() -> tuple[pd.DataFrame, str]:
    df = scored_portfolio(get_model(), cfg)
    return df, df.attrs.get("source", "unknown")


@st.cache_data(show_spinner="Loading monitoring reference…")
def get_reference() -> tuple[pd.DataFrame, str]:
    """The development population for drift monitoring: data/reference/ if present, else a synthetic one
    scored by the champion (demo deployments have no loan-level data)."""
    try:
        return load_reference(cfg), "development sample"
    except FileNotFoundError:
        df = synthetic_applications(5000, seed=11)
        return df.assign(probability_of_default=score_frame(get_model(), df, cfg)["probability_of_default"]
                         .to_numpy()), "synthetic"


@st.cache_data(show_spinner="Computing SHAP values…")
def get_global_shap(n: int = 3000) -> pd.DataFrame | None:
    """Mean |SHAP| per feature over a sample of the portfolio (None if the model type has no SHAP support)."""
    df, _ = get_portfolio()
    sample = df.sample(min(n, len(df)), random_state=0)
    c = contributions(get_model(), sample.drop(columns=["probability_of_default", "risk_grade", "decision"]))
    if c is None:
        return None
    return (pd.DataFrame({"feature": c.columns, "mean_abs_shap": c.abs().mean().to_numpy(),
                          "mean_shap": c.mean().to_numpy()})
            .sort_values("mean_abs_shap", ascending=False, ignore_index=True))


def get_metadata() -> dict:
    return model_metadata(cfg)


def setting(name: str) -> str | None:
    """An environment variable, else a Streamlit secret (Streamlit Community Cloud), else None."""
    if value := os.environ.get(name):
        return value
    try:
        return st.secrets.get(name)
    except Exception:  # noqa: BLE001 – no secrets file configured
        return None


def api_url() -> str | None:
    """API base URL if API_URL is set and the service is healthy, else None (score in-process)."""
    url = (setting("API_URL") or "").rstrip("/")
    if not url:
        return None
    try:
        urllib.request.urlopen(f"{url}/health", timeout=5)
        return url
    except Exception:  # noqa: BLE001
        return None


def sidebar_info() -> None:
    meta = get_metadata()
    st.sidebar.markdown("**Model**")
    if meta:
        st.sidebar.caption(f"{meta['model_name']} v{meta['version']} · {meta.get('algorithm')}\n\n"
                           f"Test AUC {meta['metrics']['roc_auc']:.3f} · Gini {meta['metrics']['gini']:.3f}")
    else:
        st.sidebar.caption(resolve_model_uri(cfg=cfg))
    st.sidebar.caption(f"Scoring via: {'API' if api_url() else 'in-process model'}")


def fmt_amount(x: float) -> str:
    for unit, div in (("bn", 1e9), ("m", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"{x / div:,.1f}{unit}"
    return f"{x:,.0f}"


if __name__ == "__main__":
    st.set_page_config(page_title="Credit Risk Platform", page_icon="📊", layout="wide")
    sidebar_info()
    st.title("Credit Risk Platform")
    st.markdown(
        "Probability-of-default (PD) scoring for consumer loan applications, built on the Home Credit "
        "dataset with an MLflow-managed model lifecycle."
    )
    meta = get_metadata()
    df, source = get_portfolio()
    c = st.columns(4)
    c[0].metric("Champion model", f"v{meta.get('version', '?')}", meta.get("algorithm"))
    if meta:
        c[1].metric("Test AUC", f"{meta['metrics']['roc_auc']:.3f}")
        c[2].metric("Gini", f"{meta['metrics']['gini']:.3f}")
        c[3].metric("KS", f"{meta['metrics']['ks']:.3f}")
    if source == "synthetic":
        st.info("**Demo mode:** the Home Credit data can't be redistributed, so this portfolio is **synthetic**. "
                "The model and its test metrics are real: they come from the exported champion, evaluated on "
                f"{meta.get('n_test', 0):,} held-out loans. Run locally with HomeLoanData.csv in data/raw/ "
                "to see the real portfolio.")
    st.caption(f"Portfolio: {len(df):,} applications ({source}).")

    st.subheader("Pages")
    st.markdown(
        "- **Portfolio Overview**: risk-grade mix, expected loss and default rates by segment\n"
        "- **Customer Scoring**: score an existing or new applicant, with reason codes\n"
        "- **Model Performance**: discrimination, calibration, lift, confusion matrix and SHAP explainability\n"
        "- **Strategy Simulator**: approval cut-off trade-offs, and risk-based collections prioritisation\n"
        "- **Model Monitoring**: population stability (PSI) of scores and features, with simulated drift scenarios"
    )
    st.caption("Research and portfolio demonstration, not an authorised lending or collections decision system.")
