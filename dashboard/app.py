"""Credit-risk dashboard.

    uv run streamlit run dashboard/app.py        # http://localhost:8501

This file is both the home page and the shared helper module for pages/ (they `import app`);
page content only renders when it runs as the main script.
"""
from __future__ import annotations

import os
import urllib.request

import pandas as pd
import streamlit as st

from credit_risk import load_config
from credit_risk.predict import load_model, model_metadata, resolve_model_uri, scored_portfolio

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


def get_metadata() -> dict:
    return model_metadata(cfg)


def api_url() -> str | None:
    """API base URL if API_URL is set and the service is healthy, else None (score in-process)."""
    url = os.environ.get("API_URL", "").rstrip("/")
    if not url:
        return None
    try:
        urllib.request.urlopen(f"{url}/health", timeout=2)
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
        st.warning("Raw data not found – showing a **synthetic demo portfolio**. "
                   "Put HomeLoanData.csv in data/raw/ for real results.")
    st.caption(f"Portfolio: {len(df):,} applications ({source}).")

    st.subheader("Pages")
    st.markdown(
        "- **Portfolio Overview**: risk-grade mix, expected loss and default rates by segment\n"
        "- **Customer Scoring**: score an existing or new applicant, with reason codes\n"
        "- **Model Performance**: discrimination, calibration and lift of the champion model\n"
        "- **Strategy Simulator**: choose a PD cut-off and see approval rate, bad rate and profit trade-offs\n"
        "- **Model Monitoring**: population stability (PSI) of scores and features, with stress scenarios"
    )
