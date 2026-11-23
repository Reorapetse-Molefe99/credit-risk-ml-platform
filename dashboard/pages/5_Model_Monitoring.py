import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app import STATUS_ICON, cfg, get_model, get_portfolio, sidebar_info

from credit_risk.ingestion import load_reference, standardise_types
from credit_risk.monitoring import drift_report
from credit_risk.predict import score_frame

st.set_page_config(page_title="Model Monitoring", page_icon="🛰️", layout="wide")
sidebar_info()
st.title("Model Monitoring")
st.markdown("Compares a **current** population against the **reference** population the champion was developed "
            "on (data/reference/). PSI < 0.10 stable · 0.10–0.25 monitor · > 0.25 investigate.")

try:
    reference = load_reference(cfg)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

portfolio, source = get_portfolio()
mode = st.radio("Current population", ["Hold-out portfolio", "Stress scenario", "Upload CSV"], horizontal=True)


@st.cache_data(show_spinner="Scoring scenario…")
def stressed(income_shock: float, ext_shift: float, thin_file: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = portfolio.drop(columns=["probability_of_default", "risk_grade", "decision"]).copy()
    df["AMT_INCOME_TOTAL"] *= 1 + income_shock
    for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if c in df:
            df[c] = (df[c] + ext_shift).clip(0, 1)
            df.loc[rng.random(len(df)) < thin_file, c] = np.nan
    return pd.concat([df, score_frame(get_model(), df, cfg)], axis=1)


if mode == "Hold-out portfolio":
    current = portfolio
    st.caption(f"{len(current):,} applications ({source})")
elif mode == "Stress scenario":
    c = st.columns(3)
    shock = c[0].slider("Income shock", -0.5, 0.5, -0.2, 0.05, format="%.2f")
    shift = c[1].slider("Shift in external bureau scores", -0.3, 0.3, -0.1, 0.01)
    thin = c[2].slider("Extra share with missing bureau scores", 0.0, 0.8, 0.1, 0.05)
    current = stressed(shock, shift, thin)
else:
    up = st.file_uploader("CSV with the raw application columns", type="csv")
    if up is None:
        st.stop()
    raw = standardise_types(pd.read_csv(up), cfg)
    current = pd.concat([raw, score_frame(get_model(), raw, cfg)], axis=1)

report = drift_report(reference, current, current["probability_of_default"].to_numpy(), cfg)

k = st.columns(5)
k[0].metric("Score PSI", f"{report['score_psi']:.3f}", f"{STATUS_ICON[report['score_status']]} {report['score_status']}",
            delta_color="off")
k[1].metric("Mean PD", f"{report['mean_pd_current']:.2%}",
            f"{report['mean_pd_current'] - report['mean_pd_reference']:+.2%} vs reference", delta_color="inverse")
k[2].metric("Features in alert", len(report["features_alert"]))
k[3].metric("Features in warning", len(report["features_warning"]))
if "default_rate_current" in report:
    k[4].metric("Observed default rate", f"{report['default_rate_current']:.2%}",
                f"reference {report['default_rate_reference']:.2%}", delta_color="off")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Score distribution")
    both = pd.concat([pd.DataFrame({"PD": reference["probability_of_default"], "population": "reference"}),
                      pd.DataFrame({"PD": current["probability_of_default"], "population": "current"})])
    fig = px.histogram(both, x="PD", color="population", nbins=80, histnorm="probability density",
                       barmode="overlay", opacity=0.55, range_x=[0, both["PD"].quantile(0.995)])
    fig.update_layout(height=380, xaxis_tickformat=".0%", margin=dict(t=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("Grade migration")
    grades = list(cfg["risk_grades"])
    from credit_risk.predict import assign_grade
    ref_g = assign_grade(reference["probability_of_default"], cfg["risk_grades"]).value_counts(normalize=True)
    cur_g = current["risk_grade"].value_counts(normalize=True)
    fig = go.Figure()
    fig.add_bar(x=grades, y=ref_g.reindex(grades).fillna(0), name="reference")
    fig.add_bar(x=grades, y=cur_g.reindex(grades).fillna(0), name="current")
    fig.update_layout(height=380, barmode="group", yaxis_tickformat=".0%", margin=dict(t=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

st.subheader("Feature stability")
fs = pd.DataFrame(report["feature_stability"])
fig = px.bar(fs.iloc[::-1], x="psi", y="feature", orientation="h", color="status",
             color_discrete_map={"stable": "#1a9850", "warning": "#fdae61", "alert": "#d73027", "n/a": "grey"})
for x in (cfg["monitoring"]["psi_warning"], cfg["monitoring"]["psi_alert"]):
    fig.add_vline(x=x, line_dash="dash", line_color="grey")
fig.update_layout(height=max(300, 22 * len(fs)), margin=dict(t=10), yaxis_title=None)
st.plotly_chart(fig, width="stretch")
fs["status"] = fs["status"].map(lambda s: f"{STATUS_ICON[s]} {s}")
st.dataframe(fs.style.format({"psi": "{:.4f}", "missing_ref": "{:.1%}", "missing_cur": "{:.1%}"}),
             hide_index=True, width="stretch")
st.caption(f"Reference: {report['n_reference']:,} development applications · current: {report['n_current']:,}. "
           "Run `python -m credit_risk.monitoring --current <csv> --log` to record a monitoring run in MLflow.")
