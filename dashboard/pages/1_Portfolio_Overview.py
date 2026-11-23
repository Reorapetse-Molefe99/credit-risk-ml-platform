import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app import (
    CORAL,
    DECISION_COLORS,
    GRADE_COLORS,
    INK,
    cfg,
    controls,
    fmt_amount,
    get_portfolio,
    sidebar_info,
)
from plotly.subplots import make_subplots

st.set_page_config(page_title="Portfolio Overview", page_icon="📊", layout="wide")
sidebar_info()
st.title("Portfolio Overview")

df, source = get_portfolio()
target = cfg["data"]["target"]
has_target = target in df
lgd = controls().slider("Loss given default (LGD)", 0.0, 1.0, float(cfg["strategy"]["lgd"]), 0.05)
df = df.assign(expected_loss=df["probability_of_default"] * lgd * df["AMT_CREDIT"])
st.caption(f"{len(df):,} applications · {source}. Amounts are in the lender's local currency, which the data does not disclose.")

k = st.columns(5)
k[0].metric("Applications", f"{len(df):,}")
k[1].metric("Mean PD", f"{df['probability_of_default'].mean():.2%}")
k[2].metric("Observed default rate", f"{df[target].mean():.2%}" if has_target else "n/a")
k[3].metric("Exposure (sum of loan amounts)", fmt_amount(df["AMT_CREDIT"].sum()))
k[4].metric("Expected loss", fmt_amount(df["expected_loss"].sum()),
            f"{df['expected_loss'].sum() / df['AMT_CREDIT'].sum():.2%} of exposure", delta_color="off")

left, right = st.columns(2)
with left:
    st.subheader("Risk grades")
    agg = {"applications": ("probability_of_default", "size"), "mean_pd": ("probability_of_default", "mean"),
           "exposure": ("AMT_CREDIT", "sum"), "expected_loss": ("expected_loss", "sum")}
    if has_target:
        agg["default_rate"] = (target, "mean")
    g = df.groupby("risk_grade").agg(**agg).reindex(list(GRADE_COLORS)).dropna(how="all").reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=g["risk_grade"], y=g["applications"], name="Applications",
                marker_color=[GRADE_COLORS[x] for x in g["risk_grade"]])
    fig.add_scatter(x=g["risk_grade"], y=g["mean_pd"], name="Mean PD", mode="lines+markers", secondary_y=True,
                    line_color=INK)
    if has_target:
        fig.add_scatter(x=g["risk_grade"], y=g["default_rate"], name="Observed default rate",
                        mode="lines+markers", line_dash="dash", secondary_y=True, line_color=CORAL)
    fig.update_yaxes(title_text="Rate", tickformat=".0%", secondary_y=True)
    fig.update_layout(height=380, margin=dict(t=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    bounds = ", ".join(f"{g} < {b:.0%}" for g, b in cfg["risk_grades"].items())
    st.caption(f"Grade PD upper bounds: {bounds}")

with right:
    st.subheader("PD distribution")
    plot_df = df.assign(outcome=df[target].map({0: "non-default", 1: "default"})) if has_target else df
    fig = px.histogram(plot_df, x="probability_of_default", color="outcome" if has_target else None,
                       nbins=80, histnorm="probability density", barmode="overlay", opacity=0.6,
                       range_x=[0, plot_df["probability_of_default"].quantile(0.995)])
    fig.update_layout(height=380, margin=dict(t=10), xaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

st.subheader("Segment analysis")
seg_df = df.copy()
if "DAYS_BIRTH" in seg_df:
    seg_df["AGE_BAND"] = pd.cut(-seg_df["DAYS_BIRTH"] / 365.25, [0, 25, 35, 45, 55, 65, 120],
                                labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)
options = [c for c in ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
                       "NAME_HOUSING_TYPE", "AGE_BAND", "OCCUPATION_TYPE", "REGION_RATING_CLIENT"] if c in seg_df]
seg = st.selectbox("Segment by", options)
agg = {"applications": ("probability_of_default", "size"), "mean_pd": ("probability_of_default", "mean"),
       "expected_loss": ("expected_loss", "sum")}
if has_target:
    agg["default_rate"] = (target, "mean")
s = seg_df.assign(**{seg: seg_df[seg].fillna("Missing").astype(str)}).groupby(seg).agg(**agg)
s = s[s["applications"] >= 30].sort_values("mean_pd", ascending=False).reset_index()
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_bar(x=s[seg], y=s["mean_pd"], name="Mean PD")
    if has_target:
        fig.add_bar(x=s[seg], y=s["default_rate"], name="Observed default rate")
    fig.update_layout(barmode="group", height=380, yaxis_tickformat=".0%", margin=dict(t=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.dataframe(s.style.format({"mean_pd": "{:.2%}", "default_rate": "{:.2%}", "expected_loss": "{:,.0f}"}),
                 hide_index=True, width="stretch", height=380)

st.subheader("Decision mix")
d = df["decision"].value_counts().reindex(list(DECISION_COLORS)).fillna(0).reset_index()
d.columns = ["decision", "applications"]
fig = px.bar(d, x="applications", y="decision", orientation="h", color="decision",
             color_discrete_map=DECISION_COLORS, text="applications")
fig.update_layout(height=220, showlegend=False, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
st.caption(f"Approve if PD < {cfg['decision']['approve_below']:.0%}, decline if PD ≥ "
           f"{cfg['decision']['decline_above']:.0%}, otherwise refer to an underwriter.")
