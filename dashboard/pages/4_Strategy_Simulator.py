import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app import cfg, fmt_amount, get_portfolio, sidebar_info

st.set_page_config(page_title="Strategy Simulator", page_icon="🎯", layout="wide")
sidebar_info()
st.title("Strategy Simulator")

df, source = get_portfolio()
target = cfg["data"]["target"]
has_target = target in df
s = cfg["strategy"]
pd_ = df["probability_of_default"].to_numpy()
ead = df["AMT_CREDIT"].to_numpy()
y = df[target].to_numpy() if has_target else None

tab_approval, tab_collections = st.tabs(["Approval strategy", "Collections prioritisation"])

with tab_approval:
    st.markdown("Choose an approval cut-off on predicted PD and see the trade-off between growth (approval rate), "
                "risk (bad rate, expected loss) and profitability. One-period loan economics: approved loans earn "
                "`margin × exposure` if they perform and lose `LGD × exposure` if they default.")
    a = st.columns(4)
    cutoff = a[0].slider("Approve if PD below", 0.01, 0.50, float(cfg["decision"]["approve_below"]), 0.005,
                         format="%.3f")
    lgd = a[1].slider("Loss given default (LGD)", 0.05, 1.0, float(s["lgd"]), 0.05)
    margin = a[2].slider("Net interest margin (annual)", 0.0, 0.40, float(s["annual_margin"]), 0.01)
    term = a[3].slider("Term (years)", 0.5, 5.0, float(s["term_years"]), 0.5)

    def evaluate(c: float) -> dict:
        a = pd_ < c
        out = {"cutoff": c, "approval_rate": a.mean(), "approved": int(a.sum()), "exposure": ead[a].sum(),
               "expected_loss": (pd_[a] * lgd * ead[a]).sum(),
               "expected_profit": ((1 - pd_[a]) * margin * term * ead[a] - pd_[a] * lgd * ead[a]).sum()}
        if y is not None:
            out["bad_rate"] = y[a].mean() if a.any() else np.nan
            out["defaults_avoided"] = 1 - y[a].sum() / max(y.sum(), 1)
            out["realised_profit"] = ((1 - y[a]) * margin * term * ead[a] - y[a] * lgd * ead[a]).sum()
        return out


    cur, everyone = evaluate(cutoff), evaluate(1.01)
    k = st.columns(5)
    k[0].metric("Approval rate", f"{cur['approval_rate']:.1%}", f"{cur['approved']:,} loans", delta_color="off")
    k[1].metric("Bad rate (approved)", f"{cur['bad_rate']:.2%}" if has_target else "n/a",
                f"{cur['bad_rate'] - everyone['bad_rate']:+.2%} vs approve-all" if has_target else None,
                delta_color="inverse")
    k[2].metric("Defaults avoided", f"{cur['defaults_avoided']:.1%}" if has_target else "n/a")
    k[3].metric("Expected loss", fmt_amount(cur["expected_loss"]),
                f"{cur['expected_loss'] / max(cur['exposure'], 1):.2%} of approved exposure", delta_color="off")
    profit_key = "realised_profit" if has_target else "expected_profit"
    k[4].metric("Realised profit" if has_target else "Expected profit", fmt_amount(cur[profit_key]),
                f"{cur[profit_key] - everyone[profit_key]:+,.0f} vs approve-all")

    grid = pd.DataFrame([evaluate(c) for c in np.round(np.arange(0.01, 0.505, 0.005), 3)])
    best = grid.loc[grid[profit_key].idxmax()]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Profit by cut-off")
        fig = go.Figure()
        fig.add_scatter(x=grid["cutoff"], y=grid["expected_profit"], name="Expected (model PD)")
        if has_target:
            fig.add_scatter(x=grid["cutoff"], y=grid["realised_profit"], name="Realised (actual outcomes)")
        fig.add_vline(x=cutoff, line_dash="dash", annotation_text="current")
        fig.add_vline(x=best["cutoff"], line_dash="dot", line_color="green", annotation_text="max profit",
                      annotation_position="bottom right")
        fig.update_layout(height=400, xaxis_title="PD cut-off", xaxis_tickformat=".0%", margin=dict(t=10),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Strategy curve")
        if has_target:
            fig = go.Figure()
            fig.add_scatter(x=grid["approval_rate"], y=grid["bad_rate"], mode="lines", name="cut-offs",
                            customdata=grid["cutoff"], hovertemplate="approval %{x:.1%}<br>bad rate %{y:.2%}"
                                                                     "<br>cut-off %{customdata:.3f}")
            fig.add_scatter(x=[cur["approval_rate"]], y=[cur["bad_rate"]], mode="markers", marker_size=14,
                            name="current")
            fig.update_layout(height=400, xaxis_title="Approval rate", yaxis_title="Bad rate among approved",
                              xaxis_tickformat=".0%", yaxis_tickformat=".1%", margin=dict(t=10),
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Needs observed outcomes (TARGET) in the portfolio.")

    st.success(f"Profit-maximising cut-off under these assumptions: **PD < {best['cutoff']:.1%}**: approval rate "
               f"{best['approval_rate']:.1%}" + (f", bad rate {best['bad_rate']:.2%}" if has_target else "") + ".")
    st.caption(f"Portfolio: {len(df):,} applications ({source}). Break-even PD for a single loan = "
               f"margin×term / (margin×term + LGD) = {margin * term / (margin * term + lgd):.1%}.")

with tab_collections:
    st.markdown("A collections or early-intervention team can only contact part of the book each month. Ranking "
                "loans by predicted PD and working from the top finds far more of the future defaults than "
                "picking at random. The curve below is the **cumulative gains** of the champion on this "
                "portfolio's actual outcomes.")
    capacity = st.slider("Share of highest-risk loans the team can prioritise", 0.01, 0.50, 0.20, 0.01,
                         format="%.2f")
    order = np.argsort(-pd_, kind="stable")
    n = len(order)
    k_sel = max(1, int(round(capacity * n)))
    sel = order[:k_sel]
    outcome = y if has_target else pd_          # without outcomes, fall back to expected defaults (sum of PD)
    capture = outcome[sel].sum() / max(outcome.sum(), 1e-9)
    kc = st.columns(5)
    kc[0].metric("Loans prioritised", f"{k_sel:,}", f"of {n:,}", delta_color="off")
    kc[1].metric("Default capture" if has_target else "Expected default capture", f"{capture:.1%}")
    kc[2].metric("Random selection capture", f"{k_sel / n:.1%}")
    kc[3].metric("Prioritisation lift", f"{capture / (k_sel / n):.2f}x")
    kc[4].metric("Defaulted exposure covered" if has_target else "Expected exposure at risk covered",
                 f"{(outcome[sel] * ead[sel]).sum() / max((outcome * ead).sum(), 1e-9):.1%}",
                 help="Share of the loan amount (AMT_CREDIT) of defaulting loans that falls in the prioritised group")

    share = np.arange(1, n + 1) / n
    gains = np.cumsum(outcome[order]) / max(outcome.sum(), 1e-9)
    idx = np.unique(np.linspace(0, n - 1, 300).astype(int))
    best_share = outcome.sum() / n if has_target else None
    fig = go.Figure()
    fig.add_scatter(x=np.r_[0, share[idx]], y=np.r_[0, gains[idx]], name="Risk-ranked (model)", line_width=3)
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Random selection", line=dict(dash="dash", color="grey"))
    if best_share:
        fig.add_scatter(x=[0, best_share, 1], y=[0, 1, 1], name="Perfect foresight",
                        line=dict(dash="dot", color="#1a9850"))
    fig.add_scatter(x=[k_sel / n], y=[capture], mode="markers", marker_size=14, name="Current capacity")
    fig.update_layout(height=430, xaxis_title="Share of loans prioritised (highest PD first)",
                      yaxis_title="Share of defaults captured", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      margin=dict(t=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Portfolio: {n:,} loans ({source}). **Default capture is not defaults prevented or money "
               "recovered.** A PD model ranks who is likely to default; it cannot say whether contacting a "
               "customer changes their behaviour. That needs treatment-effect (uplift) data from a controlled "
               "contact strategy.")
