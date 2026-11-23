import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app import CORAL, TEAL, cfg, get_global_shap, get_metadata, get_portfolio, sidebar_info
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
sidebar_info()
st.title("Model Performance")

meta = get_metadata()
if not meta:
    st.error("artifacts/metrics.json not found. Promote a model with `python -m credit_risk.evaluate --promote`.")
    st.stop()

m = meta["metrics"]
st.caption(f"{meta['model_name']} v{meta['version']} · {meta.get('algorithm')} · trained on "
           f"{meta.get('n_train', 0):,} rows · evaluated on {meta['n_test']:,} held-out applications · "
           f"exported {meta['exported_at']}")
k = st.columns(6)
k[0].metric("ROC AUC", f"{m['roc_auc']:.3f}")
k[1].metric("Gini", f"{m['gini']:.3f}")
k[2].metric("KS", f"{m['ks']:.3f}")
k[3].metric("PR AUC", f"{m['pr_auc']:.3f}")
k[4].metric("Brier score", f"{m['brier']:.4f}")
k[5].metric("Mean PD vs observed", f"{m['mean_predicted_pd']:.2%}", f"observed {m['observed_default_rate']:.2%}",
            delta_color="off")
d1 = meta["deciles"][0]                            # decile 1 = riskiest 10% of the test set
k = st.columns(6)
k[0].metric("Recall @ top 10%", f"{m.get('recall_top10', d1['cum_capture_rate']):.1%}",
            help="Share of all test-set defaults found among the 10% of loans with the highest PD")
k[1].metric("Lift @ top 10%", f"{m.get('lift_top10', d1['lift']):.2f}x", help="Versus picking 10% at random")
k[2].metric("Log-loss", f"{m['log_loss']:.4f}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Discrimination: ROC curve")
    roc = meta["roc_curve"]
    fig = go.Figure()
    fig.add_scatter(x=roc["fpr"], y=roc["tpr"], name=f"AUC {m['roc_auc']:.3f}", fill="tozeroy")
    fig.add_scatter(x=[0, 1], y=[0, 1], name="random", line=dict(dash="dash", color="grey"))
    fig.update_layout(height=400, xaxis_title="False positive rate", yaxis_title="True positive rate",
                      margin=dict(t=10), legend=dict(x=0.6, y=0.1))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("Calibration")
    cal = meta["calibration"]
    lim = max(cal["mean_predicted"] + cal["observed"]) * 1.05
    fig = go.Figure()
    fig.add_scatter(x=cal["mean_predicted"], y=cal["observed"], mode="lines+markers", name="model")
    fig.add_scatter(x=[0, lim], y=[0, lim], name="perfect", line=dict(dash="dash", color="grey"))
    fig.update_layout(height=400, xaxis_title="Mean predicted PD", yaxis_title="Observed default rate",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin=dict(t=10), legend=dict(x=0.05, y=0.95))
    st.plotly_chart(fig, width="stretch")

st.subheader("Rank ordering: deciles (1 = highest risk)")
dec = pd.DataFrame(meta["deciles"])
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_bar(x=dec["decile"], y=dec["default_rate"], name="Default rate", marker_color=TEAL)
fig.add_scatter(x=dec["decile"], y=dec["cum_capture_rate"], name="Cumulative share of defaults captured",
                mode="lines+markers", secondary_y=True, line_color=CORAL)
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=380, xaxis=dict(dtick=1, title="Decile"), margin=dict(t=10),
                  legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, width="stretch")
st.dataframe(dec.style.format({c: "{:.2%}" for c in ["min_pd", "max_pd", "mean_pd", "default_rate", "cum_capture_rate"]}
                              | {"lift": "{:.2f}"}), hide_index=True, width="stretch")
top = dec.iloc[:2]
st.caption(f"The riskiest 20% of applicants contain {top['defaults'].sum() / dec['defaults'].sum():.0%} "
           f"of all defaults (lift {top['lift'].mean():.1f}x).")

st.subheader("Confusion matrix at a PD cut-off")
df, source = get_portfolio()
target = cfg["data"]["target"]
if target not in df:
    st.info("Needs observed outcomes in the portfolio.")
else:
    c1, c2 = st.columns([1, 2])
    cut = c1.slider("Flag as high risk if PD ≥", 0.02, 0.50, float(cfg["decision"]["approve_below"]), 0.01,
                    format="%.2f")
    y, flag = df[target].to_numpy(), df["probability_of_default"].to_numpy() >= cut
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    c1.metric("Recall (defaults flagged)", f"{tp / max(tp + fn, 1):.1%}")
    c1.metric("Precision (flagged who default)", f"{tp / max(tp + fp, 1):.1%}")
    c1.metric("Share flagged", f"{flag.mean():.1%}")
    cm = np.array([[tn, fp], [fn, tp]])
    fig = px.imshow(cm, text_auto=",", color_continuous_scale="Teal", aspect="auto",
                    x=["Predicted low risk", "Predicted high risk"], y=["Did not default", "Defaulted"])
    fig.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
    c2.plotly_chart(fig, width="stretch")
    c2.caption(f"{len(df):,} loans ({source}). Accuracy is not shown on purpose: with an "
               f"{y.mean():.1%} default rate, approving everyone is {1 - y.mean():.1%} 'accurate'.")

st.subheader("Explainability: global SHAP importance")
shap_df = get_global_shap()
if shap_df is None:
    st.info("SHAP values are available for LightGBM, XGBoost and logistic-regression champions.")
else:
    top = shap_df.head(15).iloc[::-1]
    fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h", color="mean_shap",
                 color_continuous_scale="RdYlGn_r", color_continuous_midpoint=0,
                 labels={"mean_abs_shap": "Mean |SHAP| (log-odds)", "mean_shap": "Mean SHAP"})
    fig.update_layout(height=480, margin=dict(t=10), yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Exact TreeSHAP contributions to the log-odds of default, averaged over a sample of the {source} "
               "portfolio. One-hot categories are summed back to their source feature. The Customer Scoring "
               "page shows the same values for a single applicant (reason codes).")

st.subheader("Validation vs test")
val = meta.get("validation_metrics", {})
rows = [{"metric": k, "validation": val.get(f"val_{k}"), "test": m[k]}
        for k in ["roc_auc", "gini", "ks", "pr_auc", "brier", "log_loss", "recall_top10"] if f"val_{k}" in val and k in m]
st.dataframe(pd.DataFrame(rows).style.format({"validation": "{:.4f}", "test": "{:.4f}"}), hide_index=True)

with st.expander("Model registry (live from MLflow)"):
    try:
        import urllib.request

        import mlflow
        from mlflow import MlflowClient

        urllib.request.urlopen(f"{cfg['mlflow']['tracking_uri']}/health", timeout=2)
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        client = MlflowClient()
        name = cfg["mlflow"]["registered_model_name"]
        aliases = client.get_registered_model(name).aliases
        by_version: dict[str, list[str]] = {}
        for a, v in aliases.items():
            by_version.setdefault(str(v), []).append(a)
        reg = pd.DataFrame([{"version": int(v.version), "aliases": ", ".join(by_version.get(v.version, [])),
                             "validation_status": v.tags.get("validation_status"),
                             "test_roc_auc": v.tags.get("test_roc_auc"), "run_id": v.run_id}
                            for v in client.search_model_versions(f"name='{name}'")]).sort_values("version")
        st.dataframe(reg, hide_index=True, width="stretch")
    except Exception:  # noqa: BLE001
        st.info(f"MLflow tracking server not reachable at {cfg['mlflow']['tracking_uri']}.")
