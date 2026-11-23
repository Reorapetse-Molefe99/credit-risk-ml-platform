import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app import cfg, get_metadata, sidebar_info
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
fig.add_bar(x=dec["decile"], y=dec["default_rate"], name="Default rate")
fig.add_scatter(x=dec["decile"], y=dec["cum_capture_rate"], name="Cumulative share of defaults captured",
                mode="lines+markers", secondary_y=True)
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=380, xaxis=dict(dtick=1, title="Decile"), margin=dict(t=10),
                  legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, width="stretch")
st.dataframe(dec.style.format({c: "{:.2%}" for c in ["min_pd", "max_pd", "mean_pd", "default_rate", "cum_capture_rate"]}
                              | {"lift": "{:.2f}"}), hide_index=True, width="stretch")
top = dec.iloc[:2]
st.caption(f"The riskiest 20% of applicants contain {top['defaults'].sum() / dec['defaults'].sum():.0%} "
           f"of all defaults (lift {top['lift'].mean():.1f}x).")

st.subheader("Validation vs test")
val = meta.get("validation_metrics", {})
rows = [{"metric": k, "validation": val.get(f"val_{k}"), "test": m[k]}
        for k in ["roc_auc", "gini", "ks", "pr_auc", "brier", "log_loss"] if f"val_{k}" in val]
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
