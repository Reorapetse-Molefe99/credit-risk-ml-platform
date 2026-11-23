"""Model loading and scoring: PD, risk grade and credit decision.

    uv run python -m credit_risk.predict --input new_applications.csv
    uv run python -m credit_risk.predict --input apps.csv --model-uri models:/credit-risk-pd/4

Model resolution order: explicit argument > $MODEL_URI > artifacts/model (exported champion)
> models:/<registered_model_name>@champion.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import load_config, project_path


def resolve_model_uri(uri: str | None = None, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    uri = uri or os.environ.get("MODEL_URI")
    if uri:
        local = project_path(uri)
        return str(local) if local.exists() else uri
    exported = project_path(cfg["artifacts"]["model_dir"])
    if (exported / "MLmodel").exists():
        return str(exported)
    return f"models:/{cfg['mlflow']['registered_model_name']}@champion"


def load_model(uri: str | None = None, cfg: dict | None = None):
    import mlflow

    cfg = cfg or load_config()
    resolved = resolve_model_uri(uri, cfg)
    if resolved.startswith(("models:/", "runs:/")):
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    return mlflow.pyfunc.load_model(resolved)


def model_metadata(cfg: dict | None = None) -> dict:
    """Champion metadata from artifacts/metrics.json (empty dict if not exported yet)."""
    cfg = cfg or load_config()
    path = project_path(cfg["artifacts"]["metrics_path"])
    return json.loads(path.read_text()) if path.exists() else {}


def positive_proba(pred) -> np.ndarray:
    """Normalise pyfunc output (predict_proba array or DataFrame) to P(default)."""
    arr = np.asarray(pred)
    return arr[:, 1] if arr.ndim == 2 else arr.ravel()


def prepare_input(model, df: pd.DataFrame) -> pd.DataFrame:
    """Align a frame of applications to the model's input schema: add missing columns as null,
    drop extras, and cast to the schema types (numbers -> float64, strings -> object/None)."""
    schema = model.metadata.get_input_schema()
    if schema is None:
        return df
    X = df.reindex(columns=schema.input_names())
    for spec in schema.inputs:
        col = X[spec.name]
        if str(spec.type) in ("DataType.string", "string"):
            X[spec.name] = col.astype(object).where(col.notna(), None)
        else:
            X[spec.name] = pd.to_numeric(col, errors="coerce").astype("float64")
    return X


def assign_grade(pd_values, grades: dict[str, float]) -> pd.Series:
    bounds = sorted(grades.items(), key=lambda kv: kv[1])
    edges = [-1e-9] + [b for _, b in bounds]
    return pd.cut(pd.Series(pd_values), bins=edges, labels=[g for g, _ in bounds]).astype(str)


def decide(pd_values, decision_cfg: dict) -> pd.Series:
    p = pd.Series(pd_values)
    return pd.Series(np.select([p < decision_cfg["approve_below"], p >= decision_cfg["decline_above"]],
                               ["approve", "decline"], default="refer"), index=p.index)


def score_frame(model, df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Return probability_of_default, risk_grade and decision for each application (index preserved)."""
    cfg = cfg or load_config()
    p = positive_proba(model.predict(prepare_input(model, df)))
    out = pd.DataFrame({"probability_of_default": p}, index=df.index)
    out["risk_grade"] = assign_grade(p, cfg["risk_grades"]).to_numpy()
    out["decision"] = decide(p, cfg["decision"]).to_numpy()
    return out


def _weights_for_mean(p: np.ndarray, target: float) -> np.ndarray:
    """Sampling weights proportional to exp(lam * p), with lam chosen so the weighted mean of p is `target`."""
    lo, hi = -500.0, 500.0
    for _ in range(60):
        lam = (lo + hi) / 2
        w = np.exp(lam * (p - p.mean()))
        lo, hi = (lam, hi) if (w * p).sum() / w.sum() < target else (lo, lam)
    return w / w.sum()


def simulated_portfolio(model, n: int = 5000, seed: int = 7, cfg: dict | None = None, signal: float = 0.75,
                        noise: float = 0.5, shift: float = -0.43) -> pd.DataFrame:
    """Synthetic applications for public demos: a realistic risk mix on which the model works well, but not
    perfectly, as on any new population.

    The real data can't be redistributed, but its PD deciles (aggregates in artifacts/metrics.json) can. A pool
    of synthetic applicants, a share of them with weaker bureau scores, is scored by the champion. Equal numbers
    are then drawn from each real PD decile, weighted to match the decile's mean PD.

    Outcomes come from a "true" risk the model only partly captures:
    logit(q) = shift + signal × logit(PD) + noise × N(0, 1), then y ~ Bernoulli(q). With the defaults, the model
    reaches an AUC of about 0.70 (real test set: 0.769) and slightly under-predicts (8.0% PD vs about 9.4%
    observed). signal=1, noise=0, shift=0 gives perfectly calibrated outcomes. Without exported deciles, it
    falls back to plain synthetic applications.
    """
    from .ingestion import synthetic_applications

    cfg = cfg or load_config()
    deciles = model_metadata(cfg).get("deciles")
    if not deciles:
        return synthetic_applications(n, seed=seed)
    rng = np.random.default_rng(seed)
    pool = synthetic_applications(n * 10, seed=seed).drop(columns=[cfg["data"]["target"]])
    risky = rng.random(len(pool)) < 0.5                       # weaker bureau scores give a realistic high-risk tail
    for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if c in pool:
            pool.loc[risky, c] = np.where(pool.loc[risky, c].isna(), np.nan, rng.beta(2, 4, risky.sum()))
    p = positive_proba(model.predict(prepare_input(model, pool)))
    per, picks = n // len(deciles), []
    for d in sorted(deciles, key=lambda d: d["decile"]):
        hi = 1.0 if d["decile"] == 1 else d["max_pd"]
        band = np.flatnonzero((p >= d["min_pd"]) & (p <= hi))
        w = _weights_for_mean(p[band], d["mean_pd"])
        picks.append(rng.choice(band, per, replace=len(band) < per * 2, p=w))
    idx = np.concatenate(picks)
    out = pool.iloc[idx].reset_index(drop=True)
    logit = np.log(np.clip(p[idx], 1e-6, 1 - 1e-6) / (1 - np.clip(p[idx], 1e-6, 1 - 1e-6)))
    true_pd = 1 / (1 + np.exp(-(shift + signal * logit + noise * rng.standard_normal(len(idx)))))
    out[cfg["data"]["target"]] = (rng.random(len(out)) < true_pd).astype(int)
    return out


def scored_portfolio(model, cfg: dict | None = None, split: str = "test") -> pd.DataFrame:
    """A scored portfolio for the dashboard: the held-out split if data is available, otherwise a simulated
    portfolio with a similar risk mix (demo mode, `attrs['source'] == 'synthetic'`)."""
    from .ingestion import make_splits

    cfg = cfg or load_config()
    try:
        df = getattr(make_splits(cfg), split).reset_index(drop=True)
        source = f"{split} split"
    except FileNotFoundError:
        # Seed 29 is a typical draw: across 30 seeds the AUC averages 0.705 (real test set: 0.769).
        df, source = simulated_portfolio(model, 5000, seed=29, cfg=cfg), "synthetic"
    out = pd.concat([df, score_frame(model, df, cfg)], axis=1)
    out.attrs["source"] = source
    return out


def contributions(model, df: pd.DataFrame) -> pd.DataFrame | None:
    """Per-application, per-feature contributions to the log-odds of default (SHAP values).

    Exact TreeSHAP for LightGBM (`pred_contrib`) and XGBoost (`pred_contribs`). Coefficient × standardised value
    for logistic regression. One-hot columns are summed back to their source feature. Returns None for other
    models (e.g. random forest). For a calibrated model these explain the base model's log-odds; calibration is
    monotonic, so the ranking and direction of the drivers are unchanged.
    """
    from sklearn.pipeline import Pipeline

    from .calibration import base_estimator

    raw = model.get_raw_model()
    if not isinstance(raw, Pipeline):
        return None
    est = base_estimator(raw.steps[-1][1])
    Xt = Pipeline(raw.steps[:-1]).transform(prepare_input(model, df))
    if hasattr(est, "booster_"):                               # LightGBM
        contrib = np.asarray(est.predict(Xt, pred_contrib=True))[:, :-1]
    elif hasattr(est, "get_booster"):                          # XGBoost
        import xgboost as xgb
        contrib = est.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]
    elif hasattr(est, "coef_"):                                # logistic regression
        contrib = Xt.to_numpy(dtype=float) * est.coef_[0]
    else:
        return None
    base = [n.split("__")[0] for n in Xt.columns]
    return pd.DataFrame(contrib, index=df.index, columns=Xt.columns).T.groupby(base).sum().T


def explain(model, df: pd.DataFrame, top: int = 8) -> pd.DataFrame | None:
    """Reason codes: the largest contributions to the first application's PD (see `contributions`)."""
    c = contributions(model, df.head(1))
    if c is None:
        return None
    out = pd.DataFrame({"feature": c.columns, "contribution": c.iloc[0].to_numpy()})
    engineered = model.get_raw_model().steps[0][1].transform(prepare_input(model, df.head(1))).iloc[0]
    out["value"] = [engineered.get(f) for f in out["feature"]]
    out["direction"] = np.where(out["contribution"] > 0, "increases risk", "decreases risk")
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--model-uri", default=None)
    args = ap.parse_args()

    cfg = load_config()
    uri = resolve_model_uri(args.model_uri, cfg)
    model = load_model(uri, cfg)
    df = pd.read_csv(args.input)
    scored = score_frame(model, df, cfg)
    id_col = cfg["data"]["id_col"]
    out = pd.concat([df[[id_col]], scored], axis=1) if id_col in df else scored

    dest = Path(args.output) if args.output else project_path("data/scored") / f"scored_{Path(args.input).stem}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"Scored {len(out):,} applications with {uri} -> {dest}")
    print(out.groupby("risk_grade")["decision"].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
