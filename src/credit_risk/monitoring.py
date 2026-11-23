"""Population-stability monitoring of scores and input features.

    uv run python -m credit_risk.monitoring                       # test split vs reference sample
    uv run python -m credit_risk.monitoring --current new_month.csv --log

PSI < 0.10 stable · 0.10–0.25 monitor · > 0.25 investigate / consider redevelopment.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from . import load_config, project_path


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions (quantile bins of `expected`)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    e = expected.fillna("MISSING").astype(str).value_counts(normalize=True)
    a = actual.fillna("MISSING").astype(str).value_counts(normalize=True)
    cats = e.index.union(a.index)
    e, a = e.reindex(cats, fill_value=0).clip(lower=1e-6), a.reindex(cats, fill_value=0).clip(lower=1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float, cfg: dict | None = None) -> str:
    m = (cfg or load_config())["monitoring"]
    if np.isnan(value):
        return "n/a"
    return "alert" if value > m["psi_alert"] else "warning" if value > m["psi_warning"] else "stable"


def feature_stability(reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None,
                      cfg: dict | None = None) -> pd.DataFrame:
    """PSI and missing-rate shift per feature, sorted by PSI."""
    cfg = cfg or load_config()
    d = cfg["data"]
    skip = {d["id_col"], d["target"], "probability_of_default", *d.get("exclude_features", [])}
    features = features or [c for c in reference.columns if c in current.columns and c not in skip]
    rows = []
    for c in features:
        ref, cur = reference[c], current[c]
        numeric = pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur)
        value = psi(ref, cur) if numeric else categorical_psi(ref, cur)
        rows.append({"feature": c, "type": "numeric" if numeric else "categorical", "psi": value,
                     "missing_ref": float(ref.isna().mean()), "missing_cur": float(cur.isna().mean()),
                     "status": status(value, cfg)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, current_scores, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    score_psi = psi(reference["probability_of_default"], current_scores)
    feats = feature_stability(reference, current, cfg=cfg)
    target = cfg["data"]["target"]
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "score_psi": score_psi,
        "score_status": status(score_psi, cfg),
        "mean_pd_reference": float(reference["probability_of_default"].mean()),
        "mean_pd_current": float(np.mean(current_scores)),
        "features_alert": feats.loc[feats.status == "alert", "feature"].tolist(),
        "features_warning": feats.loc[feats.status == "warning", "feature"].tolist(),
        "feature_stability": feats.head(cfg["monitoring"]["top_features"]).round(5).to_dict(orient="records"),
    }
    if target in current and target in reference:
        report["default_rate_reference"] = float(reference[target].mean())
        report["default_rate_current"] = float(current[target].mean())
    return report


def main() -> None:
    from .ingestion import load_reference, make_splits, standardise_types
    from .predict import load_model, score_frame

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", default=None, help="CSV of new applications (default: test split)")
    ap.add_argument("--model-uri", default=None)
    ap.add_argument("--log", action="store_true", help="Also log the report to MLflow (run_type=monitoring)")
    args = ap.parse_args()

    cfg = load_config()
    reference = load_reference(cfg)
    current = (standardise_types(pd.read_csv(args.current), cfg) if args.current
               else make_splits(cfg).test)
    scores = score_frame(load_model(args.model_uri, cfg), current, cfg)["probability_of_default"]
    report = drift_report(reference, current, scores.to_numpy(), cfg)

    out = project_path("reports") / f"monitoring_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Score PSI {report['score_psi']:.4f} ({report['score_status']}) | mean PD "
          f"{report['mean_pd_reference']:.4f} -> {report['mean_pd_current']:.4f}")
    print(f"Feature alerts: {report['features_alert'] or 'none'} | warnings: {report['features_warning'] or 'none'}")
    print(f"Report -> {out}")

    if args.log:
        import mlflow

        from .train import setup_mlflow
        setup_mlflow(cfg)
        with mlflow.start_run(run_name="monitoring"):
            mlflow.set_tag("run_type", "monitoring")
            mlflow.log_metrics({"score_psi": report["score_psi"], "mean_pd_current": report["mean_pd_current"],
                                "n_features_alert": len(report["features_alert"])})
            mlflow.log_dict(report, "monitoring_report.json")


if __name__ == "__main__":
    main()
