"""Population-stability monitoring of scores and input features, with simulated drift scenarios.

    uv run python -m credit_risk.monitoring                       # test split vs reference sample
    uv run python -m credit_risk.monitoring --current new_month.csv --log
    uv run python -m credit_risk.monitoring --scenario recession  # simulated drift on the test split

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


def missing_status(shift: float, cfg: dict | None = None) -> str:
    m = (cfg or load_config())["monitoring"]
    shift = abs(shift)
    return ("alert" if shift > m.get("missing_shift_alert", 0.10)
            else "warning" if shift > m.get("missing_shift_warning", 0.05) else "stable")


_SEVERITY = {"n/a": -1, "stable": 0, "warning": 1, "alert": 2}


def feature_stability(reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None,
                      cfg: dict | None = None) -> pd.DataFrame:
    """PSI and missing-rate shift per feature, sorted by PSI. `status` is the worse of the two checks: PSI is
    computed on non-missing values only, so a feature that suddenly goes missing is caught by the second."""
    cfg = cfg or load_config()
    d = cfg["data"]
    skip = {d["id_col"], d["target"], "probability_of_default", *d.get("exclude_features", [])}
    features = features or [c for c in reference.columns if c in current.columns and c not in skip]
    rows = []
    for c in features:
        ref, cur = reference[c], current[c]
        numeric = pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur)
        value = psi(ref, cur) if numeric else categorical_psi(ref, cur)
        miss_ref, miss_cur = float(ref.isna().mean()), float(cur.isna().mean())
        worst = max(status(value, cfg), missing_status(miss_cur - miss_ref, cfg), key=_SEVERITY.__getitem__)
        rows.append({"feature": c, "type": "numeric" if numeric else "categorical", "psi": value,
                     "missing_ref": miss_ref, "missing_cur": miss_cur, "status": worst})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


# Simulated operating conditions for drift monitoring without live data. Each one perturbs the inputs of a
# real population; outcomes are then unknown, so only data drift and prediction drift are reported.
SCENARIOS: dict[str, dict[str, float]] = {
    "income_shock": {"income_shock": -0.20},
    "bureau_deterioration": {"ext_shift": -0.10},
    "thin_file_growth": {"thin_file": 0.30},
    "recession": {"income_shock": -0.20, "ext_shift": -0.10, "thin_file": 0.10},
}


def simulate_drift(df: pd.DataFrame, income_shock: float = 0.0, ext_shift: float = 0.0, thin_file: float = 0.0,
                   seed: int = 0, cfg: dict | None = None) -> pd.DataFrame:
    """A shifted copy of `df`: incomes scaled by (1 + income_shock), external bureau scores shifted by
    `ext_shift` (clipped to [0, 1]), and an extra `thin_file` share of applicants losing their bureau scores.

    The target and any previous scores are dropped. The old labels describe the old borrowers, so reusing them
    on modified inputs would fabricate a performance result.
    """
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    out = df.drop(columns=[cfg["data"]["target"], "probability_of_default", "risk_grade", "decision"],
                  errors="ignore").copy()
    if "AMT_INCOME_TOTAL" in out:
        out["AMT_INCOME_TOTAL"] = out["AMT_INCOME_TOTAL"] * (1 + income_shock)
    for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if c in out:
            out[c] = (out[c] + ext_shift).clip(0, 1)
            if thin_file:
                out.loc[rng.random(len(out)) < thin_file, c] = np.nan
    return out


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
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default=None,
                    help="Apply a simulated shift to the current population (reports data and prediction drift only)")
    ap.add_argument("--log", action="store_true", help="Also log the report to MLflow (run_type=monitoring)")
    args = ap.parse_args()

    cfg = load_config()
    reference = load_reference(cfg)
    current = (standardise_types(pd.read_csv(args.current), cfg) if args.current
               else make_splits(cfg).test)
    if args.scenario:
        current = simulate_drift(current, **SCENARIOS[args.scenario], cfg=cfg)
    scores = score_frame(load_model(args.model_uri, cfg), current, cfg)["probability_of_default"]
    report = drift_report(reference, current, scores.to_numpy(), cfg)
    report["scenario"] = args.scenario or "none"

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
            mlflow.set_tags({"run_type": "monitoring", "scenario": report["scenario"]})
            mlflow.log_metrics({"score_psi": report["score_psi"], "mean_pd_current": report["mean_pd_current"],
                                "n_features_alert": len(report["features_alert"])})
            mlflow.log_dict(report, "monitoring_report.json")


if __name__ == "__main__":
    main()
