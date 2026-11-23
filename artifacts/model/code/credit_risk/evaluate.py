"""Credit-risk metrics, champion/challenger validation, promotion and champion export.

    uv run python -m credit_risk.evaluate                      # newest version vs @champion (read-only)
    uv run python -m credit_risk.evaluate --version 3 --promote
    uv run python -m credit_risk.evaluate --export             # re-export current @champion to artifacts/

Promotion moves the @champion alias and exports the champion to artifacts/model + artifacts/metrics.json
(used by the API, dashboard and Docker image) and refreshes data/reference/ for monitoring.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from . import load_config, project_path

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous_champion"


# --------------------------------------------------------------------------- metrics
def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: max separation between the default and non-default score CDFs."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def classification_metrics(y, p, prefix: str = "") -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    m = {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "observed_default_rate": float(y.mean()),
        "mean_predicted_pd": float(p.mean()),
    }
    return {f"{prefix}{k}": float(v) for k, v in m.items()}


def decile_table(y, p, n: int = 10) -> pd.DataFrame:
    """Gains / lift table. Decile 1 = highest predicted risk."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), n, labels=range(1, n + 1))
    t = df.groupby("decile", observed=True).agg(n=("y", "size"), defaults=("y", "sum"), min_pd=("p", "min"),
                                                max_pd=("p", "max"), mean_pd=("p", "mean"))
    t["default_rate"] = t["defaults"] / t["n"]
    t["cum_capture_rate"] = t["defaults"].cumsum() / t["defaults"].sum()
    t["lift"] = t["default_rate"] / df["y"].mean()
    return t.reset_index().astype({"decile": int})


def plot_roc(y, p, title: str = "ROC curve"):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_calibration(y, p, title: str = "Calibration (reliability) curve"):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    lim = max(mean_pred.max(), frac_pos.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="grey", lw=1, label="perfect")
    ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_score_distribution(y, p, title: str = "PD distribution by outcome"):
    y, p = np.asarray(y), np.asarray(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, np.quantile(p, 0.995), 50)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, density=True, label="non-default")
    ax.hist(p[y == 1], bins=bins, alpha=0.6, density=True, label="default")
    ax.set(xlabel="Predicted PD", ylabel="Density", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def performance_summary(y, p) -> dict:
    """JSON-serialisable performance pack (metrics, ROC points, calibration bins, deciles)."""
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 101).astype(int))
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "metrics": classification_metrics(y, p),
        "roc_curve": {"fpr": fpr[idx].round(5).tolist(), "tpr": tpr[idx].round(5).tolist()},
        "calibration": {"mean_predicted": mean_pred.round(5).tolist(), "observed": frac_pos.round(5).tolist()},
        "deciles": decile_table(y, p).round(5).to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- export
def export_champion(cfg: dict | None = None) -> dict:
    """Copy @champion into artifacts/model, write artifacts/metrics.json and refresh the reference sample."""
    import mlflow
    from mlflow import MlflowClient

    from .ingestion import make_splits, save_reference
    from .predict import positive_proba
    from .train import get_alias_version

    cfg = cfg or load_config()
    name = cfg["mlflow"]["registered_model_name"]
    champ = get_alias_version(name, CHAMPION)
    if champ is None:
        raise SystemExit(f"No @{CHAMPION} alias on '{name}' – promote a version first.")

    model_dir = project_path(cfg["artifacts"]["model_dir"])
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{champ.version}")
    shutil.copytree(tmp, model_dir)

    model = mlflow.pyfunc.load_model(str(model_dir))
    splits = make_splits(cfg)
    X_te, y_te = splits.xy("test", cfg)
    summary = performance_summary(y_te, positive_proba(model.predict(X_te)))
    run = MlflowClient().get_run(champ.run_id)

    payload = {
        "model_name": name,
        "version": champ.version,
        "run_id": champ.run_id,
        "algorithm": run.data.params.get("model"),
        "n_train": int(run.data.params.get("n_train", 0)),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "evaluation_split": "test",
        "n_test": int(len(y_te)),
        **summary,
        "validation_metrics": {k: v for k, v in run.data.metrics.items() if k.startswith("val_")},
    }
    metrics_path = project_path(cfg["artifacts"]["metrics_path"])
    metrics_path.write_text(json.dumps(payload, indent=2))

    X_tr, _ = splits.xy("train", cfg)
    ref_path = save_reference(splits.train, positive_proba(model.predict(X_tr)), cfg)
    print(f"Exported {name} v{champ.version} -> {model_dir}\nMetrics -> {metrics_path}\nReference -> {ref_path}")
    return payload


# --------------------------------------------------------------------------- champion / challenger
def main() -> None:
    import mlflow
    from mlflow import MlflowClient

    from .ingestion import make_splits
    from .monitoring import psi
    from .predict import positive_proba
    from .train import get_alias_version, setup_mlflow

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=None, help="Registered version to evaluate (default: latest)")
    ap.add_argument("--promote", action="store_true", help="Move @champion to the challenger if it passes")
    ap.add_argument("--export", action="store_true", help="Only export the current @champion to artifacts/")
    ap.add_argument("--threshold", type=float, default=None,
                    help="PD cut-off for accept/decline confusion-matrix metrics (default: decision.approve_below)")
    args = ap.parse_args()

    cfg = load_config()
    setup_mlflow(cfg)
    if args.export:
        export_champion(cfg)
        return

    name = cfg["mlflow"]["registered_model_name"]
    primary = cfg["promotion"]["primary_metric"]
    threshold = args.threshold or cfg["decision"]["approve_below"]
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(f"No versions of '{name}' registered yet. Train with --register first.")
    version = args.version or str(max(int(v.version) for v in versions))
    client.set_registered_model_alias(name, CHALLENGER, version)
    champ = get_alias_version(name, CHAMPION)

    X_te, y_te = make_splits(cfg).xy("test", cfg)

    def score(v: str) -> np.ndarray:
        return positive_proba(mlflow.pyfunc.load_model(f"models:/{name}/{v}").predict(X_te))

    with mlflow.start_run(run_name=f"validate-v{version}"):
        mlflow.set_tags({"run_type": "validation", "challenger_version": version,
                         "champion_version": champ.version if champ else "none"})
        p_ch = score(version)
        ch = classification_metrics(y_te, p_ch)
        mlflow.log_metrics({f"challenger_{k}": v for k, v in ch.items()})
        fig = plot_calibration(y_te, p_ch, f"Calibration – v{version}")
        mlflow.log_figure(fig, "plots/challenger_calibration.png")
        plt.close(fig)
        mlflow.log_table(decile_table(y_te, p_ch), "tables/challenger_deciles.json")

        # MLflow's built-in classifier evaluator on the accept/decline decision at the PD cut-off.
        eval_df = pd.DataFrame({"target": y_te.to_numpy(), "prediction": (p_ch >= threshold).astype(int)})
        mlflow.models.evaluate(data=eval_df, targets="target", predictions="prediction", model_type="classifier",
                               evaluator_config={"log_model_explainability": False, "pos_label": 1})

        passed, notes = True, []
        if champ and champ.version != version:
            p_cp = score(champ.version)
            cp = classification_metrics(y_te, p_cp)
            mlflow.log_metrics({f"champion_{k}": v for k, v in cp.items()})
            delta, shift = ch[primary] - cp[primary], psi(p_cp, p_ch)
            mlflow.log_metrics({f"delta_{primary}": delta, "psi_champion_vs_challenger": shift})
            if delta < cfg["promotion"]["min_improvement"]:
                passed = False
                notes.append(f"{primary} improvement {delta:+.4f} < {cfg['promotion']['min_improvement']}")
            if shift > cfg["promotion"]["max_psi"]:
                notes.append(f"WARNING score PSI vs champion {shift:.3f} > {cfg['promotion']['max_psi']}")
            print(f"Champion v{champ.version} {primary}={cp[primary]:.4f} | Challenger v{version} "
                  f"{primary}={ch[primary]:.4f} | delta={delta:+.4f} | PSI={shift:.3f}")
        else:
            print(f"No other champion – challenger v{version} {primary}={ch[primary]:.4f}")

        verdict = "passed" if passed else "failed"
        mlflow.set_tag("validation_status", verdict)
        client.set_model_version_tag(name, version, "validation_status", verdict)
        client.set_model_version_tag(name, version, f"test_{primary}", f"{ch[primary]:.4f}")
        for n in notes:
            print(" -", n)

    if args.promote and passed:
        if champ and champ.version != version:
            client.set_registered_model_alias(name, PREVIOUS, champ.version)
        client.set_registered_model_alias(name, CHAMPION, version)
        print(f"Promoted v{version} to @{CHAMPION}")
        export_champion(cfg)
    elif args.promote:
        print(f"Not promoted: validation {verdict}")


if __name__ == "__main__":
    main()
