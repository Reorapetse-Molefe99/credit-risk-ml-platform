"""Train PD models with full MLflow tracking, optionally with Optuna tuning.

Examples
--------
    uv run python -m credit_risk.train --model logreg --sample 30000
    uv run python -m credit_risk.train --model lightgbm --register
    uv run python -m credit_risk.train --model xgboost --param max_depth=6 --param learning_rate=0.03
    uv run python -m credit_risk.train --model lightgbm --tune --trials 30 --sample 60000
    uv run python -m credit_risk.train --model rf --sample 60000
    uv run python -m credit_risk.train --model lightgbm --calibrate sigmoid     # calibrated on the validation split
"""
from __future__ import annotations

import argparse
import ast
import time
import urllib.request
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import PROJECT_ROOT, load_config, project_path
from .calibration import METHODS as CALIBRATION_METHODS
from .calibration import PrefitCalibratedClassifier, base_estimator
from .features import add_features, build_preprocessor, split_column_types
from .ingestion import Splits, data_fingerprint, make_splits, sample_splits

ALGORITHMS = ("logreg", "rf", "hgb", "xgboost", "lightgbm")

# Types that may be deserialised from the skops model file. Anything else makes logging fail loudly.
TRUSTED_TYPE_PREFIXES = ("credit_risk.", "numpy.", "sklearn.", "scipy.", "xgboost.", "lightgbm.")
TRUSTED_TYPES = {"collections.OrderedDict", "collections.defaultdict"}


# --------------------------------------------------------------------------- MLflow helpers
def setup_mlflow(cfg: dict | None = None, experiment: bool = True) -> str:
    cfg = cfg or load_config()
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=3)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Cannot reach the MLflow tracking server at {uri}.\n"
                "Start it with:  uv run mlflow server --backend-store-uri sqlite:///mlflow.db "
                "--artifacts-destination ./mlartifacts --port 5000\n"
                f"(underlying error: {exc})"
            ) from None
    mlflow.set_tracking_uri(uri)
    if experiment:
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    return uri


def get_alias_version(name: str, alias: str):
    """ModelVersion behind a registry alias, or None."""
    try:
        return MlflowClient().get_model_version_by_alias(name, alias)
    except mlflow.exceptions.MlflowException:
        return None


def trusted_types_for(model) -> list[str]:
    import skops.io as sio

    found = sio.get_untrusted_types(data=sio.dumps(model))
    bad = [t for t in found if not t.startswith(TRUSTED_TYPE_PREFIXES) and t not in TRUSTED_TYPES]
    if bad:
        raise RuntimeError(f"Refusing to trust unexpected types in the model: {bad}")
    return found


# --------------------------------------------------------------------------- pipelines
def build_estimator(model: str, params: dict[str, Any]):
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**params)
    if model == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if model == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    raise ValueError(f"Unknown model '{model}'. Choose from {ALGORITHMS}")


def build_pipeline(model: str, X_train: pd.DataFrame, overrides: dict[str, Any] | None = None,
                   cfg: dict | None = None) -> tuple[Pipeline, dict[str, Any]]:
    cfg = cfg or load_config()
    params = {**cfg["model"]["params"].get(model, {}), **(overrides or {})}
    num, cat = split_column_types(X_train)
    pipe = Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocess", build_preprocessor(num, cat, scale=(model == "logreg"))),
        ("model", build_estimator(model, params)),
    ])
    pipe.set_output(transform="pandas")
    return pipe, params


def feature_importance(pipe: Pipeline) -> pd.DataFrame | None:
    est = base_estimator(pipe.named_steps["model"])
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(est.coef_.ravel())
    else:
        return None
    return (pd.DataFrame({"feature": names, "importance": vals})
            .sort_values("importance", ascending=False, ignore_index=True))


def typical_applicant(X: pd.DataFrame) -> pd.DataFrame:
    """One synthetic 'median applicant' row – used as the model's input example so no real
    customer record is ever stored with the model artifact."""
    row = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            row[c] = float(s.median()) if s.notna().any() else np.nan
        else:
            mode = s.mode(dropna=True)
            row[c] = mode.iloc[0] if len(mode) else None
    return pd.DataFrame([row], columns=X.columns)


# --------------------------------------------------------------------------- training
def _log_datasets(splits: Splits, cfg: dict) -> None:
    src = str(project_path(cfg["data"]["raw_path"]))
    for ctx in ("train", "val", "test"):
        ds = mlflow.data.from_pandas(getattr(splits, ctx), source=src, name=f"home-credit-{ctx}",
                                     targets=cfg["data"]["target"])
        mlflow.log_input(ds, context=ctx)


def train_and_log(model: str, splits: Splits, overrides: dict[str, Any] | None = None,
                  register: bool = False, run_name: str | None = None, nested: bool = False,
                  log_model: bool = True, cfg: dict | None = None,
                  calibrate: str | None = None) -> dict[str, Any]:
    """Fit on train, select on validation, report on test.

    With `calibrate`, a calibration layer is fitted on the validation split after the base model. `val_*`
    metrics then describe the *uncalibrated* base model (the honest out-of-sample selection metric), and
    `test_*` metrics describe the final calibrated model. `test_uncal_*` records the base model on test so the
    effect of calibration is measured on data neither step has seen.
    """
    from . import evaluate as E
    from .monitoring import psi

    cfg = cfg or load_config()
    X_tr, y_tr = splits.xy("train", cfg)
    X_va, y_va = splits.xy("val", cfg)
    X_te, y_te = splits.xy("test", cfg)
    pipe, params = build_pipeline(model, X_tr, overrides, cfg)

    if calibrate and calibrate not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method '{calibrate}'. Choose from {CALIBRATION_METHODS}")
    default_name = f"{model}-{calibrate}" if calibrate else model
    with mlflow.start_run(run_name=run_name or default_name, nested=nested) as run:
        mlflow.set_tags({
            "model_family": model,
            "task": "probability_of_default",
            "data_fingerprint_train": data_fingerprint(splits.train),
            "excluded_features": ",".join(cfg["data"].get("exclude_features", [])),
            "calibration": calibrate or "none",
        })
        mlflow.log_params({f"model__{k}": v for k, v in params.items()})
        mlflow.log_params({"model": model, "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
                           "n_raw_features": X_tr.shape[1], "calibration": calibrate or "none"})
        if not nested:
            _log_datasets(splits, cfg)

        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        mlflow.log_metric("fit_seconds", time.perf_counter() - t0)

        p_va = pipe.predict_proba(X_va)[:, 1]
        results: dict[str, Any] = E.classification_metrics(y_va, p_va, "val_")
        if calibrate:
            results.update(E.classification_metrics(y_te, pipe.predict_proba(X_te)[:, 1], "test_uncal_"))
            cal = PrefitCalibratedClassifier(pipe.steps[-1][1], calibrate).fit(pipe[:-1].transform(X_va), y_va)
            pipe.steps[-1] = ("model", cal)
        p_te = pipe.predict_proba(X_te)[:, 1]
        results.update(E.classification_metrics(y_te, p_te, "test_"))
        results["val_test_psi"] = psi(pipe.predict_proba(X_va)[:, 1] if calibrate else p_va, p_te)
        mlflow.log_metrics(results)

        if log_model:
            for name, fig in {
                "plots/roc_test.png": E.plot_roc(y_te, p_te, f"ROC – {model} (test)"),
                "plots/calibration_test.png": E.plot_calibration(y_te, p_te),
                "plots/pd_distribution_test.png": E.plot_score_distribution(y_te, p_te),
            }.items():
                mlflow.log_figure(fig, name)
            mlflow.log_table(E.decile_table(y_te, p_te), "tables/decile_lift_test.json")
            fi = feature_importance(pipe)
            if fi is not None:
                mlflow.log_table(fi.head(50), "tables/feature_importance_top50.json")

            sample = X_tr.head(200)
            info = mlflow.sklearn.log_model(
                sk_model=pipe,
                name="model",
                signature=infer_signature(sample, pipe.predict_proba(sample)),
                input_example=typical_applicant(X_tr),
                pyfunc_predict_fn="predict_proba",          # pyfunc / REST returns [P(0), P(1)]
                code_paths=[str(PROJECT_ROOT / "src" / "credit_risk")],
                skops_trusted_types=trusted_types_for(pipe),
                registered_model_name=cfg["mlflow"]["registered_model_name"] if register else None,
            )
            results["model_uri"] = info.model_uri
            if register:
                results["registered_version"] = info.registered_model_version
        results["run_id"] = run.info.run_id
    return results


# --------------------------------------------------------------------------- tuning
SEARCH_SPACES = {
    "lightgbm": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": t.suggest_int("num_leaves", 15, 127, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 20, 300, log=True),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 0.9),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "n_estimators": t.suggest_int("n_estimators", 200, 1500, step=100),
    },
    "xgboost": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": t.suggest_int("max_depth", 3, 8),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 50, log=True),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 0.9),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "n_estimators": t.suggest_int("n_estimators", 200, 1500, step=100),
    },
    "logreg": lambda t: {"C": t.suggest_float("C", 1e-4, 10, log=True)},
    "rf": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": t.suggest_int("max_depth", 6, 14),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 20, 400, log=True),
        "max_features": t.suggest_float("max_features", 0.05, 0.5, log=True),
    },
}


def tune(model: str, splits: Splits, trials: int, register: bool = False, cfg: dict | None = None) -> dict:
    """Optuna search on the validation split; each trial is a nested MLflow run.
    The best configuration is refit and logged as the parent run's model."""
    import optuna

    cfg = cfg or load_config()
    if model not in SEARCH_SPACES:
        raise SystemExit(f"No search space for '{model}'. Choose from {list(SEARCH_SPACES)}")
    with mlflow.start_run(run_name=f"tune-{model}") as parent:
        mlflow.set_tags({"model_family": model, "run_type": "hyperparameter_search"})

        def objective(trial):
            res = train_and_log(model, splits, SEARCH_SPACES[model](trial), run_name=f"trial-{trial.number}",
                                nested=True, log_model=False, cfg=cfg)
            return res["val_roc_auc"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=trials, show_progress_bar=True)
        mlflow.log_params({f"best__{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_roc_auc", study.best_value)

        best = train_and_log(model, splits, study.best_params, register=register, run_name=f"best-{model}",
                             nested=True, cfg=cfg)
        mlflow.set_tag("best_child_run_id", best["run_id"])
        best["parent_run_id"] = parent.info.run_id
        best["best_params"] = study.best_params
    return best


# --------------------------------------------------------------------------- CLI
def _parse_params(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=ALGORITHMS, default=cfg["model"]["default_algorithm"])
    ap.add_argument("--sample", type=int, default=None, help="Down-sample training to N rows for quick runs")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="Hyper-parameter override")
    ap.add_argument("--tune", action="store_true", help="Run an Optuna search instead of a single fit")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--calibrate", choices=CALIBRATION_METHODS, default=None,
                    help="Fit a calibration layer on the validation split (not with --tune)")
    ap.add_argument("--register", action="store_true", help="Register the model in the MLflow Model Registry")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--rebuild-splits", action="store_true")
    args = ap.parse_args()

    setup_mlflow(cfg)
    splits = make_splits(cfg, force=args.rebuild_splits)
    if args.sample:
        splits = sample_splits(splits, args.sample, cfg)

    if args.tune and args.calibrate:
        raise SystemExit("--calibrate is not supported with --tune: tune first, then calibrate the best params.")
    if args.tune:
        res = tune(args.model, splits, args.trials, register=args.register, cfg=cfg)
        print(f"\nBest params:   {res['best_params']}\nParent run:    {res['parent_run_id']}")
    else:
        res = train_and_log(args.model, splits, _parse_params(args.param), register=args.register,
                            run_name=args.run_name, cfg=cfg, calibrate=args.calibrate)
    print(f"Run ID:        {res['run_id']}\nModel URI:     {res.get('model_uri')}")
    if "registered_version" in res:
        print(f"Registered:    {cfg['mlflow']['registered_model_name']} v{res['registered_version']}")
    for k in ("val_roc_auc", "test_roc_auc", "test_gini", "test_ks", "test_brier", "test_uncal_brier",
              "test_recall_top10", "val_test_psi"):
        if k in res:
            print(f"{k:<17} {res[k]:.4f}")


if __name__ == "__main__":
    main()
