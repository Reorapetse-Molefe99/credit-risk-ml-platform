"""Credit-risk ML platform: probability-of-default modelling with MLflow.

Modules
-------
ingestion   load raw data, build reproducible splits and the monitoring reference sample
validation  schema / range / quality checks on application data
features    domain feature engineering and preprocessing
train       model pipelines, MLflow-tracked training and Optuna tuning
evaluate    credit-risk metrics, champion/challenger validation, promotion and export
predict     model loading, PD scoring, risk grades and decisions
monitoring  population stability (PSI) of scores and features
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__version__ = "0.2.0"


def _find_root() -> Path:
    if env := os.environ.get("CREDIT_RISK_HOME"):
        return Path(env).resolve()
    src_root = Path(__file__).resolve().parents[2]
    if (src_root / "configs").is_dir():
        return src_root
    return Path.cwd()


PROJECT_ROOT = _find_root()


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("CREDIT_RISK_CONFIG", PROJECT_ROOT / "configs" / "model_config.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mlflow"]["tracking_uri"] = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    return cfg


def project_path(relative: str | os.PathLike) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
