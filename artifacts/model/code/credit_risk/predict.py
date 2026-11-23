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
