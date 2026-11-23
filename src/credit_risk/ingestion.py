"""Data ingestion: raw loading, reproducible splits, monitoring reference data and synthetic samples."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import load_config, project_path
from .validation import validate_applications


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def xy(self, name: str, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
        cfg = cfg or load_config()
        df = getattr(self, name)
        return df[feature_columns(df, cfg)], df[cfg["data"]["target"]].astype(int)


def feature_columns(df: pd.DataFrame, cfg: dict | None = None) -> list[str]:
    d = (cfg or load_config())["data"]
    drop = {d["id_col"], d["target"], *d.get("exclude_features", [])}
    return [c for c in df.columns if c not in drop]


def standardise_types(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Integer features -> float64 so the model schema tolerates missing values at inference."""
    d = (cfg or load_config())["data"]
    skip = {d["id_col"], d["target"]}
    int_cols = [c for c in df.select_dtypes("integer").columns if c not in skip]
    if int_cols:
        df = df.copy()
        df[int_cols] = df[int_cols].astype("float64")
    return df


def load_raw(cfg: dict | None = None, path: str | None = None, nrows: int | None = None,
             validate: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    d = cfg["data"]
    src = project_path(path or d["raw_path"])
    if not src.exists():
        raise FileNotFoundError(
            f"Raw data not found at {src}. Place HomeLoanData.csv in data/raw/ (see docs/data_dictionary.md)."
        )
    df = standardise_types(pd.read_csv(src, nrows=nrows), cfg)
    if validate:
        validate_applications(df, target=d["target"], id_col=d["id_col"]).raise_if_invalid()
    return df


def split_frame(df: pd.DataFrame, cfg: dict | None = None) -> Splits:
    """Stratified train / validation / test split."""
    d = (cfg or load_config())["data"]
    train_val, test = train_test_split(df, test_size=d["test_size"], stratify=df[d["target"]],
                                       random_state=d["random_state"])
    train, val = train_test_split(train_val, test_size=d["val_size"], stratify=train_val[d["target"]],
                                  random_state=d["random_state"])
    return Splits(train=train, val=val, test=test)


def make_splits(cfg: dict | None = None, force: bool = False) -> Splits:
    """Create (or load cached) parquet splits in data/processed/."""
    cfg = cfg or load_config()
    out = project_path(cfg["data"]["processed_dir"])
    paths = {k: out / f"{k}.parquet" for k in ("train", "val", "test")}
    if not force and all(p.exists() for p in paths.values()):
        return Splits(**{k: pd.read_parquet(p) for k, p in paths.items()})
    splits = split_frame(load_raw(cfg), cfg)
    out.mkdir(parents=True, exist_ok=True)
    for k, p in paths.items():
        getattr(splits, k).to_parquet(p, index=False)
    return splits


def sample_splits(splits: Splits, n_train: int, cfg: dict | None = None) -> Splits:
    """Stratified down-sample of every split by the same fraction, for quick experiments."""
    cfg = cfg or load_config()
    target, seed = cfg["data"]["target"], cfg["data"]["random_state"]
    frac = min(1.0, n_train / len(splits.train))
    if frac >= 1.0:
        return splits

    def _s(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(target, group_keys=False).sample(frac=frac, random_state=seed)

    return Splits(train=_s(splits.train), val=_s(splits.val), test=_s(splits.test))


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short content hash, logged to MLflow so every run can be tied to a data version."""
    h = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:12]


def reference_path(cfg: dict | None = None):
    cfg = cfg or load_config()
    return project_path(cfg["data"]["reference_dir"]) / "reference_sample.parquet"


def save_reference(df: pd.DataFrame, scores: np.ndarray, cfg: dict | None = None):
    """Persist the population the champion was developed on (features + PD) for drift monitoring."""
    cfg = cfg or load_config()
    n = min(cfg["data"]["reference_sample_size"], len(df))
    ref = df.assign(probability_of_default=scores).sample(n, random_state=cfg["data"]["random_state"])
    path = reference_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(path, index=False)
    return path


def load_reference(cfg: dict | None = None) -> pd.DataFrame:
    path = reference_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"No reference sample at {path}. It is created by `credit_risk.evaluate --promote`.")
    return pd.read_parquet(path)


def synthetic_applications(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Synthetic applications with the real column names and a learnable default signal.

    Used by tests and CI so nothing depends on the (non-redistributable) loan-level data.
    """
    rng = np.random.default_rng(seed)
    income = rng.lognormal(11.8, 0.5, n).round(-2)
    credit = (income * rng.uniform(1, 6, n)).round(-2)
    annuity = (credit * rng.uniform(0.03, 0.10, n)).round(1)
    employed = -rng.integers(30, 12000, n).astype(float)
    employed[rng.random(n) < 0.15] = 365243
    ext = {f"EXT_SOURCE_{i}": rng.beta(4, 3, n) for i in (1, 2, 3)}
    for i, rate in ((1, 0.5), (3, 0.2)):
        ext[f"EXT_SOURCE_{i}"][rng.random(n) < rate] = np.nan
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M"], n),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
        "CNT_CHILDREN": rng.poisson(0.4, n).astype(float),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": (credit * rng.uniform(0.8, 1.0, n)).round(-2),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner", "State servant"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education",
                                           "Incomplete higher"], n),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage"], n),
        "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents", "Rented apartment"], n),
        "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n),
        "DAYS_BIRTH": -rng.integers(21 * 365, 69 * 365, n).astype(float),
        "DAYS_EMPLOYED": employed,
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Sales staff", "Core staff", None], n),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
        "REGION_RATING_CLIENT": rng.choice([1.0, 2.0, 3.0], n, p=[0.1, 0.75, 0.15]),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "Self-employed", "School", "Government"], n),
        **ext,
    })
    ext_mean = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    logit = -2.5 - 4.0 * (ext_mean - 0.57) + 0.6 * (annuity / income) + 0.3 * (df["REGION_RATING_CLIENT"] - 2)
    df["TARGET"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
