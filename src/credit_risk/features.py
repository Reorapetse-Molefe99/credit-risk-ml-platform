"""Feature engineering and preprocessing.

`add_features` is the first step of every model pipeline, so the logged MLflow model accepts raw
application columns and applies these transforms itself (no training/serving skew). It must stay
row-wise and stateless: no fitting, no target, tolerant of missing columns and NaNs.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# In the Home Credit data DAYS_EMPLOYED == 365243 means "not employed / unknown".
DAYS_EMPLOYED_ANOMALY = 365243

ENGINEERED_FEATURES = {
    "DAYS_EMPLOYED_ANOMALY": "1 if DAYS_EMPLOYED carried the 365243 'not employed' sentinel",
    "EMPLOYED_YEARS": "Years in current employment",
    "AGE_YEARS": "Applicant age in years",
    "EMPLOYED_TO_AGE": "Share of life spent in current employment",
    "CREDIT_TO_INCOME": "Loan amount / annual income",
    "ANNUITY_TO_INCOME": "Annual repayment / annual income (debt-service burden)",
    "ANNUITY_TO_CREDIT": "Annual repayment / loan amount (inverse ~ term in years)",
    "GOODS_TO_CREDIT": "Goods price / loan amount (loan-to-value proxy)",
    "INCOME_PER_PERSON": "Income / family members",
    "EXT_SOURCE_MEAN": "Mean of available external bureau scores",
    "EXT_SOURCE_MIN": "Minimum external bureau score",
    "EXT_SOURCE_MAX": "Maximum external bureau score",
    "EXT_SOURCE_NULLS": "Number of missing external bureau scores (thin-file indicator)",
}


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = set(df.columns)

    if "DAYS_EMPLOYED" in cols:
        anomaly = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
        df["DAYS_EMPLOYED_ANOMALY"] = anomaly.astype("float64")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].mask(anomaly)
        df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    if "DAYS_BIRTH" in cols:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"} <= cols:
        df["EMPLOYED_TO_AGE"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"} <= cols:
        df["CREDIT_TO_INCOME"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"} <= cols:
        df["ANNUITY_TO_INCOME"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    if {"AMT_ANNUITY", "AMT_CREDIT"} <= cols:
        df["ANNUITY_TO_CREDIT"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    if {"AMT_GOODS_PRICE", "AMT_CREDIT"} <= cols:
        df["GOODS_TO_CREDIT"] = _safe_div(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"} <= cols:
        df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    ext = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in cols]
    if ext:
        df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext].max(axis=1)
        df["EXT_SOURCE_NULLS"] = df[ext].isna().sum(axis=1).astype("float64")

    return df


def clean_feature_name(feature: str, category) -> str:
    """One-hot column names safe for LightGBM / XGBoost (no spaces, commas, slashes...)."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", f"{feature}__{category}").strip("_")


def split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric and categorical columns *after* feature engineering."""
    engineered = add_features(X.head(200))
    cat = engineered.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num = [c for c in engineered.columns if c not in cat]
    return num, cat


def build_preprocessor(num_cols: list[str], cat_cols: list[str], scale: bool) -> ColumnTransformer:
    num_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                 sparse_output=False, dtype=np.float32,
                                 feature_name_combiner=clean_feature_name)),
    ])
    return ColumnTransformer(
        [("num", Pipeline(num_steps), num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
