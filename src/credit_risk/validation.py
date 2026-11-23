"""Data validation for loan-application data (training extracts and scoring requests)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Columns the model cannot do without – everything else may be missing and is imputed.
REQUIRED_COLUMNS = ["AMT_INCOME_TOTAL", "AMT_CREDIT"]

# (min, max) plausibility ranges; None = unbounded.
RANGES: dict[str, tuple[float | None, float | None]] = {
    "AMT_INCOME_TOTAL": (0, None),
    "AMT_CREDIT": (0, None),
    "AMT_ANNUITY": (0, None),
    "AMT_GOODS_PRICE": (0, None),
    "CNT_CHILDREN": (0, 30),
    "CNT_FAM_MEMBERS": (0, 40),
    "DAYS_BIRTH": (-36500, 0),          # up to 100 years old
    "DAYS_EMPLOYED": (-36500, 365243),  # 365243 = "not employed" sentinel
    "EXT_SOURCE_1": (0, 1),
    "EXT_SOURCE_2": (0, 1),
    "EXT_SOURCE_3": (0, 1),
    "REGION_POPULATION_RELATIVE": (0, 1),
}


@dataclass
class ValidationReport:
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> ValidationReport:
        if self.errors:
            raise ValueError("Data validation failed:\n  - " + "\n  - ".join(self.errors))
        return self

    def to_dict(self) -> dict:
        return {"n_rows": self.n_rows, "ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def validate_applications(df: pd.DataFrame, target: str | None = None, id_col: str | None = None,
                          max_missing: float = 0.95) -> ValidationReport:
    """Validate a frame of applications.

    Errors   = the data must not be used (missing required columns, bad target, duplicate IDs, out-of-range values).
    Warnings = worth investigating (very sparse columns, constant columns).
    """
    rep = ValidationReport(n_rows=len(df))
    if df.empty:
        rep.errors.append("dataframe is empty")
        return rep

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        rep.errors.append(f"missing required columns: {missing}")
    for c in REQUIRED_COLUMNS:
        if c in df.columns and df[c].isna().any():
            rep.errors.append(f"required column {c} has {int(df[c].isna().sum())} missing values")

    if target and target in df.columns:
        values = set(df[target].dropna().unique())
        if not values <= {0, 1}:
            rep.errors.append(f"target {target} must be binary 0/1, found {sorted(values)[:10]}")
        if df[target].isna().any():
            rep.errors.append(f"target {target} has missing values")
    elif target:
        rep.errors.append(f"target column {target} not found")

    if id_col and id_col in df.columns and df[id_col].duplicated().any():
        rep.errors.append(f"{int(df[id_col].duplicated().sum())} duplicate {id_col} values")

    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad = ((s < lo) if lo is not None else False) | ((s > hi) if hi is not None else False)
        if bad.any():
            rep.errors.append(f"{col}: {int(bad.sum())} values outside [{lo}, {hi}]")

    miss = df.isna().mean()
    sparse = miss[miss > max_missing].index.tolist()
    if sparse:
        rep.warnings.append(f"{len(sparse)} columns >{max_missing:.0%} missing: {sparse[:10]}")
    if len(df) > 1:
        constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant:
            rep.warnings.append(f"constant columns: {constant[:10]}")
    return rep
