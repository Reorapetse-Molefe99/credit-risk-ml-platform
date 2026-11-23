"""Probability calibration of an already-fitted classifier on held-out data.

The base model is fitted on the training split. The calibrator is then fitted on the validation split, so
the test split stays untouched. It replaces the final step of the model pipeline, so the logged MLflow model
returns calibrated PDs everywhere: API, batch scoring and dashboard.

- ``sigmoid``: logistic regression on the base model's log-odds (Platt scaling). Monotonic, so AUC is unchanged.
- ``isotonic``: non-parametric, monotonic step function. Needs plenty of data and can create ties.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODS = ("sigmoid", "isotonic")
_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class PrefitCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a fitted binary classifier and calibrates its P(1). `fit` only fits the calibrator."""

    def __init__(self, estimator=None, method: str = "sigmoid"):
        self.estimator = estimator
        self.method = method

    def fit(self, X, y):
        if self.method not in METHODS:
            raise ValueError(f"Unknown calibration method '{self.method}'. Choose from {METHODS}")
        p = self.estimator.predict_proba(X)[:, 1]
        y = np.asarray(y)
        if self.method == "sigmoid":
            self.calibrator_ = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        else:
            self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.estimator.predict_proba(X)[:, 1]
        if self.method == "sigmoid":
            q = self.calibrator_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        else:
            q = self.calibrator_.predict(p)
        q = np.clip(q, 0.0, 1.0)
        return np.column_stack([1 - q, q])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def base_estimator(est):
    """The underlying model, unwrapping a calibration layer if there is one."""
    return est.estimator if isinstance(est, PrefitCalibratedClassifier) else est
