"""Probability calibration for ElJefe (plan §10).

Wraps isotonic regression, Platt scaling, and temperature scaling behind one
picklable interface. Fit on the validation split only; freeze before test.
"""

from __future__ import annotations

import pickle
from typing import Any, Optional

import numpy as np

_EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _bce(p: np.ndarray, y: np.ndarray) -> float:
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class _TemperatureScaler:
    """Single-temperature scaling on logits; T fit by BCE grid search."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_TemperatureScaler":
        logits = _logit(p)
        y = np.asarray(y, dtype=float)
        # Coarse then fine grid over T in [0.1, 10].
        best_t, best_loss = 1.0, np.inf
        for grid in (np.linspace(0.1, 10.0, 200),):
            for t in grid:
                loss = _bce(_sigmoid(logits / t), y)
                if loss < best_loss:
                    best_loss, best_t = loss, float(t)
        lo, hi = max(0.05, best_t * 0.5), best_t * 1.5
        for t in np.linspace(lo, hi, 200):
            loss = _bce(_sigmoid(logits / t), y)
            if loss < best_loss:
                best_loss, best_t = loss, float(t)
        self.temperature = best_t
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(p) / self.temperature)


class Calibrator:
    """method in {'isotonic', 'platt', 'temperature'}; fit/predict/save/load."""

    METHODS = ("isotonic", "platt", "temperature")

    def __init__(self, method: str = "isotonic") -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        self.method = method
        self._model: Optional[Any] = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        p = _clip(np.asarray(p, dtype=float))
        y = np.asarray(y, dtype=float)
        if p.shape[0] != y.shape[0]:
            raise ValueError("p and y must have the same length")
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(p, y)
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression

            self._model = LogisticRegression(C=1e10, solver="lbfgs").fit(
                _logit(p).reshape(-1, 1), y
            )
        else:
            self._model = _TemperatureScaler().fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator is not fitted")
        p = _clip(np.asarray(p, dtype=float))
        if self.method == "isotonic":
            out = self._model.predict(p)
        elif self.method == "platt":
            out = self._model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        else:
            out = self._model.predict(p)
        return _clip(np.asarray(out, dtype=float))

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"method": self.method, "model": self._model}, f)

    @classmethod
    def load(cls, path) -> "Calibrator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        cal = cls(state["method"])
        cal._model = state["model"]
        return cal
