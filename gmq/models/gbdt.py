"""Gradient-boosted trees, retrained periodically on a purged replay buffer.

Complements the online linear models rather than replacing them. Trees find
interactions the linear model cannot (order-flow imbalance matters only when
the spread is tight; a breakout signal matters only when volume confirms),
but they cannot update per-sample, so they are always slightly stale. Running
both and letting the ensemble weight them by measured skill is the point --
neither is trusted a priori.

Retraining honours the same purge/embargo discipline as the cross-validator:
the most recent `embargo` samples are withheld because their labels overlap
the period the model will immediately be scored on.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Optional, Tuple, List

import numpy as np

from ..core.types import Prediction
from ..core.mathx import clamp
from .base import Predictor, SkillTracker

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HAVE_SK = True
except Exception:                                       # noqa: BLE001
    _HAVE_SK = False


class GBDTClassifier(Predictor):
    def __init__(self, name: str = "gbdt", horizon_s: float = 300.0,
                 buffer_size: int = 12000, retrain_every: int = 1500,
                 min_samples: int = 800, embargo: int = 120,
                 max_iter: int = 140, max_depth: int = 5,
                 learning_rate: float = 0.06, max_train_s: float = 4.0):
        self.name = name
        self.horizon_s = horizon_s
        self.X: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.y: Deque[float] = deque(maxlen=buffer_size)
        self.w: Deque[float] = deque(maxlen=buffer_size)
        self.model = None
        self.retrain_every = retrain_every
        self.min_samples = min_samples
        self.embargo = embargo
        self.since_fit = 0
        self.n = 0
        self.fits = 0
        self.last_fit_s = 0.0
        self.skill = SkillTracker()
        self.available = _HAVE_SK
        self._params = dict(max_iter=max_iter, max_depth=max_depth,
                            learning_rate=learning_rate,
                            l2_regularization=1.0, min_samples_leaf=25,
                            early_stopping=False)
        self.max_train_s = max_train_s
        self.dim = 0
        self.oos_brier: Optional[float] = None

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if self.dim == 0:
            self.dim = x.size
        if x.size == self.dim:
            return x
        out = np.zeros(self.dim)
        k = min(x.size, self.dim)
        out[:k] = x[:k]
        return out

    def partial_fit(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        self.X.append(self._pad(x).copy())
        self.y.append(float(y))
        self.w.append(float(weight))
        self.n += 1
        self.since_fit += 1
        if (self.available and self.since_fit >= self.retrain_every
                and len(self.X) >= self.min_samples):
            self.fit()

    def fit(self) -> bool:
        if not self.available or len(self.X) < self.min_samples:
            return False
        # Withhold the tail: those labels overlap the window the model is
        # about to be judged on, and training on them is leakage.
        n = len(self.X) - self.embargo
        if n < self.min_samples:
            return False
        X = np.asarray(list(self.X)[:n])
        y = np.asarray(list(self.y)[:n])
        w = np.asarray(list(self.w)[:n])
        yb = (y > 0.5).astype(int)
        if yb.min() == yb.max():
            return False                      # single-class window
        # honest holdout: last 20% chronologically, never shuffled
        cut = int(n * 0.8)
        t0 = time.perf_counter()
        try:
            m = HistGradientBoostingClassifier(**self._params)
            m.fit(X[:cut], yb[:cut], sample_weight=w[:cut])
            if cut < n - 5:
                p = m.predict_proba(X[cut:])[:, 1]
                self.oos_brier = float(((p - yb[cut:]) ** 2).mean())
            # refit on everything up to the embargo for deployment
            m2 = HistGradientBoostingClassifier(**self._params)
            m2.fit(X, yb, sample_weight=w)
            self.model = m2
        except Exception:                              # noqa: BLE001
            return False
        self.last_fit_s = time.perf_counter() - t0
        self.fits += 1
        self.since_fit = 0
        return True

    def prob(self, x: np.ndarray) -> float:
        if self.model is None:
            return 0.5
        try:
            return float(self.model.predict_proba(self._pad(x).reshape(1, -1))[0, 1])
        except Exception:                              # noqa: BLE001
            return 0.5

    def predict(self, x: np.ndarray, ctx: Optional[dict] = None) -> Prediction:
        from .online import reliability
        p = self.prob(x)
        conf = reliability(self.n, self.skill.skill_score()) \
            if self.model is not None else 0.0
        return Prediction(p_up=p, source=self.name, horizon_s=self.horizon_s,
                          confidence=conf)

    def ready(self) -> bool:
        return self.model is not None

    def state(self) -> dict:
        return {"n": self.n, "fits": self.fits, "buffer": len(self.X),
                "last_fit_s": round(self.last_fit_s, 3),
                "oos_brier": self.oos_brier,
                "available": self.available,
                "skill": self.skill.stats()}


class GBDTRegressor(Predictor):
    """Same discipline, continuous target (expected return / holding time)."""

    def __init__(self, name: str = "gbdt_reg", horizon_s: float = 300.0,
                 buffer_size: int = 12000, retrain_every: int = 1500,
                 min_samples: int = 800, embargo: int = 120):
        self.name = name
        self.horizon_s = horizon_s
        self.X: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.y: Deque[float] = deque(maxlen=buffer_size)
        self.model = None
        self.retrain_every = retrain_every
        self.min_samples = min_samples
        self.embargo = embargo
        self.since_fit = 0
        self.n = 0
        self.fits = 0
        self.dim = 0
        self.available = _HAVE_SK
        self.resid_sd = 0.0

    def _pad(self, x):
        if self.dim == 0:
            self.dim = x.size
        if x.size == self.dim:
            return x
        out = np.zeros(self.dim); k = min(x.size, self.dim); out[:k] = x[:k]
        return out

    def partial_fit(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        self.X.append(self._pad(x).copy())
        self.y.append(float(y))
        self.n += 1
        self.since_fit += 1
        if (self.available and self.since_fit >= self.retrain_every
                and len(self.X) >= self.min_samples):
            self.fit()

    def fit(self) -> bool:
        if not self.available or len(self.X) < self.min_samples:
            return False
        n = len(self.X) - self.embargo
        if n < self.min_samples:
            return False
        X = np.asarray(list(self.X)[:n])
        y = np.asarray(list(self.y)[:n])
        try:
            m = HistGradientBoostingRegressor(
                max_iter=120, max_depth=5, learning_rate=0.06,
                l2_regularization=1.0, min_samples_leaf=25, early_stopping=False)
            m.fit(X, y)
            self.model = m
            pred = m.predict(X[-min(1000, n):])
            self.resid_sd = float((y[-min(1000, n):] - pred).std())
        except Exception:                              # noqa: BLE001
            return False
        self.fits += 1
        self.since_fit = 0
        return True

    def value(self, x: np.ndarray) -> float:
        if self.model is None:
            return 0.0
        try:
            return float(self.model.predict(self._pad(x).reshape(1, -1))[0])
        except Exception:                              # noqa: BLE001
            return 0.0

    def predict(self, x: np.ndarray, ctx: Optional[dict] = None) -> Prediction:
        return Prediction(exp_return=self.value(x), exp_vol=self.resid_sd,
                          source=self.name, horizon_s=self.horizon_s,
                          confidence=1.0 if self.model is not None else 0.0)

    def ready(self) -> bool:
        return self.model is not None

    def state(self) -> dict:
        return {"n": self.n, "fits": self.fits, "resid_sd": self.resid_sd,
                "available": self.available}
