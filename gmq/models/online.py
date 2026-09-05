"""Online learners: logistic direction model and a quantile regressor.

Online rather than batch because the market is non-stationary: a model refit
nightly spends every session trading on yesterday's relationships. These learn
continuously from resolved triple-barrier labels, with

  * AdaGrad-style per-feature learning rates -- features have wildly different
    activity (a 1-week EMA distance barely moves; order-flow imbalance moves
    every tick) and a single global rate is wrong for both
  * L2 shrinkage toward zero, so a feature that stops paying decays out
  * an explicit forgetting factor, so old regimes fade rather than dominate
  * sample weighting by label quality (an ambiguous both-barriers-hit sample
    counts for less)
"""
from __future__ import annotations

import math
from typing import Optional, Dict

import numpy as np

from ..core.types import Prediction
from ..core.mathx import sigmoid, clamp
from .base import Predictor, OnlineScaler, SkillTracker


def reliability(n_samples: int, skill: float,
                warm_at: float = 400.0, skill_ceiling: float = 0.25) -> float:
    """How much should this probability be trusted? In [0, 1].

    Confidence must NOT include how far the forecast sits from 0.5. That is a
    statement about the *size* of the edge, and it is already fully expressed
    in `p_up`; folding it in again means a well-evidenced 0.55 is reported as
    less trustworthy than a wildly-guessed 0.75, and every consumer that
    multiplies edge by confidence then squares the edge without meaning to.

    What confidence should carry is exactly what `p_up` cannot: whether the
    model has seen enough data to have an opinion, and whether its opinions
    have historically been any good.

    Getting this wrong is silent. The probabilities look fine, the search
    looks fine, and the engine simply never trades -- because every entry gate
    is comparing a reliability threshold against a number that has had the
    edge magnitude multiplied into it.
    """
    warm = clamp(n_samples / max(warm_at, 1.0), 0.0, 1.0)
    sk = clamp(skill / max(skill_ceiling, 1e-9), 0.0, 1.0)
    return float(clamp(warm * (0.30 + 0.70 * sk), 0.0, 1.0))


class OnlineLogistic(Predictor):
    """Binary classifier trained by SGD with AdaGrad and L2."""

    def __init__(self, dim: int, name: str = "logistic", lr: float = 0.02,
                 l2: float = 1e-4, forget: float = 0.99995,
                 horizon_s: float = 300.0):
        self.name = name
        self.horizon_s = horizon_s
        self.dim = dim
        self.w = np.zeros(dim)
        self.b = 0.0
        self.g2 = np.full(dim, 1e-8)
        self.gb2 = 1e-8
        self.lr = lr
        self.l2 = l2
        self.forget = forget
        self.n = 0
        self.skill = SkillTracker()

    def _grow(self, dim: int) -> None:
        if dim <= self.dim:
            return
        w = np.zeros(dim); w[: self.dim] = self.w
        g = np.full(dim, 1e-8); g[: self.dim] = self.g2
        self.w, self.g2, self.dim = w, g, dim

    def _pad(self, x: np.ndarray) -> np.ndarray:
        self._grow(x.size)
        if x.size == self.dim:
            return x
        out = np.zeros(self.dim)
        out[: x.size] = x
        return out

    def raw(self, x: np.ndarray) -> float:
        x = self._pad(x)
        return float(np.dot(self.w, x) + self.b)

    def prob(self, x: np.ndarray) -> float:
        return sigmoid(clamp(self.raw(x), -20, 20))

    def partial_fit(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        x = self._pad(x)
        z = clamp(float(np.dot(self.w, x) + self.b), -20, 20)
        p = sigmoid(z)
        err = (p - y) * weight
        g = err * x + self.l2 * self.w
        self.g2 = self.g2 * self.forget + g * g
        self.w -= self.lr * g / np.sqrt(self.g2 + 1e-12)
        self.gb2 = self.gb2 * self.forget + err * err
        self.b -= self.lr * err / math.sqrt(self.gb2 + 1e-12)
        self.n += 1

    def predict(self, x: np.ndarray, ctx: Optional[dict] = None) -> Prediction:
        p = self.prob(x)
        return Prediction(
            p_up=p, source=self.name, horizon_s=self.horizon_s,
            confidence=reliability(self.n, self.skill.skill_score()),
            features_used=int(np.count_nonzero(self.w)),
        )

    def ready(self) -> bool:
        return self.n >= 150

    def top_features(self, names, k: int = 12):
        if not names:
            return []
        m = min(len(names), self.w.size)
        idx = np.argsort(-np.abs(self.w[:m]))[:k]
        return [(names[i], round(float(self.w[i]), 4)) for i in idx]

    def state(self) -> dict:
        return {"n": self.n, "dim": self.dim, "nonzero": int(np.count_nonzero(self.w)),
                "skill": self.skill.stats()}


class OnlineRidge(Predictor):
    """Online ridge regression -- used for expected return and holding time."""

    def __init__(self, dim: int, name: str = "ridge", lr: float = 0.02,
                 l2: float = 1e-4, horizon_s: float = 300.0):
        self.name = name
        self.horizon_s = horizon_s
        self.dim = dim
        self.w = np.zeros(dim)
        self.b = 0.0
        self.g2 = np.full(dim, 1e-8)
        self.gb2 = 1e-8
        self.lr = lr
        self.l2 = l2
        self.n = 0
        self.resid_var = 1e-6
        self.y_mean = 0.0

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if x.size > self.dim:
            w = np.zeros(x.size); w[: self.dim] = self.w
            g = np.full(x.size, 1e-8); g[: self.dim] = self.g2
            self.w, self.g2, self.dim = w, g, x.size
        if x.size == self.dim:
            return x
        out = np.zeros(self.dim)
        out[: x.size] = x
        return out

    def value(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, self._pad(x)) + self.b)

    def partial_fit(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        x = self._pad(x)
        pred = float(np.dot(self.w, x) + self.b)
        err = (pred - y) * weight
        g = err * x + self.l2 * self.w
        self.g2 += g * g
        self.w -= self.lr * g / np.sqrt(self.g2 + 1e-12)
        self.gb2 += err * err
        self.b -= self.lr * err / math.sqrt(self.gb2 + 1e-12)
        self.n += 1
        a = 1.0 / min(self.n, 500)
        self.y_mean += a * (y - self.y_mean)
        self.resid_var += a * ((y - pred) ** 2 - self.resid_var)

    def predict(self, x: np.ndarray, ctx: Optional[dict] = None) -> Prediction:
        v = self.value(x)
        return Prediction(exp_return=v, exp_vol=math.sqrt(max(self.resid_var, 0.0)),
                          source=self.name, horizon_s=self.horizon_s,
                          confidence=clamp(self.n / 400.0, 0, 1))

    def ready(self) -> bool:
        return self.n >= 150

    def state(self) -> dict:
        return {"n": self.n, "resid_sd": math.sqrt(max(self.resid_var, 0.0))}


class OnlineQuantile(Predictor):
    """Pinball-loss regression for a chosen quantile.

    Used for expected adverse excursion: the median MAE is not the number that
    matters when sizing a stop -- the 80th percentile is, because that is the
    excursion you actually have to survive.
    """

    def __init__(self, dim: int, q: float = 0.8, name: str = "quantile",
                 lr: float = 0.02, l2: float = 1e-4):
        self.name = f"{name}_q{int(q*100)}"
        self.q = q
        self.dim = dim
        self.w = np.zeros(dim)
        self.b = 0.0
        self.g2 = np.full(dim, 1e-8)
        self.lr = lr
        self.l2 = l2
        self.n = 0

    def _pad(self, x):
        if x.size > self.dim:
            w = np.zeros(x.size); w[: self.dim] = self.w
            g = np.full(x.size, 1e-8); g[: self.dim] = self.g2
            self.w, self.g2, self.dim = w, g, x.size
        if x.size == self.dim:
            return x
        out = np.zeros(self.dim); out[: x.size] = x
        return out

    def value(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, self._pad(x)) + self.b)

    def partial_fit(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        x = self._pad(x)
        pred = float(np.dot(self.w, x) + self.b)
        # subgradient of the pinball loss
        grad = -(self.q if y > pred else (self.q - 1.0)) * weight
        g = grad * x + self.l2 * self.w
        self.g2 += g * g
        self.w -= self.lr * g / np.sqrt(self.g2 + 1e-12)
        self.b -= self.lr * grad * 0.1
        self.n += 1

    def predict(self, x: np.ndarray, ctx: Optional[dict] = None) -> Prediction:
        return Prediction(exp_return=self.value(x), source=self.name)

    def ready(self) -> bool:
        return self.n >= 150
