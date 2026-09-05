"""Incremental / streaming maths primitives.

Every statistic the live system needs is computed in O(1) per update from a
fixed-size ring buffer. Nothing here recomputes over a growing window, which
is what keeps the tick->decision path inside its latency budget.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Iterable, Tuple

import numpy as np


class Ring:
    """Fixed-capacity float ring buffer with O(1) push and vectorised views."""

    __slots__ = ("buf", "n", "i", "cap", "_full")

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.buf = np.zeros(self.cap, dtype=np.float64)
        self.n = 0
        self.i = 0
        self._full = False

    def push(self, x: float) -> None:
        self.buf[self.i] = x
        self.i = (self.i + 1) % self.cap
        if self.n < self.cap:
            self.n += 1
        else:
            self._full = True

    def view(self) -> np.ndarray:
        """Chronological view, oldest first."""
        if self.n < self.cap:
            return self.buf[: self.n]
        return np.concatenate((self.buf[self.i:], self.buf[: self.i]))

    def last(self, k: int = 1) -> np.ndarray:
        v = self.view()
        return v[-k:] if k <= v.size else v

    @property
    def full(self) -> bool:
        return self.n >= self.cap

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return self.view()[idx]


class EWMA:
    """Exponentially weighted mean with bias correction."""

    __slots__ = ("alpha", "value", "_n", "_corr")

    def __init__(self, halflife: float = 20.0, alpha: Optional[float] = None):
        self.alpha = alpha if alpha is not None else 1.0 - 0.5 ** (1.0 / max(halflife, 1e-9))
        self.value = 0.0
        self._n = 0
        self._corr = 0.0

    def update(self, x: float) -> float:
        self._n += 1
        self.value += self.alpha * (x - self.value)
        self._corr = 1.0 - (1.0 - self.alpha) ** self._n
        return self.get()

    def get(self) -> float:
        return self.value / self._corr if self._corr > 0 else 0.0

    @property
    def ready(self) -> bool:
        return self._n >= 3


class EWVar:
    """Exponentially weighted mean + variance (Welford-style, online)."""

    __slots__ = ("alpha", "mean", "var", "_n")

    def __init__(self, halflife: float = 60.0):
        self.alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1e-9))
        self.mean = 0.0
        self.var = 0.0
        self._n = 0

    def update(self, x: float) -> None:
        self._n += 1
        if self._n == 1:
            self.mean = x
            return
        d = x - self.mean
        incr = self.alpha * d
        self.mean += incr
        self.var = (1.0 - self.alpha) * (self.var + d * incr)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    @property
    def ready(self) -> bool:
        return self._n >= 5

    def z(self, x: float) -> float:
        s = self.std
        return (x - self.mean) / s if s > 1e-12 else 0.0


class RollingQuantile:
    """Approximate streaming quantiles (P2-style, cheap and good enough for
    regime thresholds where exactness buys nothing)."""

    __slots__ = ("q", "est", "_n", "step")

    def __init__(self, q: float = 0.5, step: float = 0.01):
        self.q = q
        self.est = 0.0
        self._n = 0
        self.step = step

    def update(self, x: float) -> float:
        self._n += 1
        if self._n == 1:
            self.est = x
            return self.est
        # adaptive step tied to observed scale
        s = max(abs(self.est), 1e-9) * self.step
        if x > self.est:
            self.est += s * self.q
        elif x < self.est:
            self.est -= s * (1.0 - self.q)
        return self.est


class OnlineCorr:
    """Pairwise EW correlation, O(1) per update."""

    __slots__ = ("alpha", "mx", "my", "vx", "vy", "cxy", "_n")

    def __init__(self, halflife: float = 240.0):
        self.alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1e-9))
        self.mx = self.my = 0.0
        self.vx = self.vy = self.cxy = 0.0
        self._n = 0

    def update(self, x: float, y: float) -> float:
        self._n += 1
        a = self.alpha
        if self._n == 1:
            self.mx, self.my = x, y
            return 0.0
        dx, dy = x - self.mx, y - self.my
        self.mx += a * dx
        self.my += a * dy
        self.vx = (1 - a) * (self.vx + a * dx * dx)
        self.vy = (1 - a) * (self.vy + a * dy * dy)
        self.cxy = (1 - a) * (self.cxy + a * dx * dy)
        return self.rho

    @property
    def rho(self) -> float:
        d = math.sqrt(self.vx * self.vy)
        if d < 1e-16:
            return 0.0
        return max(-1.0, min(1.0, self.cxy / d))

    @property
    def ready(self) -> bool:
        return self._n >= 20


# --------------------------------------------------------------------------
# free functions
# --------------------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = clamp(p, 1e-9, 1 - 1e-9)
    return math.log(p / (1 - p))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-15 else default


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-15 else np.zeros_like(x)


def linreg_slope(y: np.ndarray) -> Tuple[float, float]:
    """OLS slope + R^2 against an integer index. Returns (slope, r2)."""
    n = y.size
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    sxx = float((dx * dx).sum())
    if sxx < 1e-15:
        return 0.0, 0.0
    slope = float((dx * dy).sum() / sxx)
    ss_tot = float((dy * dy).sum())
    if ss_tot < 1e-15:
        return slope, 0.0
    resid = dy - slope * dx
    r2 = 1.0 - float((resid * resid).sum()) / ss_tot
    return slope, max(0.0, r2)


def hurst(series: np.ndarray, max_lag: int = 40) -> float:
    """Rescaled-range-ish Hurst exponent via variance of lagged differences.

    H > 0.5 -> trending / persistent;  H < 0.5 -> mean reverting.
    Used by the regime detector, never on its own.
    """
    n = series.size
    if n < 40:
        return 0.5
    max_lag = min(max_lag, n // 3)
    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        d = series[lag:] - series[:-lag]
        sd = d.std()
        tau.append(sd if sd > 1e-12 else 1e-12)
    tau = np.asarray(tau)
    try:
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(clamp(m[0], 0.0, 1.0))
    except Exception:                                  # noqa: BLE001
        return 0.5


def half_life_ou(series: np.ndarray) -> float:
    """Ornstein-Uhlenbeck mean-reversion half-life in bars. inf == no MR."""
    if series.size < 30:
        return float("inf")
    y = series[1:]
    x = series[:-1]
    dy = y - x
    xm = x - x.mean()
    denom = float((xm * xm).sum())
    if denom < 1e-15:
        return float("inf")
    beta = float((xm * (dy - dy.mean())).sum() / denom)
    if beta >= -1e-9:
        return float("inf")
    return float(-math.log(2.0) / beta)


def max_drawdown(equity: np.ndarray) -> Tuple[float, int, int]:
    """Returns (max dd as positive fraction, peak idx, trough idx)."""
    if equity.size < 2:
        return 0.0, 0, 0
    running = np.maximum.accumulate(equity)
    dd = np.where(running > 0, (running - equity) / running, 0.0)
    t = int(dd.argmax())
    p = int(equity[: t + 1].argmax()) if t > 0 else 0
    return float(dd[t]), p, t


def var_es(returns: np.ndarray, alpha: float = 0.99) -> Tuple[float, float]:
    """Historical VaR and Expected Shortfall as POSITIVE loss fractions."""
    if returns.size < 20:
        s = float(returns.std()) if returns.size else 0.0
        z = 2.326 if alpha >= 0.99 else 1.645
        return z * s, z * s * 1.15
    q = float(np.quantile(returns, 1.0 - alpha))
    tail = returns[returns <= q]
    es = float(tail.mean()) if tail.size else q
    return -q, -es


def sharpe(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    if returns.size < 2:
        return 0.0
    s = returns.std(ddof=1)
    if s < 1e-15:
        return 0.0
    return float(returns.mean() / s * math.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: float = 252.0,
            mar: float = 0.0) -> float:
    if returns.size < 2:
        return 0.0
    downside = returns[returns < mar] - mar
    if downside.size == 0:
        return float("inf") if returns.mean() > 0 else 0.0
    dd = math.sqrt(float((downside ** 2).mean()))
    if dd < 1e-15:
        return 0.0
    return float((returns.mean() - mar) / dd * math.sqrt(periods_per_year))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts. Lower is better; 0.25 is
    the score of an uninformative 50/50 forecaster."""
    if probs.size == 0:
        return 0.25
    return float(((probs - outcomes) ** 2).mean())


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    if probs.size == 0:
        return math.log(2.0)
    p = np.clip(probs, 1e-9, 1 - 1e-9)
    return float(-(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)).mean())


def kelly_fraction(p_win: float, win_r: float, loss_r: float = 1.0) -> float:
    """Kelly for an asymmetric binary bet expressed in R multiples."""
    if win_r <= 0 or loss_r <= 0:
        return 0.0
    b = win_r / loss_r
    f = (p_win * (b + 1.0) - 1.0) / b
    return clamp(f, 0.0, 1.0)


def bootstrap_paths(returns: np.ndarray, n_paths: int, horizon: int,
                    rng: np.random.Generator, block: int = 5) -> np.ndarray:
    """Stationary block bootstrap -- preserves short-range autocorrelation,
    which an iid bootstrap destroys and which matters a lot for path-dependent
    quantities like 'did we touch the stop'."""
    n = returns.size
    if n < block + 2:
        return rng.normal(0, max(returns.std(), 1e-6), (n_paths, horizon))
    out = np.empty((n_paths, horizon), dtype=np.float64)
    n_blocks = int(math.ceil(horizon / block))
    starts = rng.integers(0, n - block, size=(n_paths, n_blocks))
    for i in range(n_paths):
        chunks = [returns[s: s + block] for s in starts[i]]
        out[i] = np.concatenate(chunks)[:horizon]
    return out
