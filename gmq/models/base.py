"""Model interfaces and the labelling machinery (spec §3, §8).

The single most important thing in this file is `TripleBarrierLabeler`, and
the single most important property of that class is that **a training sample
is not emitted until its outcome is fully determined**.

That sounds obvious. It is also the most common way a backtest lies. If you
label a bar with "did price rise over the next 15 minutes" and hand that
sample to an online learner at the moment the bar closes, the learner is being
told the future. Accuracy looks superb, live performance is random. Here,
every pending label sits in a queue and is only released -- with the feature
snapshot taken at decision time -- once the barrier has actually been touched
or the horizon has actually elapsed.

The `purge` and `embargo` parameters exist for the same reason at the
cross-validation level: overlapping label windows leak information between
adjacent train and test folds unless the boundary is cut out.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Any

import numpy as np

from ..core.types import Prediction, NS
from ..core.mathx import clamp, brier, log_loss


class Predictor:
    """Every prediction model implements this."""

    name: str = "base"
    #: horizon this model is trained for, seconds
    horizon_s: float = 300.0

    def predict(self, x: np.ndarray, ctx: Optional[dict] = None) -> Prediction:
        raise NotImplementedError

    def partial_fit(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        raise NotImplementedError

    def ready(self) -> bool:
        return True

    def state(self) -> dict:
        return {}


@dataclass
class Sample:
    """One (features, label) pair, released only once the label is real."""
    x: np.ndarray
    ts: int
    symbol: str
    entry_px: float
    horizon_end_ns: int
    up_barrier: float
    dn_barrier: float
    # filled in on resolution
    y_dir: float = 0.0            # 1 if up barrier hit first / price up
    y_target: float = 0.0         # 1 if the up barrier was touched first
    y_stop: float = 0.0           # 1 if the down barrier was touched first
    hold_s: float = 0.0
    ret: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    resolved: bool = False
    regime: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class TripleBarrierLabeler:
    """Path-aware labelling: which barrier is touched first, and when.

    A plain "return over N minutes" label ignores that a trade with a stop
    would have been closed long before N minutes elapsed. The triple barrier
    (profit target, stop, time limit) labels the outcome the strategy would
    actually have experienced, which is the only outcome worth predicting.
    """

    def __init__(self, up_atr: float = 1.6, dn_atr: float = 1.0,
                 horizon_s: float = 900.0, max_pending: int = 20000):
        self.up_atr = up_atr
        self.dn_atr = dn_atr
        self.horizon_s = horizon_s
        self.pending: Deque[Sample] = deque()
        self.max_pending = max_pending
        self.released: int = 0
        self.expired: int = 0

    def observe(self, x: np.ndarray, ts: int, symbol: str, px: float,
                atr: float, regime: str = "", meta: Optional[dict] = None
                ) -> None:
        """Register a new, as-yet-unlabelled observation."""
        if atr <= 0 or px <= 0:
            return
        s = Sample(
            x=x.copy(), ts=ts, symbol=symbol, entry_px=px,
            horizon_end_ns=ts + int(self.horizon_s * NS),
            up_barrier=px + self.up_atr * atr,
            dn_barrier=px - self.dn_atr * atr,
            regime=regime, meta=meta or {},
        )
        s.mfe = s.mae = 0.0
        self.pending.append(s)
        while len(self.pending) > self.max_pending:
            self.pending.popleft()
            self.expired += 1

    def update(self, symbol: str, ts: int, px: float,
               high: Optional[float] = None,
               low: Optional[float] = None) -> List[Sample]:
        """Advance every pending sample for `symbol` with a new observation.

        Returns the samples that resolved on this update -- and only those.

        `high`/`low` are for bar-shaped feeds. A bar hides its own path, so a
        bar whose range spans *both* barriers is genuinely ambiguous: we
        cannot know which was touched first, and picking one manufactures
        precision the data does not contain. Tick feeds pass price alone,
        where the question does not arise because each observation is a
        single point.
        """
        done: List[Sample] = []
        keep: Deque[Sample] = deque()
        hi = px if high is None else max(high, px)
        lo = px if low is None else min(low, px)
        for s in self.pending:
            if s.symbol != symbol:
                keep.append(s)
                continue
            if hi - s.entry_px > s.mfe:
                s.mfe = hi - s.entry_px
            if lo - s.entry_px < s.mae:
                s.mae = lo - s.entry_px
            hit_up = hi >= s.up_barrier
            hit_dn = lo <= s.dn_barrier
            timeout = ts >= s.horizon_end_ns
            if hit_up or hit_dn or timeout:
                s.hold_s = (ts - s.ts) / NS
                s.ret = (px - s.entry_px) / s.entry_px
                if hit_up and not hit_dn:
                    s.y_target, s.y_stop, s.y_dir = 1.0, 0.0, 1.0
                elif hit_dn and not hit_up:
                    s.y_target, s.y_stop, s.y_dir = 0.0, 1.0, 0.0
                elif hit_up and hit_dn:
                    # One bar's range spanned both barriers. Which came first
                    # is not recoverable from a bar, so the honest label is
                    # the ambiguous one -- and `HorizonModels.learn` halves
                    # its training weight rather than dropping it, because
                    # dropping these would quietly bias the training set
                    # toward calm periods.
                    s.y_target = s.y_stop = 0.5
                    s.y_dir = 1.0 if s.ret > 0 else 0.0
                else:
                    s.y_target, s.y_stop = 0.0, 0.0
                    s.y_dir = 1.0 if s.ret > 0 else 0.0
                s.resolved = True
                done.append(s)
                self.released += 1
            else:
                keep.append(s)
        self.pending = keep
        return done

    def pending_count(self, symbol: Optional[str] = None) -> int:
        if symbol is None:
            return len(self.pending)
        return sum(1 for s in self.pending if s.symbol == symbol)


class OnlineScaler:
    """Streaming standardisation of the feature matrix.

    Fitted incrementally on the same stream the model sees, so there is no
    scaler fitted on the full dataset -- which would itself be lookahead.
    """

    __slots__ = ("mean", "var", "n", "dim", "alpha", "clip")

    def __init__(self, dim: int, halflife: float = 4000.0, clip: float = 6.0):
        self.dim = dim
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.n = 0
        self.alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1.0))
        self.clip = clip

    def _grow(self, dim: int) -> None:
        if dim <= self.dim:
            return
        m = np.zeros(dim); m[: self.dim] = self.mean
        v = np.ones(dim); v[: self.dim] = self.var
        self.mean, self.var, self.dim = m, v, dim

    def partial_fit(self, x: np.ndarray) -> None:
        self._grow(x.size)
        x = self._pad(x)
        self.n += 1
        a = self.alpha if self.n > 30 else 1.0 / self.n
        d = x - self.mean
        self.mean += a * d
        self.var = (1 - a) * (self.var + a * d * d)

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if x.size == self.dim:
            return x
        out = np.zeros(self.dim)
        k = min(x.size, self.dim)
        out[:k] = x[:k]
        return out

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._grow(x.size)
        x = self._pad(x)
        s = np.sqrt(np.maximum(self.var, 1e-12))
        return np.clip((x - self.mean) / s, -self.clip, self.clip)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.partial_fit(x)
        return self.transform(x)

    @property
    def ready(self) -> bool:
        return self.n >= 50


class SkillTracker:
    """Rolling forecast-quality metrics for one model (spec §17).

    Brier score is the headline because it is a *proper* scoring rule: it is
    minimised only by reporting your true belief, so a model cannot improve it
    by shading its probabilities toward whatever pays off. Accuracy is not
    proper and rewards overconfidence, which is why it is reported but never
    used for weighting.
    """

    def __init__(self, window: int = 500):
        self.window = window
        self.p: Deque[float] = deque(maxlen=window)
        self.y: Deque[float] = deque(maxlen=window)
        self.n_total = 0
        self.baseline_brier: Optional[float] = None

    def add(self, p: float, y: float) -> None:
        self.p.append(float(clamp(p, 1e-6, 1 - 1e-6)))
        self.y.append(float(y))
        self.n_total += 1

    @property
    def ready(self) -> bool:
        return len(self.p) >= 60

    def base_rate(self) -> float:
        """The observed frequency of the positive label -- the value a
        skill-less forecast should collapse to."""
        if not self.y:
            return 0.5
        return float(np.mean(self.y))

    def brier(self) -> float:
        if not self.p:
            return 0.25
        return brier(np.asarray(self.p), np.asarray(self.y))

    def log_loss(self) -> float:
        if not self.p:
            return math.log(2)
        return log_loss(np.asarray(self.p), np.asarray(self.y))

    def accuracy(self) -> float:
        if not self.p:
            return 0.5
        p = np.asarray(self.p)
        y = np.asarray(self.y)
        return float(((p > 0.5) == (y > 0.5)).mean())

    def skill_score(self) -> float:
        """Brier skill score against the climatological base rate.

        Positive means the model beats simply predicting the historical
        frequency. Zero or negative means it adds nothing, no matter how good
        its raw accuracy looks -- a model that says 'up' 90% of the time in a
        market that rises 90% of the time has 90% accuracy and no skill.
        """
        if not self.ready:
            return 0.0
        y = np.asarray(self.y)
        base = float(y.mean())
        ref = float(((base - y) ** 2).mean())
        if ref < 1e-9:
            return 0.0
        return float(clamp(1.0 - self.brier() / ref, -1.0, 1.0))

    def calibration(self, bins: int = 10) -> List[Tuple[float, float, int]]:
        """Reliability curve: (mean predicted, observed frequency, count)."""
        if not self.ready:
            return []
        p = np.asarray(self.p)
        y = np.asarray(self.y)
        edges = np.linspace(0, 1, bins + 1)
        out = []
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1
                                   else p <= edges[i + 1])
            if m.sum() >= 3:
                out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
        return out

    def stats(self) -> dict:
        return {
            "n": self.n_total,
            "n_window": len(self.p),
            "brier": round(self.brier(), 4),
            "log_loss": round(self.log_loss(), 4),
            "accuracy": round(self.accuracy(), 4),
            "skill": round(self.skill_score(), 4),
            "ready": self.ready,
        }
