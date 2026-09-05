"""Probability calibration and the skill-weighted ensemble (spec §16, §17).

Two ideas do most of the work here.

**Calibration.** A model's raw output is a score, not a probability. If the
model says 0.70 and the event happens 55% of the time, then every expected
value the search engine computes downstream is wrong by 15 points -- and the
position sizer, which is fed those probabilities, will systematically bet too
big. Isotonic regression on a rolling window maps scores back onto observed
frequencies, and the reliability curve is exposed so the miscalibration is
visible rather than assumed away.

**Weighting by demonstrated skill, not by accuracy.** Weights come from the
Brier skill score against the climatological base rate. A model that is 88%
accurate in a market that rises 88% of the time has no skill and gets no
weight, which is exactly right and is not what an accuracy-weighted ensemble
would do.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Prediction
from ..core.mathx import clamp, logit, sigmoid
from .base import Predictor, SkillTracker


class IsotonicCalibrator:
    """Rolling isotonic calibration with a linear-interpolation lookup."""

    def __init__(self, window: int = 1500, min_samples: int = 200,
                 refit_every: int = 150):
        self.p: Deque[float] = deque(maxlen=window)
        self.y: Deque[float] = deque(maxlen=window)
        self.min_samples = min_samples
        self.refit_every = refit_every
        self.since = 0
        self._xs: Optional[np.ndarray] = None
        self._ys: Optional[np.ndarray] = None
        self.fits = 0

    def add(self, p: float, y: float) -> None:
        self.p.append(float(p))
        self.y.append(float(y))
        self.since += 1
        if self.since >= self.refit_every and len(self.p) >= self.min_samples:
            self.fit()

    def fit(self) -> bool:
        n = len(self.p)
        if n < self.min_samples:
            return False
        p = np.asarray(self.p)
        y = np.asarray(self.y)
        order = np.argsort(p)
        ps, ys = p[order], y[order]
        # pool-adjacent-violators
        vals = ys.astype(float).copy()
        wts = np.ones(n)
        idx = list(range(n))
        i = 0
        blocks: List[Tuple[float, float, int, int]] = []  # (val, wt, lo, hi)
        for k in range(n):
            blocks.append((vals[k], wts[k], k, k))
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                v2, w2, lo2, hi2 = blocks.pop()
                v1, w1, lo1, hi1 = blocks.pop()
                w = w1 + w2
                blocks.append(((v1 * w1 + v2 * w2) / w, w, lo1, hi2))
        xs, ys_out = [], []
        for v, w, lo, hi in blocks:
            xs.append(float(ps[lo]))
            ys_out.append(float(v))
            if hi != lo:
                xs.append(float(ps[hi]))
                ys_out.append(float(v))
        self._xs = np.asarray(xs)
        self._ys = np.clip(np.asarray(ys_out), 1e-4, 1 - 1e-4)
        self.since = 0
        self.fits += 1
        return True

    def transform(self, p: float) -> float:
        if self._xs is None or self._xs.size < 2:
            return float(clamp(p, 1e-4, 1 - 1e-4))
        return float(clamp(float(np.interp(p, self._xs, self._ys)),
                           1e-4, 1 - 1e-4))

    @property
    def ready(self) -> bool:
        return self._xs is not None

    def reliability(self, bins: int = 10):
        if len(self.p) < 60:
            return []
        p = np.asarray(self.p); y = np.asarray(self.y)
        edges = np.linspace(0, 1, bins + 1)
        out = []
        for i in range(bins):
            hi = p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1]
            m = (p >= edges[i]) & hi
            if m.sum() >= 3:
                out.append({"pred": round(float(p[m].mean()), 3),
                            "obs": round(float(y[m].mean()), 3),
                            "n": int(m.sum())})
        return out


class SkillWeightedEnsemble(Predictor):
    """Combines member classifiers in log-odds space, weighted by skill."""

    def __init__(self, members: List[Predictor], name: str = "ensemble",
                 horizon_s: float = 300.0, min_weight: float = 0.02,
                 calibrate: bool = True):
        self.name = name
        self.horizon_s = horizon_s
        self.members = members
        self.min_weight = min_weight
        self.trackers: Dict[str, SkillTracker] = {
            m.name: getattr(m, "skill", None) or SkillTracker()
            for m in members
        }
        for m in members:
            if not hasattr(m, "skill"):
                m.skill = self.trackers[m.name]           # type: ignore[attr-defined]
        self.cal = IsotonicCalibrator() if calibrate else None
        self.skill = SkillTracker()
        self.n = 0
        self.last_weights: Dict[str, float] = {}
        self.suspended: Dict[str, bool] = {m.name: False for m in members}

    # ------------------------------------------------------------------
    def weights(self) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        for m in self.members:
            if self.suspended.get(m.name):
                raw[m.name] = 0.0
                continue
            if not m.ready():
                raw[m.name] = 0.0
                continue
            t = self.trackers[m.name]
            if not t.ready:
                # Not enough evidence either way -- give it a small, fixed
                # participation so it can earn its weight, rather than either
                # trusting it blind or locking it out forever.
                raw[m.name] = 0.15
                continue
            s = t.skill_score()
            raw[m.name] = max(0.0, s) ** 1.5
        tot = sum(raw.values())
        if tot <= 1e-9:
            n = len(self.members)
            return {m.name: 1.0 / n for m in self.members}
        w = {k: v / tot for k, v in raw.items()}
        # floor + renormalise so a temporarily-cold model is not erased
        w = {k: max(v, self.min_weight if raw[k] > 0 else 0.0)
             for k, v in w.items()}
        tot = sum(w.values()) or 1.0
        return {k: v / tot for k, v in w.items()}

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray, ctx: Optional[dict] = None) -> Prediction:
        w = self.weights()
        self.last_weights = w
        # Combine in log-odds. Averaging probabilities directly pulls a
        # confident-and-correct model toward the mush of its undecided peers;
        # log-odds averaging is the form that corresponds to pooling evidence.
        z = 0.0
        conf = 0.0
        used = 0
        for m in self.members:
            wt = w.get(m.name, 0.0)
            if wt <= 0:
                continue
            pr = m.predict(x, ctx)
            z += wt * logit(pr.p_up)
            conf += wt * pr.confidence
            used += 1
        p = sigmoid(clamp(z, -12, 12))
        if self.cal is not None and self.cal.ready:
            p = self.cal.transform(p)
        agree = self._agreement(x, w)
        # Disagreement between members is information: when they diverge, the
        # ensemble should be less sure. The floor stops the agreement term
        # from collapsing confidence to zero on its own -- attenuation factors
        # multiply, and three of them in series drive any signal to nothing.
        return Prediction(
            p_up=p, source=self.name, horizon_s=self.horizon_s,
            confidence=float(clamp(conf * (0.70 + 0.30 * agree), 0, 1)),
            features_used=used,
        )

    def _agreement(self, x: np.ndarray, w: Dict[str, float]) -> float:
        ps = []
        for m in self.members:
            if w.get(m.name, 0.0) <= 0:
                continue
            ps.append(m.predict(x).p_up)
        if len(ps) < 2:
            return 1.0
        arr = np.asarray(ps)
        # 1 when all members sit on the same side with similar strength
        return float(clamp(1.0 - 2.0 * float(arr.std()), 0.0, 1.0))

    # ------------------------------------------------------------------
    def partial_fit(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        # Score each member on this sample *before* training it, so the skill
        # numbers are genuinely out-of-sample for that observation.
        for m in self.members:
            try:
                p = m.predict(x).p_up
                self.trackers[m.name].add(p, y)
            except Exception:                          # noqa: BLE001
                pass
        pe = self.predict(x).p_up
        self.skill.add(pe, y)
        if self.cal is not None:
            self.cal.add(pe, y)
        for m in self.members:
            m.partial_fit(x, y, weight)
        self.n += 1

    def ready(self) -> bool:
        return any(m.ready() for m in self.members)

    # ------------------------------------------------------------------
    def review(self, min_skill: float = -0.02, min_samples: int = 250
               ) -> Dict[str, str]:
        """Spec §17: flag or suspend members whose skill has decayed.

        A model that has demonstrably negative skill over a meaningful sample
        is not neutral -- it is actively harmful, and the ensemble weighting
        alone will not fully remove it because weights are floored.
        """
        actions: Dict[str, str] = {}
        for m in self.members:
            t = self.trackers[m.name]
            if t.n_total < min_samples or not t.ready:
                continue
            s = t.skill_score()
            if s < min_skill and not self.suspended[m.name]:
                self.suspended[m.name] = True
                actions[m.name] = f"SUSPENDED skill={s:.3f}"
            elif s > min_skill + 0.05 and self.suspended[m.name]:
                self.suspended[m.name] = False
                actions[m.name] = f"REINSTATED skill={s:.3f}"
        return actions

    def state(self) -> dict:
        return {
            "n": self.n,
            "weights": {k: round(v, 3) for k, v in self.last_weights.items()},
            "suspended": [k for k, v in self.suspended.items() if v],
            "skill": self.skill.stats(),
            "members": {m.name: m.state() for m in self.members},
            "calibration": self.cal.reliability() if self.cal else [],
        }
