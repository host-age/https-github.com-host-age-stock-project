"""Market regime detection (spec §9).

Eight regimes, per symbol, updated on every 1-minute close.

The detector is a **filtered hidden Markov model with interpretable emissions**
rather than a black box, for a specific reason: the regime gates which
strategies are allowed to trade, so when the system refuses to take a trade,
an operator has to be able to read *why*. Every emission term below is a
statistic you can point at on a chart.

  P(regime_t | data) proportional to  P(data | regime) * sum_j P(regime|j) P(j)

* the transition prior is the same persistence matrix the market is built on
  in spirit (regimes last; they do not flip every bar)
* emissions are log-scores from bounded, normalised evidence
* the posterior is smoothed, and a regime change is only *declared* when the
  posterior stays above a threshold for `confirm_bars` -- untethered flipping
  is worse than being one bar late

The detector never sees the simulator's ground truth. `SimExchange.true_regime`
exists only so the evaluation harness can score this module honestly.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Regime, Timeframe
from ..core.mathx import (
    clamp, safe_div, hurst, half_life_ou, linreg_slope, EWMA, EWVar, Ring,
)

# Statistics that are standardised against their own rolling distribution
# before entering the emission model. Absolute reference points ("Hurst above
# 0.5 means trending", "ADX above 25 means a trend") are folklore calibrated on
# daily bars of 1980s futures; on 1-minute equity bars the same estimators sit
# at completely different baselines, and every symbol has its own. Comparing a
# statistic to its own recent history is the only version of the question that
# has a stable answer.
_ZKEYS = ("vol_ratio", "drift_t", "range_exp", "hurst", "mr_speed",
          "adx", "slope_r2", "squeeze", "liquidity", "spread_z", "jump",
          "donchian_abs", "vol_z")

REGIMES: List[Regime] = [
    Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.MEAN_REVERTING,
    Regime.BREAKOUT, Regime.HIGH_VOL, Regime.LOW_VOL,
    Regime.EVENT_DRIVEN, Regime.ILLIQUID,
]
_IDX = {r: i for i, r in enumerate(REGIMES)}

# Persistence prior. Diagonal-heavy on purpose.
_TRANS = np.array([
    [0.930, 0.008, 0.022, 0.014, 0.010, 0.010, 0.003, 0.003],
    [0.008, 0.928, 0.022, 0.014, 0.014, 0.008, 0.003, 0.003],
    [0.020, 0.020, 0.900, 0.032, 0.010, 0.014, 0.002, 0.002],
    [0.090, 0.080, 0.030, 0.760, 0.032, 0.005, 0.002, 0.001],
    [0.030, 0.032, 0.040, 0.040, 0.830, 0.020, 0.006, 0.002],
    [0.022, 0.018, 0.055, 0.016, 0.010, 0.875, 0.002, 0.002],
    [0.110, 0.100, 0.050, 0.100, 0.150, 0.020, 0.450, 0.020],
    [0.030, 0.030, 0.070, 0.015, 0.030, 0.070, 0.015, 0.740],
])
_TRANS = _TRANS / _TRANS.sum(axis=1, keepdims=True)


def _stationary(P: np.ndarray, iters: int = 4000) -> np.ndarray:
    """Stationary distribution of the transition matrix -- the class mix the
    model itself implies, and therefore the mix its predictions should have."""
    v = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(iters):
        v = v @ P
    return v / v.sum()


_STATIONARY = _stationary(_TRANS)


@dataclass
class RegimeState:
    symbol: str
    posterior: np.ndarray = field(
        default_factory=lambda: np.full(len(REGIMES), 1.0 / len(REGIMES)))
    current: Regime = Regime.LOW_VOL
    initialised: bool = False      # False until enough history to standardise
    confidence: float = 0.125
    since_bars: int = 0
    candidate: Optional[Regime] = None
    candidate_bars: int = 0
    history: Deque[Regime] = field(default_factory=lambda: deque(maxlen=500))
    transitions: int = 0
    vol_ring: Ring = field(default_factory=lambda: Ring(400))
    evidence: Dict[str, float] = field(default_factory=dict)
    z: Dict[str, float] = field(default_factory=dict)
    norm: Dict[str, "EWVar"] = field(default_factory=dict)
    seen: int = 0
    # running marginal of what the detector has been predicting, used to
    # correct systematic class bias (see _bias_correction)
    marginal: np.ndarray = field(
        default_factory=lambda: np.full(len(REGIMES), 1.0 / len(REGIMES)))


class RegimeDetector:
    def __init__(self, confirm_bars: int = 3, min_conf: float = 0.34):
        self.states: Dict[str, RegimeState] = {}
        self.confirm_bars = confirm_bars
        self.min_conf = min_conf

    def get(self, symbol: str) -> RegimeState:
        st = self.states.get(symbol)
        if st is None:
            st = RegimeState(symbol=symbol)
            self.states[symbol] = st
        return st

    # ------------------------------------------------------------------
    def update(self, symbol: str, fv, mde) -> RegimeState:
        st = self.get(symbol)
        ev = self._evidence(symbol, fv, mde, st)
        if not ev:
            return st
        st.evidence = ev
        st.z = self._standardise(st, ev)
        st.seen += 1
        if st.seen < 40:
            # Not enough history to standardise against yet. Emitting a
            # confident regime from an uncalibrated scale would be worse than
            # admitting ignorance, so hold the prior and wait. `initialised`
            # stays False so callers can tell "no opinion yet" apart from
            # "considered it and concluded LOW_VOL" -- conflating those two is
            # how a warm-up artefact gets mistaken for a signal.
            return st
        st.initialised = True
        loglik = self._emissions(st.z) + self._bias_correction(st)

        prior = _TRANS.T @ st.posterior
        # Floor the prior. Without it the filter self-traps: after a run of
        # agreeing bars one state reaches ~0.999, every rival's log-prior sits
        # near -7, and no amount of contrary evidence at the emission scale can
        # ever climb back. The floor is what keeps the detector able to change
        # its mind -- which is the entire job.
        prior = np.maximum(prior, 0.004)
        prior /= prior.sum()
        logpost = np.log(prior) + loglik
        logpost -= logpost.max()
        post = np.exp(logpost)
        st.posterior = post / post.sum()

        top = int(np.argmax(st.posterior))
        cand = REGIMES[top]
        conf = float(st.posterior[top])

        if cand is st.current:
            st.candidate = None
            st.candidate_bars = 0
            st.since_bars += 1
        else:
            if cand is st.candidate:
                st.candidate_bars += 1
            else:
                st.candidate = cand
                st.candidate_bars = 1
            # A high-conviction shock regime is allowed to pre-empt the
            # confirmation delay: waiting three bars to notice an event is
            # exactly the wrong behaviour when an event is what happened.
            fast = (cand in (Regime.EVENT_DRIVEN, Regime.HIGH_VOL)
                    and conf > 0.55)
            if (st.candidate_bars >= self.confirm_bars and conf >= self.min_conf) \
                    or fast:
                st.current = cand
                st.transitions += 1
                st.since_bars = 0
                st.candidate = None
                st.candidate_bars = 0
        # Track what we have actually been emitting, so persistent class bias
        # can be corrected next bar.
        obs = np.zeros(len(REGIMES))
        obs[_IDX[cand]] = 1.0
        st.marginal = 0.995 * st.marginal + 0.005 * obs

        st.confidence = conf
        st.history.append(st.current)
        return st

    # ------------------------------------------------------------------
    def _evidence(self, symbol: str, fv, mde, st: RegimeState
                  ) -> Dict[str, float]:
        # Evidence is computed from the 1-minute series directly rather than
        # read out of the feature vector. The 5-minute block needs ~30 closed
        # bars (two and a half hours) before ADX and the Bollinger history are
        # meaningful, and a detector that emits zeros for the first two hours
        # of every session is worse than useless -- it is confidently wrong.
        ser = mde.series(symbol, Timeframe.M1)
        if ser is None or ser.n < 40:
            return {}
        c, h_, l_, v_ = ser.close, ser.high, ser.low, ser.volume
        lc = np.log(np.maximum(c[-300:], 1e-9))
        r = np.diff(lc)
        if r.size < 25:
            return {}
        # Volatility estimate: short window, bounce-corrected.
        #
        # The window length is the thing that matters most and is the easiest
        # to get wrong. An intraday volatility regime lasts on the order of
        # half an hour, so a 60-bar estimate averages across two or three
        # different regimes and reports the mean of all of them -- which is
        # why a long-window estimator will tell you, with great confidence,
        # that nothing ever changes.
        #
        # The MA(1) correction (Zhou) removes the variance bid-ask bounce adds
        # to every one-bar return: bounce shows up as negative first-order
        # autocovariance, so adding 2*sum(r_t r_{t-1}) cancels it. Floored at a
        # quarter of the naive estimate, because on a genuinely trending tape
        # the correction can overshoot into nonsense.
        W = 20
        rw_ = r[-W:]
        if rw_.size >= 8:
            rv_naive = float((rw_ ** 2).sum())
            cov1 = float((rw_[1:] * rw_[:-1]).sum())
            rv = max(rv_naive + 2.0 * cov1, 0.25 * rv_naive)
            vol = math.sqrt(rv / rw_.size)
        else:
            vol = float(r.std())
        st.vol_ring.push(vol)
        vv = st.vol_ring.view()
        vol_med = float(np.median(vv)) if vv.size > 25 else vol
        vol_ratio = safe_div(vol, vol_med, 1.0)

        h = hurst(lc)
        hl = half_life_ou(lc)
        slope, r2 = linreg_slope(lc[-60:])
        # slope is per-bar log return; express it in units of that bar's own
        # volatility so it is comparable across names and vol states
        slope_sigma = safe_div(abs(slope), max(vol, 1e-9))

        from ..features.technical import (
            adx as _adx, bollinger as _bb, donchian_position as _don,
            volume_zscore as _vz,
        )
        adx_raw, pdi, mdi = _adx(h_, l_, c, 14)
        adx_ = adx_raw / 100.0
        di = (pdi - mdi) / 100.0
        _pb, bbw, _bz = _bb(c, 20, 2.0)
        donch = _don(h_, l_, c, 20)
        volz = _vz(v_, 40)
        liq = fv.liquidity
        spread_z = fv.values.get("mx_spread_z", 0.0)
        vpin = fv.values.get("mx_vpin", 0.0)

        # squeeze: bandwidth in the low decile of its own history
        bbw_hist = getattr(st, "_bbw", None)
        if bbw_hist is None:
            st._bbw = deque(maxlen=300)          # type: ignore[attr-defined]
            bbw_hist = st._bbw                    # type: ignore[attr-defined]
        bbw_hist.append(bbw)
        squeeze = 0.0
        if len(bbw_hist) > 40:
            arr = np.asarray(bbw_hist)
            squeeze = float(clamp(1.0 - (bbw - arr.min()) /
                                  max(arr.max() - arr.min(), 1e-9), 0, 1))

        # jump evidence: largest |return| in the last 5 bars vs typical
        jump = 0.0
        if r.size > 30:
            typ = float(np.median(np.abs(r[-120:]))) or 1e-9
            jump = float(clamp(float(np.abs(r[-5:]).max()) / (typ * 8.0), 0, 3))

        # Drift t-statistic: the correct test for "is there a trend here".
        # mean return over N bars divided by its own standard error. Comparing
        # a slope to a price level, as most indicator-based classifiers do,
        # answers a different and much less useful question.
        n_d = min(90, r.size)
        rw = r[-n_d:]
        se = float(rw.std()) / math.sqrt(n_d) if n_d > 5 else 0.0
        drift_t = float(clamp(safe_div(float(rw.mean()), se), -6, 6)) if se > 1e-12 else 0.0

        # Range expansion: are bars getting bigger right now?
        rng = h_[-120:] - l_[-120:]
        rng_exp = 1.0
        if rng.size > 30:
            med = float(np.median(rng[:-5])) or 1e-9
            rng_exp = float(clamp(float(rng[-5:].mean()) / med, 0.1, 6.0))

        return {
            "vol_ratio": float(clamp(vol_ratio, 0.05, 8.0)),
            "drift_t": drift_t,
            "range_exp": rng_exp,
            "hurst": h,
            "mr_speed": float(clamp(1.0 / hl if math.isfinite(hl) and hl > 0
                                    else 0.0, 0, 1)),
            "slope_r2": r2,
            "slope_sign": float(np.sign(slope)),
            "slope_mag": float(clamp(slope_sigma * 60, 0, 6)),
            "adx": adx_,
            "di": di,
            "squeeze": squeeze,
            "donchian_abs": abs(donch),
            "vol_z": volz,
            "liquidity": liq,
            "spread_z": spread_z,
            "vpin": vpin,
            "jump": jump,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _bias_correction(st: RegimeState) -> np.ndarray:
        """Nudge the emission scores so the detector's long-run class mix
        matches the transition matrix's stationary distribution.

        Any hand-written emission model will sit systematically high on some
        class -- here it was LOW_VOL -- and an argmax filter then reports that
        class far more often than the world contains it. Comparing the running
        predicted marginal against the model's own stationary distribution and
        subtracting the log-ratio is the standard prior-correction fix. It uses
        no labels: the target comes from the transition matrix that was already
        assumed, not from any ground truth.
        """
        target = _STATIONARY
        obs = np.maximum(st.marginal, 1e-4)
        obs = obs / obs.sum()
        return -0.55 * np.log(obs / target)

    # ------------------------------------------------------------------
    def _standardise(self, st: RegimeState, e: Dict[str, float]
                     ) -> Dict[str, float]:
        """Convert raw evidence into z-scores against its own history."""
        z: Dict[str, float] = {}
        for k in _ZKEYS:
            v = e.get(k)
            if v is None:
                continue
            n = st.norm.get(k)
            if n is None:
                n = EWVar(halflife=300.0)
                st.norm[k] = n
            n.update(float(v))
            z[k] = clamp(n.z(float(v)), -4.0, 4.0) if n.ready else 0.0
        # pass through the terms that are already signed and scale-free
        for k in ("di", "slope_sign", "slope_mag", "vpin"):
            if k in e:
                z[k] = e[k]
        z["drift_t_raw"] = e.get("drift_t", 0.0)
        z["liquidity_raw"] = e.get("liquidity", 0.5)
        return z

    @staticmethod
    def _emissions(z: Dict[str, float]) -> np.ndarray:
        """Log-likelihood of the standardised evidence under each regime.

        Inputs are z-scores, so every coefficient below reads as "how many
        standard deviations of this statistic, in this symbol's own recent
        history, does this regime imply". That is a claim you can argue with;
        "ADX above 25" is not.
        """
        ll = np.zeros(len(REGIMES))
        g = z.get
        vol = g("vol_ratio", 0.0)          # z of short-window volatility
        rex = g("range_exp", 0.0)          # z of range expansion
        dtz = g("drift_t", 0.0)            # z of drift significance
        dt_raw = g("drift_t_raw", 0.0)     # raw t-stat, keeps absolute meaning
        hz = g("hurst", 0.0)
        mrz = g("mr_speed", 0.0)
        adxz = g("adx", 0.0)
        r2z = g("slope_r2", 0.0)
        sqz = g("squeeze", 0.0)
        donz = g("donchian_abs", 0.0)
        volzz = g("vol_z", 0.0)
        liqz = g("liquidity", 0.0)
        spz = g("spread_z", 0.0)
        jz = g("jump", 0.0)

        # -- TRENDING_UP / TRENDING_DOWN -------------------------------
        # Scored as one "is this trending at all" term split by a direction
        # *probability*. Scoring the two directions independently would give
        # the trending hypothesis two chances to win an argmax against every
        # rival's one, and that structural bias -- not the evidence -- is what
        # makes hand-built classifiers announce a trend all day long.
        trend_char = (0.55 * adxz + 0.40 * r2z + 0.50 * hz
                      + 0.90 * clamp(abs(dt_raw) - 1.1, -1.0, 2.5)
                      - 0.30 * abs(vol))
        dirn = clamp(0.75 * dt_raw + 1.1 * g("di", 0.0)
                     + 0.25 * g("slope_sign", 0.0) * min(g("slope_mag", 0.0), 3.0),
                     -4, 4)
        p_up = clamp(1.0 / (1.0 + math.exp(-1.3 * dirn)), 1e-4, 1 - 1e-4)
        ll[_IDX[Regime.TRENDING_UP]] = trend_char + math.log(p_up)
        ll[_IDX[Regime.TRENDING_DOWN]] = trend_char + math.log(1.0 - p_up)

        # -- MEAN_REVERTING --------------------------------------------
        ll[_IDX[Regime.MEAN_REVERTING]] = (
            0.75 * mrz - 0.60 * hz - 0.45 * adxz - 0.30 * r2z
            # absence of significant drift is positive evidence for reversion,
            # not merely absence of evidence for trend
            + 0.80 * clamp(1.2 - abs(dt_raw), -1.5, 1.2)
            - 0.45 * vol - 0.40 * clamp(rex, 0, 3)
        )

        # -- BREAKOUT ---------------------------------------------------
        # A transition, so it demands the conjunction: prior compression,
        # current expansion, price at the channel edge. Scoring any one of
        # those alone is why naive classifiers cry breakout twice an hour.
        ll[_IDX[Regime.BREAKOUT]] = (
            0.85 * clamp(sqz, -1, 3) * clamp(rex, 0, 3)
            + 0.45 * donz + 0.35 * clamp(volzz, 0, 3)
            + 0.40 * clamp(vol, 0, 3) - 0.55 * mrz - 0.9
        )

        # -- HIGH_VOL / LOW_VOL ----------------------------------------
        ll[_IDX[Regime.HIGH_VOL]] = (1.15 * clamp(vol - 0.45, -2, 3)
                                     + 0.55 * clamp(rex - 0.3, -1.5, 3)
                                     + 0.30 * spz
                                     - 0.45 * clamp(abs(dt_raw) - 1.0, 0, 3)
                                     - 0.35 * adxz)
        ll[_IDX[Regime.LOW_VOL]] = (1.15 * clamp(-0.45 - vol, -2, 3)
                                    + 0.55 * clamp(-0.3 - rex, -1.5, 3)
                                    + 0.35 * sqz - 0.40 * adxz
                                    - 0.35 * clamp(abs(dt_raw) - 1.0, 0, 3))

        # -- EVENT_DRIVEN ----------------------------------------------
        ll[_IDX[Regime.EVENT_DRIVEN]] = (
            1.05 * clamp(jz - 1.4, -1, 3) + 0.60 * clamp(spz - 0.8, -1, 3)
            + 0.75 * clamp(g("vpin", 0.0) - 0.5, -0.5, 0.6) * 2
            + 0.55 * clamp(vol - 1.2, -1, 3) - 1.1
        )

        # -- ILLIQUID ---------------------------------------------------
        ll[_IDX[Regime.ILLIQUID]] = (
            -1.10 * clamp(liqz + 0.4, -3, 1.5)
            + 0.55 * clamp(spz - 0.8, -1, 3)
            + 0.35 * clamp(1.0 - g("liquidity_raw", 0.5) * 3, -1, 2) - 1.0
        )

        # Centre and temper. Centring removes any baseline advantage a
        # particular formula happens to carry; the temperature sets how much
        # one bar of evidence is allowed to move the filter.
        ll -= ll.mean()
        s = ll.std()
        if s > 1e-9:
            ll = ll / s * 2.35
        return ll

    # ------------------------------------------------------------------
    def snapshot(self, symbol: str) -> dict:
        st = self.get(symbol)
        return {
            "regime": st.current.value,
            "confidence": round(st.confidence, 4),
            "initialised": st.initialised,
            "bars_in_regime": st.since_bars,
            "transitions": st.transitions,
            "posterior": {REGIMES[i].value: round(float(p), 4)
                          for i, p in enumerate(st.posterior)},
            "evidence": {k: round(v, 4) for k, v in st.evidence.items()},
        }

    def stability(self, symbol: str, window: int = 60) -> float:
        """Fraction of the last `window` bars spent in the current regime."""
        st = self.get(symbol)
        if not st.history:
            return 0.0
        h = list(st.history)[-window:]
        return sum(1 for x in h if x is st.current) / len(h)
