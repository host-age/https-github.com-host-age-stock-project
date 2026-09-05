"""Feature assembly (spec §16, block 2) and multi-timeframe alignment (§4).

Two jobs:

1. **Assemble** the full feature vector for one symbol at one instant from the
   bar, microstructure, cross-sectional, derivatives and calendar blocks --
   with a *stable ordering*, because a model trained on one column order and
   scored on another fails silently rather than loudly.

2. **Align timeframes.** The spec is explicit that the agent must not take a
   short-horizon position that fights a materially stronger higher-timeframe
   structure unless the strategy allows it. That is computed here as a signed
   alignment score plus an explicit `conflict` flag the strategy layer reads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Timeframe, TF_LADDER, Tick
from ..core.mathx import clamp, safe_div
from .technical import timeframe_features, atr, realised_vol
from .microstructure import MicrostructureEngine
from .crosssectional import CrossSectionalEngine
from .derivatives import DerivativesEngine

# Weight of each timeframe in the alignment score. Higher timeframes dominate,
# which is the whole point of the constraint.
TF_WEIGHT: Dict[str, float] = {
    "1m": 0.05, "5m": 0.10, "15m": 0.15,
    "1h": 0.22, "4h": 0.20, "1d": 0.20, "1w": 0.08,
}


@dataclass
class FeatureVector:
    symbol: str
    ts: int
    values: Dict[str, float] = field(default_factory=dict)
    alignment: float = 0.0            # [-1, 1], signed multi-timeframe bias
    alignment_conflict: bool = False  # short-term fights a stronger long-term
    dominant_tf: str = ""
    atr: float = 0.0                  # 5m ATR in price units -- the risk unit
    atr_pct: float = 0.0
    price: float = 0.0
    liquidity: float = 0.5
    ready: bool = False

    def vec(self, names: List[str]) -> np.ndarray:
        return np.fromiter((self.values.get(n, 0.0) for n in names),
                           dtype=np.float64, count=len(names))


class FeatureEngine:
    """Owns the feature blocks and produces FeatureVectors on demand."""

    def __init__(self, symbols: List[str], index_symbol: str = "NIFTY",
                 timeframes: Optional[List[Timeframe]] = None):
        self.symbols = list(symbols)
        self.timeframes = timeframes or [
            Timeframe.M1, Timeframe.M5, Timeframe.M15,
            Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1,
        ]
        self.micro = MicrostructureEngine()
        self.cross = CrossSectionalEngine(symbols, index_symbol)
        self.derivs = DerivativesEngine()
        self._names: Optional[List[str]] = None
        self._name_set: set = set()
        self.session_features_on = True
        # Last full build per symbol. Bar-level features only change when a bar
        # closes, so recomputing seven timeframes of indicators on every
        # decision is pure waste -- and it is the single largest cost in the
        # hot path. The cached vector is refreshed with the fast-moving
        # microstructure block instead.
        self._cache: Dict[str, FeatureVector] = {}

    # ------------------------------------------------------------------
    def feature_names(self) -> List[str]:
        """Stable, sorted column order. Frozen after first assembly so that
        train-time and score-time vectors cannot silently disagree."""
        return self._names or []

    def _freeze_names(self, values: Dict[str, float]) -> None:
        if self._names is None:
            self._names = sorted(values.keys())
            self._name_set = set(self._names)
        else:
            new = set(values.keys()) - self._name_set
            if new:
                # A block came online late (e.g. the options chain arrived).
                # Extend rather than reorder, so existing model weights keep
                # pointing at the same columns.
                self._names = self._names + sorted(new)
                self._name_set |= new

    # ------------------------------------------------------------------
    def build(self, symbol: str, ts: int, mde, calendar: Optional[dict] = None
              ) -> FeatureVector:
        fv = FeatureVector(symbol=symbol, ts=ts)
        px = mde.ltp(symbol)
        if px <= 0:
            return fv
        fv.price = px
        vals: Dict[str, float] = {}

        # ---- per-timeframe technical blocks
        tf_scores: Dict[str, float] = {}
        atr5 = 0.0
        for tf in self.timeframes:
            ser = mde.series(symbol, tf)
            if ser is None or ser.n < 25:
                continue
            o, h, l, c, v = (ser.open, ser.high, ser.low, ser.close, ser.volume)
            blk = timeframe_features(o, h, l, c, v, prefix=f"{tf.value}_")
            vals.update(blk)
            tf_scores[tf.value] = self._tf_direction(blk, tf.value)
            if tf is Timeframe.M5:
                atr5 = atr(h, l, c, 14)
        if atr5 <= 0:
            ser = mde.series(symbol, Timeframe.M1)
            if ser is not None and ser.n > 20:
                atr5 = atr(ser.high, ser.low, ser.close, 14) * math.sqrt(5)
        fv.atr = atr5 if atr5 > 0 else px * 0.0025
        fv.atr_pct = fv.atr / px

        # ---- microstructure
        ms = self.micro.get(symbol)
        vals.update(ms.features(px))
        fv.liquidity = ms.liquidity_score()
        vals["mx_liquidity"] = fv.liquidity

        # ---- cross-sectional
        vals.update(self.cross.features(symbol))

        # ---- derivatives
        ser1d = mde.series(symbol, Timeframe.D1)
        rv_ann = 0.0
        if ser1d is not None and ser1d.n > 10:
            rv_ann = realised_vol(ser1d.close, 20) * math.sqrt(252)
        else:
            rv_ann = fv.atr_pct * math.sqrt(252 * 75)
        vals.update(self.derivs.features(symbol, px, rv_ann,
                                         mde.day_change_pct(symbol)))
        vals["dv_rv_ann"] = clamp(rv_ann, 0, 3)

        # ---- calendar / session context
        if calendar:
            vals.update({f"cal_{k}": float(v) for k, v in calendar.items()})
        vals["ctx_day_change"] = clamp(mde.day_change_pct(symbol), -12, 12)
        vals["ctx_stale"] = 1.0 if mde.is_stale(symbol) else 0.0

        # ---- multi-timeframe alignment
        fv.alignment, fv.alignment_conflict, fv.dominant_tf = \
            self._alignment(tf_scores)
        vals["mtf_alignment"] = fv.alignment
        vals["mtf_conflict"] = 1.0 if fv.alignment_conflict else 0.0
        for tfv, sc in tf_scores.items():
            vals[f"mtf_{tfv}_dir"] = sc

        self._freeze_names(vals)
        fv.values = vals
        fv.ready = len(tf_scores) >= 2 and ms.n > 50
        self._cache[symbol] = fv
        return fv

    # ------------------------------------------------------------------
    def refresh(self, symbol: str, ts: int, mde) -> Optional[FeatureVector]:
        """Cheap update of the cached vector between bar closes.

        Only the blocks that actually move tick-to-tick are recomputed:
        microstructure, price, and the day-change context. Everything derived
        from closed bars is carried over unchanged, because it is unchanged --
        recomputing it would produce identical numbers at roughly a hundred
        times the cost.
        """
        fv = self._cache.get(symbol)
        if fv is None:
            return None
        px = mde.ltp(symbol)
        if px <= 0:
            return fv
        ms = self.micro.get(symbol)
        out = FeatureVector(
            symbol=symbol, ts=ts, values=dict(fv.values),
            alignment=fv.alignment, alignment_conflict=fv.alignment_conflict,
            dominant_tf=fv.dominant_tf, atr=fv.atr,
            atr_pct=fv.atr / px if px > 0 else fv.atr_pct,
            price=px, liquidity=ms.liquidity_score(), ready=fv.ready)
        out.values.update(ms.features(px))
        out.values["mx_liquidity"] = out.liquidity
        out.values["ctx_day_change"] = clamp(mde.day_change_pct(symbol), -12, 12)
        out.values["ctx_stale"] = 1.0 if mde.is_stale(symbol) else 0.0
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _tf_direction(blk: Dict[str, float], tfv: str) -> float:
        """Collapse one timeframe's block into a single signed bias in [-1,1].

        Deliberately a transparent weighted vote of orthogonal-ish components
        (trend, momentum, location, structure) rather than a learned score --
        this feeds a *constraint*, and a constraint you cannot read is not a
        constraint you can trust.
        """
        p = f"{tfv}_"
        def g(k, d=0.0): return blk.get(p + k, d)

        trend = 0.0
        for span, w in ((9, 0.15), (21, 0.25), (50, 0.35), (200, 0.25)):
            trend += w * math.tanh(g(f"ema{span}_dist_atr") / 2.0)
        slope = math.tanh(g("slope_atr") / 2.0) * max(g("slope_r2"), 0.0)
        adx_w = clamp(g("adx") * 2.5, 0.0, 1.0)
        di = g("di_diff")
        momentum = 0.5 * math.tanh(g("rsi") * 1.5) + 0.5 * math.tanh(g("macd_hist"))
        location = 0.6 * (2 * clamp(g("bb_pctb", 0.5), 0, 1) - 1) + \
            0.4 * g("donchian")
        score = (0.34 * trend + 0.24 * slope + 0.18 * adx_w * di +
                 0.14 * momentum + 0.10 * location)
        return float(clamp(score, -1.0, 1.0))

    @staticmethod
    def _alignment(tf_scores: Dict[str, float]) -> Tuple[float, bool, str]:
        if not tf_scores:
            return 0.0, False, ""
        num = den = 0.0
        for tfv, sc in tf_scores.items():
            w = TF_WEIGHT.get(tfv, 0.1)
            num += w * sc
            den += w
        align = num / den if den else 0.0

        # Conflict test: is the *higher*-timeframe consensus both strong and
        # opposed to the short-term read?
        higher = [(tfv, sc) for tfv, sc in tf_scores.items()
                  if tfv in ("1h", "4h", "1d", "1w")]
        lower = [(tfv, sc) for tfv, sc in tf_scores.items()
                 if tfv in ("1m", "5m", "15m")]
        conflict = False
        if higher and lower:
            hw = sum(TF_WEIGHT.get(t, 0.1) for t, _ in higher)
            h = sum(TF_WEIGHT.get(t, 0.1) * s for t, s in higher) / max(hw, 1e-9)
            lw = sum(TF_WEIGHT.get(t, 0.1) for t, _ in lower)
            lo = sum(TF_WEIGHT.get(t, 0.1) * s for t, s in lower) / max(lw, 1e-9)
            conflict = (abs(h) > 0.30 and np.sign(h) != np.sign(lo)
                        and abs(lo) > 0.15)
        dominant = max(tf_scores.items(),
                       key=lambda kv: abs(kv[1]) * TF_WEIGHT.get(kv[0], 0.1))[0]
        return float(clamp(align, -1, 1)), bool(conflict), dominant

    # ------------------------------------------------------------------
    def on_tick(self, t: Tick) -> None:
        self.micro.on_tick(t)

    def on_depth(self, d) -> None:
        self.micro.on_depth(d)

    def on_minute(self, prices: Dict[str, float]) -> None:
        self.cross.on_bar_close(prices)

    def reset_day(self) -> None:
        self.micro.reset_day()
