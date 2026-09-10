"""Feature assembly and time-aware stock knowledge.

The feature engine combines technical/microstructure data with the persistent
knowledge layer. Past observations, present market state and known future events
are converted into bounded model inputs; the hot tick path only updates O(1)
state, while knowledge is checkpointed from the minute boundary.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Timeframe, Tick
from ..core.mathx import clamp
from ..core.config import sector_of
from ..quant_math import descriptive_stats, log_returns, linear_regression, rate_of_change, acceleration
from ..knowledge import KnowledgeStore, PreTradeKnowledge
from .technical import timeframe_features, atr, realised_vol
from .microstructure import MicrostructureEngine
from .crosssectional import CrossSectionalEngine
from .derivatives import DerivativesEngine

TF_WEIGHT: Dict[str, float] = {
    "1m": 0.05, "5m": 0.10, "15m": 0.15,
    "1h": 0.22, "4h": 0.20, "1d": 0.20, "1w": 0.08,
}


@dataclass
class FeatureVector:
    symbol: str
    ts: int
    values: Dict[str, float] = field(default_factory=dict)
    alignment: float = 0.0
    alignment_conflict: bool = False
    dominant_tf: str = ""
    atr: float = 0.0
    atr_pct: float = 0.0
    price: float = 0.0
    liquidity: float = 0.5
    knowledge_score: float = 0.5
    knowledge_event_risk: float = 0.0
    knowledge_win_rate: float = 0.5
    ready: bool = False

    def vec(self, names: List[str]) -> np.ndarray:
        return np.fromiter((self.values.get(n, 0.0) for n in names),
                           dtype=np.float64, count=len(names))


class FeatureEngine:
    def __init__(self, symbols: List[str], index_symbol: str = "NIFTY",
                 timeframes: Optional[List[Timeframe]] = None,
                 knowledge_path: Optional[str] = None):
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
        self._cache: Dict[str, FeatureVector] = {}

        # Past/present/future knowledge. Persistence is only touched from the
        # minute boundary, never from the per-tick hot path.
        self.knowledge_path = knowledge_path or os.environ.get(
            "GMQ_KNOWLEDGE_PATH", "runs/stock_knowledge.json")
        self.knowledge = KnowledgeStore(self.knowledge_path)
        self.pretrade = PreTradeKnowledge(self.knowledge)
        self._knowledge_minutes = 0
        for s in self.symbols:
            self.knowledge.register(s, sector=sector_of(s), instrument_type="EQ")

    def feature_names(self) -> List[str]:
        return self._names or []

    def _freeze_names(self, values: Dict[str, float]) -> None:
        if self._names is None:
            self._names = sorted(values.keys())
            self._name_set = set(self._names)
        else:
            new = set(values.keys()) - self._name_set
            if new:
                self._names = self._names + sorted(new)
                self._name_set |= new

    def _quant_math_features(self, ser) -> Dict[str, float]:
        vals: Dict[str, float] = {}
        if ser is None or ser.n < 10:
            return vals
        closes = np.asarray(ser.close, dtype=np.float64)
        rets = log_returns(closes[-120:])
        if rets.size >= 8:
            ds = descriptive_stats(rets)
            vals["qtm_ret_mean"] = clamp(ds["mean"] * 1e4, -50.0, 50.0)
            vals["qtm_ret_std"] = clamp(ds["std"] * 1e4, 0.0, 100.0)
            vals["qtm_ret_mad"] = clamp(ds["mad"] * 1e4, 0.0, 100.0)
            vals["qtm_ret_iqr"] = clamp(ds["iqr"] * 1e4, 0.0, 100.0)
            vals["qtm_ret_robust_z"] = clamp(
                (rets[-1] - ds["median"]) / max(1.4826 * ds["mad"], 1e-9), -8.0, 8.0)
        if closes.size >= 3:
            vals["qtm_price_velocity"] = clamp(
                rate_of_change(float(closes[-1]), float(closes[-2]), 60.0)
                / max(float(closes[-1]), 1e-9) * 1e4, -100.0, 100.0)
            vals["qtm_price_acceleration"] = clamp(
                acceleration(float(closes[-1]), float(closes[-2]), float(closes[-3]), 60.0)
                / max(float(closes[-1]), 1e-9) * 1e6, -100.0, 100.0)
        if closes.size >= 20:
            window = closes[-60:] if closes.size >= 60 else closes
            x = np.arange(window.size, dtype=np.float64)
            slope, intercept, r2, resid = linear_regression(
                x, np.log(np.maximum(window, 1e-9)))
            vals["qtm_ols_slope"] = clamp(slope * 1e4, -100.0, 100.0)
            vals["qtm_ols_r2"] = clamp(r2, 0.0, 1.0)
            vals["qtm_ols_resid"] = clamp(resid * 1e4, 0.0, 100.0)
        return vals

    def _knowledge_features(self, symbol: str, ts: int, px: float,
                            spread_bps: float, liquidity: float) -> Dict[str, float]:
        k = self.knowledge.snapshot(symbol, ts)
        hist = self.knowledge.historical_win_rate(symbol)
        obs_quality = clamp(k.observed_ticks / 500.0, 0.0, 1.0)
        event_risk = clamp(k.event_risk, 0.0, 1.0)
        spread_quality = 1.0 - clamp(spread_bps / 40.0, 0.0, 1.0)
        knowledge_score = clamp(
            0.35 * obs_quality
            + 0.25 * hist
            + 0.20 * liquidity
            + 0.20 * spread_quality
            - 0.30 * event_risk,
            0.0, 1.0)
        return {
            "kn_observations": obs_quality,
            "kn_hist_win_rate": hist,
            "kn_event_risk": event_risk,
            "kn_spread_quality": spread_quality,
            "kn_knowledge_score": knowledge_score,
            "kn_hist_return_mean": clamp(k.return_mean * 1e4, -50.0, 50.0),
            "kn_hist_return_std": clamp(k.return_std * 1e4, 0.0, 100.0),
            "kn_hist_trend_slope": clamp(k.trend_slope * 1e4, -100.0, 100.0),
            "kn_hist_trend_r2": clamp(k.trend_r2, 0.0, 1.0),
        }

    def build(self, symbol: str, ts: int, mde, calendar: Optional[dict] = None
              ) -> FeatureVector:
        fv = FeatureVector(symbol=symbol, ts=ts)
        px = mde.ltp(symbol)
        if px <= 0:
            return fv
        fv.price = px
        vals: Dict[str, float] = {}
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

        ms = self.micro.get(symbol)
        vals.update(ms.features(px))
        fv.liquidity = ms.liquidity_score()
        vals["mx_liquidity"] = fv.liquidity

        vals.update(self.cross.features(symbol))

        ser1d = mde.series(symbol, Timeframe.D1)
        rv_ann = 0.0
        if ser1d is not None and ser1d.n > 10:
            rv_ann = realised_vol(ser1d.close, 20) * math.sqrt(252)
        else:
            rv_ann = fv.atr_pct * math.sqrt(252 * 75)
        vals.update(self.derivs.features(symbol, px, rv_ann,
                                         mde.day_change_pct(symbol)))
        vals["dv_rv_ann"] = clamp(rv_ann, 0, 3)

        vals.update(self._quant_math_features(mde.series(symbol, Timeframe.M1)))
        kn = self._knowledge_features(symbol, ts, px,
                                      getattr(mde.last_tick.get(symbol), "spread_bps", 0.0),
                                      fv.liquidity)
        vals.update(kn)
        fv.knowledge_score = kn["kn_knowledge_score"]
        fv.knowledge_event_risk = kn["kn_event_risk"]
        fv.knowledge_win_rate = kn["kn_hist_win_rate"]

        if calendar:
            vals.update({f"cal_{k}": float(v) for k, v in calendar.items()})
        vals["ctx_day_change"] = clamp(mde.day_change_pct(symbol), -12, 12)
        vals["ctx_stale"] = 1.0 if mde.is_stale(symbol) else 0.0

        fv.alignment, fv.alignment_conflict, fv.dominant_tf = \
            self._alignment(tf_scores)
        vals["mtf_alignment"] = fv.alignment
        vals["mtf_conflict"] = 1.0 if fv.alignment_conflict else 0.0
        for tfv, sc in tf_scores.items():
            vals[f"mtf_{tfv}_dir"] = sc

        self._freeze_names(vals)
        fv.values = vals
        # Knowledge admissibility is a hard pre-trade safety gate: lack of
        # evidence, extreme upcoming-event risk, or very poor liquidity should
        # prevent the decision layer from producing an order at all.
        knowledge_admissible = (
            fv.knowledge_score >= 0.25
            and fv.knowledge_event_risk < 0.90
            and fv.liquidity >= 0.15
        )
        fv.ready = len(tf_scores) >= 2 and ms.n > 50 and knowledge_admissible
        self._cache[symbol] = fv
        return fv

    def refresh(self, symbol: str, ts: int, mde) -> Optional[FeatureVector]:
        fv = self._cache.get(symbol)
        if fv is None:
            return None
        px = mde.ltp(symbol)
        if px <= 0:
            return fv
        ms = self.micro.get(symbol)
        tick = mde.last_tick.get(symbol)
        spread_bps = getattr(tick, "spread_bps", 0.0)
        kn = self._knowledge_features(symbol, ts, px, spread_bps, ms.liquidity_score())
        knowledge_admissible = (
            kn["kn_knowledge_score"] >= 0.25
            and kn["kn_event_risk"] < 0.90
            and ms.liquidity_score() >= 0.15
        )
        out = FeatureVector(
            symbol=symbol, ts=ts, values=dict(fv.values),
            alignment=fv.alignment, alignment_conflict=fv.alignment_conflict,
            dominant_tf=fv.dominant_tf, atr=fv.atr,
            atr_pct=fv.atr / px if px > 0 else fv.atr_pct,
            price=px, liquidity=ms.liquidity_score(),
            knowledge_score=kn["kn_knowledge_score"],
            knowledge_event_risk=kn["kn_event_risk"],
            knowledge_win_rate=kn["kn_hist_win_rate"],
            ready=fv.ready and knowledge_admissible)
        out.values.update(ms.features(px))
        out.values.update(kn)
        out.values["mx_liquidity"] = out.liquidity
        out.values["ctx_day_change"] = clamp(mde.day_change_pct(symbol), -12, 12)
        out.values["ctx_stale"] = 1.0 if mde.is_stale(symbol) else 0.0
        return out

    @staticmethod
    def _tf_direction(blk: Dict[str, float], tfv: str) -> float:
        p = f"{tfv}_"
        def g(k, d=0.0): return blk.get(p + k, d)
        trend = 0.0
        for span, w in ((9, 0.15), (21, 0.25), (50, 0.35), (200, 0.25)):
            trend += w * math.tanh(g(f"ema{span}_dist_atr") / 2.0)
        slope = math.tanh(g("slope_atr") / 2.0) * max(g("slope_r2"), 0.0)
        adx_w = clamp(g("adx") * 2.5, 0.0, 1.0)
        di = g("di_diff")
        momentum = 0.5 * math.tanh(g("rsi") * 1.5) + 0.5 * math.tanh(g("macd_hist"))
        location = 0.6 * (2 * clamp(g("bb_pctb", 0.5), 0, 1) - 1) + 0.4 * g("donchian")
        score = (0.34 * trend + 0.24 * slope + 0.18 * adx_w * di + 0.14 * momentum + 0.10 * location)
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
        higher = [(tfv, sc) for tfv, sc in tf_scores.items() if tfv in ("1h", "4h", "1d", "1w")]
        lower = [(tfv, sc) for tfv, sc in tf_scores.items() if tfv in ("1m", "5m", "15m")]
        conflict = False
        if higher and lower:
            hw = sum(TF_WEIGHT.get(t, 0.1) for t, _ in higher)
            h = sum(TF_WEIGHT.get(t, 0.1) * s for t, s in higher) / max(hw, 1e-9)
            lw = sum(TF_WEIGHT.get(t, 0.1) for t, _ in lower)
            lo = sum(TF_WEIGHT.get(t, 0.1) * s for t, s in lower) / max(lw, 1e-9)
            conflict = (abs(h) > 0.30 and np.sign(h) != np.sign(lo) and abs(lo) > 0.15)
        dominant = max(tf_scores.items(), key=lambda kv: abs(kv[1]) * TF_WEIGHT.get(kv[0], 0.1))[0]
        return float(clamp(align, -1, 1)), bool(conflict), dominant

    def on_tick(self, t: Tick) -> None:
        self.micro.on_tick(t)
        self.knowledge.observe(
            t.symbol, t.ts, t.ltp, volume=t.volume,
            spread_bps=t.spread_bps, liquidity=self.micro.get(t.symbol).liquidity_score())

    def on_depth(self, d) -> None:
        self.micro.on_depth(d)

    def on_minute(self, prices: Dict[str, float]) -> None:
        self.cross.on_bar_close(prices)
        self._knowledge_minutes += 1
        if self._knowledge_minutes % 5 == 0:
            last_ts = max((t.last_ts_ns for t in self.knowledge._stocks.values()), default=0)
            self.knowledge.prune_expired_events(last_ts)
            self.knowledge.checkpoint(self.knowledge_path)

    def add_future_event(self, symbol: str, ts_ns: int, kind: str,
                         severity: float = 0.5, source: str = "", note: str = "") -> None:
        from ..knowledge.store import FutureEvent
        self.knowledge.add_event(FutureEvent(
            symbol=symbol, ts_ns=ts_ns, kind=kind, severity=severity,
            source=source, note=note))

    def checkpoint_knowledge(self) -> str:
        return self.knowledge.checkpoint(self.knowledge_path)

    def reset_day(self) -> None:
        self.micro.reset_day()
