"""Live position monitoring and losing-trade intelligence (spec §6).

When a position moves against us, "it's down" is not a diagnosis. The response
should differ completely depending on *why*, and this module's job is to work
out which of the seven causes applies:

    NORMAL_VOLATILITY     it is inside the noise band; nothing has happened
    TEMPORARY_PULLBACK    counter-move within an intact structure
    THESIS_WEAKENING      the evidence that justified entry is decaying
    THESIS_INVALIDATED    the evidence has reversed; the reason is gone
    REGIME_CHANGE         the market changed character under the position
    EVENT_SHOCK           news or a jump repriced the instrument
    LIQUIDITY_EXECUTION   the loss is spread, impact or a bad fill, not direction

The distinction that does the most work is NORMAL_VOLATILITY versus
THESIS_INVALIDATED. Treating noise as invalidation means being stopped out of
every good trade by its ordinary wiggle; treating invalidation as noise is how
a small loss becomes the one that matters. Neither error is symmetric with the
other, and neither is detectable from the P&L alone -- which is why this reads
the *evidence*, not the drawdown.

The explicit goal, per the spec: not to force every losing trade into a
profit. It is to make the reason for holding or exiting a reasoned one, and to
prevent any single loss from becoming catastrophic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import (
    LossCause, Regime, Position, Prediction, MoveType, Side, NS,
)
from ..core.mathx import clamp, safe_div


@dataclass
class Diagnosis:
    cause: LossCause
    confidence: float
    recommendation: str          # HOLD | REDUCE | EXIT | TIGHTEN | REASSESS
    evidence: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return {"cause": self.cause.value,
                "confidence": round(self.confidence, 3),
                "recommendation": self.recommendation,
                "evidence": {k: round(v, 4) for k, v in self.evidence.items()},
                "note": self.note}


@dataclass
class Thesis:
    """What we believed at entry, kept so it can be checked later.

    Storing the entry evidence rather than just the entry price is what makes
    'has the thesis changed?' a answerable question instead of a feeling.
    """
    decision_id: str
    symbol: str
    side: int
    entry_px: float
    entry_ns: int
    p_up: float
    confidence: float
    regime: str
    alignment: float
    expected_hold_s: float
    expected_move: float
    key_signals: Dict[str, float] = field(default_factory=dict)
    stop: float = 0.0
    target: float = 0.0
    initial_risk: float = 0.0


class TradeMonitor:
    def __init__(self, noise_atr: float = 0.85,
                 weaken_edge_drop: float = 0.28,
                 invalidate_edge_flip: float = 0.12):
        self.theses: Dict[str, Thesis] = {}
        self.noise_atr = noise_atr
        self.weaken_edge_drop = weaken_edge_drop
        self.invalidate_edge_flip = invalidate_edge_flip
        self.diagnoses: List[Tuple[int, str, Diagnosis]] = []

    # ------------------------------------------------------------------
    def open_thesis(self, t: Thesis) -> None:
        self.theses[t.symbol] = t

    def close_thesis(self, symbol: str) -> Optional[Thesis]:
        return self.theses.pop(symbol, None)

    def get(self, symbol: str) -> Optional[Thesis]:
        return self.theses.get(symbol)

    # ------------------------------------------------------------------
    def diagnose(self, symbol: str, px: float, atr: float,
                 pred: Prediction, regime: Regime, alignment: float,
                 spread_bps: float, liquidity: float, now_ns: int,
                 recent_news: bool = False,
                 realised_slippage_bps: float = 0.0,
                 bar_return_z: float = 0.0) -> Optional[Diagnosis]:
        th = self.theses.get(symbol)
        if th is None or atr <= 0:
            return None

        move = (px - th.entry_px) * th.side
        move_atr = move / atr
        hold_s = (now_ns - th.entry_ns) / NS
        entry_edge = (2.0 * th.p_up - 1.0) * th.side
        now_edge = (2.0 * pred.p_up - 1.0) * th.side
        edge_drop = entry_edge - now_edge
        regime_changed = (regime.value != th.regime)
        align_now = alignment * th.side
        align_then = th.alignment * th.side

        ev = {
            "move_atr": move_atr,
            "entry_edge": entry_edge,
            "current_edge": now_edge,
            "edge_drop": edge_drop,
            "alignment_then": align_then,
            "alignment_now": align_now,
            "hold_s": hold_s,
            "spread_bps": spread_bps,
            "liquidity": liquidity,
            "slippage_bps": realised_slippage_bps,
            "confidence": pred.confidence,
        }

        # ---- ordered from most specific cause to least ----------------

        # 1. Execution/liquidity: the position is barely down in price terms
        #    but the round trip has become expensive. This is a cost problem
        #    wearing the costume of a directional one, and exiting "because
        #    it's losing" would realise the cost for nothing.
        if (abs(move_atr) < 0.5 and
                (spread_bps > 18.0 or liquidity < 0.25 or
                 abs(realised_slippage_bps) > 25.0)):
            return self._log(now_ns, symbol, Diagnosis(
                LossCause.LIQUIDITY_EXECUTION,
                confidence=clamp(0.5 + spread_bps / 60.0, 0.4, 0.9),
                recommendation="HOLD" if liquidity > 0.15 else "REDUCE",
                evidence=ev,
                note="loss is execution cost, not direction; exiting now pays "
                     "the same spread again"))

        # 2. Event shock: a jump plus news plus a volatility regime change.
        if recent_news and (abs(bar_return_z) > 3.0 or
                            regime is Regime.EVENT_DRIVEN):
            return self._log(now_ns, symbol, Diagnosis(
                LossCause.EVENT_SHOCK, confidence=0.75,
                recommendation="EXIT" if move_atr < -1.0 else "TIGHTEN",
                evidence=ev,
                note="repriced by news; the pre-event model no longer applies"))

        # 3. Regime change: the market changed character under the position.
        if regime_changed and pred.confidence > 0.35:
            hostile = regime in (Regime.HIGH_VOL, Regime.EVENT_DRIVEN,
                                 Regime.ILLIQUID)
            return self._log(now_ns, symbol, Diagnosis(
                LossCause.REGIME_CHANGE,
                confidence=clamp(0.45 + pred.confidence * 0.4, 0, 0.9),
                recommendation="REDUCE" if hostile else "TIGHTEN",
                evidence=ev,
                note=f"regime moved {th.regime} -> {regime.value}; the stop was "
                     f"sized for the old one"))

        # 4. Thesis invalidated: the edge has not merely faded, it has
        #    reversed, with confidence behind it.
        if now_edge < -self.invalidate_edge_flip and pred.confidence > 0.4:
            return self._log(now_ns, symbol, Diagnosis(
                LossCause.THESIS_INVALIDATED,
                confidence=clamp(0.5 + abs(now_edge), 0, 0.95),
                recommendation="EXIT", evidence=ev,
                note="the model now favours the other side; the reason for "
                     "being here is gone"))

        # 5. Thesis weakening: still nominally onside, but decaying.
        if edge_drop > self.weaken_edge_drop or \
                (align_then > 0.3 and align_now < 0.0):
            return self._log(now_ns, symbol, Diagnosis(
                LossCause.THESIS_WEAKENING,
                confidence=clamp(0.4 + edge_drop, 0, 0.85),
                recommendation="TIGHTEN" if move_atr > -1.0 else "REDUCE",
                evidence=ev,
                note="evidence decaying; cap what it can still cost"))

        # 6. Normal volatility: inside the noise band. Nothing has happened.
        if abs(move_atr) <= self.noise_atr:
            return self._log(now_ns, symbol, Diagnosis(
                LossCause.NORMAL_VOLATILITY,
                confidence=clamp(1.0 - abs(move_atr) / self.noise_atr, 0.3, 0.9),
                recommendation="HOLD", evidence=ev,
                note="within one ATR of entry; this is the noise the stop was "
                     "placed outside of"))

        # 7. Temporary pullback: structure intact, edge intact, just adverse.
        return self._log(now_ns, symbol, Diagnosis(
            LossCause.TEMPORARY_PULLBACK,
            confidence=clamp(0.35 + now_edge, 0.2, 0.8),
            recommendation="HOLD" if now_edge > 0.05 else "TIGHTEN",
            evidence=ev,
            note="adverse but the thesis still holds"))

    def _log(self, ts: int, symbol: str, d: Diagnosis) -> Diagnosis:
        self.diagnoses.append((ts, symbol, d))
        if len(self.diagnoses) > 5000:
            del self.diagnoses[:2500]
        return d

    # ------------------------------------------------------------------
    def thesis_was_correct(self, symbol: str, exit_px: float) -> Optional[bool]:
        """Did the market do what we predicted, regardless of the P&L?

        Deliberately separate from whether the trade made money. A trade can be
        stopped out by noise while the prediction was right, and can make money
        while the prediction was wrong. Scoring the model on P&L conflates the
        two and teaches it the wrong lesson from both.
        """
        th = self.theses.get(symbol)
        if th is None:
            return None
        realised = (exit_px - th.entry_px) * th.side
        predicted_dir = 1 if th.p_up > 0.5 else -1
        return (realised > 0) == (predicted_dir * th.side > 0)

    def cause_summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _ts, _sym, d in self.diagnoses:
            out[d.cause.value] = out.get(d.cause.value, 0) + 1
        return out

    def latest(self, symbol: str) -> Optional[Diagnosis]:
        for ts, sym, d in reversed(self.diagnoses):
            if sym == symbol:
                return d
        return None
