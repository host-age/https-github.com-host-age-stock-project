"""Move generation -- the legal move list at each node (spec §2).

The nine actions from the specification, turned into concrete candidate moves
with real quantities, stops and targets. Two design points:

* **Candidates are parameterised, not abstract.** "REDUCE" is not one move; it
  is reduce-by-a-third and reduce-by-two-thirds, which have different expected
  values. The search compares them as separate moves.

* **Illegal moves are removed here, not scored badly later.** Scaling into a
  position that is already at its size cap, or reversing a position opened
  four seconds ago, should not appear in the move list at all. Filtering by
  legality up front keeps the branching factor honest and stops the search
  from spending its time budget on moves it could never play.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..core.types import MoveType, Side, Regime, Prediction
from ..core.mathx import clamp
from ..analytics.excursion import GivebackPolicy


@dataclass(slots=True)
class Candidate:
    """A concrete move the engine may play."""
    move: MoveType
    qty: int = 0                 # signed change in position
    stop: float = 0.0
    target: float = 0.0
    limit_px: float = 0.0        # 0 -> marketable
    passive: bool = False
    tag: str = ""

    def describe(self) -> str:
        if self.move is MoveType.WAIT:
            return "wait"
        return f"{self.move.value.lower()} {abs(self.qty)}{' (passive)' if self.passive else ''}"


@dataclass(slots=True)
class MoveContext:
    """Everything the generator needs about the current situation."""
    symbol: str
    px: float
    bid: float
    ask: float
    atr: float
    qty: int                     # current signed position
    avg_price: float
    stop: float
    target: float
    max_qty: int                 # hard cap from the risk engine
    unit_qty: int                # risk-sized entry quantity
    regime: Regime
    alignment: float
    alignment_conflict: bool
    pred: Prediction
    hold_s: float = 0.0
    partials: int = 0
    scale_ins: int = 0
    liquidity: float = 1.0
    r_multiple: float = 0.0
    seconds_to_close: float = 1e9
    can_open: bool = True        # risk engine may forbid new risk entirely
    expected_mae_pct: float = 0.0
    peak_r: float = 0.0          # max favourable excursion so far, in R
    risk_per_share: float = 0.0  # |entry - initial stop|, for R arithmetic
    s_since_exit: float = 1e9    # seconds since this symbol was last flattened


class MoveGenerator:
    def __init__(self, max_scale_ins: int = 2, max_partials: int = 3,
                 min_hold_for_reverse_s: float = 45.0,
                 stop_atr: float = 1.0, target_atr: float = 1.8,
                 trail_atr: float = 1.3, min_entry_confidence: float = 0.20,
                 min_entry_edge: float = 0.06,
                 reentry_cooldown_s: float = 120.0,
                 giveback: Optional["GivebackPolicy"] = None):
        self.min_entry_confidence = min_entry_confidence
        self.min_entry_edge = min_entry_edge
        self.reentry_cooldown_s = reentry_cooldown_s
        self.giveback = giveback
        self.max_scale_ins = max_scale_ins
        self.max_partials = max_partials
        self.min_hold_for_reverse_s = min_hold_for_reverse_s
        self.stop_atr = stop_atr
        self.target_atr = target_atr
        self.trail_atr = trail_atr

    # ------------------------------------------------------------------
    def generate(self, c: MoveContext) -> List[Candidate]:
        moves: List[Candidate] = [Candidate(MoveType.WAIT, tag="hold")]
        if c.atr <= 0 or c.px <= 0:
            return moves
        flat = c.qty == 0
        long_ = c.qty > 0
        short = c.qty < 0

        if flat:
            moves.extend(self._entries(c))
        else:
            moves.extend(self._management(c))
            moves.extend(self._reversal(c))
        return moves

    # ------------------------------------------------------------------
    def _entries(self, c: MoveContext) -> List[Candidate]:
        out: List[Candidate] = []
        if not c.can_open or c.unit_qty <= 0:
            return out
        # Forced flat near the close: opening new intraday risk minutes before
        # square-off is a guaranteed round-trip cost with no time to be right.
        if c.seconds_to_close < 420:
            return out
        # Re-entry cooldown. Having just decided this name was not worth
        # holding, deciding four seconds later that it is worth owning again
        # is not a new thesis -- it is the same estimator crossing the same
        # threshold from the other side, and each crossing costs a round trip.
        # The cooldown does not forbid the trade, it forbids the *immediate*
        # repeat, which is the only version of it that cannot possibly be
        # informed by anything new.
        if c.s_since_exit < self.reentry_cooldown_s:
            return out
        base = min(c.unit_qty, c.max_qty)
        if base <= 0:
            return out
        # No opinion, no trade. Generating entries the objective will reject
        # anyway just burns the latency budget, and every one of them is a
        # chance for Monte Carlo noise to promote a coin flip into a position.
        edge = abs(2.0 * c.pred.p_up - 1.0)
        if c.pred.confidence < self.min_entry_confidence or \
                edge < self.min_entry_edge:
            return out

        for side, mt in ((Side.BUY, MoveType.ENTER_LONG),
                         (Side.SELL, MoveType.ENTER_SHORT)):
            sgn = side.sign
            # The multi-timeframe constraint from §4: a short-horizon entry
            # against a materially stronger higher-timeframe structure is
            # allowed only at reduced size, never at full size.
            against = (c.alignment_conflict and
                       np.sign(c.alignment) == -sgn and abs(c.alignment) > 0.3)
            sizes = [(base, "full")]
            if against:
                sizes = [(max(1, int(base * 0.4)), "reduced_vs_htf")]
            elif abs(c.alignment) > 0.45 and np.sign(c.alignment) == sgn:
                sizes = [(base, "full"), (max(1, int(base * 0.6)), "half")]
            stop = c.px - sgn * self.stop_atr * c.atr
            target = c.px + sgn * self.target_atr * c.atr
            for q, tag in sizes:
                out.append(Candidate(mt, qty=sgn * q, stop=stop, target=target,
                                     tag=tag))
                # A passive entry at the touch: cheaper if it fills, and in a
                # mean-reverting or illiquid regime the spread saved is a large
                # fraction of the whole edge.
                if c.regime in (Regime.MEAN_REVERTING, Regime.LOW_VOL,
                                Regime.ILLIQUID) and c.bid > 0 and c.ask > 0:
                    px = c.bid if side is Side.BUY else c.ask
                    out.append(Candidate(mt, qty=sgn * q, stop=stop,
                                         target=target, limit_px=px,
                                         passive=True, tag=tag + "_passive"))
        return out

    # ------------------------------------------------------------------
    def _management(self, c: MoveContext) -> List[Candidate]:
        out: List[Candidate] = []
        sgn = 1 if c.qty > 0 else -1
        held = abs(c.qty)

        # -- EXIT
        out.append(Candidate(MoveType.EXIT, qty=-c.qty, tag="flatten"))

        # -- REDUCE / TAKE_PARTIAL
        for frac, mt, tag in ((0.33, MoveType.TAKE_PARTIAL, "partial_third"),
                              (0.50, MoveType.REDUCE, "reduce_half")):
            q = int(held * frac)
            if q >= 1 and q < held:
                if mt is MoveType.TAKE_PARTIAL and c.partials >= self.max_partials:
                    continue
                # Taking profit only makes sense in profit; "taking a partial"
                # at a loss is just a reduction, and is generated as one.
                if mt is MoveType.TAKE_PARTIAL and c.r_multiple < 0.5:
                    continue
                out.append(Candidate(mt, qty=-sgn * q, tag=tag))

        # -- INCREASE (pyramiding), only with the position working
        if (c.scale_ins < self.max_scale_ins and held < c.max_qty
                and c.can_open and c.r_multiple > 0.6
                and np.sign(c.alignment) == sgn
                and c.seconds_to_close > 600):
            add = min(max(1, int(c.unit_qty * 0.5)), c.max_qty - held)
            if add >= 1:
                # Scaling in raises the average price; the stop moves up with
                # it so total risk does not grow with the position.
                new_avg = (c.avg_price * held + c.px * add) / (held + add)
                stop = new_avg - sgn * self.stop_atr * c.atr
                if sgn > 0:
                    stop = max(stop, c.stop)
                else:
                    stop = min(stop, c.stop) if c.stop else stop
                out.append(Candidate(MoveType.INCREASE, qty=sgn * add,
                                     stop=stop, tag="pyramid"))

        # -- MOVE_STOP
        out.extend(self._stop_moves(c, sgn))
        return out

    def _stop_moves(self, c: MoveContext, sgn: int) -> List[Candidate]:
        out: List[Candidate] = []
        cands: List[tuple] = []
        # trail at a volatility multiple
        trail = c.px - sgn * self.trail_atr * c.atr
        cands.append((trail, "trail_atr"))
        # break-even once the trade has paid for itself
        if c.r_multiple >= 1.0 and c.avg_price > 0:
            be = c.avg_price + sgn * (c.atr * 0.05)
            cands.append((be, "breakeven"))
        # tighten to the model's expected adverse excursion
        if c.expected_mae_pct > 0:
            mae = c.px - sgn * c.expected_mae_pct * c.px * 1.25
            cands.append((mae, "expected_mae"))
        # protect a fitted share of the peak once the peak is big enough
        if (self.giveback is not None and c.peak_r > 0
                and c.risk_per_share > 0 and c.avg_price > 0):
            lvl_r = self.giveback.stop_r(c.peak_r)
            if lvl_r is not None:
                cands.append((c.avg_price + sgn * lvl_r * c.risk_per_share,
                              "giveback"))

        for lvl, tag in cands:
            if lvl <= 0:
                continue
            # A stop may only ever move in the direction that reduces risk.
            # This is the hard rule from §5: widening a stop to avoid booking
            # a loss is the single most reliable way to turn a small loss into
            # an account-ending one, so the move is not generated at all --
            # it is not scored badly, it does not exist.
            if c.stop > 0:
                if sgn > 0 and lvl <= c.stop:
                    continue
                if sgn < 0 and lvl >= c.stop:
                    continue
            # never place a stop through the current price
            if sgn > 0 and lvl >= c.px:
                continue
            if sgn < 0 and lvl <= c.px:
                continue
            out.append(Candidate(MoveType.MOVE_STOP, qty=0, stop=lvl, tag=tag))
        return out

    # ------------------------------------------------------------------
    def _reversal(self, c: MoveContext) -> List[Candidate]:
        if not c.can_open or c.hold_s < self.min_hold_for_reverse_s:
            return []
        if c.seconds_to_close < 600 or c.unit_qty <= 0:
            return []
        sgn = 1 if c.qty > 0 else -1
        edge = 2.0 * c.pred.p_up - 1.0
        # Reversing pays two spreads and admits the thesis was wrong. It needs
        # a genuinely strong opposing signal, not merely a weak current one.
        if np.sign(edge) == sgn or abs(edge) < 0.25 or c.pred.confidence < 0.4:
            return []
        new_sgn = -sgn
        q = min(c.unit_qty, c.max_qty)
        if q <= 0:
            return []
        stop = c.px - new_sgn * self.stop_atr * c.atr
        target = c.px + new_sgn * self.target_atr * c.atr
        return [Candidate(MoveType.REVERSE, qty=-c.qty + new_sgn * q,
                          stop=stop, target=target, tag="reverse")]
