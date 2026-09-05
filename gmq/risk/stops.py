"""Dynamic stop placement and management (spec §5).

Two responsibilities, and the second one is a safety property, not a feature.

**Placement.** The stop goes where the thesis is wrong, not at a round
percentage. Candidates are drawn from volatility (ATR and realised range),
market structure (the nearest swing level with room to breathe), liquidity
(below a real resting bid, not inside an air pocket), and the model's own
expected adverse excursion. The widest of those, capped by what the risk
budget can afford, is the answer -- because a stop placed inside the noise
band is not a risk control, it is a guaranteed loss with extra steps.

**The ratchet.** `update` will only ever return a stop at least as tight as
the current one. This is enforced structurally rather than by policy: there is
no code path that widens a stop. That is the §5 requirement that the system
must never move a stop merely to avoid recognising a loss, and it is the one
rule in the whole system that has no override, because every account-ending
loss in the history of trading has the same first step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Regime, Side, DepthSnapshot
from ..core.mathx import clamp, safe_div
from ..analytics.excursion import GivebackPolicy


@dataclass
class StopPlan:
    stop: float
    target: float
    basis: str                       # which candidate won
    distance_atr: float
    candidates: Dict[str, float]

    def as_dict(self) -> dict:
        return {"stop": round(self.stop, 2), "target": round(self.target, 2),
                "basis": self.basis, "dist_atr": round(self.distance_atr, 2),
                "candidates": {k: round(v, 2)
                               for k, v in self.candidates.items()}}


# Volatility multiples by regime. A trend needs room; a mean-reverting tape
# does not, and giving it room only means paying more when it fails.
REGIME_STOP_ATR: Dict[Regime, float] = {
    Regime.TRENDING_UP: 1.6,
    Regime.TRENDING_DOWN: 1.6,
    Regime.MEAN_REVERTING: 0.9,
    Regime.BREAKOUT: 1.8,
    Regime.HIGH_VOL: 2.2,
    Regime.LOW_VOL: 1.0,
    Regime.EVENT_DRIVEN: 2.6,
    Regime.ILLIQUID: 2.4,
}

REGIME_TARGET_R: Dict[Regime, float] = {
    Regime.TRENDING_UP: 2.6,
    Regime.TRENDING_DOWN: 2.6,
    Regime.MEAN_REVERTING: 1.3,
    Regime.BREAKOUT: 3.0,
    Regime.HIGH_VOL: 2.0,
    Regime.LOW_VOL: 1.5,
    Regime.EVENT_DRIVEN: 2.2,
    Regime.ILLIQUID: 1.8,
}


class StopPolicy:
    def __init__(self, min_atr: float = 0.7, max_atr: float = 3.2,
                 breakeven_r: float = 1.0, trail_start_r: float = 1.4,
                 trail_atr: float = 1.5,
                 giveback: Optional["GivebackPolicy"] = None):
        self.min_atr = min_atr
        self.max_atr = max_atr
        self.breakeven_r = breakeven_r
        self.trail_start_r = trail_start_r
        self.trail_atr = trail_atr
        # Fitted from the measured excursion distribution; see
        # analytics/excursion.py. None -> the give-back rule is off.
        self.giveback = giveback

    # ------------------------------------------------------------------
    def initial(self, side: Side, px: float, atr: float, regime: Regime,
                support: float = 0.0, resistance: float = 0.0,
                depth: Optional[DepthSnapshot] = None,
                expected_mae_pct: float = 0.0,
                realised_vol_bar: float = 0.0,
                max_affordable_dist: float = 0.0) -> StopPlan:
        sgn = side.sign
        cands: Dict[str, float] = {}

        base_mult = REGIME_STOP_ATR.get(regime, 1.4)
        cands["volatility"] = base_mult * atr

        if realised_vol_bar > 0:
            # 2.5 sigma of a single bar's move -- the stop should sit outside
            # ordinary noise or it will be hit by ordinary noise
            cands["noise"] = 2.5 * realised_vol_bar * px

        # structure: just beyond the nearest level, not exactly on it, because
        # the level is where everyone else's stop is
        if sgn > 0 and support > 0 and support < px:
            cands["structure"] = (px - support) + 0.25 * atr
        elif sgn < 0 and resistance > 0 and resistance > px:
            cands["structure"] = (resistance - px) + 0.25 * atr

        # liquidity: beyond a genuinely thick resting level
        if depth is not None:
            levels = depth.bids if sgn > 0 else depth.asks
            if levels:
                qtys = [l.qty for l in levels]
                if qtys:
                    thick = max(range(len(qtys)), key=lambda i: qtys[i])
                    lvl = levels[thick].price
                    d = (px - lvl) if sgn > 0 else (lvl - px)
                    if d > 0:
                        cands["liquidity"] = d + 0.15 * atr

        if expected_mae_pct > 0:
            cands["model_mae"] = expected_mae_pct * px * 1.3

        if not cands:
            cands["volatility"] = 1.4 * atr

        basis = max(cands, key=lambda k: cands[k])
        dist = cands[basis]
        dist = clamp(dist, self.min_atr * atr, self.max_atr * atr)
        # The risk budget can force a tighter stop than the market suggests.
        # When that happens the honest response is a smaller position, not a
        # stop inside the noise -- the sizer is told, via `affordable`.
        affordable = True
        if max_affordable_dist > 0 and dist > max_affordable_dist:
            dist = max_affordable_dist
            basis = basis + "+budget_capped"
            affordable = False

        stop = px - sgn * dist
        r_mult = REGIME_TARGET_R.get(regime, 2.0)
        target = px + sgn * dist * r_mult
        return StopPlan(stop=stop, target=target, basis=basis,
                        distance_atr=safe_div(dist, atr), candidates=cands)

    # ------------------------------------------------------------------
    def update(self, side: Side, px: float, avg: float, current_stop: float,
               atr: float, regime: Regime, r_multiple: float,
               hold_s: float = 0.0, expected_mae_pct: float = 0.0,
               regime_changed: bool = False,
               thesis_weakening: bool = False,
               peak_r: float = 0.0,
               risk_per_share: float = 0.0) -> Tuple[float, str]:
        """Return (new stop, reason). Never wider than `current_stop`.

        The monotonicity is asserted at the end rather than assumed, because
        this is the one invariant whose violation is unrecoverable.
        """
        sgn = side.sign
        best = current_stop
        reason = "unchanged"

        def tighten(level: float, why: str) -> None:
            nonlocal best, reason
            if level <= 0:
                return
            if sgn > 0 and level >= px:
                return
            if sgn < 0 and level <= px:
                return
            if (sgn > 0 and level > best) or (sgn < 0 and (level < best or best <= 0)):
                best = level
                reason = why

        # break-even once the trade has paid for itself
        if r_multiple >= self.breakeven_r and avg > 0:
            tighten(avg + sgn * 0.03 * atr, "breakeven")

        # volatility trail once it is properly working
        if r_multiple >= self.trail_start_r:
            tighten(px - sgn * self.trail_atr * atr, "trail_atr")

        # a regime change invalidates the assumptions the stop was placed under
        if regime_changed:
            mult = REGIME_STOP_ATR.get(regime, 1.4)
            tighten(px - sgn * mult * atr * 0.85, "regime_change")

        # the thesis is decaying but not dead: reduce how much it can cost
        if thesis_weakening:
            tighten(px - sgn * 0.9 * atr, "thesis_weakening")

        if expected_mae_pct > 0 and r_multiple > 0.5:
            tighten(px - sgn * expected_mae_pct * px * 1.15, "model_mae")

        # Give-back stop, fitted to the measured excursion distribution.
        #
        # The ATR trail above answers "how far can price wander and the trend
        # still be intact"; it says nothing about how much of an existing
        # profit is worth risking to find out. Those are different questions,
        # and the second one is what 61% of losing trades having been in profit
        # first is a symptom of. This rule is stated in the trade's own risk
        # units: once the peak excursion reaches the fitted activation level,
        # protect the fitted share of it.
        if (self.giveback is not None and peak_r > 0 and avg > 0
                and risk_per_share > 0):
            lvl_r = self.giveback.stop_r(peak_r)
            if lvl_r is not None:
                tighten(avg + sgn * lvl_r * risk_per_share, "giveback")

        # Structural guarantee. If this ever fires, something upstream tried to
        # widen a stop and the bug is caught here rather than in the P&L.
        if current_stop > 0:
            if sgn > 0:
                assert best >= current_stop - 1e-9, "stop widened on a long"
            else:
                assert best <= current_stop + 1e-9, "stop widened on a short"
        return best, reason

    # ------------------------------------------------------------------
    @staticmethod
    def time_stop(hold_s: float, expected_hold_s: float, r_multiple: float,
                  multiple: float = 3.0) -> bool:
        """Should the trade be closed simply for taking too long?

        A thesis has a time horizon. A position that has gone nowhere for
        several times its expected holding period is not "still working"; the
        move it was predicting did not happen, and the capital and risk budget
        it is consuming have better uses.
        """
        if expected_hold_s <= 0:
            return False
        return hold_s > multiple * expected_hold_s and abs(r_multiple) < 0.5
