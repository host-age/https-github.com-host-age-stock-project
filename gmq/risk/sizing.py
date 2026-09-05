"""Position sizing (spec §5, §14).

Size is decided by the *binding* constraint, not by a single formula. Four
candidate sizes are computed and the smallest wins:

  1. **Risk budget.** Rupees between entry and stop must not exceed the
     per-trade risk limit. This is the constraint that should usually bind.
  2. **Volatility target.** Scale down in high volatility so that the position's
     contribution to portfolio variance is roughly constant. Without this,
     a fixed-risk sizer takes wildly different amounts of real risk in a calm
     tape versus a violent one.
  3. **Fractional Kelly.** Bet more when the edge is larger and better
     evidenced. Quarter-Kelly at most -- full Kelly is the growth-optimal bet
     only if the probabilities are exactly right, and they are not. Kelly is
     brutally sensitive to overestimated edge: a modest overestimate turns the
     growth-optimal bet into a negative-growth one, so the fraction is small
     and shrinks further when the model's demonstrated skill is weak.
  4. **Liquidity.** Never take more than a modest share of visible depth; the
     impact cost of unwinding it otherwise exceeds any edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..core.types import Prediction
from ..core.config import RiskLimits
from ..core.mathx import clamp, kelly_fraction, safe_div


@dataclass
class SizeDecision:
    qty: int
    binding: str                     # which constraint set the size
    risk_rupees: float
    detail: Dict[str, int]

    def as_dict(self) -> dict:
        return {"qty": self.qty, "binding": self.binding,
                "risk": round(self.risk_rupees, 2), "candidates": self.detail}


class PositionSizer:
    def __init__(self, limits: RiskLimits, target_vol_ann: float = 0.14,
                 kelly_cap: float = 0.25, max_depth_participation: float = 0.20):
        self.limits = limits
        self.target_vol_ann = target_vol_ann
        self.kelly_cap = kelly_cap
        self.max_depth_participation = max_depth_participation

    # ------------------------------------------------------------------
    def size(self, equity: float, px: float, stop: float,
             pred: Prediction, atr: float, ann_vol: float,
             top_depth_qty: int = 0, lot_size: int = 1,
             model_skill: float = 0.0,
             existing_exposure_pct: float = 0.0) -> SizeDecision:
        if px <= 0 or equity <= 0:
            return SizeDecision(0, "invalid", 0.0, {})
        stop_dist = abs(px - stop)
        if stop_dist <= 0:
            stop_dist = max(atr, px * 0.002)

        cands: Dict[str, int] = {}

        # 1 -- risk budget
        risk_budget = equity * self.limits.max_risk_per_trade_pct / 100.0
        cands["risk"] = int(risk_budget / stop_dist)

        # 2 -- volatility target
        if ann_vol > 1e-6:
            target_notional = equity * self.target_vol_ann / ann_vol
            cands["vol_target"] = int(target_notional / px)
        else:
            cands["vol_target"] = cands["risk"]

        # 3 -- fractional Kelly on the barrier probabilities
        p_t, p_s = pred.p_target, pred.p_stop
        if p_t + p_s > 1e-6:
            p_win = p_t / (p_t + p_s)
        else:
            p_win = clamp(pred.p_up, 0.01, 0.99)
        # payoff ratio implied by where the target and stop actually sit
        win_r = 1.6
        f = kelly_fraction(p_win, win_r, 1.0)
        # Shrink toward zero by how much skill the model has actually shown.
        # Kelly on an unvalidated probability is not aggressive, it is reckless.
        shrink = clamp(0.25 + 0.75 * clamp(model_skill * 3.0, 0.0, 1.0), 0.0, 1.0)
        shrink *= clamp(pred.confidence, 0.0, 1.0)
        f_eff = clamp(f * self.kelly_cap * shrink, 0.0, self.kelly_cap)
        kelly_risk = equity * f_eff
        cands["kelly"] = int(kelly_risk / stop_dist) if stop_dist > 0 else 0

        # 4 -- liquidity
        if top_depth_qty > 0:
            cands["liquidity"] = int(top_depth_qty * self.max_depth_participation)
        else:
            cands["liquidity"] = cands["risk"]

        # 5 -- single-name concentration cap
        room_pct = max(0.0, self.limits.max_position_pct - existing_exposure_pct)
        cands["concentration"] = int(equity * room_pct / 100.0 / px)

        binding = min(cands, key=lambda k: cands[k])
        qty = max(0, cands[binding])
        if lot_size > 1:
            qty = (qty // lot_size) * lot_size
        return SizeDecision(qty=qty, binding=binding,
                            risk_rupees=qty * stop_dist, detail=cands)

    # ------------------------------------------------------------------
    def heat_scale(self, open_risk_pct: float, drawdown_pct: float) -> float:
        """Multiplier applied to new positions as the book heats up.

        Two independent brakes. Portfolio heat: as total open risk approaches
        the budget, new positions shrink so the last trade of the day cannot
        be the same size as the first. Drawdown: risk taken *down* is reduced,
        because the alternative -- trading bigger to recover faster -- is the
        single most reliable way to turn a drawdown into a ruin.
        """
        heat = clamp(1.0 - open_risk_pct / 3.0, 0.15, 1.0)
        dd = clamp(1.0 - drawdown_pct / self.limits.max_drawdown_pct * 0.8,
                   0.25, 1.0)
        return float(heat * dd)
