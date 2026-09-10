"""Pre-trade knowledge gate for GMQ.

The gate is intentionally conservative. It does not create directional alpha by
itself; it checks whether the existing signal is economically sensible given
what GMQ currently knows about the security, its recent behaviour and upcoming
known events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..core.mathx import clamp
from .store import KnowledgeStore


@dataclass(slots=True)
class PreTradeVerdict:
    allowed: bool
    score: float
    reason: str
    warnings: List[str] = field(default_factory=list)
    win_rate: float = 0.5
    event_risk: float = 0.0
    data_quality: float = 0.0

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "score": round(self.score, 4),
            "reason": self.reason,
            "warnings": list(self.warnings),
            "win_rate": round(self.win_rate, 4),
            "event_risk": round(self.event_risk, 4),
            "data_quality": round(self.data_quality, 4),
        }


class PreTradeKnowledge:
    """Evaluate the strength/quality of the information available now."""

    def __init__(self, store: KnowledgeStore,
                 *, min_score: float = 0.45,
                 min_liquidity: float = 0.15,
                 max_event_risk: float = 0.90,
                 min_observations: int = 50):
        self.store = store
        self.min_score = min_score
        self.min_liquidity = min_liquidity
        self.max_event_risk = max_event_risk
        self.min_observations = min_observations

    def evaluate(self, symbol: str, now_ns: int, *, direction: int,
                 confidence: float, expected_edge_bps: float,
                 current_spread_bps: float, liquidity: float,
                 data_fresh: bool = True) -> PreTradeVerdict:
        k = self.store.snapshot(symbol, now_ns)
        hist = self.store.historical_win_rate(symbol)
        warnings: List[str] = []

        data_quality = 1.0 if data_fresh else 0.0
        if k.observed_ticks < self.min_observations:
            data_quality *= clamp(k.observed_ticks / self.min_observations, 0.0, 1.0)
            warnings.append("limited_security_history")
        if not data_fresh:
            warnings.append("stale_market_data")
        if liquidity < self.min_liquidity:
            warnings.append("low_liquidity")
        if current_spread_bps > 25.0:
            warnings.append("wide_spread")
        if k.event_risk > 0.50:
            warnings.append("material_upcoming_event")
        if abs(direction) == 0:
            warnings.append("no_direction")

        # Blend observable quality, current edge, model confidence and the
        # symbol's own historical outcomes. Event risk subtracts from the
        # score rather than being treated as a directional signal.
        edge_term = clamp(expected_edge_bps / 20.0, -1.0, 1.0)
        edge_score = 0.5 + 0.5 * edge_term
        conf_score = clamp(confidence, 0.0, 1.0)
        liq_score = clamp(liquidity, 0.0, 1.0)
        spread_penalty = clamp(current_spread_bps / 40.0, 0.0, 1.0)
        score = (
            0.30 * hist
            + 0.25 * conf_score
            + 0.20 * edge_score
            + 0.15 * liq_score
            + 0.10 * data_quality
        )
        score -= 0.20 * spread_penalty
        score -= 0.30 * clamp(k.event_risk, 0.0, 1.0)

        blocked = []
        if not data_fresh:
            blocked.append("stale_market_data")
        if liquidity < self.min_liquidity:
            blocked.append("low_liquidity")
        if k.event_risk > self.max_event_risk:
            blocked.append("event_risk_limit")
        if direction == 0:
            blocked.append("no_direction")
        if expected_edge_bps <= 0:
            blocked.append("non_positive_edge")

        allowed = not blocked and score >= self.min_score
        if not allowed and not blocked:
            blocked.append("knowledge_score_below_threshold")
        reason = "approved" if allowed else ",".join(blocked)
        return PreTradeVerdict(
            allowed=allowed,
            score=float(clamp(score, 0.0, 1.0)),
            reason=reason,
            warnings=warnings,
            win_rate=hist,
            event_risk=k.event_risk,
            data_quality=data_quality,
        )
