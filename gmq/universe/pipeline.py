"""Three-stage NSE equity universe selection.

Stages:
  1. knowledge: retain every eligible equity for which reliable data exists;
  2. tradable: apply liquidity/data/tradability/risk-budget gates;
  3. candidates: rank the surviving equities for expensive deep analysis.

This module is deliberately deterministic and dependency-light so it can be
used by simulation, replay and live orchestration alike. It does not place
orders and it does not assume NIFTY-100 membership.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class SecuritySnapshot:
    """Point-in-time facts used for universe selection."""

    symbol: str
    isin: str = ""
    sector: str = ""
    industry: str = ""
    last_price: float = 0.0
    avg_daily_value: float = 0.0
    spread_bps: float = 0.0
    volatility: float = 0.0
    data_quality: float = 1.0
    tradable: bool = True
    halted: bool = False
    auction: bool = False
    price_available: bool = True
    circuit_blocked: bool = False
    position_feasible: bool = True
    instrument_eligible: bool = True
    knowledge_complete: bool = True
    signal_score: float = 0.0
    news_score: float = 0.0
    fundamental_score: float = 0.0
    historical_edge: float = 0.0
    risk_score: float = 0.0
    candidate_score: float = field(init=False)

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must be non-empty")
        if self.avg_daily_value < 0:
            raise ValueError("avg_daily_value must be non-negative")
        if self.spread_bps < 0 or self.volatility < 0:
            raise ValueError("spread and volatility must be non-negative")
        for name, value in (("data_quality", self.data_quality), ("risk_score", self.risk_score)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        score = (
            0.30 * self.signal_score
            + 0.20 * self.news_score
            + 0.20 * self.fundamental_score
            + 0.20 * self.historical_edge
            + 0.10 * (1.0 - self.risk_score)
        )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "candidate_score", float(score))


@dataclass(frozen=True)
class UniverseFilterConfig:
    """Conservative defaults; tune from data, not intuition alone."""

    min_avg_daily_value: float = 50_000_000.0
    max_spread_bps: float = 40.0
    min_data_quality: float = 0.90
    max_volatility: float = 0.25
    min_candidate_score: float = 0.20
    max_candidates: int = 20

    def __post_init__(self) -> None:
        if self.min_avg_daily_value < 0 or self.max_spread_bps < 0:
            raise ValueError("liquidity thresholds must be non-negative")
        if not 0 <= self.min_data_quality <= 1:
            raise ValueError("min_data_quality must be between 0 and 1")
        if self.max_volatility < 0:
            raise ValueError("max_volatility must be non-negative")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")


@dataclass(frozen=True)
class UniverseResult:
    knowledge: List[SecuritySnapshot]
    tradable: List[SecuritySnapshot]
    candidates: List[SecuritySnapshot]

    @property
    def counts(self) -> dict:
        return {
            "knowledge": len(self.knowledge),
            "tradable": len(self.tradable),
            "candidates": len(self.candidates),
        }


class UniversePipeline:
    """Implement the Knowledge -> Tradable -> Candidates funnel."""

    def __init__(self, config: UniverseFilterConfig | None = None):
        self.config = config or UniverseFilterConfig()

    def knowledge_universe(self, securities: Iterable[SecuritySnapshot]) -> list[SecuritySnapshot]:
        """Keep all securities with an identifiable symbol and usable knowledge."""
        out = []
        seen = set()
        for security in securities:
            if not security.knowledge_complete or not security.instrument_eligible:
                continue
            if security.symbol in seen:
                continue
            seen.add(security.symbol)
            out.append(security)
        return sorted(out, key=lambda x: x.symbol)

    def tradable_universe(self, securities: Iterable[SecuritySnapshot]) -> list[SecuritySnapshot]:
        """Filter for execution feasibility and data quality."""
        c = self.config
        out = []
        for s in securities:
            if not s.tradable or s.halted or s.auction or s.circuit_blocked:
                continue
            if not s.price_available or not s.position_feasible:
                continue
            if s.data_quality < c.min_data_quality:
                continue
            if s.avg_daily_value < c.min_avg_daily_value:
                continue
            if s.spread_bps > c.max_spread_bps:
                continue
            if s.volatility > c.max_volatility:
                continue
            out.append(s)
        return sorted(out, key=lambda x: (-x.candidate_score, x.symbol))

    def candidate_universe(self, securities: Iterable[SecuritySnapshot]) -> list[SecuritySnapshot]:
        """Rank tradable names before expensive model/LLM/Grandmaster work."""
        c = self.config
        ranked = [s for s in securities if s.candidate_score >= c.min_candidate_score]
        ranked.sort(key=lambda x: (-x.candidate_score, x.symbol))
        return ranked[: c.max_candidates]

    def run(self, securities: Sequence[SecuritySnapshot]) -> UniverseResult:
        knowledge = self.knowledge_universe(securities)
        tradable = self.tradable_universe(knowledge)
        candidates = self.candidate_universe(tradable)
        return UniverseResult(knowledge=knowledge, tradable=tradable, candidates=candidates)
