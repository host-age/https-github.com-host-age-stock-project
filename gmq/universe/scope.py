"""NSE trading-universe scope and cross-segment context policy.

Equities/cash remain the primary decision axis. Other NSE segments are
contextual inputs linked to an equity whenever a measurable/economically
relevant relationship exists. A context instrument is never promoted to a
trade target by this policy alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Iterable


class Segment(str, Enum):
    EQUITY = "equity"
    EQUITY_DERIVATIVES = "equity_derivatives"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    OTHER = "other"


class Role(str, Enum):
    PRIMARY = "primary"
    CONTEXT = "context"


@dataclass(frozen=True)
class InstrumentLink:
    equity_symbol: str
    instrument_symbol: str
    segment: Segment
    role: Role = Role.CONTEXT
    relevance: float = 1.0
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.equity_symbol.strip() or not self.instrument_symbol.strip():
            raise ValueError("symbols must be non-empty")
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError("relevance must be between 0 and 1")
        if self.segment is Segment.EQUITY and self.role is not Role.PRIMARY:
            raise ValueError("equity instruments must be primary")


@dataclass(frozen=True)
class UniverseScope:
    primary_segment: Segment = Segment.EQUITY
    contextual_segments: FrozenSet[Segment] = frozenset({
        Segment.EQUITY_DERIVATIVES,
        Segment.CURRENCY,
        Segment.COMMODITY,
    })
    allow_context_to_trade: bool = False

    def __post_init__(self) -> None:
        if self.primary_segment is not Segment.EQUITY:
            raise ValueError("GMQ primary decision segment must be NSE equity")
        if Segment.EQUITY in self.contextual_segments:
            raise ValueError("equity cannot be a contextual-only segment")

    def is_primary(self, segment: Segment) -> bool:
        return segment is self.primary_segment

    def is_contextual(self, segment: Segment) -> bool:
        return segment in self.contextual_segments

    def admissible_for_trade(self, segment: Segment) -> bool:
        return self.is_primary(segment) or (
            self.allow_context_to_trade and self.is_contextual(segment)
        )

    def required_segments(self) -> FrozenSet[Segment]:
        return frozenset({self.primary_segment, *self.contextual_segments})


def linked_context(
    equity_symbol: str,
    links: Iterable[InstrumentLink],
    *,
    min_relevance: float = 0.0,
) -> list[InstrumentLink]:
    """Return relevant contextual instruments for one equity, strongest first."""
    symbol = equity_symbol.strip().upper()
    if not symbol:
        return []
    return sorted(
        (
            link
            for link in links
            if link.equity_symbol.upper() == symbol
            and link.role is Role.CONTEXT
            and link.relevance >= min_relevance
        ),
        key=lambda link: link.relevance,
        reverse=True,
    )


def context_feature_namespace(link: InstrumentLink) -> str:
    """Stable feature prefix for contextual data, e.g. ``fo.NIFTY:...``."""
    return f"{link.segment.value}.{link.instrument_symbol.upper()}"
