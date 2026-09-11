"""Point-in-time market knowledge queries for leak-free replay."""
from __future__ import annotations

from dataclasses import dataclass

from ..data.feed import ReplayRow
from ..data.nse.archive import NSEArchive


@dataclass(frozen=True)
class KnowledgeSnapshot:
    symbol: str
    at_ts_ns: int
    bars: list[ReplayRow]

    @property
    def last_close(self) -> float:
        return self.bars[-1].c if self.bars else 0.0


class PointInTimeKnowledge:
    """Only returns records whose timestamp is <= the requested decision time."""

    def __init__(self, archive: NSEArchive):
        self.archive = archive

    def market(self, symbol: str, at_ts_ns: int, *, lookback: int = 500) -> KnowledgeSnapshot:
        return KnowledgeSnapshot(
            symbol=symbol.upper(),
            at_ts_ns=at_ts_ns,
            bars=self.archive.snapshot(symbol.upper(), at_ts_ns, limit=lookback),
        )
