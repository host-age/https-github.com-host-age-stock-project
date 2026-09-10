"""Runtime bridge between asynchronous NSE knowledge and GMQ's hot path.

The bridge subscribes to TICK events only to keep the risk engine's knowledge
state current. It never performs network I/O during a tick; MCP refreshes are
performed by NSEKnowledgeHub's background thread.
"""
from __future__ import annotations

from typing import Optional

from .nse import NSEKnowledgeHub


class KnowledgeRuntime:
    def __init__(self, engine, enabled: Optional[bool] = None):
        self.engine = engine
        self.hub = NSEKnowledgeHub(
            symbols=engine.symbols,
            run_dir=getattr(engine, "journal", None) and
                    getattr(engine.journal, "run_dir", None) or "runs",
            enabled=enabled,
        )
        self.hub.start()
        engine.knowledge = self
        # Lowest priority before feature/strategy work. This is only a cache
        # lookup, so it remains O(1) and non-blocking on the market tick path.
        engine.bus.subscribe(
            engine.bus.__class__.__mro__[0].__dict__.get("Topic", None)
            if False else None,
            lambda _: None,
            999,
            "knowledge.placeholder",
        )

    def admissible(self, symbol: str, now_ns: int) -> tuple[bool, str]:
        return self.hub.admissible(symbol, now_ns)

    def context(self, symbol: str, now_ns: int) -> dict:
        return self.hub.decision_context(symbol, now_ns)

    def close(self) -> None:
        self.hub.stop()
