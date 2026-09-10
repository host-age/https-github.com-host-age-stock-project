"""Persistent, time-aware NSE security knowledge built off asynchronous sources.

The knowledge layer is deliberately non-blocking for trading: MCP calls happen
outside the tick/decision path and update a thread-safe cache. GMQ can therefore
combine fast Kite data with slower historical/current exchange context without
turning network latency into trading latency.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.nse_mcp import StreamableHttpMcpClient, default_clients


@dataclass
class SecurityKnowledge:
    symbol: str
    observed_ns: int = 0
    identity: Dict[str, Any] = field(default_factory=dict)
    historical: Dict[str, float] = field(default_factory=dict)
    current: Dict[str, float] = field(default_factory=dict)
    future_events: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    stale: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SecurityKnowledge":
        return cls(**d)


class NSEKnowledgeHub:
    """Cached NSE context with optional asynchronous MCP refresh."""

    def __init__(self, symbols: List[str], run_dir: Optional[str] = None,
                 enabled: Optional[bool] = None, refresh_s: float = 300.0,
                 mcp_timeout_s: float = 8.0):
        self.symbols = list(symbols)
        self.enabled = (os.getenv("NSE_MCP_ENABLED", "0").lower()
                        in {"1", "true", "yes", "on"}) if enabled is None else bool(enabled)
        self.refresh_s = max(float(refresh_s), 30.0)
        self.run_dir = Path(run_dir or os.getenv("RUN_DIR", "runs"))
        self.path = self.run_dir / "stock_knowledge.json"
        self._items: Dict[str, SecurityKnowledge] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.clients: Dict[str, StreamableHttpMcpClient] = default_clients(mcp_timeout_s)
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for symbol, item in raw.get("items", {}).items():
                    self._items[symbol] = SecurityKnowledge.from_dict(item)
        except Exception:
            # Historical knowledge is valuable but must never prevent the
            # trading process from starting.
            self._items = {}

    def _save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {"saved_at": datetime.now(timezone.utc).isoformat(),
                       "items": {s: x.to_dict() for s, x in self._items.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(self.path)

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop,
                                        name="nse-knowledge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        for c in self.clients.values():
            c.close()
        self._save()

    def snapshot(self, symbol: str, now_ns: Optional[int] = None,
                 max_age_s: float = 900.0) -> SecurityKnowledge:
        now_ns = int(now_ns or time.time_ns())
        with self._lock:
            item = self._items.get(symbol.upper())
            if item is None:
                return SecurityKnowledge(symbol=symbol.upper(), observed_ns=0,
                                         notes=["no_cached_nse_knowledge"])
            age_s = (now_ns - item.observed_ns) / 1e9 if item.observed_ns else float("inf")
            item.stale = age_s > max_age_s
            return SecurityKnowledge.from_dict(item.to_dict())

    def admissible(self, symbol: str, now_ns: Optional[int] = None,
                   max_age_s: float = 900.0) -> tuple[bool, str]:
        item = self.snapshot(symbol, now_ns, max_age_s)
        if item.observed_ns == 0:
            # Knowledge enrichment is optional while MCP credentials/connectivity
            # are unavailable; existing market-data/risk controls remain active.
            return True, "nse_knowledge_unavailable"
        if item.stale:
            return False, "nse_knowledge_stale"
        if item.confidence < 0.25:
            return False, "nse_knowledge_low_confidence"
        return True, "ok"

    def decision_context(self, symbol: str, now_ns: Optional[int] = None) -> dict:
        item = self.snapshot(symbol, now_ns)
        return {
            "knowledge_age_s": ((now_ns - item.observed_ns) / 1e9
                                 if now_ns and item.observed_ns else None),
            "knowledge_confidence": item.confidence,
            "knowledge_stale": item.stale,
            "identity": item.identity,
            "historical": item.historical,
            "current": item.current,
            "future_events": item.future_events[:20],
            "notes": item.notes[-10:],
            "sources": item.sources[-10:],
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            for symbol in self.symbols:
                if self._stop.is_set():
                    break
                try:
                    self.refresh_symbol(symbol)
                except Exception as exc:
                    with self._lock:
                        item = self._items.setdefault(symbol, SecurityKnowledge(symbol=symbol))
                        item.notes.append(f"refresh_error:{type(exc).__name__}")
            self._save()
            self._stop.wait(self.refresh_s)

    def refresh_symbol(self, symbol: str) -> SecurityKnowledge:
        symbol = symbol.upper()
        now = time.time_ns()
        item = self.snapshot(symbol, now, max_age_s=10**9)
        sources = set(item.sources)
        notes = list(item.notes)[-20:]

        # Tool selection is description-driven because the NSE MCP server's
        # public tool names may evolve. We prefer market/current tools for the
        # current snapshot and bhavcopy/history tools for historical context.
        cm = self.clients["cm_market"]
        bh = self.clients["bhavcopy"]
        current_tool = cm.choose_tool("quote", "market", "equity", "security", "symbol")
        history_tool = bh.choose_tool("bhavcopy", "historical", "equity", "security", "price")

        current = dict(item.current)
        historical = dict(item.historical)
        identity = dict(item.identity)

        if current_tool:
            result = cm.call_tool(current_tool.name, self._symbol_args(current_tool.input_schema, symbol))
            self._merge_result(result, current=current, identity=identity)
            sources.add(f"nse-mcp:{current_tool.name}")
        else:
            notes.append("cm_market_tool_not_discovered")

        if history_tool:
            result = bh.call_tool(history_tool.name, self._symbol_args(history_tool.input_schema, symbol))
            self._merge_result(result, historical=historical, identity=identity)
            sources.add(f"nse-mcp:{history_tool.name}")
        else:
            notes.append("bhavcopy_tool_not_discovered")

        nonempty = sum(bool(x) for x in (identity, historical, current))
        confidence = min(1.0, 0.25 * nonempty)
        out = SecurityKnowledge(
            symbol=symbol, observed_ns=now, identity=identity,
            historical=historical, current=current,
            future_events=item.future_events, sources=sorted(sources),
            confidence=confidence, stale=False, notes=notes[-20:])
        with self._lock:
            self._items[symbol] = out
        return out

    @staticmethod
    def _symbol_args(schema: dict, symbol: str) -> dict:
        """Construct conservative arguments from the advertised schema."""
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        args = {}
        for name in props:
            lname = name.lower()
            if lname in {"symbol", "tradingsymbol", "security", "ticker", "scrip"}:
                args[name] = symbol
            elif lname in {"symbols", "tickers"}:
                args[name] = [symbol]
        # Many simple MCP tools accept no args and infer market state; don't
        # fabricate additional fields when the server does not advertise them.
        return args

    @staticmethod
    def _merge_result(result: Any, current: Optional[dict] = None,
                      historical: Optional[dict] = None,
                      identity: Optional[dict] = None) -> None:
        """Best-effort normalization of JSON-like MCP tool results."""
        if result is None:
            return
        if isinstance(result, dict):
            raw = result.get("structuredContent", result)
            if isinstance(raw, dict):
                result = raw
            elif isinstance(raw, str):
                try:
                    result = json.loads(raw)
                except Exception:
                    result = {"raw": raw[:500]}
        if isinstance(result, dict):
            if identity is not None:
                for k in ("symbol", "tradingsymbol", "isin", "company", "name", "sector", "industry"):
                    if k in result:
                        identity[k] = result[k]
            target = current if current is not None else historical
            if target is not None:
                for k, v in result.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        target[str(k)] = float(v)


class PreTradeKnowledge:
    """A small deterministic gate over cached knowledge, not a prediction model."""

    def evaluate(self, hub: NSEKnowledgeHub, symbol: str, now_ns: int) -> dict:
        item = hub.snapshot(symbol, now_ns)
        ok, reason = hub.admissible(symbol, now_ns)
        return {
            "approved": ok,
            "reason": reason,
            "confidence": item.confidence,
            "stale": item.stale,
            "event_count": len(item.future_events),
            "identity": item.identity,
        }
