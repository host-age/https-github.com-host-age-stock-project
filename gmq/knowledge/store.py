"""Time-aware persistent stock knowledge.

This module separates past, present and future-known information. The hot-path
market observation remains in memory; persistence is atomic and asynchronous.
When NSE_MCP_ENABLED is enabled, a background worker enriches the store from
NSE's Streamable-HTTP MCP endpoints. It never runs network I/O in a tick or
pre-trade decision call.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import numpy as np

from ..core.mathx import clamp


@dataclass(slots=True)
class FutureEvent:
    symbol: str
    ts_ns: int
    kind: str
    severity: float = 0.5
    source: str = ""
    note: str = ""


@dataclass(slots=True)
class StockKnowledge:
    symbol: str
    exchange: str = "NSE"
    sector: str = "UNKNOWN"
    instrument_type: str = "EQ"
    tick_size: float = 0.05
    lot_size: int = 1
    observed_ticks: int = 0
    current_price: float = 0.0
    current_volume: int = 0
    current_spread_bps: float = 0.0
    current_liquidity: float = 0.5
    return_mean: float = 0.0
    return_std: float = 0.0
    return_median: float = 0.0
    return_mad: float = 0.0
    trend_slope: float = 0.0
    trend_r2: float = 0.0
    wins: int = 0
    losses: int = 0
    realised_pnl: float = 0.0
    last_event_ts_ns: int = 0
    event_risk: float = 0.0
    event_kinds: List[str] = field(default_factory=list)


class _Track:
    __slots__ = ("profile", "prices", "returns", "wins", "losses", "pnl",
                 "events", "last_ts_ns", "current_volume", "current_spread_bps",
                 "current_liquidity", "mcp_last_success_ns", "mcp_attempts",
                 "mcp_successes", "mcp_last_error")

    def __init__(self, profile: StockKnowledge, window: int):
        self.profile = profile
        self.prices: Deque[float] = deque(maxlen=window)
        self.returns: Deque[float] = deque(maxlen=window)
        self.wins = 0
        self.losses = 0
        self.pnl = 0.0
        self.events: List[FutureEvent] = []
        self.last_ts_ns = 0
        self.current_volume = 0
        self.current_spread_bps = 0.0
        self.current_liquidity = 0.5
        # NSE MCP sync health -- never persisted, never gates a decision by
        # itself; purely so an operator can see whether the "collected" data
        # actually arrived, and how stale it is, instead of the sync failing
        # silently into an unchanged (and therefore indistinguishable-looking)
        # profile.
        self.mcp_last_success_ns = 0
        self.mcp_attempts = 0
        self.mcp_successes = 0
        self.mcp_last_error = ""


class KnowledgeStore:
    """In-memory stock memory with explicit persistence and optional NSE sync."""

    def __init__(self, path: Optional[str] = None, window: int = 2048):
        self.path = path
        self.window = max(int(window), 128)
        self._stocks: Dict[str, _Track] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._mcp_thread: Optional[threading.Thread] = None
        self._mcp_clients = None
        self._mcp_interval_s = max(float(os.getenv("NSE_MCP_REFRESH_S", "300")), 60.0)
        self._mcp_last_cycle_ns = 0
        self._mcp_cycles = 0
        if path:
            self.load(path)
        self._start_mcp_if_enabled()

    def register(self, symbol: str, *, exchange: str = "NSE",
                 sector: str = "UNKNOWN", instrument_type: str = "EQ",
                 tick_size: float = 0.05, lot_size: int = 1) -> None:
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self._stocks:
                self._stocks[symbol] = _Track(
                    StockKnowledge(
                        symbol=symbol, exchange=exchange, sector=sector,
                        instrument_type=instrument_type, tick_size=tick_size,
                        lot_size=max(int(lot_size), 1)),
                    self.window,
                )

    def observe(self, symbol: str, ts_ns: int, price: float,
                volume: int = 0, spread_bps: float = 0.0,
                liquidity: float = 0.5) -> None:
        symbol = symbol.upper()
        self.register(symbol)
        with self._lock:
            tr = self._stocks[symbol]
            price = float(price)
            if price <= 0:
                return
            if tr.prices:
                prev = tr.prices[-1]
                if prev > 0:
                    tr.returns.append(float(np.log(price / prev)))
            tr.prices.append(price)
            tr.last_ts_ns = int(ts_ns)
            tr.current_volume = max(int(volume), 0)
            tr.current_spread_bps = max(float(spread_bps), 0.0)
            tr.current_liquidity = clamp(float(liquidity), 0.0, 1.0)
            tr.profile.observed_ticks += 1
            tr.profile.current_price = price
            tr.profile.current_volume = tr.current_volume
            tr.profile.current_spread_bps = tr.current_spread_bps
            tr.profile.current_liquidity = tr.current_liquidity
            tr.profile.last_event_ts_ns = tr.last_ts_ns

    def record_outcome(self, symbol: str, pnl: float, won: Optional[bool] = None) -> None:
        symbol = symbol.upper()
        self.register(symbol)
        with self._lock:
            tr = self._stocks[symbol]
            pnl = float(pnl)
            tr.pnl += pnl
            if won is None:
                won = pnl > 0.0
            if won:
                tr.wins += 1
            else:
                tr.losses += 1

    def add_event(self, event: FutureEvent) -> None:
        symbol = event.symbol.upper()
        self.register(symbol)
        with self._lock:
            tr = self._stocks[symbol]
            tr.events.append(event)
            tr.events = [e for e in tr.events if e.ts_ns >= event.ts_ns - 90 * 86400 * 1_000_000_000]

    def prune_expired_events(self, now_ns: int) -> None:
        with self._lock:
            for tr in self._stocks.values():
                tr.events = [e for e in tr.events if e.ts_ns >= now_ns]

    def snapshot(self, symbol: str, now_ns: int) -> StockKnowledge:
        symbol = symbol.upper()
        self.register(symbol)
        with self._lock:
            tr = self._stocks[symbol]
            out = StockKnowledge(**asdict(tr.profile))
            out.current_volume = tr.current_volume
            out.current_spread_bps = tr.current_spread_bps
            out.current_liquidity = tr.current_liquidity
            if tr.returns:
                r = np.asarray(tr.returns, dtype=np.float64)
                out.return_mean = float(r.mean())
                out.return_std = float(r.std(ddof=1)) if r.size > 1 else 0.0
                out.return_median = float(np.median(r))
                out.return_mad = float(np.median(np.abs(r - out.return_median)))
            if len(tr.prices) >= 20:
                y = np.log(np.maximum(np.asarray(tr.prices, dtype=np.float64), 1e-12))
                x = np.arange(y.size, dtype=np.float64)
                xm, ym = x.mean(), y.mean()
                dx = x - xm
                sxx = float((dx * dx).sum())
                if sxx > 1e-12:
                    slope = float((dx * (y - ym)).sum() / sxx)
                    out.trend_slope = slope
                    fitted = ym + slope * dx
                    ss_tot = float(((y - ym) ** 2).sum())
                    ss_res = float(((y - fitted) ** 2).sum())
                    out.trend_r2 = clamp(1.0 - ss_res / ss_tot, 0.0, 1.0) if ss_tot > 1e-12 else 0.0
            out.wins = tr.wins
            out.losses = tr.losses
            out.realised_pnl = tr.pnl
            future = [e for e in tr.events if e.ts_ns >= now_ns]
            out.event_risk = max((clamp(e.severity, 0.0, 1.0) for e in future), default=0.0)
            out.event_kinds = sorted({e.kind for e in future})
            return out

    def historical_win_rate(self, symbol: str, prior: float = 0.5) -> float:
        symbol = symbol.upper()
        self.register(symbol)
        with self._lock:
            tr = self._stocks[symbol]
            n = tr.wins + tr.losses
            return float((tr.wins + 2.0 * prior) / (n + 2.0))

    def checkpoint(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ValueError("checkpoint path is required")
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            payload = {"version": 2, "created_utc": datetime.now(timezone.utc).isoformat(), "stocks": []}
            for symbol in sorted(self._stocks):
                tr = self._stocks[symbol]
                snap = asdict(tr.profile)
                snap.update({
                    "prices": list(tr.prices), "returns": list(tr.returns),
                    "wins": tr.wins, "losses": tr.losses, "pnl": tr.pnl,
                    "events": [asdict(e) for e in tr.events],
                    "last_ts_ns": tr.last_ts_ns,
                    "current_volume": tr.current_volume,
                    "current_spread_bps": tr.current_spread_bps,
                    "current_liquidity": tr.current_liquidity,
                })
                payload["stocks"].append(snap)
        fd, tmp = tempfile.mkstemp(prefix=".knowledge.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
                fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self.path = path
        return path

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for raw in payload.get("stocks", []):
            symbol = str(raw.get("symbol", "")).upper()
            if not symbol:
                continue
            profile = StockKnowledge(
                symbol=symbol, exchange=raw.get("exchange", "NSE"),
                sector=raw.get("sector", "UNKNOWN"),
                instrument_type=raw.get("instrument_type", "EQ"),
                tick_size=float(raw.get("tick_size", 0.05)),
                lot_size=int(raw.get("lot_size", 1)),
                observed_ticks=int(raw.get("observed_ticks", 0)),
                current_price=float(raw.get("current_price", 0.0)),
                current_volume=int(raw.get("current_volume", 0)),
                current_spread_bps=float(raw.get("current_spread_bps", 0.0)),
                current_liquidity=float(raw.get("current_liquidity", 0.5)),
                last_event_ts_ns=int(raw.get("last_event_ts_ns", 0)),
            )
            tr = _Track(profile, self.window)
            tr.prices.extend(float(x) for x in raw.get("prices", []))
            tr.returns.extend(float(x) for x in raw.get("returns", []))
            tr.wins = int(raw.get("wins", 0)); tr.losses = int(raw.get("losses", 0))
            tr.pnl = float(raw.get("pnl", 0.0))
            tr.events = [FutureEvent(**e) for e in raw.get("events", [])]
            tr.last_ts_ns = int(raw.get("last_ts_ns", 0))
            tr.current_volume = int(raw.get("current_volume", 0))
            tr.current_spread_bps = float(raw.get("current_spread_bps", 0.0))
            tr.current_liquidity = float(raw.get("current_liquidity", 0.5))
            with self._lock:
                self._stocks[symbol] = tr

    # ------------------------------------------------------------------
    # NSE MCP background enrichment
    # ------------------------------------------------------------------
    def _start_mcp_if_enabled(self) -> None:
        enabled = os.getenv("NSE_MCP_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
        if not enabled or (self._mcp_thread and self._mcp_thread.is_alive()):
            return
        from ..data.nse_mcp import default_clients
        self._mcp_clients = default_clients(float(os.getenv("NSE_MCP_TIMEOUT_S", "8")))
        self._stop.clear()
        self._mcp_thread = threading.Thread(target=self._mcp_loop, name="nse-mcp", daemon=True)
        self._mcp_thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._mcp_thread and self._mcp_thread.is_alive():
            self._mcp_thread.join(timeout=2.0)
        if self._mcp_clients:
            for c in self._mcp_clients.values():
                c.close()
        if self.path:
            self.checkpoint(self.path)

    def _mcp_loop(self) -> None:
        while not self._stop.is_set():
            symbols = list(self._stocks.keys())
            for symbol in symbols:
                if self._stop.is_set():
                    break
                with self._lock:
                    tr = self._stocks.get(symbol)
                    if tr is not None:
                        tr.mcp_attempts += 1
                try:
                    self._refresh_from_nse(symbol)
                    with self._lock:
                        tr = self._stocks.get(symbol)
                        if tr is not None:
                            tr.mcp_successes += 1
                            tr.mcp_last_success_ns = time.time_ns()
                            tr.mcp_last_error = ""
                except Exception as e:
                    # MCP failure must degrade to existing local/Kite knowledge;
                    # it must never stop the market engine -- but it must not
                    # vanish either, or "collecting data" and "silently not
                    # collecting data" look identical from the outside.
                    with self._lock:
                        tr = self._stocks.get(symbol)
                        if tr is not None:
                            tr.mcp_last_error = f"{type(e).__name__}: {e}"
                    continue
            with self._lock:
                self._mcp_cycles += 1
                self._mcp_last_cycle_ns = time.time_ns()
            try:
                if self.path:
                    self.checkpoint(self.path)
            except Exception:
                pass
            self._stop.wait(self._mcp_interval_s)

    def coverage_report(self, now_ns: int) -> dict:
        """Is the NSE MCP sync actually delivering fresh data, per symbol?

        Read-only and side-effect-free -- purely for operators/dashboards to
        answer "did the collection step work" rather than inferring it from
        an unmoving profile. ``stale_after_s`` is generous (3x the refresh
        interval) since a single missed cycle is normal jitter, not a fault.
        """
        stale_after_s = 3.0 * self._mcp_interval_s
        with self._lock:
            mcp_enabled = bool(self._mcp_thread and self._mcp_thread.is_alive())
            symbols = {}
            synced = 0
            stale = 0
            for symbol, tr in self._stocks.items():
                age_s = None
                if tr.mcp_last_success_ns:
                    age_s = max((now_ns - tr.mcp_last_success_ns) / 1e9, 0.0)
                is_stale = age_s is None or age_s > stale_after_s
                if tr.mcp_successes:
                    synced += 1
                if is_stale:
                    stale += 1
                symbols[symbol] = {
                    "observed_ticks": tr.profile.observed_ticks,
                    "mcp_attempts": tr.mcp_attempts,
                    "mcp_successes": tr.mcp_successes,
                    "mcp_synced": tr.mcp_successes > 0,
                    "mcp_age_s": round(age_s, 1) if age_s is not None else None,
                    "mcp_stale": is_stale,
                    "mcp_last_error": tr.mcp_last_error,
                }
            total = len(self._stocks)
            return {
                "mcp_enabled": mcp_enabled,
                "mcp_cycles": self._mcp_cycles,
                "mcp_last_cycle_age_s": round((now_ns - self._mcp_last_cycle_ns) / 1e9, 1)
                if self._mcp_last_cycle_ns else None,
                "symbols_tracked": total,
                "symbols_synced": synced,
                "symbols_stale": stale,
                "fraction_synced": round(synced / total, 4) if total else 0.0,
                "symbols": symbols,
            }

    def _refresh_from_nse(self, symbol: str) -> None:
        if not self._mcp_clients:
            return
        cm = self._mcp_clients["cm_market"]
        bh = self._mcp_clients["bhavcopy"]
        current_tool = cm.choose_tool("quote", "market", "equity", "security", "symbol")
        history_tool = bh.choose_tool("bhavcopy", "historical", "equity", "security", "price")
        now = time.time_ns()
        if current_tool:
            result = cm.call_tool(current_tool.name, self._tool_args(current_tool.input_schema, symbol))
            self._merge_numeric(symbol, result, "current")
        if history_tool:
            result = bh.call_tool(history_tool.name, self._tool_args(history_tool.input_schema, symbol))
            self._merge_numeric(symbol, result, "historical")

    @staticmethod
    def _tool_args(schema: dict, symbol: str) -> dict:
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        args = {}
        for name in props:
            lname = str(name).lower()
            if lname in {"symbol", "tradingsymbol", "security", "ticker", "scrip"}:
                args[name] = symbol
            elif lname in {"symbols", "tickers"}:
                args[name] = [symbol]
        return args

    def _merge_numeric(self, symbol: str, result, section: str) -> None:
        symbol = symbol.upper()
        self.register(symbol)
        if isinstance(result, dict):
            raw = result.get("structuredContent", result)
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except Exception: raw = {}
            result = raw
        if not isinstance(result, dict):
            return
        with self._lock:
            tr = self._stocks[symbol]
            for key, value in result.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                k = str(key).lower()
                v = float(value)
                if section == "current":
                    if k in {"ltp", "price", "last_price", "close"}:
                        tr.profile.current_price = v
                    elif k in {"volume", "total_traded_volume"}:
                        tr.current_volume = max(int(v), 0)
                        tr.profile.current_volume = tr.current_volume
                    elif k in {"spread_bps", "spread"}:
                        tr.current_spread_bps = max(v, 0.0)
                        tr.profile.current_spread_bps = tr.current_spread_bps
                elif section == "historical":
                    if k in {"return_mean", "mean_return"}:
                        tr.profile.return_mean = v
                    elif k in {"return_std", "volatility", "std"}:
                        tr.profile.return_std = max(v, 0.0)
                    elif k in {"trend_slope", "slope"}:
                        tr.profile.trend_slope = v
                    elif k in {"r2", "trend_r2"}:
                        tr.profile.trend_r2 = clamp(v, 0.0, 1.0)
