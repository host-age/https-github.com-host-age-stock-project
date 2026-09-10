"""Time-aware persistent stock knowledge.

This module deliberately separates three clocks of information:
  past     - rolling observations and completed trade outcomes
  present  - latest market state seen by GMQ
  future   - information known now about a later timestamp (events/calendar)

Hot-path observation is O(1) and in-memory. Persistence is explicit and uses an
atomic JSON checkpoint, so disk I/O never blocks tick processing.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List, Optional

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
    __slots__ = (
        "profile", "prices", "returns", "wins", "losses", "pnl", "events",
        "last_ts_ns", "current_volume", "current_spread_bps", "current_liquidity"
    )

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


class KnowledgeStore:
    """In-memory knowledge with explicit persistence/checkpointing."""

    def __init__(self, path: Optional[str] = None, window: int = 2048):
        self.path = path
        self.window = max(int(window), 128)
        self._stocks: Dict[str, _Track] = {}
        if path:
            self.load(path)

    def register(self, symbol: str, *, exchange: str = "NSE",
                 sector: str = "UNKNOWN", instrument_type: str = "EQ",
                 tick_size: float = 0.05, lot_size: int = 1) -> None:
        symbol = symbol.upper()
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
        tr.profile.last_event_ts_ns = tr.last_ts_ns

    def record_outcome(self, symbol: str, pnl: float, won: Optional[bool] = None) -> None:
        symbol = symbol.upper()
        self.register(symbol)
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
        tr = self._stocks[symbol]
        tr.events.append(event)
        # Keep only future-relevant events for a bounded footprint.
        tr.events = [e for e in tr.events if e.ts_ns >= event.ts_ns - 90 * 86400 * 1_000_000_000]

    def prune_expired_events(self, now_ns: int) -> None:
        for tr in self._stocks.values():
            tr.events = [e for e in tr.events if e.ts_ns >= now_ns]

    def snapshot(self, symbol: str, now_ns: int) -> StockKnowledge:
        symbol = symbol.upper()
        self.register(symbol)
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
            sxx = float(((x - xm) ** 2).sum())
            if sxx > 1e-12:
                slope = float(((x - xm) * (y - ym)).sum() / sxx)
                out.trend_slope = slope
                fitted = ym + slope * (x - xm)
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
        tr = self._stocks[symbol]
        n = tr.wins + tr.losses
        # Mild Laplace prior keeps a new symbol at a neutral prior.
        return float((tr.wins + 2.0 * prior) / (n + 2.0))

    def checkpoint(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ValueError("checkpoint path is required")
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        payload = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "stocks": []}
        for symbol in sorted(self._stocks):
            tr = self._stocks[symbol]
            snap = asdict(tr.profile)
            snap.update({
                "prices": list(tr.prices),
                "returns": list(tr.returns),
                "wins": tr.wins,
                "losses": tr.losses,
                "pnl": tr.pnl,
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
                fh.flush()
                os.fsync(fh.fileno())
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
                symbol=symbol,
                exchange=raw.get("exchange", "NSE"),
                sector=raw.get("sector", "UNKNOWN"),
                instrument_type=raw.get("instrument_type", "EQ"),
                tick_size=float(raw.get("tick_size", 0.05)),
                lot_size=int(raw.get("lot_size", 1)),
                observed_ticks=int(raw.get("observed_ticks", 0)),
                current_price=float(raw.get("current_price", 0.0)),
            )
            tr = _Track(profile, self.window)
            tr.prices.extend(float(x) for x in raw.get("prices", []))
            tr.returns.extend(float(x) for x in raw.get("returns", []))
            tr.wins = int(raw.get("wins", 0))
            tr.losses = int(raw.get("losses", 0))
            tr.pnl = float(raw.get("pnl", 0.0))
            tr.events = [FutureEvent(**e) for e in raw.get("events", [])]
            tr.last_ts_ns = int(raw.get("last_ts_ns", 0))
            tr.current_volume = int(raw.get("current_volume", 0))
            tr.current_spread_bps = float(raw.get("current_spread_bps", 0.0))
            tr.current_liquidity = float(raw.get("current_liquidity", 0.5))
            self._stocks[symbol] = tr
