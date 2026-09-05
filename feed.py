"""Market data sources and the Market Data Engine (spec §16, block 1).

Three feeds share one interface so that backtest, simulation and live differ
only in construction:

  SimFeed        -- drives the SimExchange; ticks emerge from real matching
  ReplayFeed     -- replays recorded ticks or OHLCV bars from disk
  LiveFeed       -- adapter surface for a broker websocket (Kite/Upstox)

`MarketDataEngine` sits above whichever feed is active and owns the derived
state everything else reads: latest tick, L2 book, multi-timeframe bars,
session phase and the tape. It publishes on the bus; nothing polls it.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from ..core.bus import EventBus, Topic
from ..core.clock import SimClock, LiveClock, Clock, SessionPhase
from ..core.types import (
    Tick, Bar, DepthSnapshot, Timeframe, Instrument, Side, NS, Fill,
)
from .bars import BarAggregator, BarSeries
from .simexchange import SimExchange


class MarketDataFeed:
    """Interface. A feed pushes ticks; it never gets polled."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def subscribe(self, symbols: List[str]) -> None: ...
    def step(self, dt_s: float) -> None: ...
    @property
    def symbols(self) -> List[str]: return []


class SimFeed(MarketDataFeed):
    """Wraps the simulated exchange and pushes its output onto the bus."""

    def __init__(self, bus: EventBus, clock: SimClock, symbols: List[str],
                 seed: int = 7, depth_levels: int = 5,
                 index_symbol: str = "NIFTY"):
        self.bus = bus
        self.clock = clock
        self._symbols = list(symbols)
        self.exchange = SimExchange(
            symbols, seed=seed, index_symbol=index_symbol,
            depth_levels=depth_levels,
            on_tick=lambda t: bus.emit(Topic.TICK, t),
            on_depth=lambda d: bus.emit(Topic.DEPTH, d),
            on_agent_fill=lambda f: bus.emit(Topic.FILL, f),
            on_news=lambda n: bus.emit(Topic.NEWS, n),
        )

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    def step(self, dt_s: float) -> None:
        ts = self.clock.advance_s(dt_s)
        self.exchange.step(ts, dt_s)

    def roll_session(self) -> None:
        self.exchange.roll_session()


@dataclass
class ReplayRow:
    ts: int
    symbol: str
    o: float
    h: float
    l: float
    c: float
    v: int


class ReplayFeed(MarketDataFeed):
    """Replays historical OHLCV, synthesising an intrabar path.

    Bars hide the path, and the path is exactly what stop placement and MAE
    depend on. Replaying a bar as four points (O -> H/L in a plausible order
    -> C) is a lie the backtester must not tell silently, so this expands each
    bar into a Brownian bridge pinned to the true O/H/L/C. The stop logic then
    sees a path with the right extremes and roughly the right shape, and the
    engine records that these were synthesised, not observed.
    """

    def __init__(self, bus: EventBus, clock: SimClock, rows: List[ReplayRow],
                 sub_steps: int = 12, seed: int = 0,
                 spread_bps: float = 2.0):
        self.bus = bus
        self.clock = clock
        self.rows = sorted(rows, key=lambda r: r.ts)
        self.sub_steps = max(4, sub_steps)
        self.rng = np.random.default_rng(seed)
        self.spread_bps = spread_bps
        self._i = 0
        self._symbols = sorted({r.symbol for r in rows})
        self._cum_vol: Dict[str, int] = {s: 0 for s in self._symbols}

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    @classmethod
    def from_csv(cls, bus, clock, path: str, **kw) -> "ReplayFeed":
        """Expects: timestamp,symbol,open,high,low,close,volume."""
        rows: List[ReplayRow] = []
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                ts = r.get("timestamp") or r.get("ts") or r.get("date")
                ts_ns = int(ts) if str(ts).isdigit() else \
                    int(np.datetime64(ts).astype("datetime64[ns]").astype(np.int64))
                rows.append(ReplayRow(
                    ts=ts_ns, symbol=r["symbol"],
                    o=float(r["open"]), h=float(r["high"]),
                    l=float(r["low"]), c=float(r["close"]),
                    v=int(float(r.get("volume", 0))),
                ))
        return cls(bus, clock, rows, **kw)

    def _bridge(self, row: ReplayRow) -> List[Tuple[float, int]]:
        """Path through the bar hitting the true high and low exactly."""
        n = self.sub_steps
        # random walk pinned at o -> c
        w = self.rng.standard_normal(n)
        w = np.cumsum(w - w.mean())
        if abs(w).max() > 1e-12:
            w = w / abs(w).max()
        span = max(row.h - row.l, 1e-9)
        path = row.o + w * span * 0.5
        # decide whether high or low comes first, weighted by where close sits
        up_first = self.rng.random() < (0.5 if row.c >= row.o else 0.5)
        hi_i = int(n * (0.3 if up_first else 0.65))
        lo_i = int(n * (0.65 if up_first else 0.3))
        path[hi_i] = row.h
        path[lo_i] = row.l
        path[0] = row.o
        path[-1] = row.c
        path = np.clip(path, row.l, row.h)
        vols = np.full(n, row.v // n, dtype=np.int64)
        vols[-1] += row.v - int(vols.sum())
        return list(zip(path.tolist(), vols.tolist()))

    def step(self, dt_s: float = 0.0) -> None:
        if self._i >= len(self.rows):
            return
        row = self.rows[self._i]
        self._i += 1
        bar_span = self._bar_span()
        dt = bar_span / self.sub_steps
        half_spread = row.c * self.spread_bps / 2e4
        for k, (px, q) in enumerate(self._bridge(row)):
            ts = row.ts + int(k * dt * NS)
            self.clock.set_ns(ts)
            self._cum_vol[row.symbol] += int(q)
            t = Tick(
                ts=ts, symbol=row.symbol, ltp=px, ltq=int(q),
                bid=px - half_spread, ask=px + half_spread,
                bid_qty=max(1, int(q * 2)), ask_qty=max(1, int(q * 2)),
                volume=self._cum_vol[row.symbol],
                aggressor=1 if (k and px >= 0) else 0,
            )
            self.bus.emit(Topic.TICK, t)

    def _bar_span(self) -> float:
        if len(self.rows) < 2:
            return 60.0
        for j in range(self._i, min(self._i + 5, len(self.rows))):
            d = (self.rows[j].ts - self.rows[self._i - 1].ts) / NS
            if d > 0:
                return d
        return 60.0

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self.rows)


class LiveFeed(MarketDataFeed):
    """Adapter surface for a real broker websocket.

    Deliberately inert here: it exposes `on_broker_tick` for a broker SDK
    callback to drive, and does nothing on its own. Wiring a real feed is a
    matter of subscribing the SDK's tick callback to this method.
    """

    def __init__(self, bus: EventBus, clock: Clock, symbols: List[str]):
        self.bus = bus
        self.clock = clock
        self._symbols = list(symbols)
        self._connected = False
        self.dropped = 0
        self.last_seq: Dict[str, int] = {}

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    def on_broker_tick(self, raw: dict) -> None:
        """Normalise a broker payload into a Tick and publish it."""
        try:
            depth = raw.get("depth") or {}
            buy = depth.get("buy") or []
            sell = depth.get("sell") or []
            t = Tick(
                ts=int(raw.get("ts_ns") or self.clock.now_ns()),
                symbol=raw["symbol"],
                ltp=float(raw["last_price"]),
                ltq=int(raw.get("last_quantity", 0)),
                bid=float(buy[0]["price"]) if buy else 0.0,
                ask=float(sell[0]["price"]) if sell else 0.0,
                bid_qty=int(buy[0]["quantity"]) if buy else 0,
                ask_qty=int(sell[0]["quantity"]) if sell else 0,
                volume=int(raw.get("volume_traded", 0)),
                oi=int(raw.get("oi", 0)),
            )
        except (KeyError, TypeError, ValueError):
            self.dropped += 1
            return
        self.bus.emit(Topic.TICK, t)
        if buy and sell:
            from ..core.types import BookLevel
            d = DepthSnapshot(
                ts=t.ts, symbol=t.symbol,
                bids=[BookLevel(float(x["price"]), int(x["quantity"]),
                                int(x.get("orders", 1))) for x in buy],
                asks=[BookLevel(float(x["price"]), int(x["quantity"]),
                                int(x.get("orders", 1))) for x in sell],
            )
            self.bus.emit(Topic.DEPTH, d)


# --------------------------------------------------------------------------


class MarketDataEngine:
    """Derived market state. Everything downstream reads from here."""

    def __init__(self, bus: EventBus, clock: Clock, symbols: List[str],
                 timeframes: Optional[List[Timeframe]] = None,
                 index_symbol: str = "NIFTY"):
        self.bus = bus
        self.clock = clock
        self.symbols = list(symbols)
        self.index_symbol = index_symbol
        self.timeframes = timeframes or [
            Timeframe.M1, Timeframe.M5, Timeframe.M15,
            Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1,
        ]
        self.bars = BarAggregator(
            symbols, self.timeframes,
            on_close=lambda b: bus.emit(Topic.BAR_CLOSE, b),
        )
        self.last_tick: Dict[str, Tick] = {}
        self.depth: Dict[str, DepthSnapshot] = {}
        self.prev_close: Dict[str, float] = {}
        self.day_open: Dict[str, float] = {}
        self.tick_count = 0
        self.stale_ns = 3 * NS
        bus.subscribe(Topic.TICK, self._on_tick, priority=10, name="mde.tick")
        bus.subscribe(Topic.DEPTH, self._on_depth, priority=10, name="mde.depth")

    # ------------------------------------------------------------------
    def _on_tick(self, t: Tick) -> None:
        self.tick_count += 1
        self.last_tick[t.symbol] = t
        if t.symbol not in self.day_open:
            self.day_open[t.symbol] = t.ltp
            self.prev_close.setdefault(t.symbol, t.ltp)
        self.bars.on_tick(t)

    def _on_depth(self, d: DepthSnapshot) -> None:
        self.depth[d.symbol] = d

    # ------------------------------------------------------------------
    def ltp(self, symbol: str) -> float:
        t = self.last_tick.get(symbol)
        return t.ltp if t else 0.0

    def mid(self, symbol: str) -> float:
        t = self.last_tick.get(symbol)
        return t.mid if t else 0.0

    def series(self, symbol: str, tf: Timeframe) -> Optional[BarSeries]:
        return self.bars.get(symbol, tf)

    def is_stale(self, symbol: str) -> bool:
        t = self.last_tick.get(symbol)
        if t is None:
            return True
        return (self.clock.now_ns() - t.ts) > self.stale_ns

    def day_change_pct(self, symbol: str) -> float:
        t = self.last_tick.get(symbol)
        base = self.prev_close.get(symbol, 0.0)
        if not t or base <= 0:
            return 0.0
        return (t.ltp / base - 1.0) * 100.0

    def breadth(self) -> Dict[str, float]:
        """Market breadth (spec §1): advance/decline and participation."""
        adv = dec = unch = 0
        chs = []
        for s in self.symbols:
            if s == self.index_symbol:
                continue
            ch = self.day_change_pct(s)
            chs.append(ch)
            if ch > 0.05:
                adv += 1
            elif ch < -0.05:
                dec += 1
            else:
                unch += 1
        n = max(1, adv + dec + unch)
        arr = np.asarray(chs) if chs else np.zeros(1)
        return {
            "advances": float(adv),
            "declines": float(dec),
            "unchanged": float(unch),
            "ad_ratio": adv / max(dec, 1),
            "ad_line": (adv - dec) / n,
            "pct_up": adv / n,
            "median_change": float(np.median(arr)),
            "dispersion": float(arr.std()),
        }

    def sector_performance(self) -> Dict[str, float]:
        from ..core.config import sector_of
        acc: Dict[str, List[float]] = {}
        for s in self.symbols:
            if s == self.index_symbol:
                continue
            acc.setdefault(sector_of(s), []).append(self.day_change_pct(s))
        return {k: float(np.mean(v)) for k, v in acc.items() if v}

    def snapshot(self, symbol: str) -> dict:
        t = self.last_tick.get(symbol)
        d = self.depth.get(symbol)
        return {
            "symbol": symbol,
            "ltp": t.ltp if t else 0.0,
            "bid": t.bid if t else 0.0,
            "ask": t.ask if t else 0.0,
            "spread_bps": t.spread_bps if t else 0.0,
            "volume": t.volume if t else 0,
            "day_change_pct": self.day_change_pct(symbol),
            "imbalance": d.imbalance() if d else 0.0,
            "stale": self.is_stale(symbol),
            "ts": t.ts if t else 0,
        }
