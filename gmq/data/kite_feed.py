"""Zerodha Kite live market-data feed (paper-trading on real prices).

This is the piece that puts the whole engine on the *real* NSE while risking
nothing: real quotes come in, the SimBroker fills against them, and no order
ever reaches a real account. The order path (KiteBroker) stays locked; this
file only reads.

Design that matters:

* **The converter is pure and separate from the socket.** `to_tick` and
  `to_depth` turn one Kite tick dict into this system's `Tick` and
  `DepthSnapshot` with no I/O, so they can be tested against recorded Kite
  payloads with no network, no credentials, and no market hours. The live
  socket is a thin shell around them. A live integration you cannot test off
  the wire is one you find out is wrong at 09:15 with money (paper or not) on
  the line.

* **Nothing is polled.** Kite pushes ticks over a websocket; this feed
  converts each and emits it on the same bus `SimFeed` uses, so everything
  downstream -- features, models, search, risk -- runs byte-for-byte unchanged
  and does not know the prices got real.

* **Symbols are mapped once.** Kite speaks in integer `instrument_token`s; we
  speak in tradingsymbols. The map is built once from Kite's instrument dump
  and reversed here, so a rename breaks loudly at startup rather than silently
  mis-routing a RELIANCE tick onto TCS.

Kite 'full' mode delivers, per tick: last_price, last_traded_quantity,
volume_traded (cumulative), total_buy/sell_quantity, ohlc, oi,
exchange_timestamp, and a 5-level `depth` with buy/sell {price, quantity,
orders}. That is everything this engine's microstructure block needs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional

from ..core.bus import EventBus, Topic
from ..core.clock import Clock, LiveClock
from ..core.types import Tick, DepthSnapshot, BookLevel, NS


def _ts_ns(kite_tick: dict, fallback_ns: int) -> int:
    """Nanosecond timestamp from a Kite tick, falling back to the local clock.

    Kite gives `exchange_timestamp` as a datetime (or None on the first few
    ticks). The exchange stamp is preferred because it is when the event
    actually happened at NSE, not when it reached us -- latency measurement
    downstream depends on that distinction.
    """
    ets = kite_tick.get("exchange_timestamp")
    if isinstance(ets, datetime):
        return int(ets.timestamp() * NS)
    return fallback_ns


def to_depth(kite_tick: dict, symbol: str, ts_ns: int) -> Optional[DepthSnapshot]:
    """Convert Kite 'full'-mode depth into a DepthSnapshot, or None if absent.

    Light modes carry no depth; the caller then simply omits the depth event,
    exactly as a venue that only quoted top-of-book would.
    """
    depth = kite_tick.get("depth")
    if not depth:
        return None
    def side(rows) -> List[BookLevel]:
        out = []
        for r in rows or []:
            px = float(r.get("price", 0.0) or 0.0)
            if px <= 0:
                continue
            out.append(BookLevel(price=px, qty=int(r.get("quantity", 0) or 0),
                                 orders=int(r.get("orders", 1) or 1)))
        return out
    bids = side(depth.get("buy"))
    asks = side(depth.get("sell"))
    if not bids and not asks:
        return None
    return DepthSnapshot(ts=ts_ns, symbol=symbol, bids=bids, asks=asks)


def to_tick(kite_tick: dict, symbol: str, ts_ns: int,
            prev_mid: float = 0.0) -> Tick:
    """Convert one Kite tick dict into a Tick.

    Best bid/ask are taken from the top of the depth book when present, and
    fall back to the last price if the tick is a light-mode update. The
    aggressor is inferred by the tick rule -- a trade printing at or above the
    prior mid is buy-initiated -- because Kite does not label trade direction
    and the order-flow features need it. It is an estimate, and is marked as
    one (0 when the prior mid is unknown).
    """
    ltp = float(kite_tick.get("last_price", 0.0) or 0.0)
    depth = kite_tick.get("depth") or {}
    buys = depth.get("buy") or []
    sells = depth.get("sell") or []
    bid = float(buys[0]["price"]) if buys and buys[0].get("price") else 0.0
    ask = float(sells[0]["price"]) if sells and sells[0].get("price") else 0.0
    bid_qty = int(buys[0]["quantity"]) if buys and buys[0].get("quantity") else 0
    ask_qty = int(sells[0]["quantity"]) if sells and sells[0].get("quantity") else 0
    # total_buy/sell_quantity is the whole-book pressure Kite reports; use it as
    # a fallback size when only top-of-book price is present.
    if bid_qty == 0:
        bid_qty = int(kite_tick.get("total_buy_quantity", 0) or 0)
    if ask_qty == 0:
        ask_qty = int(kite_tick.get("total_sell_quantity", 0) or 0)

    mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0) else ltp
    aggressor = 0
    if prev_mid > 0 and ltp > 0:
        aggressor = 1 if ltp >= prev_mid else -1

    return Tick(
        ts=ts_ns, symbol=symbol, ltp=ltp,
        ltq=int(kite_tick.get("last_traded_quantity", 0) or 0),
        bid=bid or (ltp and ltp * 0.99995), ask=ask or (ltp and ltp * 1.00005),
        bid_qty=bid_qty, ask_qty=ask_qty,
        volume=int(kite_tick.get("volume_traded", 0) or 0),
        oi=int(kite_tick.get("oi", 0) or 0),
        aggressor=aggressor,
    )


class KiteFeed:
    """Live NSE feed over the Kite websocket. Read-only; never places orders.

    The socket client is injected, not created, so a recorded-tick fake can
    drive the exact same code path in a test. In production the caller passes a
    real `KiteTicker`; in a test it passes anything with `on_ticks`/`connect`.
    """

    def __init__(self, bus: EventBus, symbols: List[str],
                 token_to_symbol: Dict[int, str],
                 clock: Optional[Clock] = None,
                 ticker: Optional[object] = None,
                 emit_depth: bool = True):
        self.bus = bus
        self._symbols = list(symbols)
        self.token_to_symbol = dict(token_to_symbol)
        self.symbol_to_token = {v: k for k, v in token_to_symbol.items()}
        self.clock = clock or LiveClock()
        self.ticker = ticker
        self.emit_depth = emit_depth
        self._prev_mid: Dict[str, float] = {}
        self._ticks_in = 0
        self._unmapped = 0

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    def tokens(self) -> List[int]:
        """Instrument tokens to subscribe, for the symbols we were given."""
        return [self.symbol_to_token[s] for s in self._symbols
                if s in self.symbol_to_token]

    # ------------------------------------------------------------------
    def on_ticks(self, ticks: List[dict]) -> None:
        """Kite's websocket callback: a batch of tick dicts. Convert + emit."""
        for kt in ticks:
            self._handle(kt)

    def _handle(self, kt: dict) -> None:
        self._ticks_in += 1
        token = kt.get("instrument_token")
        symbol = self.token_to_symbol.get(token)
        if symbol is None:
            self._unmapped += 1
            return
        now = self.clock.now_ns()
        ts = _ts_ns(kt, now)
        t = to_tick(kt, symbol, ts, self._prev_mid.get(symbol, 0.0))
        self._prev_mid[symbol] = t.mid
        self.bus.emit(Topic.TICK, t)
        if self.emit_depth:
            d = to_depth(kt, symbol, ts)
            if d is not None:
                self.bus.emit(Topic.DEPTH, d)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Wire the callbacks and open the socket (if a real ticker is set).

        Kept deliberately small: all the logic worth testing is in the pure
        converters above, so this method has nothing in it that a test needs to
        reach.
        """
        if self.ticker is None:
            raise RuntimeError(
                "KiteFeed.start needs a ticker (a KiteTicker instance, or a "
                "fake in tests). None was provided.")
        tk = self.ticker

        def _on_connect(ws, response=None):
            toks = self.tokens()
            ws.subscribe(toks)
            # 'full' mode = last price + 5-level depth + volume + OHLC + OI,
            # which is the whole microstructure block this engine consumes.
            if hasattr(ws, "set_mode") and hasattr(ws, "MODE_FULL"):
                ws.set_mode(ws.MODE_FULL, toks)

        tk.on_ticks = lambda ws, ticks: self.on_ticks(ticks)
        tk.on_connect = _on_connect
        if hasattr(tk, "connect"):
            tk.connect(threaded=True)

    def stop(self) -> None:
        if self.ticker is not None and hasattr(self.ticker, "close"):
            self.ticker.close()

    def stats(self) -> dict:
        return {"ticks_in": self._ticks_in, "unmapped": self._unmapped,
                "symbols": len(self._symbols),
                "subscribed_tokens": len(self.tokens())}
