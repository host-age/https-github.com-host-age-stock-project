"""A matching surface backed by live quotes, for paper trading on real prices.

`SimBroker` fills orders against anything that exposes `submit_market`,
`submit_limit`, `cancel`, `modify`, an `on_agent_fill` callback, a `.ts`, and a
`.state[symbol]` with `.book.halted` and `.inst.tick_size`. The simulator's
exchange is one such thing. `LiveQuoteBook` is another -- it holds the latest
real top-of-book per symbol (fed from the Kite websocket) and fills against it.

This is the whole trick to paper-trading on live data without a second, parallel
execution model: the broker, its latency model, its stop-trigger logic, its fee
charging and its accounting are the *same tested code* that ran against the
simulator. Only the price a fill lands at now comes from the real market instead
of a synthetic book.

The fill model is deliberately unflattering, matching the philosophy the rest of
the system holds:

* a **market** order crosses the spread and pays a square-root impact for any
  size beyond what is showing at the touch -- filling a large order at the quote
  is the most common way a paper account lies to itself;
* a **marketable limit** fills at the touch (you never pay worse than your
  limit, and the book gives you the quote), the rest resting;
* a **passive limit** rests and fills only when the tape actually trades to it --
  which, being adverse selection, is disproportionately just before the market
  keeps going through it. That is modelled by filling a resting order when a
  later tick prints at or through its price, never by assuming it fills for free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..core.types import Tick, Fill, Side


@dataclass
class _Inst:
    tick_size: float = 0.05


@dataclass
class _Book:
    halted: bool = False


@dataclass
class _State:
    inst: _Inst = field(default_factory=_Inst)
    book: _Book = field(default_factory=_Book)


@dataclass
class _Quote:
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    ltp: float = 0.0


@dataclass
class _Resting:
    eid: int
    coid: str
    symbol: str
    side: Side
    price: float
    qty: int


class LiveQuoteBook:
    """Fills against the latest live quote. Duck-types SimExchange for SimBroker."""

    def __init__(self, symbols: List[str], impact_coef: float = 0.5,
                 max_impact_bps: float = 60.0, tick_size: float = 0.05):
        self.state: Dict[str, _State] = {
            s: _State(inst=_Inst(tick_size=tick_size)) for s in symbols}
        self.quotes: Dict[str, _Quote] = {s: _Quote() for s in symbols}
        self.on_agent_fill: Optional[Callable[[Fill], None]] = None
        self.ts: int = 0
        self.impact_coef = impact_coef
        self.max_impact_bps = max_impact_bps
        self._resting: Dict[int, _Resting] = {}
        self._eid = 0

    # ------------------------------------------------------------------
    def update(self, t: Tick) -> None:
        """Take a live tick: refresh the quote and fill any resting order the
        tape has now traded through."""
        self.ts = t.ts
        st = self.state.get(t.symbol)
        if st is None:
            self.state[t.symbol] = _State()
            self.quotes[t.symbol] = _Quote()
        q = self.quotes[t.symbol]
        q.bid = t.bid or q.bid
        q.ask = t.ask or q.ask
        q.bid_qty = t.bid_qty or q.bid_qty
        q.ask_qty = t.ask_qty or q.ask_qty
        q.ltp = t.ltp or q.ltp
        self._match_resting(t)

    def halt(self, symbol: str, halted: bool = True) -> None:
        if symbol in self.state:
            self.state[symbol].book.halted = halted

    # ------------------------------------------------------------------
    def _impact_frac(self, qty: int, top_qty: int) -> float:
        top = max(float(top_qty), 1.0)
        part = qty / top
        bps = min(self.impact_coef * 100.0 * math.sqrt(max(part, 0.0)),
                  self.max_impact_bps)
        return bps / 1e4

    def submit_market(self, symbol: str, side: Side, qty: int,
                      coid: str) -> Tuple[List[Fill], int, str]:
        q = self.quotes.get(symbol)
        if q is None or (side is Side.BUY and q.ask <= 0) or \
                (side is Side.SELL and q.bid <= 0):
            return [], qty, "NO_QUOTE"
        touch = q.ask if side is Side.BUY else q.bid
        top = q.ask_qty if side is Side.BUY else q.bid_qty
        imp = self._impact_frac(qty, top)
        px = touch * (1 + imp) if side is Side.BUY else touch * (1 - imp)
        f = Fill(order_id=coid, symbol=symbol, side=side, qty=qty,
                 price=round(px, 2), ts=self.ts, liquidity="TAKER")
        return [f], 0, ""

    def submit_limit(self, symbol: str, side: Side, price: float, qty: int,
                     coid: str, ioc: bool = False
                     ) -> Tuple[List[Fill], Optional[int], str]:
        q = self.quotes.get(symbol)
        if q is None:
            return [], None, "NO_QUOTE"
        marketable = (side is Side.BUY and q.ask > 0 and price >= q.ask) or \
                     (side is Side.SELL and q.bid > 0 and price <= q.bid)
        if marketable:
            touch = q.ask if side is Side.BUY else q.bid
            # you never pay worse than your limit
            px = min(price, touch) if side is Side.BUY else max(price, touch)
            f = Fill(order_id=coid, symbol=symbol, side=side, qty=qty,
                     price=round(px, 2), ts=self.ts, liquidity="TAKER")
            return [f], None, ""
        if ioc:
            return [], None, ""             # IOC that cannot fill is cancelled
        # rest it; it becomes a maker fill when the tape trades to it
        self._eid += 1
        self._resting[self._eid] = _Resting(self._eid, coid, symbol, side,
                                             round(price, 2), qty)
        return [], self._eid, ""

    def _match_resting(self, t: Tick) -> None:
        if not self._resting:
            return
        done: List[int] = []
        for eid, r in self._resting.items():
            if r.symbol != t.symbol:
                continue
            # a resting buy fills when price trades down to it; a resting sell
            # when price trades up to it. Use the traded price and the touch.
            hit = ((r.side is Side.BUY and (t.ltp <= r.price or
                                            (t.ask and t.ask <= r.price))) or
                   (r.side is Side.SELL and (t.ltp >= r.price or
                                             (t.bid and t.bid >= r.price))))
            if not hit:
                continue
            f = Fill(order_id=r.coid, symbol=r.symbol, side=r.side, qty=r.qty,
                     price=r.price, ts=t.ts, liquidity="MAKER")
            if self.on_agent_fill:
                self.on_agent_fill(f)
            done.append(eid)
        for eid in done:
            self._resting.pop(eid, None)

    # ------------------------------------------------------------------
    def cancel(self, symbol: str, eid: int) -> bool:
        return self._resting.pop(eid, None) is not None

    def modify(self, symbol: str, eid: int, qty=None, price=None
               ) -> Tuple[bool, Optional[int]]:
        r = self._resting.get(eid)
        if r is None:
            return False, None
        if qty is not None:
            r.qty = int(qty)
        if price is not None:
            r.price = round(float(price), 2)
        return True, eid
