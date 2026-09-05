"""Limit order book with NSE-style price-time priority matching.

This is the exchange side of the replica. It is a genuine matching engine:
resting orders queue at each price level in arrival order, aggressive orders
walk the book consuming levels, and partial fills leave the residual resting
(or cancel it, for IOC). Circuit limits and tick-size validation are enforced
here, the way the exchange does it -- not by the client.

Design notes
------------
* Price levels are dicts keyed by integer ticks, with the ladder of occupied
  prices kept sorted incrementally via bisect. The sorted view is read on
  every order but written only when a price level is created or emptied, so
  maintaining it beats both re-sorting and a heap.
* Every mutation bumps a sequence number so consumers can detect gaps.
"""
from __future__ import annotations

import itertools
from bisect import insort, bisect_left
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Callable

from ..core.types import (
    Side, OrderType, OrderStatus, Fill, DepthSnapshot, BookLevel, Instrument,
)

_eid = itertools.count(1)


@dataclass(slots=True)
class RestingOrder:
    eid: int
    symbol: str
    side: Side
    price_ticks: int
    qty: int
    remaining: int
    ts: int
    owner: str = "MKT"          # "AGENT" for our own orders, else background
    client_order_id: str = ""
    hidden: int = 0             # iceberg reserve
    otype: OrderType = OrderType.LIMIT


class OrderBook:
    """One instrument's book."""

    __slots__ = (
        "inst", "tick", "bids", "asks", "_bid_keys", "_ask_keys",
        "orders", "seq", "last_price_ticks",
        "prev_close_ticks", "volume", "turnover", "trades", "halted",
        "upper_ticks", "lower_ticks", "_on_trade", "buy_volume", "sell_volume",
        "open_ticks", "high_ticks", "low_ticks",
    )

    def __init__(self, inst: Instrument, prev_close: float,
                 on_trade: Optional[Callable] = None):
        self.inst = inst
        self.tick = inst.tick_size
        self.bids: Dict[int, Deque[RestingOrder]] = {}
        self.asks: Dict[int, Deque[RestingOrder]] = {}
        # Both key lists are kept sorted ASCENDING and maintained incrementally
        # with bisect. Re-sorting the whole price ladder on every insertion is
        # the single most expensive thing a naive book does: level creation is
        # rare, but the sorted view is read on every single order, so the cost
        # lands in the hottest loop in the system.
        self._bid_keys: List[int] = []
        self._ask_keys: List[int] = []
        self.orders: Dict[int, RestingOrder] = {}
        self.seq = 0
        self.prev_close_ticks = self.to_ticks(prev_close)
        self.last_price_ticks = self.prev_close_ticks
        self.open_ticks = 0
        self.high_ticks = 0
        self.low_ticks = 0
        self.volume = 0
        self.buy_volume = 0
        self.sell_volume = 0
        self.turnover = 0.0
        self.trades = 0
        self.halted = False
        band = inst.circuit_pct
        self.upper_ticks = int(self.prev_close_ticks * (1 + band))
        self.lower_ticks = max(1, int(self.prev_close_ticks * (1 - band)))
        self._on_trade = on_trade

    # -- price conversions ---------------------------------------------
    def to_ticks(self, px: float) -> int:
        return int(round(px / self.tick))

    def to_price(self, t: int) -> float:
        return round(t * self.tick, 4)

    # -- top of book ----------------------------------------------------
    def _add_key(self, side: Side, pt: int) -> None:
        keys = self._bid_keys if side is Side.BUY else self._ask_keys
        insort(keys, pt)

    def _drop_key(self, side: Side, pt: int) -> None:
        keys = self._bid_keys if side is Side.BUY else self._ask_keys
        i = bisect_left(keys, pt)
        if i < len(keys) and keys[i] == pt:
            del keys[i]

    def _sorted_bids(self) -> List[int]:
        """Bid prices, best first."""
        return self._bid_keys[::-1]

    def _sorted_asks(self) -> List[int]:
        """Ask prices, best first."""
        return self._ask_keys

    @property
    def best_bid_ticks(self) -> int:
        return self._bid_keys[-1] if self._bid_keys else 0

    @property
    def best_ask_ticks(self) -> int:
        return self._ask_keys[0] if self._ask_keys else 0

    @property
    def best_bid(self) -> float:
        t = self.best_bid_ticks
        return self.to_price(t) if t else 0.0

    @property
    def best_ask(self) -> float:
        t = self.best_ask_ticks
        return self.to_price(t) if t else 0.0

    @property
    def ltp(self) -> float:
        return self.to_price(self.last_price_ticks)

    @property
    def mid(self) -> float:
        b, a = self.best_bid, self.best_ask
        if b and a:
            return 0.5 * (b + a)
        return b or a or self.ltp

    def level_qty(self, side: Side, ticks: int) -> int:
        book = self.bids if side is Side.BUY else self.asks
        q = book.get(ticks)
        return sum(o.remaining for o in q) if q else 0

    def depth(self, ts: int, levels: int = 5) -> DepthSnapshot:
        bk, ak = self._bid_keys, self._ask_keys
        bids = []
        for t in reversed(bk[-levels:]) if bk else ():
            q = self.bids.get(t)
            if q:
                bids.append(BookLevel(self.to_price(t),
                                      sum(o.remaining for o in q), len(q)))
        asks = []
        for t in ak[:levels]:
            q = self.asks.get(t)
            if q:
                asks.append(BookLevel(self.to_price(t),
                                      sum(o.remaining for o in q), len(q)))
        return DepthSnapshot(ts=ts, symbol=self.inst.symbol, bids=bids, asks=asks)

    # -- validation -----------------------------------------------------
    def validate(self, price_ticks: int, qty: int, otype: OrderType
                 ) -> Tuple[bool, str]:
        if self.halted:
            return False, "TRADING_HALTED"
        if qty <= 0:
            return False, "INVALID_QTY"
        if self.inst.lot_size > 1 and qty % self.inst.lot_size:
            return False, "LOT_SIZE_MISMATCH"
        if otype in (OrderType.LIMIT, OrderType.SL, OrderType.IOC):
            if price_ticks <= 0:
                return False, "INVALID_PRICE"
            if price_ticks > self.upper_ticks:
                return False, "ABOVE_UPPER_CIRCUIT"
            if price_ticks < self.lower_ticks:
                return False, "BELOW_LOWER_CIRCUIT"
        return True, ""

    # -- mutation -------------------------------------------------------
    def add_limit(self, side: Side, price: float, qty: int, ts: int,
                  owner: str = "MKT", client_order_id: str = "",
                  ioc: bool = False) -> Tuple[List[Fill], Optional[int], str]:
        """Insert an order, matching aggressively first.

        Returns (fills, resting_eid_or_None, reject_reason).
        """
        pt = self.to_ticks(price)
        ok, why = self.validate(pt, qty, OrderType.IOC if ioc else OrderType.LIMIT)
        if not ok:
            return [], None, why

        fills, remaining = self._match(side, pt, qty, ts, owner, client_order_id)

        if remaining <= 0:
            return fills, None, ""
        if ioc:
            return fills, None, ""

        eid = next(_eid)
        ro = RestingOrder(eid, self.inst.symbol, side, pt, qty, remaining, ts,
                          owner, client_order_id)
        book = self.bids if side is Side.BUY else self.asks
        q = book.get(pt)
        if q is None:
            q = deque()
            book[pt] = q
            self._add_key(side, pt)
        q.append(ro)
        self.orders[eid] = ro
        self.seq += 1
        return fills, eid, ""

    def add_market(self, side: Side, qty: int, ts: int, owner: str = "MKT",
                   client_order_id: str = "") -> Tuple[List[Fill], int, str]:
        """Market order: sweeps up to the circuit band, residual is cancelled."""
        ok, why = self.validate(0, qty, OrderType.MARKET)
        if not ok:
            return [], qty, why
        limit_ticks = self.upper_ticks if side is Side.BUY else self.lower_ticks
        fills, remaining = self._match(side, limit_ticks, qty, ts, owner,
                                       client_order_id)
        return fills, remaining, ""

    def _match(self, side: Side, limit_ticks: int, qty: int, ts: int,
               owner: str, coid: str) -> Tuple[List[Fill], int]:
        fills: List[Fill] = []
        remaining = qty
        buying = side is Side.BUY
        opp = self.asks if buying else self.bids
        opp_side = Side.SELL if buying else Side.BUY
        # Walk the opposing ladder from the touch outward. The key list is
        # mutated as levels empty, so step by index from the appropriate end
        # rather than iterating it directly.
        while remaining > 0:
            keys = self._ask_keys if buying else self._bid_keys
            if not keys:
                break
            pt = keys[0] if buying else keys[-1]
            if buying and pt > limit_ticks:
                break
            if (not buying) and pt < limit_ticks:
                break
            q = opp.get(pt)
            if not q:
                opp.pop(pt, None)
                self._drop_key(opp_side, pt)
                continue
            px = self.to_price(pt)
            while q and remaining > 0:
                head = q[0]
                take = min(remaining, head.remaining)
                head.remaining -= take
                remaining -= take
                self._book_trade(px, take, ts, side)
                # aggressor fill
                fills.append(Fill(order_id=coid, symbol=self.inst.symbol,
                                  side=side, qty=take, price=px, ts=ts,
                                  liquidity="TAKER"))
                # passive fill -- reported so our own resting orders get filled
                if head.owner == "AGENT" and self._on_trade:
                    self._on_trade(Fill(order_id=head.client_order_id,
                                        symbol=self.inst.symbol,
                                        side=head.side, qty=take, price=px,
                                        ts=ts, liquidity="MAKER"))
                if head.remaining <= 0:
                    q.popleft()
                    self.orders.pop(head.eid, None)
            if not q:
                opp.pop(pt, None)
                self._drop_key(opp_side, pt)
        if fills:
            self.seq += 1
        return fills, remaining

    def _book_trade(self, px: float, qty: int, ts: int, aggressor: Side) -> None:
        pt = self.to_ticks(px)
        self.last_price_ticks = pt
        if not self.open_ticks:
            self.open_ticks = pt
            self.high_ticks = self.low_ticks = pt
        self.high_ticks = max(self.high_ticks, pt)
        self.low_ticks = min(self.low_ticks, pt)
        self.volume += qty
        if aggressor is Side.BUY:
            self.buy_volume += qty
        else:
            self.sell_volume += qty
        self.turnover += px * qty
        self.trades += 1
        # dynamic circuit breaker: NSE halts on large moves from prev close
        move = abs(pt - self.prev_close_ticks) / max(self.prev_close_ticks, 1)
        if move >= self.inst.circuit_pct:
            self.halted = True

    def cancel(self, eid: int) -> bool:
        ro = self.orders.pop(eid, None)
        if ro is None:
            return False
        book = self.bids if ro.side is Side.BUY else self.asks
        q = book.get(ro.price_ticks)
        if q:
            try:
                q.remove(ro)
            except ValueError:
                pass
            if not q:
                book.pop(ro.price_ticks, None)
                self._drop_key(ro.side, ro.price_ticks)
        self.seq += 1
        return True

    def modify(self, eid: int, new_qty: Optional[int] = None,
               new_price: Optional[float] = None, ts: int = 0
               ) -> Tuple[bool, Optional[int]]:
        """Quantity-down keeps time priority; price change loses it (as at NSE)."""
        ro = self.orders.get(eid)
        if ro is None:
            return False, None
        if new_price is not None and self.to_ticks(new_price) != ro.price_ticks:
            side, qty, owner, coid = ro.side, (new_qty or ro.remaining), ro.owner, ro.client_order_id
            self.cancel(eid)
            _, new_eid, err = self.add_limit(side, new_price, qty, ts, owner, coid)
            return (not err), new_eid
        if new_qty is not None:
            if new_qty <= 0:
                self.cancel(eid)
                return True, None
            if new_qty < ro.remaining:
                ro.remaining = new_qty          # priority preserved
            else:
                # increasing size goes to the back of the queue
                extra = new_qty - ro.remaining
                book = self.bids if ro.side is Side.BUY else self.asks
                q = book.get(ro.price_ticks)
                if q is not None:
                    q.remove(ro)
                    ro.remaining = new_qty
                    ro.qty = new_qty
                    q.append(ro)
            self.seq += 1
        return True, eid

    def cancel_all(self, owner: str = "AGENT") -> int:
        eids = [e for e, o in self.orders.items() if o.owner == owner]
        for e in eids:
            self.cancel(e)
        return len(eids)

    # -- analytics ------------------------------------------------------
    def queue_position(self, eid: int) -> int:
        """Volume ahead of our order at its price level -- drives realistic
        fill probability estimates for passive orders."""
        ro = self.orders.get(eid)
        if ro is None:
            return -1
        book = self.bids if ro.side is Side.BUY else self.asks
        q = book.get(ro.price_ticks)
        if not q:
            return -1
        ahead = 0
        for o in q:
            if o.eid == ro.eid:
                return ahead
            ahead += o.remaining
        return ahead

    def sweep_cost_bps(self, side: Side, qty: int) -> float:
        """Cost of taking `qty` immediately, in bps versus mid."""
        d = self.depth(0, levels=20)
        avg, filled = d.sweep_cost(side, qty)
        m = self.mid
        if not filled or m <= 0:
            return 999.0
        shortfall = (avg - m) * side.sign / m * 1e4
        if filled < qty:
            shortfall += 25.0 * (1 - filled / qty)      # unfillable penalty
        return shortfall

    def stats(self) -> Dict[str, float]:
        d = self.depth(0)
        return {
            "ltp": self.ltp,
            "bid": self.best_bid,
            "ask": self.best_ask,
            "spread": self.best_ask - self.best_bid if (self.best_bid and self.best_ask) else 0.0,
            "imbalance": d.imbalance(),
            "volume": float(self.volume),
            "vwap": self.turnover / self.volume if self.volume else self.ltp,
            "trades": float(self.trades),
            "halted": float(self.halted),
            "buy_vol": float(self.buy_volume),
            "sell_vol": float(self.sell_volume),
        }
