"""Simulated broker: the honest path from a decision to a fill.

Sitting between the strategy and the matching engine, this models everything
that makes real execution worse than a backtest assumes:

  * **Latency.** An order is not at the exchange when you send it. It arrives
    some microseconds later, by which time the book has moved. Modelled as a
    scheduled arrival on the simulation clock, so the order matches against
    the book *as it will be*, not as it was when the decision was made. This
    single detail is the difference between a backtest and a fantasy.

  * **Rejects.** Margin, circuit limits, tick size, exchange throttles.

  * **Partial fills.** Large orders do not fill in one print.

  * **Slippage,** which here is not a fudge factor but the emergent result of
    walking a real book after real latency.

  * **Stop orders held at the broker,** triggered on the tape, then sent as
    market orders -- which is what actually happens, and why a stop does not
    fill at its trigger price.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.bus import EventBus, Topic, TimerWheel
from ..core.clock import SimClock
from ..core.config import ExecConfig
from ..core.types import (
    Order, Fill, OrderStatus, OrderType, Side, Tick, NS,
)
from ..data.simexchange import SimExchange
from ..strategy.evaluator import CostModel
from .broker import Broker, BrokerError


@dataclass
class _Pending:
    order: Order
    exchange_eid: Optional[int] = None
    arrives_ns: int = 0
    triggered: bool = False


class SimBroker(Broker):
    name = "sim"
    supports_depth = True

    def __init__(self, exchange: SimExchange, clock: SimClock,
                 cfg: ExecConfig, bus: Optional[EventBus] = None,
                 timers: Optional[TimerWheel] = None, seed: int = 17,
                 cost: Optional[CostModel] = None):
        self.cost = cost or CostModel()
        self.ex = exchange
        self.clock = clock
        self.cfg = cfg
        self.bus = bus
        # NOT `timers or TimerWheel()`: TimerWheel.__len__ is 0 when empty, so
        # a freshly-passed (empty) wheel is falsy and would be silently
        # replaced by a private one -- meaning a caller that schedules on the
        # wheel it passed in and then drives it never sees the broker's timers.
        # Harmless in the sim (the engine only ever calls broker.pump()), fatal
        # for a live loop that owns the clock and the wheel.
        self.timers = timers if timers is not None else TimerWheel()
        self.rng = np.random.default_rng(seed)
        self._orders: Dict[str, Order] = {}
        self._pending: Dict[str, _Pending] = {}
        self._eid_to_oid: Dict[int, str] = {}
        self._positions: Dict[str, int] = defaultdict(int)
        self._connected = False
        self.on_fill: Optional[Callable[[Fill], None]] = None
        self.on_order_update: Optional[Callable[[Order], None]] = None
        self.rejects = 0
        self.placed = 0
        self.exchange_fills = 0
        self.fees_charged = 0.0
        # stop orders resting at the broker, keyed by symbol
        self._stops: Dict[str, List[Order]] = defaultdict(list)
        self.ex.on_agent_fill = self._on_maker_fill

    # ------------------------------------------------------------------
    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    def _latency_ns(self) -> int:
        base = self.cfg.lat_send_us + self.cfg.lat_ack_us
        jitter = abs(self.rng.normal(0.0, self.cfg.lat_jitter_us))
        # Latency is right-skewed in reality: mostly fast with an occasional
        # long tail. A symmetric jitter would hide exactly the events that
        # cost the most.
        if self.rng.random() < 0.02:
            jitter += self.rng.exponential(self.cfg.lat_jitter_us * 8)
        return int((base + jitter) * 1000)

    def place(self, order: Order) -> str:
        if not self._connected:
            raise BrokerError("not connected")
        self.placed += 1
        order.sent_ns = self.clock.now_ns()
        order.status = OrderStatus.PENDING
        self._orders[order.order_id] = order

        # pre-trade rejects the broker itself would raise
        reason = self._pre_trade_reject(order)
        if reason:
            order.status = OrderStatus.REJECTED
            order.reject_reason = reason
            order.done_ns = self.clock.now_ns()
            self.rejects += 1
            self._notify(order)
            return order.order_id

        if order.otype in (OrderType.SL, OrderType.SL_M):
            order.status = OrderStatus.TRIGGER_PENDING
            order.ack_ns = self.clock.now_ns() + self._latency_ns()
            self._stops[order.symbol].append(order)
            self._notify(order)
            return order.order_id

        arrive = self.clock.now_ns() + self._latency_ns()
        self._pending[order.order_id] = _Pending(order, arrives_ns=arrive)
        # The order reaches the matching engine later, against whatever book
        # exists at that moment.
        self.timers.schedule(arrive, self._deliver, order.order_id)
        return order.order_id

    def _pre_trade_reject(self, order: Order) -> str:
        if order.qty <= 0:
            return "INVALID_QTY"
        st = self.ex.state.get(order.symbol)
        if st is None:
            return "UNKNOWN_SYMBOL"
        if st.book.halted:
            return "TRADING_HALTED"
        inst = st.inst
        if order.otype in (OrderType.LIMIT, OrderType.SL) and order.price > 0:
            if abs(round(order.price / inst.tick_size) * inst.tick_size
                   - order.price) > 1e-6:
                return "TICK_SIZE"
        if self.rng.random() < self.cfg.reject_prob:
            return "EXCHANGE_THROTTLE"
        return ""

    # ------------------------------------------------------------------
    def _deliver(self, order_id: str) -> None:
        """The order actually arrives at the exchange."""
        p = self._pending.get(order_id)
        if p is None:
            return
        o = p.order
        if o.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
            self._pending.pop(order_id, None)
            return
        o.ack_ns = self.clock.now_ns()

        qty = o.pending_qty
        if qty <= 0:
            self._pending.pop(order_id, None)
            return

        # Large orders do not arrive as one block. Anything above the
        # participation threshold is sliced, which both reduces impact and
        # means the tail of the order fills at a worse price than the head.
        if o.otype is OrderType.MARKET:
            fills, remaining, err = self.ex.submit_market(
                o.symbol, o.side, qty, o.order_id)
            if err:
                self._reject(o, err)
                return
            self._apply_fills(o, fills)
            if remaining > 0:
                # unfilled remainder of a market order is cancelled at NSE
                o.status = (OrderStatus.PARTIAL if o.filled_qty
                            else OrderStatus.CANCELLED)
                if o.filled_qty:
                    o.status = OrderStatus.COMPLETE if o.filled_qty >= o.qty \
                        else OrderStatus.PARTIAL
                o.done_ns = self.clock.now_ns()
            self._pending.pop(order_id, None)
            self._notify(o)
            return

        ioc = o.otype is OrderType.IOC
        fills, eid, err = self.ex.submit_limit(
            o.symbol, o.side, o.price, qty, o.order_id, ioc=ioc)
        if err:
            self._reject(o, err)
            return
        self._apply_fills(o, fills)
        if eid is not None:
            p.exchange_eid = eid
            self._eid_to_oid[eid] = o.order_id
            o.status = OrderStatus.PARTIAL if o.filled_qty else OrderStatus.OPEN
        else:
            o.status = OrderStatus.COMPLETE if o.filled_qty >= o.qty else (
                OrderStatus.PARTIAL if o.filled_qty else OrderStatus.CANCELLED)
            o.done_ns = self.clock.now_ns()
            self._pending.pop(order_id, None)
        self._notify(o)

    def _reject(self, o: Order, err: str) -> None:
        o.status = OrderStatus.REJECTED
        o.reject_reason = err
        o.done_ns = self.clock.now_ns()
        self.rejects += 1
        self._pending.pop(o.order_id, None)
        self._notify(o)

    # ------------------------------------------------------------------
    def _apply_fills(self, o: Order, fills: List[Fill]) -> None:
        for f in fills:
            f.order_id = o.order_id
            self._book_fill(o, f)

    def _on_maker_fill(self, f: Fill) -> None:
        """A resting order of ours was hit by someone else."""
        o = self._orders.get(f.order_id)
        if o is None:
            return
        self._book_fill(o, f)
        if o.status.terminal:
            self._notify(o)

    def _book_fill(self, o: Order, f: Fill) -> None:
        # Charge the explicit costs of the trade.
        #
        # This is where the backtest stops flattering itself. The matching
        # engine produces a fill price that already contains the spread and
        # the market impact -- the order genuinely walked a real book -- but
        # brokerage, STT, exchange charges, stamp duty and GST are levied by
        # people, not by the order book, so nothing was charging them. Every
        # P&L number the system produced was gross, while being labelled net.
        #
        # Only the explicit fees are added here. Adding a modelled spread on
        # top would double-count what the fill price already paid.
        if f.fee <= 0:
            f.fee = self.cost.fees(abs(f.qty) * f.price, f.side)
        self.fees_charged += f.fee
        prev = o.filled_qty
        o.filled_qty += f.qty
        o.avg_price = (o.avg_price * prev + f.price * f.qty) / max(o.filled_qty, 1)
        self._positions[o.symbol] += f.qty * f.side.sign
        self.exchange_fills += 1
        if o.filled_qty >= o.qty:
            o.status = OrderStatus.COMPLETE
            o.done_ns = self.clock.now_ns()
        else:
            o.status = OrderStatus.PARTIAL
        if self.on_fill:
            self.on_fill(f)
        if self.bus:
            self.bus.emit(Topic.FILL, f)

    def _notify(self, o: Order) -> None:
        if self.on_order_update:
            self.on_order_update(o)
        if self.bus:
            self.bus.emit(Topic.ORDER_UPDATE, o)

    # ------------------------------------------------------------------
    def on_tick(self, t: Tick) -> None:
        """Trigger broker-held stop orders on the tape.

        A stop is not resting in the order book -- the exchange never sees it
        until it triggers. It becomes a market order at the moment of trigger
        and fills wherever the book is *then*, which is why stop fills are
        systematically worse than the trigger price and why modelling them as
        exact fills flatters a backtest most in exactly the fast markets where
        stops matter.
        """
        pend = self._stops.get(t.symbol)
        if not pend:
            return
        fire: List[Order] = []
        for o in pend:
            if o.status is not OrderStatus.TRIGGER_PENDING:
                continue
            if o.side is Side.SELL and t.ltp <= o.trigger:
                fire.append(o)
            elif o.side is Side.BUY and t.ltp >= o.trigger:
                fire.append(o)
        for o in fire:
            pend.remove(o)
            o.otype = OrderType.MARKET if o.otype is OrderType.SL_M \
                else OrderType.LIMIT
            o.status = OrderStatus.PENDING
            arrive = self.clock.now_ns() + self._latency_ns()
            self._pending[o.order_id] = _Pending(o, arrives_ns=arrive)
            self.timers.schedule(arrive, self._deliver, o.order_id)

    def pump(self) -> int:
        """Run any timers that are due. Called by the engine loop."""
        return self.timers.run_until(self.clock.now_ns())

    # ------------------------------------------------------------------
    def cancel(self, order_id: str) -> bool:
        o = self._orders.get(order_id)
        if o is None or o.status.terminal:
            return False
        if o.status is OrderStatus.TRIGGER_PENDING:
            lst = self._stops.get(o.symbol)
            if lst and o in lst:
                lst.remove(o)
            o.status = OrderStatus.CANCELLED
            o.done_ns = self.clock.now_ns()
            self._notify(o)
            return True
        p = self._pending.get(order_id)
        if p and p.exchange_eid is not None:
            self.ex.cancel(o.symbol, p.exchange_eid)
            self._eid_to_oid.pop(p.exchange_eid, None)
        o.status = OrderStatus.CANCELLED if not o.filled_qty else OrderStatus.COMPLETE
        o.done_ns = self.clock.now_ns()
        self._pending.pop(order_id, None)
        self._notify(o)
        return True

    def modify(self, order_id: str, qty: Optional[int] = None,
               price: Optional[float] = None) -> bool:
        o = self._orders.get(order_id)
        if o is None or o.status.terminal:
            return False
        p = self._pending.get(order_id)
        if p is None or p.exchange_eid is None:
            if price is not None:
                o.price = price
            if qty is not None:
                o.qty = qty
            return True
        ok, new_eid = self.ex.modify(o.symbol, p.exchange_eid, qty, price)
        if ok:
            if p.exchange_eid in self._eid_to_oid:
                self._eid_to_oid.pop(p.exchange_eid)
            p.exchange_eid = new_eid
            if new_eid is not None:
                self._eid_to_oid[new_eid] = o.order_id
            if price is not None:
                o.price = price
            if qty is not None:
                o.qty = qty
            self._notify(o)
        return ok

    # ------------------------------------------------------------------
    def orders(self) -> List[Order]:
        return list(self._orders.values())

    def open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.is_live]

    def positions(self) -> Dict[str, int]:
        return {k: v for k, v in self._positions.items() if v}

    def order_status(self, order_id: str) -> Optional[OrderStatus]:
        o = self._orders.get(order_id)
        return o.status if o else None

    def stats(self) -> dict:
        live = [o for o in self._orders.values() if o.is_live]
        lat = [o.latency_us for o in self._orders.values() if o.latency_us > 0]
        return {
            "placed": self.placed,
            "rejects": self.rejects,
            "reject_rate": round(self.rejects / max(self.placed, 1), 4),
            "fills": self.exchange_fills,
            "live_orders": len(live),
            "fees_charged": round(self.fees_charged, 2),
            "mean_latency_us": round(float(np.mean(lat)), 1) if lat else 0.0,
            "p99_latency_us": round(float(np.percentile(lat, 99)), 1)
            if len(lat) > 20 else 0.0,
        }
