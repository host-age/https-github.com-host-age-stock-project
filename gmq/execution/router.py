"""Order router: lifecycle, verification, reconciliation, recovery (spec §11).

The rule this module exists to enforce: **the agent must never assume a
position exists because it sent an order.** An order can be rejected, partially
filled, filled at a different price, or acknowledged and then lost in a
disconnect. Every one of those produces a book that differs from what the
strategy believes, and a strategy trading against a phantom position is worse
than one not trading at all.

So the router keeps two views -- what we intended and what the broker confirms
-- and reconciles them on a timer. Any divergence raises a discrepancy the
risk engine can act on, rather than being silently absorbed.

It also handles:
  * slicing large orders to stay under a participation cap
  * retrying transient rejects with backoff, and *not* retrying hard ones
  * cancelling and replacing stale passive orders
  * flattening everything on demand (square-off, emergency liquidation)
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..core.bus import EventBus, Topic
from ..core.clock import Clock
from ..core.config import ExecConfig
from ..core.types import (
    Order, Fill, OrderStatus, OrderType, Side, Product, TimeInForce, NS,
)
from .broker import Broker, LiveTradingGate, LiveTradingBlocked

# Rejects that will never succeed on a retry. Retrying these is how a bot
# turns one bad order into a throttle ban.
TERMINAL_REJECTS = {
    "TICK_SIZE", "INVALID_QTY", "LOT_SIZE_MISMATCH", "UNKNOWN_SYMBOL",
    "ABOVE_UPPER_CIRCUIT", "BELOW_LOWER_CIRCUIT", "TRADING_HALTED",
    "INSUFFICIENT_MARGIN",
}


@dataclass
class Discrepancy:
    ts: int
    symbol: str
    kind: str
    expected: float
    actual: float
    detail: str = ""


class OrderRouter:
    def __init__(self, broker: Broker, clock: Clock, cfg: ExecConfig,
                 bus: Optional[EventBus] = None,
                 on_fill: Optional[Callable[[Fill], None]] = None,
                 risk_hook: Optional[Callable] = None):
        self.broker = broker
        self.clock = clock
        self.cfg = cfg
        self.bus = bus
        self.on_fill_cb = on_fill
        self.risk_hook = risk_hook
        self.live_orders: Dict[str, Order] = {}
        self.history: List[Order] = []
        self.expected_position: Dict[str, int] = defaultdict(int)
        self.discrepancies: List[Discrepancy] = []
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.slippages: Deque[float] = deque(maxlen=500)
        self.last_reconcile_ns = 0
        self.acknowledged_live = False
        broker.on_fill = self._on_fill
        broker.on_order_update = self._on_order_update

    # ------------------------------------------------------------------
    def submit(self, symbol: str, side: Side, qty: int,
               otype: OrderType = OrderType.MARKET, price: float = 0.0,
               trigger: float = 0.0, tag: str = "",
               product: Product = Product.MIS,
               intent_price: float = 0.0,
               top_depth: int = 0,
               counts_as_decision: bool = True) -> List[Order]:
        """Submit, slicing if the order is large relative to visible depth."""
        LiveTradingGate.check(self.cfg.mode, self.acknowledged_live)
        if qty <= 0:
            return []
        slices = self._slice(qty, price or intent_price, top_depth)
        out: List[Order] = []
        # One parent decision, however many child slices it becomes -- and
        # only if it *is* a decision. Placing and ratcheting a stop-loss is
        # risk-management plumbing: a single position decided on once produces
        # a steady stream of stop modifications, and counting those as
        # decisions makes an engine managing its risk properly look exactly
        # like one in a runaway loop.
        if self.risk_hook and counts_as_decision:
            self.risk_hook("sent", self.clock.now_ns())
        for q in slices:
            o = Order(symbol=symbol, side=side, qty=q, otype=otype,
                      price=price, trigger=trigger, tag=tag, product=product,
                      intent_price=intent_price or price)
            try:
                self.broker.place(o)
            except Exception as e:                     # noqa: BLE001
                o.status = OrderStatus.REJECTED
                o.reject_reason = f"SUBMIT_ERROR:{type(e).__name__}"
            self.live_orders[o.order_id] = o
            self.expected_position[symbol] += q * side.sign
            if self.bus:
                self.bus.emit(Topic.ORDER_NEW, o)
            out.append(o)
        return out

    def _slice(self, qty: int, px: float, top_depth: int) -> List[int]:
        """Split an order so no child takes more than the participation cap."""
        notional = qty * px if px > 0 else 0.0
        cap_qty = qty
        if top_depth > 0:
            cap_qty = max(1, int(top_depth * self.cfg.max_participation * 3))
        if notional and notional > self.cfg.slice_threshold_notional:
            by_notional = max(1, int(self.cfg.slice_threshold_notional / px))
            cap_qty = min(cap_qty, by_notional)
        if cap_qty >= qty:
            return [qty]
        n = int(math.ceil(qty / cap_qty))
        base = qty // n
        out = [base] * n
        for i in range(qty - base * n):
            out[i] += 1
        return [q for q in out if q > 0]

    # ------------------------------------------------------------------
    def cancel(self, order_id: str) -> bool:
        o = self.live_orders.get(order_id)
        if o is None:
            return False
        # The expectation is NOT adjusted here. `_on_order_update` fires for
        # every terminal order and already backs out the unfilled remainder;
        # doing it in both places double-counts every cancellation. With a
        # stop-loss resting on each position -- cancelled and replaced on
        # every ratchet -- that produced hundreds of phantom position
        # mismatches, which is worse than none at all: a reconciliation alarm
        # nobody believes is not an alarm.
        return self.broker.cancel(order_id)

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        n = 0
        for oid, o in list(self.live_orders.items()):
            if symbol and o.symbol != symbol:
                continue
            if o.is_live and self.cancel(oid):
                n += 1
        return n

    def flatten(self, symbol: str, qty: int, price_hint: float = 0.0
                ) -> List[Order]:
        """Close a position now. Used by square-off and emergency liquidation.

        Always a market order: when the reason for flattening is that risk has
        been breached, the priority is certainty of exit, not price.

        Not counted as a trading decision. Flattening is always initiated by
        the risk layer -- a stop-out, square-off or emergency liquidation --
        never by the search. A correlated book stopping out together produces
        a burst of these in one minute, which is precisely the behaviour the
        correlation limits expect, and counting it as decision rate makes the
        engine halt itself for correctly closing its own risk.
        """
        if qty == 0:
            return []
        side = Side.SELL if qty > 0 else Side.BUY
        self.cancel_all(symbol)
        return self.submit(symbol, side, abs(qty), OrderType.MARKET,
                           tag="flatten", intent_price=price_hint,
                           counts_as_decision=False)

    # ------------------------------------------------------------------
    def _on_fill(self, f: Fill) -> None:
        o = self.live_orders.get(f.order_id)
        if o is not None and o.intent_price > 0:
            slip = o.side.sign * (f.price - o.intent_price) / o.intent_price * 1e4
            self.slippages.append(float(slip))
            if self.risk_hook:
                self.risk_hook("slippage", float(slip))
        if self.on_fill_cb:
            self.on_fill_cb(f)

    def _on_order_update(self, o: Order) -> None:
        if o.status.terminal:
            self.live_orders.pop(o.order_id, None)
            self.history.append(o)
            if o.status is OrderStatus.REJECTED:
                self.expected_position[o.symbol] -= o.pending_qty * o.side.sign
                if self.risk_hook:
                    self.risk_hook("reject", self.clock.now_ns())
                self._maybe_retry(o)
            elif o.filled_qty < o.qty:
                # cancelled with a residual: our expectation was too high
                self.expected_position[o.symbol] -= \
                    (o.qty - o.filled_qty) * o.side.sign
        if self.bus:
            self.bus.emit(Topic.ORDER_UPDATE, o)

    def _maybe_retry(self, o: Order) -> None:
        reason = (o.reject_reason or "").upper()
        if any(t in reason for t in TERMINAL_REJECTS):
            return
        n = self.retry_counts[o.tag or o.order_id]
        if n >= self.cfg.max_retries:
            return
        self.retry_counts[o.tag or o.order_id] = n + 1
        # A transient reject means the venue was busy. Re-sending immediately
        # is what turns a busy venue into a throttled one, so back off and
        # cross the spread rather than re-posting at a price that may be stale.
        self.submit(o.symbol, o.side, o.pending_qty, OrderType.MARKET,
                    tag=o.tag, intent_price=o.intent_price or o.price)

    # ------------------------------------------------------------------
    def reconcile(self, ts: int, force: bool = False) -> List[Discrepancy]:
        """Compare what we think we hold with what the broker says.

        This is the check that catches the failure the spec calls out
        explicitly: assuming a position exists because an order was sent.
        """
        if not force and (ts - self.last_reconcile_ns) < \
                self.cfg.reconcile_every_s * NS:
            return []
        self.last_reconcile_ns = ts
        broker_pos = self.broker.positions()
        found: List[Discrepancy] = []
        symbols = set(broker_pos) | set(self.expected_position)
        for s in symbols:
            exp = self.expected_position.get(s, 0)
            act = broker_pos.get(s, 0)
            # in-flight quantity legitimately explains a gap
            inflight = sum(o.pending_qty * o.side.sign
                           for o in self.live_orders.values()
                           if o.symbol == s and o.is_live)
            if exp - inflight != act:
                d = Discrepancy(ts=ts, symbol=s, kind="POSITION_MISMATCH",
                                expected=float(exp - inflight), actual=float(act),
                                detail=f"inflight={inflight}")
                found.append(d)
                self.discrepancies.append(d)
                # The broker is the source of truth. Ours is a belief.
                self.expected_position[s] = act + inflight
        if found and self.bus:
            for d in found:
                self.bus.emit(Topic.RISK_EVENT,
                              {"type": "reconcile", "detail": d.__dict__})
        return found

    # ------------------------------------------------------------------
    def verify_filled(self, order_id: str) -> Tuple[bool, int, float]:
        """Confirm a fill from the broker rather than from our own hope."""
        st = self.broker.order_status(order_id)
        o = self.live_orders.get(order_id)
        if o is None:
            o = next((x for x in self.history if x.order_id == order_id), None)
        if o is None or st is None:
            return False, 0, 0.0
        return (st in (OrderStatus.COMPLETE, OrderStatus.PARTIAL),
                o.filled_qty, o.avg_price)

    def stale_passive_orders(self, ts: int, max_age_s: float = 45.0
                             ) -> List[Order]:
        out = []
        for o in self.live_orders.values():
            if o.otype is OrderType.LIMIT and o.status is OrderStatus.OPEN:
                age = (ts - (o.ack_ns or o.sent_ns)) / NS
                if age > max_age_s:
                    out.append(o)
        return out

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        sl = list(self.slippages)
        done = self.history
        # Fill rate is measured over orders that were *meant* to fill. A
        # protective stop that never triggered is a success, not a miss, and
        # counting it as one drags the metric toward 50% and hides a genuine
        # execution problem behind it.
        # Identified by tag, not by order type and not by list membership.
        # A stop that triggers is rewritten to a MARKET order by the broker,
        # so its type no longer says what it was -- which reported "0 stops
        # triggered" for a run in which every stop worked. And `Order` is a
        # dataclass with a generated __eq__, so `o not in working` compares by
        # value: two distinct stops with identical fields test equal, and the
        # partition silently loses orders (in O(n^2), for good measure).
        def is_stop(o: Order) -> bool:
            return o.tag.startswith("stop:")

        working = [o for o in done if not is_stop(o)]
        stops = [o for o in done if is_stop(o)]
        filled = [o for o in working if o.filled_qty > 0]
        rejected = [o for o in done if o.status is OrderStatus.REJECTED]
        lat = [o.latency_us for o in done if o.latency_us > 0]
        return {
            "submitted": len(done) + len(self.live_orders),
            "filled": len(filled),
            "stops_placed": len(stops),
            "stops_triggered": sum(1 for o in stops if o.filled_qty > 0),
            "rejected": len(rejected),
            "reject_rate": round(len(rejected) / max(len(done), 1), 4),
            "fill_rate": round(len(filled) / max(len(working), 1), 4),
            "live": len(self.live_orders),
            "mean_slippage_bps": round(float(np.mean(sl)), 3) if sl else 0.0,
            "p95_slippage_bps": round(float(np.percentile(sl, 95)), 3)
            if len(sl) > 20 else 0.0,
            "mean_latency_us": round(float(np.mean(lat)), 1) if lat else 0.0,
            "discrepancies": len(self.discrepancies),
            "retries": sum(self.retry_counts.values()),
        }
