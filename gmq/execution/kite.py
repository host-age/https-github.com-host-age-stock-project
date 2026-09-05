"""Zerodha Kite adapter -- structure only, deliberately inert.

This file exists so the shape of a real broker integration is visible and the
rest of the system needs no change to use one. It does **not** place orders.
Every method that would touch a live account raises unless the multi-part
`LiveTradingGate` is satisfied, and even then the HTTP calls are left
unimplemented, because shipping a working live-order path inside a system
whose models have only ever been validated against a simulator would be the
single most dangerous thing in this repository.

To make this real you would:
  1. `pip install kiteconnect`
  2. supply api_key / access_token from a completed login flow
  3. implement the three marked methods against the KiteConnect client
  4. re-validate everything -- the latency, slippage and fill-probability
     models in this system are calibrated against the simulator, and none of
     those numbers transfer to a real venue without measurement

Read the notes at the bottom before doing any of that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..core.types import Order, Fill, OrderStatus, OrderType, Side, Product
from .broker import Broker, BrokerError, LiveTradingGate, LiveTradingBlocked


# Kite's vocabulary differs from ours in every field. Mapping it in one place
# means a rename upstream breaks loudly here rather than silently mis-routing.
_OTYPE = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "SL",
    OrderType.SL_M: "SL-M",
    OrderType.IOC: "LIMIT",
}
_STATUS = {
    "COMPLETE": OrderStatus.COMPLETE,
    "OPEN": OrderStatus.OPEN,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "TRIGGER PENDING": OrderStatus.TRIGGER_PENDING,
    "PUT ORDER REQ RECEIVED": OrderStatus.PENDING,
    "VALIDATION PENDING": OrderStatus.PENDING,
    "OPEN PENDING": OrderStatus.PENDING,
    "MODIFY VALIDATION PENDING": OrderStatus.OPEN,
}


@dataclass
class KiteCredentials:
    api_key: str = ""
    access_token: str = ""
    user_id: str = ""

    @property
    def complete(self) -> bool:
        return bool(self.api_key and self.access_token)


class KiteBroker(Broker):
    name = "kite"
    supports_depth = True

    def __init__(self, creds: Optional[KiteCredentials] = None,
                 mode: str = "paper", acknowledged: bool = False):
        self.creds = creds or KiteCredentials()
        self.mode = mode
        self.acknowledged = acknowledged
        self._client = None
        self._connected = False
        self.on_fill: Optional[Callable[[Fill], None]] = None
        self.on_order_update: Optional[Callable[[Order], None]] = None

    # ------------------------------------------------------------------
    def _guard(self) -> None:
        """Called before anything that could reach a real account."""
        LiveTradingGate.check(self.mode, self.acknowledged)
        if self.mode == "live" and not self.creds.complete:
            raise BrokerError("live mode without complete credentials")

    def connect(self) -> bool:
        self._guard()
        if self.mode != "live":
            # Paper mode connects to nothing; the simulated venue is used
            # instead, so the code path above it is identical either way.
            self._connected = True
            return True
        try:
            from kiteconnect import KiteConnect       # noqa: F401
        except ImportError as e:
            raise BrokerError(
                "kiteconnect is not installed; `pip install kiteconnect`") from e
        raise NotImplementedError(
            "Live Kite order placement is intentionally not implemented. "
            "See the notes at the bottom of gmq/execution/kite.py.")

    @property
    def connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self._connected = False

    # ------------------------------------------------------------------
    def place(self, order: Order) -> str:
        self._guard()
        raise NotImplementedError(
            "KiteBroker.place is a stub. Implement against KiteConnect."
            " Sketch:\n"
            "    oid = self._client.place_order(\n"
            "        variety='regular',\n"
            "        exchange='NSE',\n"
            "        tradingsymbol=order.symbol,\n"
            "        transaction_type='BUY' if order.side is Side.BUY else 'SELL',\n"
            "        quantity=order.qty,\n"
            "        product=order.product.value,\n"
            "        order_type=_OTYPE[order.otype],\n"
            "        price=order.price or None,\n"
            "        trigger_price=order.trigger or None,\n"
            "        validity='IOC' if order.otype is OrderType.IOC else 'DAY',\n"
            "        tag=order.tag[:20])\n"
            "    order.broker_order_id = str(oid)")

    def cancel(self, order_id: str) -> bool:
        self._guard()
        raise NotImplementedError("KiteBroker.cancel is a stub.")

    def modify(self, order_id: str, qty: Optional[int] = None,
               price: Optional[float] = None) -> bool:
        self._guard()
        raise NotImplementedError("KiteBroker.modify is a stub.")

    # ------------------------------------------------------------------
    # Read-only calls are safe and are implemented as normalisers, so the
    # reconciliation path can be tested against recorded payloads.
    # ------------------------------------------------------------------
    @staticmethod
    def normalise_order(raw: dict) -> Order:
        o = Order(
            symbol=raw.get("tradingsymbol", ""),
            side=Side.BUY if raw.get("transaction_type") == "BUY" else Side.SELL,
            qty=int(raw.get("quantity", 0)),
            otype=OrderType(_OTYPE.get(raw.get("order_type"), "MARKET"))
            if raw.get("order_type") in ("MARKET", "LIMIT") else OrderType.MARKET,
            price=float(raw.get("price", 0) or 0),
            trigger=float(raw.get("trigger_price", 0) or 0),
            product=Product(raw.get("product", "MIS")),
            tag=raw.get("tag", "") or "",
        )
        o.broker_order_id = str(raw.get("order_id", ""))
        o.filled_qty = int(raw.get("filled_quantity", 0) or 0)
        o.avg_price = float(raw.get("average_price", 0) or 0)
        o.status = _STATUS.get(str(raw.get("status", "")).upper(),
                               OrderStatus.PENDING)
        o.reject_reason = raw.get("status_message", "") or ""
        return o

    @staticmethod
    def normalise_positions(raw: dict) -> Dict[str, int]:
        """Kite returns {'net': [...], 'day': [...]}. Net is the truth."""
        out: Dict[str, int] = {}
        for p in (raw or {}).get("net", []):
            q = int(p.get("quantity", 0) or 0)
            if q:
                out[p.get("tradingsymbol", "")] = q
        return out

    def positions(self) -> Dict[str, int]:
        if self._client is None:
            return {}
        return self.normalise_positions(self._client.positions())

    def orders(self) -> List[Order]:
        if self._client is None:
            return []
        return [self.normalise_order(r) for r in self._client.orders()]


# ---------------------------------------------------------------------------
# Before wiring this to a real account
# ---------------------------------------------------------------------------
#
# The models in this system have been validated against a simulator, and a
# simulator validates the *code*, not the *edge*. Specifically:
#
# * Every latency, slippage, impact and fill-probability number here is
#   calibrated to the simulated venue. None of them are measurements of a real
#   one. Run in observation mode against live data and measure them before
#   trusting any expected-value calculation that depends on them.
#
# * The simulated market is a model of a market. It has no earnings surprises
#   that gap 15%, no exchange outages, no auction mechanics, no settlement, no
#   corporate actions, no margin calls. A strategy that works against it has
#   demonstrated that it can exploit *that* model, which is weaker evidence
#   than it feels like.
#
# * SEBI and exchange rules on algorithmic trading apply to automated order
#   flow from retail accounts, including approval requirements for algos
#   placing orders through a broker's API. That is a question for the broker
#   and the regulator, and it is not answered by this code compiling.
#
# * Start with capital you are fully prepared to lose, keep the hard risk
#   limits tight, and watch it. An autonomous system's first live session is
#   an experiment, not a deployment.
