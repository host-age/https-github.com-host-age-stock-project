"""Broker interface and the live-trading safety gate.

Every venue -- the simulated exchange, a paper account, a real broker -- sits
behind this interface, so the rest of the system cannot tell which one it is
talking to.

`LiveTradingGate` is the reason this file exists as more than an abstract base
class. Placing real orders with real money requires an explicit, deliberate,
multi-part opt-in that cannot be satisfied by a config typo, a default value,
or a stray environment variable. The system ships locked, and the lock is
checked at the point of order submission -- not at construction, where a later
mutation could slip past it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ..core.types import Order, Fill, OrderStatus, Side, Position


class BrokerError(Exception):
    pass


class LiveTradingBlocked(BrokerError):
    pass


class LiveTradingGate:
    """Three independent conditions, all required, none of them a default.

    Deliberately awkward. A single `live=True` flag is one careless edit away
    from an account trading by accident, and the failure mode is not a stack
    trace -- it is real orders in a real market.
    """

    ENV_FLAG = "GMQ_ALLOW_LIVE_TRADING"
    ENV_TOKEN = "GMQ_LIVE_CONFIRM"
    REQUIRED_TOKEN = "I-UNDERSTAND-THIS-TRADES-REAL-MONEY"

    @classmethod
    def check(cls, mode: str, acknowledged: bool = False) -> None:
        if mode != "live":
            return
        if os.environ.get(cls.ENV_FLAG, "").lower() not in ("1", "true", "yes"):
            raise LiveTradingBlocked(
                f"live mode requires {cls.ENV_FLAG}=1 in the environment")
        if os.environ.get(cls.ENV_TOKEN, "") != cls.REQUIRED_TOKEN:
            raise LiveTradingBlocked(
                f"live mode requires {cls.ENV_TOKEN}='{cls.REQUIRED_TOKEN}'")
        if not acknowledged:
            raise LiveTradingBlocked(
                "live mode requires acknowledged=True at the call site")

    @classmethod
    def is_live_possible(cls) -> bool:
        return (os.environ.get(cls.ENV_FLAG, "").lower() in ("1", "true", "yes")
                and os.environ.get(cls.ENV_TOKEN, "") == cls.REQUIRED_TOKEN)


class Broker:
    """What every venue must provide."""

    name = "base"
    supports_depth = False

    # -- lifecycle
    def connect(self) -> bool: raise NotImplementedError
    def disconnect(self) -> None: ...
    @property
    def connected(self) -> bool: return False

    # -- orders
    def place(self, order: Order) -> str: raise NotImplementedError
    def cancel(self, order_id: str) -> bool: raise NotImplementedError
    def modify(self, order_id: str, qty: Optional[int] = None,
               price: Optional[float] = None) -> bool: raise NotImplementedError

    # -- state, for reconciliation
    def orders(self) -> List[Order]: return []
    def positions(self) -> Dict[str, int]: return {}
    def order_status(self, order_id: str) -> Optional[OrderStatus]: return None

    # -- callbacks the router installs
    on_fill: Optional[Callable[[Fill], None]] = None
    on_order_update: Optional[Callable[[Order], None]] = None
