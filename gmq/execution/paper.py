"""Paper broker for live trading: real quotes in, no real orders out.

`make_paper_broker` returns a SimBroker wired to a LiveQuoteBook instead of the
simulated exchange. It is the same broker the whole system was validated
against -- same latency model, same stop-trigger semantics, same fee schedule,
same accounting -- filling against live prices. The KiteBroker order path stays
untouched and locked; nothing here can reach a real account.

The returned broker exposes its `.ex` (the LiveQuoteBook); feed the live ticks
to `broker.ex.update(tick)` (and to `broker.on_tick(tick)` for stops) and the
paper account trades the real market.
"""
from __future__ import annotations

from typing import List, Optional

from ..core.clock import Clock, LiveClock
from ..core.config import ExecConfig
from ..core.bus import EventBus
from ..core.bus import TimerWheel
from .simbroker import SimBroker
from .livebook import LiveQuoteBook


def make_paper_broker(symbols: List[str], cfg: Optional[ExecConfig] = None,
                      bus: Optional[EventBus] = None,
                      clock: Optional[Clock] = None,
                      timers: Optional[TimerWheel] = None,
                      seed: int = 17) -> SimBroker:
    cfg = cfg or ExecConfig(mode="paper", broker="kite")
    book = LiveQuoteBook(symbols, impact_coef=cfg.taker_impact_coef)
    broker = SimBroker(book, clock or LiveClock(), cfg, bus=bus,
                       timers=timers, seed=seed)
    return broker
