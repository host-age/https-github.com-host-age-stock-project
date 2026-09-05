"""Paper broker filling against live quotes -- through the real SimBroker.

These prove the honest properties of paper-trading on real prices: a market
order crosses the spread and pays impact, a marketable limit never fills worse
than its limit, a passive limit rests until the tape trades to it, and a
broker-held stop triggers on the live tape. All of it runs through the exact
SimBroker code path the system was validated on; only the quotes are live.
"""
import sys
sys.path.insert(0, ".")
import pytest

from gmq.core.bus import EventBus, TimerWheel
from gmq.core.clock import LiveClock
from gmq.core.config import ExecConfig
from gmq.core.types import Tick, Fill, Side, OrderType, Product, Order
from gmq.execution.paper import make_paper_broker
from gmq.execution.livebook import LiveQuoteBook


def tick(sym="RELIANCE", ltp=2478.4, bid=2478.35, ask=2478.45,
         bidq=200, askq=180, ts=1):
    return Tick(ts=ts, symbol=sym, ltp=ltp, bid=bid, ask=ask,
                bid_qty=bidq, ask_qty=askq, volume=1000)


def _broker():
    clock = LiveClock()
    timers = TimerWheel()
    fills = []
    b = make_paper_broker(["RELIANCE"], cfg=ExecConfig(mode="paper"),
                          clock=clock, timers=timers)
    b.on_fill = fills.append
    b.connect()
    b.ex.update(tick())        # seed a live quote
    return b, fills, clock, timers


def _drive(b, timers, clock, ns=5_000_000):
    """Advance the clock and run due timers, as the live loop would."""
    clock  # LiveClock advances on its own; just run timers now
    timers.run_until(clock.now_ns() + ns)


# ------------------------------------------------------------ market orders
def test_market_buy_crosses_spread_and_pays_impact():
    b, fills, clock, timers = _broker()
    o = Order(symbol="RELIANCE", side=Side.BUY,
              qty=200, otype=OrderType.MARKET, product=Product.MIS)
    b.place(o)
    b.pump(); timers.run_until(clock.now_ns() + 10_000_000)
    assert len(fills) == 1
    # fills at or above the ask (crossed the spread), never below
    assert fills[0].price >= 2478.45
    assert fills[0].side is Side.BUY and fills[0].qty == 200


def test_market_large_order_pays_more_impact():
    b, fills, clock, timers = _broker()
    small = b.ex.submit_market("RELIANCE", Side.BUY, 50, "a")[0][0]
    big = b.ex.submit_market("RELIANCE", Side.BUY, 5000, "b")[0][0]
    assert big.price > small.price     # sqrt impact grows with size


# ------------------------------------------------------------ limit orders
def test_marketable_limit_never_fills_worse_than_limit():
    b, _, _, _ = _broker()
    # a buy limit above the ask is marketable; fills at the ask, not the limit
    fills, eid, err = b.ex.submit_limit("RELIANCE", Side.BUY, 2479.0, 100, "c")
    assert err == "" and eid is None and len(fills) == 1
    assert fills[0].price <= 2479.0 and fills[0].price == pytest.approx(2478.45)
    assert fills[0].liquidity == "TAKER"


def test_passive_limit_rests_then_fills_when_tape_trades_to_it():
    b, fills, clock, timers = _broker()
    made = []
    b.ex.on_agent_fill = made.append
    # a buy limit below the market rests
    _f, eid, err = b.ex.submit_limit("RELIANCE", Side.BUY, 2477.0, 100, "d")
    assert err == "" and eid is not None and _f == []
    # market ticks down through the resting price -> maker fill
    b.ex.update(tick(ltp=2476.9, bid=2476.85, ask=2476.95, ts=2))
    assert len(made) == 1
    assert made[0].price == pytest.approx(2477.0)
    assert made[0].liquidity == "MAKER"


def test_passive_limit_does_not_fill_while_market_stays_away():
    b, _, _, _ = _broker()
    made = []
    b.ex.on_agent_fill = made.append
    b.ex.submit_limit("RELIANCE", Side.BUY, 2470.0, 100, "e")
    b.ex.update(tick(ltp=2479.0, bid=2478.9, ask=2479.1, ts=2))  # stays above
    assert made == []


# ------------------------------------------------------------ stops on tape
def test_broker_stop_triggers_on_live_tape():
    b, fills, clock, timers = _broker()
    # long position implied; place a protective sell stop below market
    o = Order(symbol="RELIANCE", side=Side.SELL,
              qty=200, otype=OrderType.SL_M, trigger=2470.0,
              product=Product.MIS)
    b.place(o)
    assert fills == []                         # not triggered yet
    # a live tick prints through the trigger
    t = tick(ltp=2469.5, bid=2469.45, ask=2469.55, ts=3)
    b.ex.update(t)
    b.on_tick(t)                               # broker checks its stops
    timers.run_until(clock.now_ns() + 10_000_000)
    assert len(fills) == 1 and fills[0].side is Side.SELL
    # stop fills at or through the trigger, not above it (honest slippage)
    assert fills[0].price <= 2470.0


def test_no_quote_rejects_cleanly():
    b, _, _, _ = _broker()
    fills, rem, err = b.ex.submit_market("UNKNOWN", Side.BUY, 100, "z")
    assert err == "NO_QUOTE" and fills == []
