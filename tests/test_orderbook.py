"""Matching-engine correctness. These are the invariants everything else
assumes; if any of them breaks, every downstream P&L number is fiction."""
import sys
sys.path.insert(0, ".")
import pytest

from gmq.core.types import Side, Instrument, InstrumentType, OrderType
from gmq.data.orderbook import OrderBook


def mk(prev=100.0, circuit=0.10, tick=0.05):
    inst = Instrument(symbol="TEST", token=1, itype=InstrumentType.EQ,
                      tick_size=tick, circuit_pct=circuit)
    return OrderBook(inst, prev)


def test_price_time_priority():
    """Two orders at the same price fill in arrival order."""
    b = mk()
    b.add_limit(Side.BUY, 100.0, 50, ts=1, owner="A", client_order_id="A")
    b.add_limit(Side.BUY, 100.0, 50, ts=2, owner="B", client_order_id="B")
    fills, rem, _ = b.add_market(Side.SELL, 60, ts=3)
    assert rem == 0
    assert sum(f.qty for f in fills) == 60
    # first order fully consumed, second partially -> 10 left at the level
    assert b.level_qty(Side.BUY, b.to_ticks(100.0)) == 40


def test_better_price_first():
    b = mk()
    b.add_limit(Side.SELL, 101.0, 100, ts=1)
    b.add_limit(Side.SELL, 100.5, 100, ts=2)
    fills, rem, _ = b.add_market(Side.BUY, 150, ts=3)
    assert rem == 0
    # 100 at the better price, then 50 at the worse one
    assert fills[0].price == pytest.approx(100.5)
    assert sum(f.qty for f in fills if f.price == pytest.approx(100.5)) == 100
    assert sum(f.qty for f in fills if f.price == pytest.approx(101.0)) == 50


def test_limit_does_not_cross_beyond_its_price():
    b = mk()
    b.add_limit(Side.SELL, 101.0, 100, ts=1)
    fills, eid, err = b.add_limit(Side.BUY, 100.0, 100, ts=2)
    assert not fills and eid is not None and not err
    assert b.best_bid == pytest.approx(100.0)
    assert b.best_ask == pytest.approx(101.0)


def test_partial_fill_rests_remainder():
    b = mk()
    b.add_limit(Side.SELL, 100.0, 30, ts=1)
    fills, eid, err = b.add_limit(Side.BUY, 100.0, 100, ts=2)
    assert sum(f.qty for f in fills) == 30
    assert eid is not None
    assert b.level_qty(Side.BUY, b.to_ticks(100.0)) == 70


def test_ioc_cancels_remainder():
    b = mk()
    b.add_limit(Side.SELL, 100.0, 30, ts=1)
    fills, eid, err = b.add_limit(Side.BUY, 100.0, 100, ts=2, ioc=True)
    assert sum(f.qty for f in fills) == 30
    assert eid is None
    assert b.level_qty(Side.BUY, b.to_ticks(100.0)) == 0


def test_circuit_limits_reject():
    b = mk(prev=100.0, circuit=0.10)
    _f, _e, err = b.add_limit(Side.BUY, 111.0, 10, ts=1)
    assert err == "ABOVE_UPPER_CIRCUIT"
    _f, _e, err = b.add_limit(Side.SELL, 89.0, 10, ts=1)
    assert err == "BELOW_LOWER_CIRCUIT"


def test_market_order_stops_at_circuit():
    """A market order must never print outside the band, however thin the book."""
    b = mk(prev=100.0, circuit=0.10)
    b.add_limit(Side.SELL, 100.0, 10, ts=1)
    b.add_limit(Side.SELL, 109.0, 10, ts=1)
    fills, rem, _ = b.add_market(Side.BUY, 500, ts=2)
    assert all(f.price <= 110.0 for f in fills)
    assert rem == 480          # unfilled remainder is cancelled, not printed


def test_cancel_removes_level_and_key():
    b = mk()
    _f, eid, _ = b.add_limit(Side.BUY, 99.0, 10, ts=1)
    assert b.best_bid == pytest.approx(99.0)
    assert b.cancel(eid)
    assert b.best_bid == 0.0
    assert b._bid_keys == []


def test_modify_down_keeps_priority():
    b = mk()
    _f, e1, _ = b.add_limit(Side.BUY, 100.0, 100, ts=1, client_order_id="first")
    b.add_limit(Side.BUY, 100.0, 100, ts=2, client_order_id="second")
    assert b.queue_position(e1) == 0
    b.modify(e1, new_qty=50)
    assert b.queue_position(e1) == 0        # still at the front
    assert b.level_qty(Side.BUY, b.to_ticks(100.0)) == 150


def test_modify_up_loses_priority():
    b = mk()
    _f, e1, _ = b.add_limit(Side.BUY, 100.0, 100, ts=1)
    b.add_limit(Side.BUY, 100.0, 100, ts=2)
    b.modify(e1, new_qty=200)
    assert b.queue_position(e1) == 100      # went to the back


def test_volume_and_vwap_accounting():
    b = mk()
    b.add_limit(Side.SELL, 100.0, 100, ts=1)
    b.add_limit(Side.SELL, 102.0, 100, ts=1)
    b.add_market(Side.BUY, 150, ts=2)
    assert b.volume == 150
    assert b.stats()["vwap"] == pytest.approx((100 * 100 + 50 * 102) / 150)
    assert b.buy_volume == 150 and b.sell_volume == 0


def test_sweep_cost_monotonic_in_size():
    b = mk()
    for i, px in enumerate([100.0, 100.5, 101.0, 101.5]):
        b.add_limit(Side.SELL, px, 100, ts=1)
    b.add_limit(Side.BUY, 99.5, 400, ts=1)
    c_small = b.sweep_cost_bps(Side.BUY, 50)
    c_big = b.sweep_cost_bps(Side.BUY, 350)
    assert c_big > c_small >= 0


def test_maker_callback_fires_for_agent_orders():
    got = []
    inst = Instrument(symbol="TEST", token=1, tick_size=0.05)
    b = OrderBook(inst, 100.0, on_trade=got.append)
    b.add_limit(Side.BUY, 100.0, 100, ts=1, owner="AGENT",
                client_order_id="MINE")
    b.add_market(Side.SELL, 40, ts=2, owner="OTHER")
    assert len(got) == 1
    assert got[0].order_id == "MINE"
    assert got[0].qty == 40
    assert got[0].liquidity == "MAKER"
    assert got[0].side is Side.BUY


def test_no_crossed_book_after_random_flow():
    import random
    rng = random.Random(4)
    b = mk(prev=100.0, circuit=0.5)
    for i in range(4000):
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        if rng.random() < 0.35:
            b.add_market(side, rng.randint(1, 80), ts=i)
        else:
            px = round(100 + rng.gauss(0, 1.0), 1)
            b.add_limit(side, px, rng.randint(1, 120), ts=i)
        if b.best_bid and b.best_ask:
            assert b.best_bid < b.best_ask, f"crossed book at step {i}"
