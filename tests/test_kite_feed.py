"""Tests for the Kite live-data adapter, driven by recorded Kite payloads.

No network, no credentials, no market hours. The whole point of separating the
pure converter from the socket is that the conversion -- the only part that can
be subtly wrong -- is provable off the wire. The sample dicts below match the
structure the KiteTicker 'full' mode actually delivers.
"""
import sys
from datetime import datetime

sys.path.insert(0, ".")
import pytest

from gmq.core.bus import EventBus, Topic
from gmq.core.types import Tick, DepthSnapshot
from gmq.data.kite_feed import KiteFeed, to_tick, to_depth, _ts_ns


# A real-shaped Kite 'full'-mode tick for RELIANCE (token 738561).
def full_tick(token=738561, ltp=2478.40):
    return {
        "tradable": True, "mode": "full", "instrument_token": token,
        "last_price": ltp, "last_traded_quantity": 5,
        "average_traded_price": 2470.5, "volume_traded": 1234567,
        "total_buy_quantity": 45000, "total_sell_quantity": 52000,
        "ohlc": {"open": 2465.0, "high": 2489.0, "low": 2460.0, "close": 2468.0},
        "oi": 0,
        "exchange_timestamp": datetime(2026, 9, 2, 10, 30, 0),
        "depth": {
            "buy": [
                {"price": 2478.35, "quantity": 120, "orders": 3},
                {"price": 2478.30, "quantity": 300, "orders": 5},
                {"price": 2478.25, "quantity": 250, "orders": 4},
                {"price": 2478.20, "quantity": 400, "orders": 6},
                {"price": 2478.15, "quantity": 150, "orders": 2},
            ],
            "sell": [
                {"price": 2478.45, "quantity": 100, "orders": 2},
                {"price": 2478.50, "quantity": 280, "orders": 4},
                {"price": 2478.55, "quantity": 320, "orders": 5},
                {"price": 2478.60, "quantity": 200, "orders": 3},
                {"price": 2478.65, "quantity": 180, "orders": 3},
            ],
        },
    }


def light_tick(token=738561, ltp=2479.0):
    """LTP mode: no depth, no ohlc -- what a light subscription delivers."""
    return {"mode": "ltp", "instrument_token": token, "last_price": ltp}


# ---------------------------------------------------------------- converter
def test_to_tick_maps_price_and_book():
    t = to_tick(full_tick(), "RELIANCE", 123, prev_mid=2478.0)
    assert isinstance(t, Tick)
    assert t.symbol == "RELIANCE"
    assert t.ltp == pytest.approx(2478.40)
    assert t.bid == pytest.approx(2478.35)
    assert t.ask == pytest.approx(2478.45)
    assert t.bid_qty == 120 and t.ask_qty == 100
    assert t.volume == 1234567
    # spread is one tick (0.10) -> a small positive bps
    assert 0 < t.spread_bps < 5


def test_to_tick_aggressor_by_tick_rule():
    up = to_tick(full_tick(ltp=2479.0), "RELIANCE", 1, prev_mid=2478.0)
    dn = to_tick(full_tick(ltp=2477.0), "RELIANCE", 1, prev_mid=2478.0)
    unknown = to_tick(full_tick(), "RELIANCE", 1, prev_mid=0.0)
    assert up.aggressor == 1
    assert dn.aggressor == -1
    assert unknown.aggressor == 0        # no prior mid -> honestly unknown


def test_to_tick_light_mode_has_no_book_but_still_prices():
    t = to_tick(light_tick(ltp=2479.0), "RELIANCE", 1, prev_mid=0.0)
    assert t.ltp == pytest.approx(2479.0)
    # synthesises a hair of spread around ltp rather than reporting 0/0
    assert t.bid > 0 and t.ask > t.bid


def test_to_depth_five_levels_each_side():
    d = to_depth(full_tick(), "RELIANCE", 999)
    assert isinstance(d, DepthSnapshot)
    assert len(d.bids) == 5 and len(d.asks) == 5
    assert d.best_bid == pytest.approx(2478.35)
    assert d.best_ask == pytest.approx(2478.45)
    # book is monotone: bids descending, asks ascending
    assert [l.price for l in d.bids] == sorted((l.price for l in d.bids),
                                               reverse=True)
    assert [l.price for l in d.asks] == sorted(l.price for l in d.asks)


def test_to_depth_absent_in_light_mode():
    assert to_depth(light_tick(), "RELIANCE", 1) is None


def test_exchange_timestamp_preferred_over_local_clock():
    ns = _ts_ns(full_tick(), fallback_ns=42)
    assert ns == int(datetime(2026, 9, 2, 10, 30, 0).timestamp() * 1_000_000_000)
    # missing stamp -> fall back
    assert _ts_ns(light_tick(), fallback_ns=42) == 42


# ---------------------------------------------------------------- feed wiring
class FakeTicker:
    """Stand-in for KiteTicker: records subscriptions, lets a test push ticks."""
    MODE_FULL = "full"

    def __init__(self):
        self.on_ticks = None
        self.on_connect = None
        self.subscribed = []
        self.mode = None
        self.closed = False

    def connect(self, threaded=True):
        if self.on_connect:
            self.on_connect(self)

    def subscribe(self, tokens):
        self.subscribed = list(tokens)

    def set_mode(self, mode, tokens):
        self.mode = mode

    def close(self):
        self.closed = True

    def push(self, ticks):
        self.on_ticks(self, ticks)


def _feed(**kw):
    bus = EventBus(swallow_errors=False)
    got = {"ticks": [], "depth": []}
    bus.subscribe(Topic.TICK, lambda t: got["ticks"].append(t), 10, "t")
    bus.subscribe(Topic.DEPTH, lambda d: got["depth"].append(d), 10, "d")
    feed = KiteFeed(bus, symbols=["RELIANCE", "TCS"],
                    token_to_symbol={738561: "RELIANCE", 2953217: "TCS"}, **kw)
    return feed, got


def test_feed_emits_tick_and_depth_on_the_bus():
    feed, got = _feed()
    feed.on_ticks([full_tick()])
    assert len(got["ticks"]) == 1 and got["ticks"][0].symbol == "RELIANCE"
    assert len(got["depth"]) == 1 and got["depth"][0].best_bid > 0


def test_feed_ignores_unmapped_tokens():
    feed, got = _feed()
    feed.on_ticks([full_tick(token=999999)])   # not in the map
    assert got["ticks"] == []
    assert feed.stats()["unmapped"] == 1


def test_feed_tracks_prev_mid_for_aggressor_across_ticks():
    feed, got = _feed()
    feed.on_ticks([full_tick(ltp=2478.40)])    # first: no prior mid
    feed.on_ticks([full_tick(ltp=2490.00)])    # jumps above prior mid -> buy
    assert got["ticks"][0].aggressor == 0
    assert got["ticks"][1].aggressor == 1


def test_start_subscribes_the_right_tokens_in_full_mode():
    tk = FakeTicker()
    feed, got = _feed(ticker=tk)
    feed.start()
    assert set(tk.subscribed) == {738561, 2953217}
    assert tk.mode == FakeTicker.MODE_FULL
    # and a pushed tick flows through to the bus
    tk.push([full_tick()])
    assert len(got["ticks"]) == 1


def test_start_without_ticker_fails_loudly():
    feed, _ = _feed()
    with pytest.raises(RuntimeError):
        feed.start()
