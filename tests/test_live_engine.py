"""End-to-end smoke test of the live paper-trading path.

Builds the REAL TradingEngine in live mode -- LiveClock, paper broker over a
live-quote book, external feed -- attaches a KiteFeed driven by a fake ticker,
and runs the actual run_live loop for a bounded few seconds while a background
thread streams Kite-format ticks. Proves the whole live wiring turns over
without error and consumes real-shaped ticks. It does not assert P&L (that is a
market-hours measurement, not a unit test); it asserts the machine runs.
"""
import sys
import threading
import time
from datetime import datetime, timedelta

sys.path.insert(0, ".")
import numpy as np
import pytest

from gmq.core.config import Config
from gmq.app.engine import TradingEngine
from gmq.data.kite_feed import KiteFeed


TOKENS = {738561: "RELIANCE", 2953217: "TCS", 341249: "HDFCBANK"}


class FakeTicker:
    MODE_FULL = "full"
    def __init__(self): self.on_ticks=None; self.on_connect=None; self.closed=False
    def connect(self, threaded=True):
        if self.on_connect: self.on_connect(self)
    def subscribe(self, tokens): pass
    def set_mode(self, m, t): pass
    def close(self): self.closed = True


def _kt(sym, token, px, ts):
    half = round(px * 3e-5, 2) or 0.05
    def book(sign):
        return [{"price": round(px + sign*half*k, 2), "quantity": 200,
                 "orders": 3} for k in range(1, 6)]
    return {"mode": "full", "instrument_token": token, "last_price": round(px, 2),
            "last_traded_quantity": 50, "volume_traded": 100000,
            "total_buy_quantity": 5000, "total_sell_quantity": 5200,
            "ohlc": {"open": px, "high": px*1.001, "low": px*0.999, "close": px},
            "oi": 0, "exchange_timestamp": ts,
            "depth": {"buy": book(-1), "sell": book(+1)}}


def test_live_engine_runs_and_consumes_kite_ticks():
    cfg = Config()
    cfg.data.symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    eng = TradingEngine(cfg, live=True, journal=False)

    tk = FakeTicker()
    feed = KiteFeed(eng.bus, cfg.data.symbols,
                    {k: v for k, v in TOKENS.items()},
                    clock=eng.clock, ticker=tk)
    eng.attach_live_feed(feed)

    # background thread streams live-shaped ticks while run_live turns
    stop = threading.Event()
    px = {"RELIANCE": 2478.0, "TCS": 3890.0, "HDFCBANK": 1650.0}
    rng = np.random.default_rng(0)

    def stream():
        base = datetime(2026, 9, 2, 10, 30, 0)
        i = 0
        while not stop.is_set():
            i += 1
            batch = []
            for sym, tok in (("RELIANCE", 738561), ("TCS", 2953217),
                             ("HDFCBANK", 341249)):
                px[sym] *= float(np.exp(rng.normal(0, 6e-4)))
                batch.append(_kt(sym, tok, px[sym],
                                 base + timedelta(milliseconds=i * 20)))
            if tk.on_ticks:
                tk.on_ticks(tk, batch)
            time.sleep(0.005)

    th = threading.Thread(target=stream, daemon=True)
    th.start()
    try:
        rep = eng.run_live(poll_s=0.05, max_seconds=2.0)
    finally:
        stop.set(); th.join(timeout=1)

    # the machine turned over and saw real-shaped quotes
    assert isinstance(rep, dict) and "equity" in rep
    assert eng.mde.last_tick["RELIANCE"].ltp > 0
    assert eng.mde.tick_count > 50
    assert feed.stats()["ticks_in"] > 0 and feed.stats()["unmapped"] == 0
    assert tk.closed                       # feed.stop() ran in the finally


def test_live_mode_uses_paper_broker_and_no_sim_feed():
    cfg = Config(); cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, live=True, journal=False)
    from gmq.execution.livebook import LiveQuoteBook
    assert eng.feed is None                       # no sim feed in live mode
    assert isinstance(eng.broker.ex, LiveQuoteBook)
    assert type(eng.clock).__name__ == "LiveClock"


def test_run_live_requires_live_flag():
    cfg = Config(); cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, live=False, journal=False)
    with pytest.raises(RuntimeError):
        eng.run_live(max_seconds=0.1)
