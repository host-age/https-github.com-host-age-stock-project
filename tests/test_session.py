"""Session and calendar handling.

The bug these exist to catch is silent: nothing errors, the engine simply
stops trading after the first day because from its point of view the closing
bell is always a few minutes away.
"""
import sys
sys.path.insert(0, ".")
from datetime import datetime, timedelta

import pytest

from gmq.core.clock import SimClock, IST, SessionPhase, OPEN_TIME, load_holidays
from gmq.core.config import Config
from gmq.core.types import NS


def test_clock_phases():
    c = SimClock(datetime(2026, 8, 17, 9, 20, tzinfo=IST))    # a Monday
    assert c.phase() == SessionPhase.OPEN
    c2 = SimClock(datetime(2026, 8, 17, 15, 20, tzinfo=IST))
    assert c2.phase() == SessionPhase.SQUARE_OFF
    c3 = SimClock(datetime(2026, 8, 17, 20, 0, tzinfo=IST))
    assert c3.phase() == SessionPhase.CLOSED
    c4 = SimClock(datetime(2026, 8, 22, 11, 0, tzinfo=IST))   # Saturday
    assert c4.phase() == SessionPhase.CLOSED


def test_day_roll_advances_to_the_next_session_open():
    """A multi-day run must actually reach day two's open."""
    from gmq.app.engine import TradingEngine
    cfg = Config(run_id="t_roll", seed=3)
    cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, run_dir="runs/_t_roll",
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                        journal=False)
    # pretend a session just ended
    eng.clock.set_ns(int(datetime(2026, 8, 17, 15, 30, tzinfo=IST)
                         .timestamp() * NS))
    eng._advance_to_next_session()
    d = eng.clock.now_dt()
    assert d.date() == datetime(2026, 8, 18).date()
    assert (d.hour, d.minute) == (OPEN_TIME.hour, OPEN_TIME.minute)
    assert eng.clock.phase() == SessionPhase.OPEN
    assert eng.clock.seconds_to_close() > 6 * 3600


def test_day_roll_skips_the_weekend():
    from gmq.app.engine import TradingEngine
    cfg = Config(run_id="t_roll2", seed=3)
    cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, run_dir="runs/_t_roll2",
                        start=datetime(2026, 8, 21, 9, 15, tzinfo=IST),
                        journal=False)
    # Friday 21 Aug 2026 -> next session must be Monday 24th
    eng.clock.set_ns(int(datetime(2026, 8, 21, 15, 30, tzinfo=IST)
                         .timestamp() * NS))
    eng._advance_to_next_session()
    assert eng.clock.now_dt().date() == datetime(2026, 8, 24).date()
    assert eng.clock.now_dt().weekday() == 0


def test_day_roll_skips_an_exchange_holiday():
    from gmq.app.engine import TradingEngine
    load_holidays(["2026-08-18"])
    try:
        cfg = Config(run_id="t_roll3", seed=3)
        cfg.data.symbols = cfg.data.symbols[:2]
        eng = TradingEngine(cfg, run_dir="runs/_t_roll3",
                            start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                            journal=False)
        eng.clock.set_ns(int(datetime(2026, 8, 17, 15, 30, tzinfo=IST)
                             .timestamp() * NS))
        eng._advance_to_next_session()
        assert eng.clock.now_dt().date() == datetime(2026, 8, 19).date()
    finally:
        load_holidays([])


def test_engine_still_opens_positions_after_a_day_roll():
    """The regression itself: entries must remain legal on day two."""
    from gmq.app.engine import TradingEngine
    from gmq.strategy.moves import MoveGenerator, MoveContext
    from gmq.core.types import Prediction, Regime, MoveType
    cfg = Config(run_id="t_roll4", seed=3)
    cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, run_dir="runs/_t_roll4",
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                        journal=False)
    eng.clock.set_ns(int(datetime(2026, 8, 17, 15, 30, tzinfo=IST)
                         .timestamp() * NS))
    assert eng.clock.seconds_to_close() < 420    # entries correctly refused
    eng._advance_to_next_session()
    gen = MoveGenerator()
    ctx = MoveContext(
        symbol="X", px=1000.0, bid=999.9, ask=1000.1, atr=3.0, qty=0,
        avg_price=0.0, stop=0.0, target=0.0, max_qty=600, unit_qty=200,
        regime=Regime.TRENDING_UP, alignment=0.5, alignment_conflict=False,
        pred=Prediction(p_up=0.72, confidence=0.8, exp_vol=0.003),
        liquidity=0.9, seconds_to_close=eng.clock.seconds_to_close())
    entries = [m for m in gen.generate(ctx)
               if m.move in (MoveType.ENTER_LONG, MoveType.ENTER_SHORT)]
    assert entries, "no entries are legal after the day roll"


def test_daily_halts_clear_at_the_next_session_but_others_do_not():
    """A daily-loss halt is a decision about a day; a drawdown halt is not."""
    from gmq.app.engine import TradingEngine
    from gmq.risk.engine import HaltReason
    cfg = Config(run_id="t_roll5", seed=3)
    cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, run_dir="runs/_t_roll5",
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                        journal=False)
    eng.risk.halt(HaltReason.DAILY_LOSS, 1)
    eng.portfolio.consecutive_losses = 9
    eng._advance_to_next_session()
    assert not eng.risk.halted, "a daily-loss halt survived the day roll"
    assert eng.portfolio.consecutive_losses == 0

    eng.risk.halt(HaltReason.DRAWDOWN, 2)
    eng._advance_to_next_session()
    assert eng.risk.halted, "a drawdown halt silently expired overnight"
    assert eng.risk.halt_reason == HaltReason.DRAWDOWN


def test_a_stop_is_actually_placed_at_the_broker():
    """A stop that exists only in a variable is not a stop.

    The engine sizes every position against its stop distance and reports that
    distance to the risk engine as 1R. If no order is ever placed, the number
    the entire risk framework is denominated in is enforced by nothing.
    """
    from gmq.app.engine import TradingEngine
    from gmq.core.types import Side, OrderType
    cfg = Config(run_id="t_stop", seed=3)
    cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, run_dir="runs/_t_stop",
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                        journal=False)
    for _ in range(40):
        eng.feed.step(0.5)
    sym = cfg.data.symbols[0]
    px = eng.mde.ltp(sym)
    pos = eng.portfolio.position(sym)
    pos.qty, pos.avg_price, pos.stop = 50, px, px * 0.99

    eng._sync_stop_order(sym, pos)
    oid = eng._stop_orders.get(sym)
    assert oid, "no stop order was placed"
    o = eng.router.live_orders[oid]
    assert o.otype is OrderType.SL_M
    assert o.side is Side.SELL and o.qty == 50
    assert o.trigger == pytest.approx(pos.stop)


def test_moving_the_stop_moves_the_resting_order():
    from gmq.app.engine import TradingEngine
    cfg = Config(run_id="t_stop2", seed=3)
    cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, run_dir="runs/_t_stop2",
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                        journal=False)
    for _ in range(40):
        eng.feed.step(0.5)
    sym = cfg.data.symbols[0]
    px = eng.mde.ltp(sym)
    pos = eng.portfolio.position(sym)
    pos.qty, pos.avg_price, pos.stop = 50, px, px * 0.99
    eng._sync_stop_order(sym, pos)
    first = eng._stop_orders[sym]

    pos.stop = px * 0.995                      # ratchet tighter
    eng._sync_stop_order(sym, pos)
    second = eng._stop_orders[sym]
    assert second != first, "the resting order did not follow the stop"
    assert eng.router.live_orders[second].trigger == pytest.approx(pos.stop)


def test_closing_a_position_cancels_its_stop():
    from gmq.app.engine import TradingEngine
    cfg = Config(run_id="t_stop3", seed=3)
    cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, run_dir="runs/_t_stop3",
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                        journal=False)
    for _ in range(40):
        eng.feed.step(0.5)
    sym = cfg.data.symbols[0]
    px = eng.mde.ltp(sym)
    pos = eng.portfolio.position(sym)
    pos.qty, pos.avg_price, pos.stop = 50, px, px * 0.99
    eng._sync_stop_order(sym, pos)
    assert sym in eng._stop_orders
    eng._cancel_stop_order(sym)
    assert sym not in eng._stop_orders, "an orphaned stop order was left resting"


def test_initial_risk_is_fixed_at_entry_not_recomputed():
    """1R must mean the risk taken at entry, whatever the position becomes.

    Recomputing it as the position is partially exited makes realised P&L
    (whole trade) and risk (residual shares) describe different trades, which
    is how a run reports positive rupee expectancy and negative R at once.
    """
    from gmq.core.types import Fill, Side
    from gmq.risk.portfolio import Portfolio
    pf = Portfolio(1_000_000)
    p = pf.position("A")
    pf.apply_fill(Fill("o1", "A", Side.BUY, 100, 500.0, ts=1, fee=0.0))
    p = pf.position("A")
    p.stop = 495.0
    p.initial_risk = abs(500.0 - 495.0) * 100          # 500 at entry
    risk_at_entry = p.initial_risk
    # partial exits whittle the position down
    pf.apply_fill(Fill("o2", "A", Side.SELL, 60, 506.0, ts=2, fee=0.0))
    assert pf.position("A").initial_risk == pytest.approx(risk_at_entry)
    pf.apply_fill(Fill("o3", "A", Side.SELL, 39, 507.0, ts=3, fee=0.0))
    assert pf.position("A").initial_risk == pytest.approx(risk_at_entry)
    rec = pf.apply_fill(Fill("o4", "A", Side.SELL, 1, 508.0, ts=4, fee=0.0))
    assert rec is not None
    # ~640 of P&L against 500 of entry risk -> a sane R, not a 76R artefact
    assert 0.5 < rec.r_multiple < 3.0, rec.r_multiple


def test_unmeasurable_risk_reports_none_not_a_sentinel():
    """Dividing by 1e-9 to avoid a crash produces an expectancy of 1.7e7 R."""
    from gmq.core.types import Fill, Side
    from gmq.risk.portfolio import Portfolio
    from gmq.analytics.metrics import trade_stats
    pf = Portfolio(1_000_000)
    pf.apply_fill(Fill("o1", "A", Side.BUY, 10, 100.0, ts=1, fee=0.0))
    # never established entry risk
    rec = pf.apply_fill(Fill("o2", "A", Side.SELL, 10, 101.0, ts=2, fee=0.0))
    assert rec.r_multiple is None

    p = pf.position("B")
    pf.apply_fill(Fill("o3", "B", Side.BUY, 10, 100.0, ts=3, fee=0.0))
    pf.position("B").initial_risk = 50.0
    ok = pf.apply_fill(Fill("o4", "B", Side.SELL, 10, 105.0, ts=4, fee=0.0))
    assert ok.r_multiple == pytest.approx(1.0)

    st = trade_stats(pf.trades)
    assert st["r_unmeasurable"] == 1 and st["r_measurable"] == 1
    assert st["expectancy_r"] == pytest.approx(1.0), \
        "an unmeasurable trade contaminated the R aggregate"


def test_a_reversal_resets_the_new_leg_risk():
    """Flipping through zero starts a new trade; it must not inherit the old
    leg's risk units."""
    from gmq.core.types import Fill, Side, Position
    p = Position(symbol="A")
    p.apply_fill(Fill("o1", "A", Side.BUY, 10, 100.0, ts=1))
    p.initial_risk, p.initial_stop = 50.0, 95.0
    p.apply_fill(Fill("o2", "A", Side.SELL, 25, 101.0, ts=2))   # flip to short
    assert p.qty == -15
    assert p.initial_risk == 0.0, "new leg inherited the old leg's 1R"
    assert p.initial_stop == 0.0
