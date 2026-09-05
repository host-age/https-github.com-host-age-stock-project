"""Risk engine tests.

These are the most important tests in the repository. Every other component
can be wrong and cost money; this one being wrong is what ends an account. In
particular `test_risk_engine_has_no_override_api` is a test about the *shape*
of the code, not its behaviour -- it fails if someone ever adds a way for the
model layer to talk its way past a limit.
"""
import sys
sys.path.insert(0, ".")
import pytest

from gmq.core.config import RiskLimits
from gmq.core.types import MoveType, Side, Fill, Position, Regime
from gmq.risk.portfolio import Portfolio
from gmq.risk.engine import RiskEngine, TradeIntent, HaltReason
from gmq.risk.sizing import PositionSizer
from gmq.risk.stops import StopPolicy
from gmq.core.types import Prediction


def setup(equity=1_000_000.0, **lim):
    limits = RiskLimits(**lim)
    pf = Portfolio(equity)
    return limits, pf, RiskEngine(limits, pf)


# ---------------------------------------------------------------- limits


def test_entry_without_a_stop_is_refused_outright():
    """Unbounded loss has no acceptable size."""
    _l, _pf, re = setup()
    v = re.check(TradeIntent("RELIANCE", MoveType.ENTER_LONG, 100, 1000.0,
                             stop=0.0))
    assert not v.approved and "NO_STOP" in v.breached


def test_per_trade_risk_scales_size_down():
    # other caps opened up so the per-trade risk limit is the one that binds
    _l, _pf, re = setup(max_risk_per_trade_pct=0.5, max_position_pct=99.0,
                        max_net_exposure_pct=999.0, max_sector_exposure_pct=99.0,
                        max_correlated_exposure_pct=99.0,
                        max_gross_exposure_pct=999.0, max_leverage=99.0)
    # 1,000,000 * 0.5% = 5,000 risk budget; 10 rupees of stop distance -> 500
    v = re.check(TradeIntent("RELIANCE", MoveType.ENTER_LONG, 5000, 1000.0,
                             stop=990.0))
    assert v.approved and v.max_qty == 500
    assert "MAX_RISK_PER_TRADE" in v.breached


def test_single_name_concentration_cap():
    _l, pf, re = setup(max_position_pct=10.0, max_risk_per_trade_pct=99.0)
    v = re.check(TradeIntent("RELIANCE", MoveType.ENTER_LONG, 10_000, 1000.0,
                             stop=999.0))
    assert v.max_qty == 100          # 10% of 1,000,000 / 1000
    assert "MAX_POSITION_PCT" in v.breached


def test_max_open_positions_blocks_a_new_name():
    _l, pf, re = setup(max_open_positions=2)
    for s in ("A", "B"):
        p = pf.position(s)
        p.qty = 10
        p.avg_price = 100.0
        pf.mark(s, 100.0)
    v = re.check(TradeIntent("C", MoveType.ENTER_LONG, 10, 100.0, stop=99.0))
    assert not v.approved and "MAX_OPEN_POSITIONS" in v.breached


def test_sector_concentration():
    _l, pf, re = setup(max_sector_exposure_pct=20.0,
                       max_risk_per_trade_pct=99.0, max_position_pct=99.0)
    # HDFCBANK is BANK; load it up, then try ICICIBANK (also BANK)
    p = pf.position("HDFCBANK")
    p.qty = 100
    p.avg_price = 1500.0
    pf.cash -= 150_000.0
    pf.mark("HDFCBANK", 1500.0)      # 150,000 = 15% of equity
    v = re.check(TradeIntent("ICICIBANK", MoveType.ENTER_LONG, 1000, 1000.0,
                             stop=995.0))
    assert v.max_qty <= 50           # only 5% of equity left in BANK
    assert "MAX_SECTOR_EXPOSURE" in v.breached


def test_leverage_cap():
    _l, pf, re = setup(max_leverage=2.0, max_risk_per_trade_pct=99.0,
                       max_position_pct=99.0, max_gross_exposure_pct=999.0,
                       max_net_exposure_pct=999.0)
    p = pf.position("A")
    p.qty = 1800
    p.avg_price = 1000.0
    pf.cash = 1_000_000.0 - 1_800_000.0    # bought on margin
    pf.mark("A", 1000.0)                   # 1.8m gross on 1m equity
    assert pf.leverage() == pytest.approx(1.8)
    v = re.check(TradeIntent("B", MoveType.ENTER_LONG, 1000, 1000.0, stop=995.0))
    assert v.max_qty <= 200
    assert "MAX_LEVERAGE" in v.breached


# ---------------------------------------------------------------- halts


def test_daily_loss_halt():
    _l, pf, re = setup(max_daily_loss_pct=2.0)
    pf.cash = 970_000.0              # -3%
    assert re.monitor(ts=1) == HaltReason.DAILY_LOSS
    v = re.check(TradeIntent("A", MoveType.ENTER_LONG, 10, 100.0, stop=99.0))
    assert not v.approved


def test_drawdown_halt():
    _l, pf, re = setup(max_drawdown_pct=5.0)
    pf.peak_equity = 1_200_000.0
    pf.cash = 1_100_000.0            # 8.3% below the high-water mark
    assert re.monitor(ts=1) == HaltReason.DRAWDOWN


def test_consecutive_loss_halt():
    """The configured value is a floor, not the trigger.

    The actual threshold is whichever is larger: the configured floor, or the
    shortest streak that would be statistically surprising given the win rate
    and how many trades were taken. Here nothing has traded, so the floor and
    the statistical threshold are both small.
    """
    _l, pf, re = setup(consecutive_loss_halt=3)
    pf.consecutive_losses = 3
    assert re.monitor(ts=1) is None, "halted on a streak that is not surprising"
    pf.consecutive_losses = 8
    assert re.monitor(ts=1) == HaltReason.CONSECUTIVE_LOSSES


def test_order_rate_anomaly_halt():
    """A model in a feedback loop is caught by its behaviour, not its P&L.

    Measured as a sustained rate over five minutes, so a legitimate burst --
    a correlated book stopping out together, a session squaring off -- does
    not look like a malfunction.
    """
    _l, _pf, re = setup(max_orders_per_min=10)
    from gmq.core.types import NS
    base = 1_000 * NS
    for i in range(200):                       # 200 in 5 min vs a cap of 50
        re.on_order_sent(base + i)
    assert re.monitor(ts=base + 100) == HaltReason.ORDER_RATE


def test_a_legitimate_burst_does_not_trip_the_rate_detector():
    """Eight correlated positions stopping out inside one minute is the
    correlation limits working, not a runaway loop."""
    _l, _pf, re = setup(max_orders_per_min=10)
    from gmq.core.types import NS
    base = 1_000 * NS
    for i in range(30):                        # a burst, then quiet
        re.on_order_sent(base + i)
    assert re.monitor(ts=base + int(120 * NS)) is None


def test_model_degradation_halt():
    _l, _pf, re = setup()
    re.on_model_health(False, "brier 0.34")
    assert re.monitor(ts=1) == HaltReason.MODEL_DEGRADED


def test_excess_slippage_halt():
    _l, _pf, re = setup(max_slippage_bps_halt=20.0)
    for _ in range(30):
        re.on_slippage(45.0)
    assert re.monitor(ts=1) == HaltReason.SLIPPAGE


def test_halt_still_allows_closing_positions():
    """A halt must never trap the book in a position it cannot exit."""
    _l, pf, re = setup()
    re.halt(HaltReason.MANUAL, 1)
    v = re.check(TradeIntent("A", MoveType.EXIT, -100, 100.0, is_reducing=True))
    assert v.approved, "halt blocked an exit -- the control became the risk"


def test_stale_data_blocks_entry():
    _l, _pf, re = setup()
    re.on_data_stale("A", True)
    v = re.check(TradeIntent("A", MoveType.ENTER_LONG, 10, 100.0, stop=99.0))
    assert not v.approved and "DATA_STALE" in v.breached


# --------------------------------------------------- structural guarantee


def test_risk_engine_has_no_override_api():
    """The model layer must have no way to relax a limit.

    This is a test about the shape of the code. If someone adds `force=`,
    `override()` or a settable limits attribute path that the strategy can
    reach, this fails -- which is the point. A limit that can be argued with
    is not a limit.
    """
    banned = {"override", "force", "bypass", "disable", "relax", "allow_all",
              "set_limit", "raise_limit"}
    names = {n.lower() for n in dir(RiskEngine)}
    assert not (names & banned), f"override-shaped API found: {names & banned}"
    import inspect
    sig = inspect.signature(RiskEngine.check)
    assert set(sig.parameters) == {"self", "intent", "ts"}, \
        "RiskEngine.check grew a parameter -- check it is not an escape hatch"


def test_risk_engine_holds_no_reference_to_the_model_layer():
    _l, _pf, re = setup()
    for attr in vars(re).values():
        mod = type(attr).__module__ if attr is not None else ""
        assert "models" not in mod and "strategy" not in mod, \
            "risk engine reaches into the decision layer"


# ---------------------------------------------------------------- stops


def test_stop_never_widens_long():
    sp = StopPolicy()
    stop = 990.0
    for px, r in [(1000, 0.5), (1010, 1.5), (995, 0.2), (1020, 2.5), (990, 0.0)]:
        new, _why = sp.update(Side.BUY, float(px), 1000.0, stop, atr=5.0,
                              regime=Regime.TRENDING_UP, r_multiple=r)
        assert new >= stop - 1e-9, f"stop widened: {stop} -> {new}"
        stop = new


def test_stop_never_widens_short():
    sp = StopPolicy()
    stop = 1010.0
    for px, r in [(1000, 0.5), (990, 1.5), (1005, 0.2), (980, 2.5)]:
        new, _why = sp.update(Side.SELL, float(px), 1000.0, stop, atr=5.0,
                              regime=Regime.TRENDING_DOWN, r_multiple=r)
        assert new <= stop + 1e-9, f"stop widened: {stop} -> {new}"
        stop = new


def test_stop_moves_to_breakeven_when_paid_for():
    sp = StopPolicy(breakeven_r=1.0)
    new, why = sp.update(Side.BUY, 1010.0, 1000.0, 990.0, atr=5.0,
                         regime=Regime.TRENDING_UP, r_multiple=1.2)
    assert new > 990.0 and new >= 1000.0


def test_initial_stop_respects_volatility_floor():
    sp = StopPolicy(min_atr=0.7)
    plan = sp.initial(Side.BUY, 1000.0, atr=5.0, regime=Regime.LOW_VOL)
    assert (1000.0 - plan.stop) >= 0.7 * 5.0 - 1e-9
    assert plan.target > 1000.0


def test_time_stop_fires_on_a_stalled_trade():
    sp = StopPolicy()
    assert sp.time_stop(hold_s=4000, expected_hold_s=900, r_multiple=0.1)
    assert not sp.time_stop(hold_s=4000, expected_hold_s=900, r_multiple=1.8)


# ---------------------------------------------------------------- sizing


def test_sizer_picks_the_binding_constraint():
    limits = RiskLimits(max_risk_per_trade_pct=0.5, max_position_pct=100.0)
    s = PositionSizer(limits)
    d = s.size(equity=1_000_000, px=1000.0, stop=990.0,
               pred=Prediction(p_up=0.6, confidence=0.6, p_target=0.5,
                               p_stop=0.3),
               atr=10.0, ann_vol=0.20, top_depth_qty=100_000,
               model_skill=0.15)
    assert d.qty == min(d.detail.values())
    assert d.binding in d.detail


def test_sizer_shrinks_when_the_model_has_no_demonstrated_skill():
    limits = RiskLimits()
    s = PositionSizer(limits)
    kw = dict(equity=1_000_000, px=1000.0, stop=990.0, atr=10.0, ann_vol=0.20,
              top_depth_qty=100_000)
    pred = Prediction(p_up=0.65, confidence=0.8, p_target=0.55, p_stop=0.25)
    skilled = s.size(pred=pred, model_skill=0.30, **kw)
    unskilled = s.size(pred=pred, model_skill=0.0, **kw)
    assert unskilled.detail["kelly"] < skilled.detail["kelly"]


def test_heat_scale_reduces_size_in_drawdown():
    s = PositionSizer(RiskLimits(max_drawdown_pct=8.0))
    calm = s.heat_scale(open_risk_pct=0.2, drawdown_pct=0.0)
    hurt = s.heat_scale(open_risk_pct=0.2, drawdown_pct=6.0)
    assert hurt < calm
    assert s.heat_scale(2.8, 0.0) < calm     # portfolio heat also brakes


# ---------------------------------------------------------------- portfolio


def test_correlated_positions_count_as_one_cluster():
    class FakeCross:
        def correlation(self, a, b):
            return 0.9 if {a, b} <= {"HDFCBANK", "ICICIBANK", "SBIN"} else 0.1
        def correlated_cluster(self, s, t):
            return [x for x in ("HDFCBANK", "ICICIBANK", "SBIN") if x != s]

    pf = Portfolio(1_000_000)
    for s in ("HDFCBANK", "ICICIBANK", "SBIN"):
        p = pf.position(s)
        p.qty = 100
        p.avg_price = 1000.0
        pf.cash -= 100_000.0
        pf.mark(s, 1000.0)
    worst, clusters = pf.correlated_exposure(FakeCross(), threshold=0.65)
    assert worst == pytest.approx(30.0)          # all three, not 10% each
    assert len(clusters) == 1 and len(clusters[0]) == 3


def test_opposite_signs_are_a_spread_not_a_concentration():
    class FakeCross:
        def correlation(self, a, b):
            return 0.9
    pf = Portfolio(1_000_000)
    long = pf.position("HDFCBANK")
    long.qty, long.avg_price = 100, 1000.0
    pf.cash -= 100_000.0
    short = pf.position("ICICIBANK")
    short.qty, short.avg_price = -100, 1000.0
    pf.cash += 100_000.0
    pf.mark("HDFCBANK", 1000.0)
    pf.mark("ICICIBANK", 1000.0)
    worst, clusters = pf.correlated_exposure(FakeCross(), threshold=0.65)
    assert worst == pytest.approx(10.0), "a hedged pair was treated as one bet"
    assert not clusters


def test_position_accounting_round_trip():
    pf = Portfolio(1_000_000)
    p = pf.position("A")
    p.initial_risk = 1000.0
    pf.apply_fill(Fill("o1", "A", Side.BUY, 100, 500.0, ts=1, fee=50.0))
    pf.mark("A", 500.0)
    assert pf.position("A").qty == 100
    assert pf.equity() == pytest.approx(1_000_000 - 50.0)
    rec = pf.apply_fill(Fill("o2", "A", Side.SELL, 100, 510.0, ts=2, fee=50.0))
    assert rec is not None
    assert rec.pnl == pytest.approx(1000.0)
    assert pf.equity() == pytest.approx(1_000_000 + 1000.0 - 100.0)
    assert pf.position("A").qty == 0


def test_open_risk_is_measured_to_the_stop():
    pf = Portfolio(1_000_000)
    p = pf.position("A")
    p.qty, p.avg_price, p.stop = 100, 1000.0, 990.0
    pf.cash -= 100_000.0
    pf.mark("A", 1000.0)
    assert pf.open_risk() == pytest.approx(1000.0)
    assert pf.open_risk_pct() == pytest.approx(0.1)


def test_model_halt_lifts_when_skill_recovers():
    """Model quality is a condition, not an event.

    A daily-loss halt is a judgement about a day and should stand. A halt
    because forecast skill collapsed must lift when skill returns, or the
    first noisy patch disables the system for good.
    """
    _l, _pf, re = setup()
    re.on_model_health(False, "skill -0.08")
    assert re.monitor(ts=1) == HaltReason.MODEL_DEGRADED
    assert re.halted
    re.on_model_health(True, "skill +0.06")
    assert not re.halted, "model halt never recovers"
    assert re.check(TradeIntent("A", MoveType.ENTER_LONG, 10, 100.0,
                                stop=99.0)).approved


def test_model_recovery_cannot_clear_a_different_halt():
    """The auto-un-halt owns exactly one reason and must not clear others."""
    _l, pf, re = setup(max_daily_loss_pct=2.0)
    pf.cash = 960_000.0
    assert re.monitor(ts=1) == HaltReason.DAILY_LOSS
    re.on_model_health(True, "fine")
    assert re.halted, "a model-health signal cleared a daily-loss halt"
    assert re.halt_reason == HaltReason.DAILY_LOSS


def test_order_rate_cap_scales_with_universe():
    """A flat cap fires on normal operation with a large universe.

    One symbol sustaining 400 decisions in five minutes is a malfunction;
    twenty symbols doing the same is four per symbol per minute, which is
    just an active book.
    """
    from gmq.core.types import NS
    small = RiskEngine(RiskLimits(), Portfolio(1e6), n_symbols=1)
    big = RiskEngine(RiskLimits(), Portfolio(1e6), n_symbols=20)
    for i in range(400):
        small.on_order_sent(1000 * NS + i)
        big.on_order_sent(1000 * NS + i)
    assert small.monitor(ts=1000 * NS + 10) == HaltReason.ORDER_RATE
    assert big.monitor(ts=1000 * NS + 10) is None


def test_loss_streak_threshold_scales_with_trade_count():
    """A fixed streak limit fires on chance alone once the engine is active.

    At a 50% win rate a run of five appears somewhere in sixty trades about
    83% of the time, so a flat "halt at 5" halts a working strategy daily and
    detects nothing.
    """
    _l, pf, re = setup(consecutive_loss_halt=5)
    pf.day_trades = 1
    quiet = re._loss_streak_threshold()
    pf.day_trades = 80
    busy = re._loss_streak_threshold()
    assert busy > quiet, "threshold ignores how many trades were taken"
    assert busy >= 9, f"still fires on chance at 80 trades (needs {busy})"
    assert quiet >= 5, "configured floor not respected"


def test_loss_streak_threshold_tightens_for_a_high_win_rate_strategy():
    """Five losses in a row means far more from an 80%-win-rate strategy."""
    from gmq.risk.portfolio import TradeRecord

    def mk(pf, n, win_rate):
        for i in range(n):
            good = (i % 100) < int(win_rate * 100)
            pf.trades.append(TradeRecord(
                symbol="A", side=1, qty=1, entry_px=100.0, exit_px=101.0,
                entry_ns=0, exit_ns=1, pnl=(10.0 if good else -10.0),
                fees=0.0, r_multiple=0.0, mfe=0.0, mae=0.0, exit_reason=""))

    _l, hi, re_hi = setup()
    mk(hi, 120, 0.80)
    hi.day_trades = 60
    _l2, lo, re_lo = setup()
    mk(lo, 120, 0.40)
    lo.day_trades = 60
    assert re_hi._loss_streak_threshold() < re_lo._loss_streak_threshold()


def test_loss_streak_still_halts_on_a_genuinely_surprising_run():
    _l, pf, re = setup(consecutive_loss_halt=5)
    pf.day_trades = 20
    pf.consecutive_losses = 40
    assert re.monitor(ts=1) == HaltReason.CONSECUTIVE_LOSSES


def test_stop_management_does_not_count_toward_the_decision_rate():
    """Ratcheting a stop is not a new trading decision.

    A position being managed properly generates a steady stream of stop
    modifications. Counting those makes correct risk management look identical
    to a runaway loop, and halts the engine for doing its job.
    """
    from gmq.core.config import ExecConfig
    from gmq.core.clock import SimClock
    from gmq.core.types import OrderType
    from gmq.execution.router import OrderRouter

    sent = []

    class DummyBroker:
        on_fill = None
        on_order_update = None
        def place(self, o): return o.order_id
        def cancel(self, oid): return True
        def positions(self): return {}
        def order_status(self, oid): return None

    r = OrderRouter(DummyBroker(), SimClock(), ExecConfig(),
                    risk_hook=lambda kind, v: sent.append(kind))
    r.submit("A", Side.BUY, 10, OrderType.MARKET, intent_price=100.0)
    assert sent.count("sent") == 1
    for _ in range(25):
        r.submit("A", Side.SELL, 10, OrderType.SL_M, trigger=99.0,
                 intent_price=99.0, counts_as_decision=False)
    assert sent.count("sent") == 1, "stop plumbing counted as trading decisions"


def test_cancelling_an_order_does_not_double_count_the_expectation():
    """A reconciliation alarm nobody believes is not an alarm.

    `cancel()` and `_on_order_update()` both used to back out the unfilled
    remainder. With a stop resting on every position -- cancelled and replaced
    on every ratchet -- that produced hundreds of phantom mismatches.
    """
    from gmq.core.config import ExecConfig
    from gmq.core.clock import SimClock
    from gmq.core.types import OrderType, OrderStatus
    from gmq.execution.router import OrderRouter

    class DummyBroker:
        on_fill = None
        on_order_update = None
        def __init__(self): self.placed = {}
        def place(self, o):
            self.placed[o.order_id] = o
            return o.order_id
        def cancel(self, oid):
            o = self.placed.get(oid)
            if o is None:
                return False
            o.status = OrderStatus.CANCELLED
            self.on_order_update(o)
            return True
        def positions(self): return {}
        def order_status(self, oid): return None

    b = DummyBroker()
    r = OrderRouter(b, SimClock(), ExecConfig())
    orders = r.submit("A", Side.BUY, 100, OrderType.LIMIT, price=100.0,
                      intent_price=100.0)
    assert r.expected_position["A"] == 100
    r.cancel(orders[0].order_id)
    assert r.expected_position["A"] == 0, \
        f"expectation double-counted: {r.expected_position['A']}"


def test_untriggered_stops_do_not_count_against_fill_rate():
    from gmq.core.config import ExecConfig
    from gmq.core.clock import SimClock
    from gmq.core.types import OrderType, OrderStatus, Order
    from gmq.execution.router import OrderRouter

    class DummyBroker:
        on_fill = None
        on_order_update = None
        def place(self, o): return o.order_id
        def cancel(self, oid): return True
        def positions(self): return {}
        def order_status(self, oid): return None

    r = OrderRouter(DummyBroker(), SimClock(), ExecConfig())
    entry = Order(symbol="A", side=Side.BUY, qty=10, otype=OrderType.MARKET,
                  tag="d1")
    entry.filled_qty, entry.status = 10, OrderStatus.COMPLETE
    r.history.append(entry)
    for _ in range(20):                       # identical stops, never triggered
        s = Order(symbol="A", side=Side.SELL, qty=10, otype=OrderType.SL_M,
                  trigger=99.0, tag="stop:d1")
        s.status = OrderStatus.CANCELLED
        r.history.append(s)
    st = r.stats()
    assert st["fill_rate"] == pytest.approx(1.0), st["fill_rate"]
    # 20 value-identical orders: a membership test would collapse them
    assert st["stops_placed"] == 20 and st["stops_triggered"] == 0


def test_a_triggered_stop_is_still_counted_as_a_stop():
    """The broker rewrites a triggered SL-M into a MARKET order, so order type
    no longer says what it was. Counting by type reports zero stops fired in a
    run where every stop worked."""
    from gmq.core.config import ExecConfig
    from gmq.core.clock import SimClock
    from gmq.core.types import OrderType, OrderStatus, Order
    from gmq.execution.router import OrderRouter

    class DummyBroker:
        on_fill = None
        on_order_update = None
        def place(self, o): return o.order_id
        def cancel(self, oid): return True
        def positions(self): return {}
        def order_status(self, oid): return None

    r = OrderRouter(DummyBroker(), SimClock(), ExecConfig())
    fired = Order(symbol="A", side=Side.SELL, qty=10,
                  otype=OrderType.MARKET,      # rewritten on trigger
                  tag="stop:d1")
    fired.filled_qty, fired.status = 10, OrderStatus.COMPLETE
    r.history.append(fired)
    st = r.stats()
    assert st["stops_placed"] == 1 and st["stops_triggered"] == 1
