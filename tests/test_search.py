"""Behavioural tests for the grandmaster search.

These assert *decisions*, not internals: given a situation a trader would
recognise, does the engine do the recognisable thing? They are the tests that
catch a mis-calibrated objective, which is otherwise silent -- an engine that
never trades looks exactly like an engine with no opportunities.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pytest

from gmq.core.types import Prediction, Regime, MoveType
from gmq.core.config import SearchConfig
from gmq.strategy.search import GrandmasterSearch
from gmq.strategy.moves import MoveContext, MoveGenerator


MARKET = dict(spread_bps=2.0, top_qty=800, liquidity=0.9, trade_rate=3.0,
              imbalance=0.1, queue_ahead=400)


def engine(**kw):
    cfg = SearchConfig(**kw)
    gs = GrandmasterSearch(cfg, seed=3)
    gs.scen.set_history("X", np.random.default_rng(1).standard_normal(400) * 8e-4)
    return gs


def ctx(**kw):
    d = dict(symbol="X", px=1000.0, bid=999.9, ask=1000.1, atr=3.0, qty=0,
             avg_price=0.0, stop=0.0, target=0.0, max_qty=600, unit_qty=200,
             regime=Regime.TRENDING_UP, alignment=0.5,
             alignment_conflict=False,
             pred=Prediction(p_up=0.5, confidence=0.5, exp_vol=0.003),
             liquidity=0.9, seconds_to_close=10000)
    d.update(kw)
    return MoveContext(**d)


def best(c, market=None, **cfgkw):
    gs = engine(**cfgkw)
    b, _all, st = gs.search(c, equity=1_000_000, market=market or MARKET)
    return b, st


def test_enters_long_on_strong_bull():
    b, _ = best(ctx(pred=Prediction(p_up=0.74, confidence=0.85, exp_vol=0.003,
                                    exp_return=0.004)))
    assert b.move is MoveType.ENTER_LONG and b.qty > 0


def test_enters_short_on_strong_bear():
    b, _ = best(ctx(pred=Prediction(p_up=0.26, confidence=0.85, exp_vol=0.003,
                                    exp_return=-0.004),
                    alignment=-0.5, regime=Regime.TRENDING_DOWN))
    assert b.move is MoveType.ENTER_SHORT and b.qty < 0


def test_waits_when_there_is_no_edge():
    b, _ = best(ctx(pred=Prediction(p_up=0.51, confidence=0.05, exp_vol=0.003)))
    assert b.move is MoveType.WAIT


def test_waits_when_edge_is_too_small_to_pay_costs():
    """A 2% edge cannot pay a 2bp spread plus taxes. It must not trade."""
    b, _ = best(ctx(pred=Prediction(p_up=0.52, confidence=0.6, exp_vol=0.003,
                                    exp_return=0.0002)))
    assert b.move is MoveType.WAIT


def test_refuses_to_open_near_the_close():
    b, _ = best(ctx(pred=Prediction(p_up=0.75, confidence=0.9, exp_vol=0.003,
                                    exp_return=0.004),
                    seconds_to_close=180))
    assert b.move is MoveType.WAIT


def test_never_crosses_a_wide_spread():
    """In a 30bp-wide illiquid name, crossing the spread cannot be right.

    The engine is allowed to *post* -- that is what a desk would do -- but it
    must not pay 15bp of half-spread for a 20% edge. The assertion is
    therefore about how it trades, not whether it trades.
    """
    m = dict(spread_bps=30.0, top_qty=60, liquidity=0.15, trade_rate=0.2,
             imbalance=0.0, queue_ahead=200)
    gs = engine()
    c = ctx(pred=Prediction(p_up=0.70, confidence=0.7, exp_vol=0.004),
            regime=Regime.ILLIQUID, liquidity=0.15)
    b, allm, _st = gs.search(c, equity=1_000_000, market=m)
    if b.move in (MoveType.ENTER_LONG, MoveType.ENTER_SHORT):
        assert "passive" in b.rationale, "crossed a 30bp spread"


def test_passive_entry_is_priced_for_adverse_selection():
    """A passive fill must not be modelled as free money."""
    from gmq.strategy.evaluator import CostModel
    from gmq.core.types import Side
    cm = CostModel()
    passive = cm.total_bps(100, 1000.0, Side.BUY, spread_bps=30.0,
                           top_qty=500, passive=True)
    crossing = cm.total_bps(100, 1000.0, Side.BUY, spread_bps=30.0,
                            top_qty=500, passive=False)
    assert passive > 0, "posting is modelled as costless"
    assert passive < crossing, "posting should still beat crossing"


def test_acts_on_a_broken_thesis():
    b, _ = best(ctx(qty=200, avg_price=1005.0, stop=998.0, target=1012.0,
                    hold_s=600, r_multiple=-0.7, alignment=-0.6,
                    regime=Regime.TRENDING_DOWN,
                    pred=Prediction(p_up=0.22, confidence=0.8, exp_vol=0.003,
                                    exp_return=-0.005)))
    assert b.move in (MoveType.EXIT, MoveType.REVERSE, MoveType.REDUCE)


def test_never_widens_a_stop():
    """The §5 rule. A widened stop must not even be generated."""
    gen = MoveGenerator()
    c = ctx(qty=200, avg_price=1000.0, stop=997.0, target=1006.0,
            hold_s=300, r_multiple=-0.5, px=998.0,
            pred=Prediction(p_up=0.45, confidence=0.5, exp_vol=0.003))
    for cand in gen.generate(c):
        if cand.move is MoveType.MOVE_STOP:
            assert cand.stop > c.stop, "generated a stop further from price"


def test_stop_moves_are_generated_when_in_profit():
    gen = MoveGenerator()
    c = ctx(qty=200, avg_price=994.0, stop=991.0, target=1006.0,
            hold_s=900, r_multiple=2.0, px=1000.0,
            pred=Prediction(p_up=0.6, confidence=0.6, exp_vol=0.003))
    stops = [m for m in gen.generate(c) if m.move is MoveType.MOVE_STOP]
    assert stops, "no trailing stop candidates in a winning position"
    assert all(s.stop > 991.0 for s in stops)


def test_no_reversal_immediately_after_entry():
    gen = MoveGenerator()
    c = ctx(qty=200, avg_price=1000.0, stop=997.0, hold_s=5.0,
            pred=Prediction(p_up=0.2, confidence=0.9, exp_vol=0.003))
    assert not [m for m in gen.generate(c) if m.move is MoveType.REVERSE]


def test_htf_conflict_reduces_size():
    gen = MoveGenerator()
    aligned = ctx(pred=Prediction(p_up=0.72, confidence=0.8, exp_vol=0.003),
                  alignment=0.6, alignment_conflict=False)
    conflict = ctx(pred=Prediction(p_up=0.72, confidence=0.8, exp_vol=0.003),
                   alignment=-0.6, alignment_conflict=True)
    a = max((m.qty for m in gen.generate(aligned)
             if m.move is MoveType.ENTER_LONG), default=0)
    c_ = max((m.qty for m in gen.generate(conflict)
              if m.move is MoveType.ENTER_LONG), default=0)
    assert 0 < c_ < a


def test_respects_latency_budget():
    lat = []
    for p in (0.74, 0.30, 0.55, 0.62):
        _b, st = best(ctx(pred=Prediction(p_up=p, confidence=0.8,
                                          exp_vol=0.003)))
        lat.append(st.ms)
    assert max(lat) <= 25.0 * 1.15, f"budget overrun: {max(lat):.1f}ms"


def test_timeout_never_returns_worse_than_a_completed_shallow_search():
    """A tiny budget must still return a sane move, not a partial-level artefact."""
    c = ctx(pred=Prediction(p_up=0.74, confidence=0.85, exp_vol=0.003,
                            exp_return=0.004))
    fast, _ = best(c, time_budget_ms=1.0)
    slow, _ = best(c, time_budget_ms=40.0)
    assert fast.move in (MoveType.ENTER_LONG, MoveType.WAIT)
    assert slow.move is MoveType.ENTER_LONG


def test_node_budget_mode_is_exactly_reproducible():
    """Backtests must replay bit-for-bit.

    Under a wall-clock budget the search explores however many paths it has
    time for, so the same input can give a different answer on a busy machine
    -- which quietly makes two walk-forward folds incomparable. `node_budget`
    replaces the clock with a deterministic node count for research runs.
    """
    c = ctx(pred=Prediction(p_up=0.70, confidence=0.8, exp_vol=0.003,
                            exp_return=0.003))
    a, _ = best(c, node_budget=400, time_budget_ms=1e6)
    b, _ = best(c, node_budget=400, time_budget_ms=1e6)
    assert a.move is b.move and a.qty == b.qty
    assert a.ev == pytest.approx(b.ev, abs=1e-12)
    assert a.score == pytest.approx(b.score, abs=1e-12)


def test_wall_clock_mode_still_returns_a_sane_move():
    c = ctx(pred=Prediction(p_up=0.74, confidence=0.85, exp_vol=0.003,
                            exp_return=0.004))
    a, _ = best(c)
    assert a.move in (MoveType.ENTER_LONG, MoveType.WAIT)
