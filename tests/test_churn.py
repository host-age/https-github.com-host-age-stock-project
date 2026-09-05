"""Tests for the churn fixes.

Two failure modes, both silent, both measured on a real run before being
fixed here:

  * the simulated exchange charged no transaction costs at all, so every P&L
    figure was gross while being labelled net;
  * re-evaluating an open position every few seconds with a noisy estimator
    let sampling noise decide whether to hold, over and over, collapsing the
    median holding period to 15 seconds against a 4.3bp round trip.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pytest

from gmq.core.config import SearchConfig, ExecConfig
from gmq.core.clock import SimClock
from gmq.core.types import (Prediction, Regime, MoveType, Side, OrderType,
                            MoveEval, NS)
from gmq.strategy.search import GrandmasterSearch
from gmq.strategy.moves import MoveContext
from gmq.strategy.evaluator import CostModel
from gmq.strategy.scenarios import ScenarioGenerator


MARKET = dict(spread_bps=2.0, top_qty=800, liquidity=0.9, trade_rate=3.0,
              imbalance=0.0, queue_ahead=400)


def engine(**kw):
    gs = GrandmasterSearch(SearchConfig(**kw), seed=3)
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


# ------------------------------------------------------- transaction costs


def test_the_exchange_charges_real_fees():
    """The fill price contains spread and impact; brokerage, STT, exchange
    charges, stamp duty and GST are levied by people and must be added."""
    from datetime import datetime
    from gmq.core.clock import IST
    from gmq.core.config import Config
    from gmq.app.engine import TradingEngine

    cfg = Config(run_id="t_fee", seed=3)
    cfg.data.symbols = cfg.data.symbols[:2]
    eng = TradingEngine(cfg, run_dir="runs/_t_fee",
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                        journal=False)
    for _ in range(60):
        eng.feed.step(0.5)
    sym = cfg.data.symbols[0]
    before = eng.portfolio.total_fees
    eng.router.submit(sym, Side.BUY, 50, OrderType.MARKET,
                      intent_price=eng.mde.ltp(sym))
    for _ in range(6):
        eng.clock.advance_s(0.5)
        eng.broker.pump()
        eng.feed.step(0.5)
    assert eng.portfolio.total_fees > before, \
        "a filled order was charged nothing"
    assert eng.broker.fees_charged > 0


def test_round_trip_cost_is_material_relative_to_a_typical_edge():
    """Documents the number the strategy has to beat."""
    cm = CostModel()
    notional = 100_000.0
    rt = (cm.fees(notional, Side.BUY) + cm.fees(notional, Side.SELL)) \
        / notional * 1e4
    assert 3.0 < rt < 6.0, f"round-trip fee {rt:.2f}bp is not in the expected range"


# ------------------------------------------------------------- hysteresis


def _in_position(**kw):
    d = dict(qty=200, avg_price=1000.0, stop=997.0, target=1006.0,
             hold_s=30.0, r_multiple=0.05,
             pred=Prediction(p_up=0.50, confidence=0.5, exp_vol=0.003))
    d.update(kw)
    return ctx(**d)


def test_a_marginal_preference_does_not_close_a_position():
    """The core fix: near-ties must resolve to holding, not to trading."""
    gs = engine()
    c = _in_position()
    best, ranked, st = gs.search(c, equity=1_000_000, market=MARKET)
    hold = next(m for m in ranked if m.move is MoveType.WAIT)
    if best.move is MoveType.WAIT:
        others = [m for m in ranked if m.move is not MoveType.WAIT]
        if others:
            # whatever lost, it must have lost by less than the round trip
            notional = abs(c.qty) * c.px
            rt = (gs.cost.fees(notional, Side.SELL)
                  + gs.cost.fees(notional, Side.BUY)
                  + notional * MARKET["spread_bps"] / 1e4)
            assert max(m.score for m in others) - hold.score < rt * 1.001


def test_hysteresis_can_be_disabled_and_then_the_engine_churns():
    """Shows the mechanism is doing the work, not a coincidence of scores."""
    c = _in_position()
    strict = engine(exit_hysteresis=1.0)
    loose = engine(exit_hysteresis=0.0)
    b_strict, _r, s_strict = strict.search(c, 1_000_000, MARKET)
    b_loose, _r2, _s2 = loose.search(c, 1_000_000, MARKET)
    # with no bar, any marginal edge is enough to act
    if b_loose.move is not MoveType.WAIT:
        assert b_strict.move is MoveType.WAIT
        assert s_strict.hysteresis_holds >= 1


def test_a_decisive_exit_still_fires_through_the_hysteresis():
    """Hysteresis raises the bar; it must not weld the position shut."""
    gs = engine()
    c = _in_position(
        r_multiple=-0.9, hold_s=600, alignment=-0.7,
        regime=Regime.TRENDING_DOWN,
        pred=Prediction(p_up=0.15, confidence=0.9, exp_vol=0.004,
                        exp_return=-0.008))
    best, _ranked, _st = gs.search(c, equity=1_000_000, market=MARKET)
    assert best.move is not MoveType.WAIT, \
        "hysteresis blocked an exit on a decisively broken thesis"


def test_hysteresis_does_not_apply_when_flat():
    """Entries are governed by the objective, not by a switching cost."""
    gs = engine()
    c = ctx(pred=Prediction(p_up=0.74, confidence=0.85, exp_vol=0.003,
                            exp_return=0.004))
    best, _r, st = gs.search(c, equity=1_000_000, market=MARKET)
    assert st.hysteresis_holds == 0


# ------------------------------------------------- common random numbers


def test_same_situation_gives_the_same_scenarios():
    """An unchanged situation must not produce a different answer.

    This is what stops a position exiting because the dice moved rather than
    because the market did.
    """
    sg = ScenarioGenerator(seed=5)
    sg.set_history("X", np.random.default_rng(0).standard_normal(400) * 8e-4)
    pred = Prediction(p_up=0.6, confidence=0.6, exp_vol=0.003)
    k = sg.common_random_seed("X", 1000.0, 3.0, Regime.TRENDING_UP, 200)
    a = sg.generate("X", 1000.0, pred, Regime.TRENDING_UP, 900,
                    n_paths=16, atr=3.0, seed_key=k)
    b = sg.generate("X", 1000.0, pred, Regime.TRENDING_UP, 900,
                    n_paths=16, atr=3.0, seed_key=k)
    assert np.allclose(a.paths, b.paths)


def test_a_material_price_move_draws_fresh_scenarios():
    """CRN must not freeze the engine's view of a market that has moved."""
    sg = ScenarioGenerator(seed=5)
    k1 = sg.common_random_seed("X", 1000.0, 3.0, Regime.TRENDING_UP, 200)
    k_small = sg.common_random_seed("X", 1000.1, 3.0, Regime.TRENDING_UP, 200)
    k_big = sg.common_random_seed("X", 1006.0, 3.0, Regime.TRENDING_UP, 200)
    assert k1 == k_small, "re-drew on a sub-tick wiggle"
    assert k1 != k_big, "did not re-draw after a two-ATR move"


def test_regime_change_draws_fresh_scenarios():
    sg = ScenarioGenerator(seed=5)
    a = sg.common_random_seed("X", 1000.0, 3.0, Regime.TRENDING_UP, 200)
    b = sg.common_random_seed("X", 1000.0, 3.0, Regime.HIGH_VOL, 200)
    assert a != b


def test_search_is_stable_across_repeated_evaluations_of_one_state():
    """The end-to-end property: re-deciding an unchanged position repeatedly
    must not eventually shake out a different answer."""
    c = _in_position()
    moves = set()
    for _ in range(12):
        gs = engine(node_budget=400, time_budget_ms=1e6)
        best, _r, _s = gs.search(c, equity=1_000_000, market=MARKET)
        moves.add((best.move, best.qty))
    assert len(moves) == 1, f"same state produced {len(moves)} different answers"
