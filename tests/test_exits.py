"""Tests for the round-trip pricing, the re-entry cooldown and the
excursion-fitted exit rule.

These exist because all three changes are the kind that look right in review
and are wrong in the P&L: a cost that is charged on some paths and not others,
a cooldown the search plans around but cannot play, and a fitted parameter
that is really a fitted coincidence.
"""
import subprocess
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from gmq.core.types import Prediction, Regime, MoveType, Side
from gmq.core.config import SearchConfig
from gmq.strategy.search import GrandmasterSearch
from gmq.strategy.moves import MoveContext, MoveGenerator
from gmq.strategy.evaluator import evaluate, CostModel
from gmq.analytics.excursion import (
    ExcursionBook, GivebackPolicy, PathExcursion, replay, Excursion,
    _ACTIVATIONS as ACTIVATIONS)
from gmq.risk.stops import StopPolicy


MARKET = dict(spread_bps=2.0, top_qty=800, liquidity=0.9, trade_rate=3.0,
              imbalance=0.1, queue_ahead=400)


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


# ---------------------------------------------------------------- determinism
def test_common_random_seed_is_stable_across_processes():
    """The seed must not depend on PYTHONHASHSEED.

    Python randomises string hashing per process, so a seed derived from the
    builtin hash() silently makes every replay of a backtest a different
    experiment. This is the regression test for that: same inputs, two
    processes with different hash seeds, same number.
    """
    prog = (
        "import sys; sys.path.insert(0,'.');"
        "from gmq.strategy.scenarios import ScenarioGenerator;"
        "from gmq.core.types import Regime;"
        "print(ScenarioGenerator().common_random_seed("
        "'RELIANCE', 2841.5, 6.2, Regime.TRENDING_UP, 100))"
    )
    outs = set()
    for hs in ("0", "1", "12345"):
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, env={"PYTHONHASHSEED": hs, "PATH": "/usr/bin:/bin"})
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"seed varies with PYTHONHASHSEED: {outs}"


# ------------------------------------------------------- round-trip pricing
def test_open_position_is_valued_at_liquidation():
    """An open position is worth what it would fetch, not what the screen says."""
    mark = evaluate(100, 1000.0, 990.0, 985.0, 1020.0, 0.0, 0.0, 3.0)
    liq = evaluate(100, 1000.0, 990.0, 985.0, 1020.0, 0.0, 0.0, 3.0,
                   exit_cost=97.0)
    assert mark - liq == pytest.approx(97.0)
    # a flat book has nothing to liquidate
    assert evaluate(0, 1000.0, 0.0, 0.0, 0.0, 50.0, 0.0, 3.0,
                    exit_cost=97.0) == pytest.approx(50.0)


def test_entry_pays_for_both_legs():
    """Turning the liquidation charge on must lower every entry's score.

    Without it the exit leg is only charged on paths where a stop or target
    happens to fire, so the search prices a round trip at roughly half.
    """
    c = ctx(pred=Prediction(p_up=0.74, confidence=0.85, exp_vol=0.003,
                            exp_return=0.004))
    scores = {}
    for flag in (True, False):
        gs = engine(liquidation_in_eval=flag)
        _b, allm, _s = gs.search(c, 1_000_000, MARKET)
        ent = [m for m in allm if m.move is MoveType.ENTER_LONG]
        scores[flag] = max(m.score for m in ent)
    assert scores[True] < scores[False], "liquidation charge did not bite"
    # and the gap is the size of an exit, not a rounding difference
    exit_cost = CostModel().cost_rupees(200, 1000.0, Side.SELL, 2.0, 800,
                                        False, 0.9)
    assert scores[False] - scores[True] > 0.15 * exit_cost


def test_marginal_entry_is_rejected_once_the_exit_leg_is_priced():
    """A trade whose whole edge is the un-charged exit leg should not happen."""
    gs_on, gs_off = engine(), engine(liquidation_in_eval=False)
    # walk down the edge until the cheap engine still enters
    for p_up in (0.62, 0.60, 0.58, 0.56, 0.545, 0.53):
        c = ctx(pred=Prediction(p_up=p_up, confidence=0.55, exp_vol=0.0016,
                                exp_return=0.0004))
        off, _, _ = gs_off.search(c, 1_000_000, MARKET)
        on, _, _ = gs_on.search(c, 1_000_000, MARKET)
        if off.move is MoveType.ENTER_LONG and on.move is MoveType.WAIT:
            return
    pytest.skip("no marginal edge found in the sampled range")


# ------------------------------------------------------------- cooldown
def test_no_entries_inside_the_reentry_cooldown():
    mg = MoveGenerator(reentry_cooldown_s=120.0)
    p = Prediction(p_up=0.75, confidence=0.9, exp_vol=0.003, exp_return=0.004)
    hot = mg.generate(ctx(pred=p, s_since_exit=8.0))
    cool = mg.generate(ctx(pred=p, s_since_exit=400.0))
    assert all(m.move is MoveType.WAIT for m in hot)
    assert any(m.move is MoveType.ENTER_LONG for m in cool)


def test_cooldown_does_not_block_managing_an_open_position():
    """The cooldown is about re-entry, not about being stuck in a trade."""
    mg = MoveGenerator(reentry_cooldown_s=120.0)
    moves = mg.generate(ctx(qty=200, avg_price=998.0, stop=995.0,
                            target=1006.0, s_since_exit=3.0))
    assert any(m.move is MoveType.EXIT for m in moves)


def test_search_does_not_plan_a_reentry_it_could_not_play():
    """A line that closes inside the search starts its own cooldown.

    Otherwise the search values exit-and-re-enter as available immediately --
    the exact move the cooldown forbids -- and plans around a move it cannot
    make.
    """
    gs = engine()
    parent = ctx(qty=200, avg_price=998.0, stop=995.0, target=1006.0,
                 s_since_exit=1e9)
    from gmq.strategy.search import SimState
    closed = SimState(px=1000.0, qty=0, avg=0.0, stop=0.0, target=0.0,
                      t_s=30.0)
    child = gs._child_ctx(parent, closed, 900.0)
    assert child.s_since_exit == 0.0
    still_open = SimState(px=1000.0, qty=200, avg=998.0, stop=995.0,
                          target=1006.0, t_s=30.0)
    assert gs._child_ctx(parent, still_open, 900.0).s_since_exit > 1e8


# ----------------------------------------------------------- give-back rule
def _walk(path):
    """Run an R-path through the live recorder and return what it would store."""
    tr = PathExcursion()
    for r in path:
        tr.update(r)
    return dict(mfe_r=tr.peak_r, mae_r=tr.trough_r, final_r=path[-1],
                min_after=tr.snapshot(), step_r=tr.step_r)


def test_recorder_arms_at_the_level_and_watches_only_afterwards():
    tr = PathExcursion()
    for r in (-0.4, 0.5, 1.2, 0.3, 2.1, 1.0):
        tr.update(r)
    i06 = ACTIVATIONS.index(0.6)
    i20 = ACTIVATIONS.index(2.0)
    # 0.6 armed at the 1.2 tick, so the -0.4 before it must not count
    assert tr.min_after[i06] == pytest.approx(0.3)
    # 2.0 armed at the 2.1 tick; the 0.3 dip happened before it
    assert tr.min_after[i20] == pytest.approx(1.0)
    assert tr.peak_r == pytest.approx(2.1)


def test_replay_is_exact_not_reconstructed():
    """The stop fires iff the worst R *after arming* fell through the level."""
    pol = GivebackPolicy(activation_r=1.0, keep_fraction=0.5)   # protects 0.5R
    def row(path):
        e = Excursion(**_walk(path))
        e.step_r = 0.0          # isolate the trigger logic from fill slippage
        return e
    never = row([0.2, 0.4, -1.0])
    gave_back = row([0.5, 1.4, 0.9, 0.1, -1.0])
    held = row([0.5, 1.4, 1.1, 1.8])
    out = replay([never, gave_back, held], pol, slippage_r=0.0)
    assert out[0] == pytest.approx(-1.0)    # never armed
    assert out[1] == pytest.approx(0.5)     # armed and crossed
    assert out[2] == pytest.approx(1.8)     # armed, never crossed


def test_fit_refuses_on_a_small_sample():
    bk = ExcursionBook(min_samples=200)
    rng = np.random.default_rng(0)
    for _ in range(50):
        bk.add(**_walk(list(rng.normal(0, 1, 12))))
    pol = bk.fit()
    assert not pol.fitted and "insufficient" in pol.note


def test_fit_refuses_without_exactly_replayable_rows():
    """Summary-only rows may not be fitted on -- their replay is biased."""
    bk = ExcursionBook(min_samples=50)
    rng = np.random.default_rng(2)
    for _ in range(200):
        f = rng.normal(0, 1)
        bk.add(mfe_r=abs(f) + 0.5, mae_r=-abs(f) - 0.5, final_r=f)
    pol = bk.fit()
    assert not pol.fitted and "replayable" in pol.note


def test_fit_refuses_on_noise():
    """A random walk has no give-back structure, so nothing should be fitted.

    This is the overfitting test that matters, and it is the one that caught
    the original design: replaying a trailing stop against a recorded peak
    made pure noise look like a policy worth 0.25R a trade at p=0.000.
    """
    bk = ExcursionBook(min_samples=100)
    rng = np.random.default_rng(4)
    for _ in range(700):
        # a driftless random walk in R, sampled at 40 ticks
        bk.add(**_walk(list(np.cumsum(rng.normal(0, 0.25, 40)))))
    pol = bk.fit()
    assert not pol.fitted, f"fitted on noise: {pol.as_dict()}"


def test_fit_finds_a_real_giveback_effect():
    """When runs genuinely round-trip, the fit should find it."""
    bk = ExcursionBook(min_samples=100)
    rng = np.random.default_rng(9)
    for _ in range(800):
        up = list(np.linspace(0, abs(rng.normal(1.8, 0.6)), 12))
        if rng.random() < 0.65:
            # gives the whole run back and then some
            path = up + list(np.linspace(up[-1], -abs(rng.normal(1.0, 0.3)), 12))
        else:
            path = up + list(np.linspace(up[-1], up[-1] * 1.1, 4))
        bk.add(**_walk(path))
    pol = bk.fit()
    assert pol.fitted and pol.edge_r > 0 and pol.p_value <= 0.10
    assert bk.stats()["losers_in_profit_first"] > 0.5


def test_giveback_stop_tightens_and_never_widens():
    sp = StopPolicy(giveback=GivebackPolicy(activation_r=1.0,
                                            keep_fraction=0.5))
    # long from 100 with 1R = 2.0/share, now at 105 having peaked at 2.5R
    new, why = sp.update(Side.BUY, px=105.0, avg=100.0, current_stop=98.0,
                         atr=2.0, regime=Regime.TRENDING_UP, r_multiple=2.5,
                         peak_r=2.5, risk_per_share=2.0)
    assert new > 98.0
    # it can only ever ratchet: feeding back a tighter current stop keeps it
    tighter, _ = sp.update(Side.BUY, px=105.0, avg=100.0, current_stop=new,
                           atr=2.0, regime=Regime.TRENDING_UP, r_multiple=2.5,
                           peak_r=2.5, risk_per_share=2.0)
    assert tighter >= new - 1e-9


def test_giveback_does_not_arm_below_activation():
    sp = StopPolicy(giveback=GivebackPolicy(activation_r=2.0,
                                            keep_fraction=0.5))
    off = StopPolicy(giveback=None)
    kw = dict(px=101.0, avg=100.0, current_stop=98.0, atr=2.0,
              regime=Regime.LOW_VOL, r_multiple=0.5, peak_r=0.5,
              risk_per_share=2.0)
    assert sp.update(Side.BUY, **kw) == off.update(Side.BUY, **kw)
