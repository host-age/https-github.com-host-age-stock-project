"""Tests for the self-improvement controller (the ABULIDN loop).

The controller decides whether a change to the *system* is allowed and whether
it helped. These tests hold it to the two properties that make it safe: it
never lets a change loosen a hard risk limit, and it never promotes a change
the evidence does not support -- not even a change that happens to look good on
a handful of runs.

A synthetic evaluator stands in for the trading engine so the loop logic is
tested in milliseconds. The real engine substrate is exercised separately.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from gmq.core.config import Config
from gmq.evolve import (Proposal, ProposalState, KnobChange, KnobRegistry,
                        BoundaryError, EvolutionController, significance_gate)
from gmq.evolve.boundaries import KnobSpec
from gmq.evolve.trader import apply_patch


def prop(pid, path, after, before=None):
    return Proposal(id=pid, problem="p", root_cause="rc",
                    changes=[KnobChange(path=path, before=before, after=after)])


# ------------------------------------------------------------ boundaries
def test_registry_rejects_unknown_knob():
    reg = KnobRegistry()
    with pytest.raises(BoundaryError):
        reg.validate("search.some_made_up_knob", 1.0)


def test_registry_rejects_out_of_range():
    reg = KnobRegistry()
    with pytest.raises(BoundaryError):
        reg.validate("search.exit_hysteresis", 99.0)


def test_guarded_limit_cannot_be_loosened():
    reg = KnobRegistry()
    # max_drawdown_pct: smaller is stricter. Raising it is loosening -> refused.
    with pytest.raises(BoundaryError):
        reg.validate("risk.max_drawdown_pct", 12.0, current_value=8.0)


def test_guarded_limit_can_be_tightened():
    reg = KnobRegistry()
    # lowering the drawdown cap is a tightening -> allowed
    assert reg.validate("risk.max_drawdown_pct", 6.0, current_value=8.0) == 6.0


def test_guarded_change_requires_current_value():
    reg = KnobRegistry()
    with pytest.raises(BoundaryError):
        reg.validate("risk.max_leverage", 2.0)   # no current -> cannot check


def test_controller_rejects_boundary_violation_before_experiment():
    calls = []
    ev = lambda patch, seeds: calls.append(patch) or [0.0] * len(seeds)
    ctl = EvolutionController(ev)
    # in range [0.5, 2.0] but a loosening: 1.0 -> 1.5 raises the loss cap
    p = prop("P1", "risk.max_daily_loss_pct", 1.5, before=1.0)
    ctl.submit(p, current_values={"risk.max_daily_loss_pct": 1.0})
    assert p.state is ProposalState.REJECTED
    assert "loosen" in p.decision.lower() or "REFUSED" in p.decision
    assert calls == []                    # never even measured


# ------------------------------------------------------------ statistics
def test_gate_inconclusive_on_small_sample():
    r = significance_gate([1, 2], [3, 4], min_samples=8)
    assert r.verdict == "INCONCLUSIVE"


def test_gate_validates_a_real_improvement():
    rng = np.random.default_rng(0)
    before = rng.normal(0, 1, 60)
    after = before + 0.8            # paired, uniformly better
    r = significance_gate(before, after, min_samples=8)
    assert r.verdict == "VALIDATED" and r.p_value < 0.10


def test_gate_rejects_a_real_regression():
    rng = np.random.default_rng(1)
    before = rng.normal(0, 1, 60)
    after = before - 0.8
    r = significance_gate(before, after, min_samples=8)
    assert r.verdict == "REJECTED"


def test_gate_inconclusive_on_noise():
    rng = np.random.default_rng(2)
    before = rng.normal(0, 1, 60)
    after = rng.normal(0, 1, 60)     # unpaired noise, no real effect
    r = significance_gate(before, after, min_samples=8)
    assert r.verdict == "INCONCLUSIVE"


# ------------------------------------------------------------ full loop
def _paired_evaluator(effect):
    """A deterministic substrate: objective = seed noise + effect*patch_flag."""
    def ev(patch, seeds):
        out = []
        flag = patch.get("search.exit_hysteresis", 1.0)
        for s in seeds:
            base = np.random.default_rng(s).normal(0, 1.0)
            out.append(base + effect * (flag - 1.0))
        return out
    return ev


def test_loop_promotes_a_change_that_helps():
    ctl = EvolutionController(_paired_evaluator(effect=1.0), min_samples=8)
    p = prop("P2", "search.exit_hysteresis", 2.0, before=1.0)
    ctl.submit(p, current_values={"search.exit_hysteresis": 1.0})
    ctl.experiment(p, seeds=list(range(20)))
    assert p.state is ProposalState.VALIDATED
    ctl.promote(p)
    assert p.state is ProposalState.PROMOTED
    assert ctl.active["search.exit_hysteresis"] == 2.0


def test_loop_rejects_a_change_that_hurts():
    ctl = EvolutionController(_paired_evaluator(effect=-1.0), min_samples=8)
    p = prop("P3", "search.exit_hysteresis", 2.0, before=1.0)
    ctl.submit(p, current_values={"search.exit_hysteresis": 1.0})
    ctl.experiment(p, seeds=list(range(20)))
    assert p.state is ProposalState.REJECTED
    assert "search.exit_hysteresis" not in ctl.active


def test_cannot_promote_without_validation():
    ctl = EvolutionController(_paired_evaluator(0.0))
    p = prop("P4", "search.exit_hysteresis", 2.0, before=1.0)
    ctl.submit(p, current_values={"search.exit_hysteresis": 1.0})
    with pytest.raises(ValueError):
        ctl.promote(p)              # still PROPOSED


def test_rollback_restores_prior_value():
    ctl = EvolutionController(_paired_evaluator(1.0), min_samples=8)
    p = prop("P5", "search.exit_hysteresis", 2.0, before=1.0)
    ctl.submit(p, current_values={"search.exit_hysteresis": 1.0})
    ctl.experiment(p, seeds=list(range(20)))
    ctl.promote(p)
    ctl.rollback(p, "manual")
    assert p.state is ProposalState.ROLLED_BACK
    assert ctl.active.get("search.exit_hysteresis") == 1.0


def test_journal_records_every_transition():
    ctl = EvolutionController(_paired_evaluator(1.0), min_samples=8)
    p = prop("P6", "search.exit_hysteresis", 2.0, before=1.0)
    ctl.submit(p, current_values={"search.exit_hysteresis": 1.0})
    ctl.experiment(p, seeds=list(range(20)))
    ctl.promote(p)
    events = [e["event"] for e in ctl.journal.for_id("P6")]
    assert events == ["PROPOSE", "EXPERIMENT_START", "EXPERIMENT_VALIDATED",
                      "PROMOTE"]


# ------------------------------------------------------------ config patch
def test_apply_patch_sets_nested_config():
    cfg = Config()
    out = apply_patch(cfg, {"search.exit_hysteresis": 1.7,
                            "risk.max_drawdown_pct": 6.0})
    assert out.search.exit_hysteresis == 1.7
    assert out.risk.max_drawdown_pct == 6.0
    # original untouched (deep copy)
    assert cfg.search.exit_hysteresis == 1.0


def test_apply_patch_rejects_unknown_path():
    with pytest.raises(AttributeError):
        apply_patch(Config(), {"search.nope": 1})
