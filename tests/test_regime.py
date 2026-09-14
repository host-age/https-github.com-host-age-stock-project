"""Tests for the regime detector's filter mechanics.

The detector's own accuracy is a modelling question, evaluated empirically by
tools/eval_regime.py against the simulator's ground truth -- that is not
something a unit test should assert on. What belongs here is the filter's
*structural* behaviour: the warm-up gate, the confirmation delay that keeps it
from flipping on noise, the fast-path for high-conviction shocks, and the
prior floor that stops it from getting permanently stuck -- a bug this build
already found and fixed once (see README's "Bugs this build found and fixed"),
with no regression test until now.

_emissions and _evidence are patched throughout so each test drives the
Bayesian update directly with a chosen log-likelihood, rather than fighting
the rolling z-score normalisation that _evidence/_standardise would otherwise
need dozens of realistic bars to warm up.
"""
import sys
sys.path.insert(0, ".")
from unittest.mock import patch

import numpy as np
import pytest

from gmq.core.types import Regime
from gmq.regime.detector import RegimeDetector, REGIMES, _IDX


def _dominant_ll(regime: Regime, weight: float = 12.0) -> np.ndarray:
    ll = np.zeros(len(REGIMES))
    ll[_IDX[regime]] = weight
    return ll


def _drive(rd: RegimeDetector, symbol: str, ll: np.ndarray, n: int = 1):
    st = None
    with patch.object(RegimeDetector, "_evidence", return_value={"vol_ratio": 1.0}), \
         patch.object(RegimeDetector, "_emissions", staticmethod(lambda z: ll)):
        for _ in range(n):
            st = rd.update(symbol, None, None)
    return st


def test_stays_uninitialised_until_the_warmup_bar_count_is_reached():
    rd = RegimeDetector()
    ll = _dominant_ll(Regime.TRENDING_UP)
    for _ in range(39):
        st = _drive(rd, "X", ll)
        assert st.initialised is False, "reported an opinion before warm-up finished"
    st = _drive(rd, "X", ll)
    assert st.initialised is True


def test_regime_change_needs_confirm_bars_before_it_flips():
    """The core anti-flip mechanism: even overwhelming evidence must persist
    for confirm_bars before the *declared* regime changes."""
    rd = RegimeDetector(confirm_bars=3, min_conf=0.34)
    ll = _dominant_ll(Regime.TRENDING_UP)
    st = _drive(rd, "X", ll, n=40)          # first initialised update
    assert st.current == Regime.LOW_VOL, "flipped on the very first observation"
    assert st.candidate == Regime.TRENDING_UP
    assert st.candidate_bars == 1

    st = _drive(rd, "X", ll, n=1)
    assert st.current == Regime.LOW_VOL, "flipped before confirm_bars was reached"
    assert st.candidate_bars == 2

    st = _drive(rd, "X", ll, n=1)
    assert st.current == Regime.TRENDING_UP, "did not flip once confirm_bars was met"


def test_a_wavering_candidate_does_not_accumulate_confirmations():
    """Alternating candidates must not sneak past the confirmation delay by
    summing bars that were never consecutive."""
    rd = RegimeDetector(confirm_bars=3, min_conf=0.34)
    up = _dominant_ll(Regime.TRENDING_UP)
    down = _dominant_ll(Regime.TRENDING_DOWN)
    st = _drive(rd, "X", up, n=40)
    assert st.candidate == Regime.TRENDING_UP and st.candidate_bars == 1
    st = _drive(rd, "X", down, n=1)
    assert st.candidate == Regime.TRENDING_DOWN and st.candidate_bars == 1
    st = _drive(rd, "X", up, n=1)
    assert st.candidate == Regime.TRENDING_UP and st.candidate_bars == 1
    assert st.current == Regime.LOW_VOL, "never held one candidate for 3 bars running"


@pytest.mark.parametrize("shock_regime", [Regime.EVENT_DRIVEN, Regime.HIGH_VOL])
def test_high_confidence_shock_regimes_preempt_the_confirmation_delay(shock_regime):
    """Sections 9/17: waiting confirm_bars to notice an event is exactly wrong
    when an event is what happened -- EVENT_DRIVEN/HIGH_VOL at high confidence
    should flip on the very first observation."""
    rd = RegimeDetector(confirm_bars=3, min_conf=0.34)
    ll = _dominant_ll(shock_regime)
    st = _drive(rd, "X", ll, n=40)
    assert st.current == shock_regime, "high-conviction shock waited for confirmation"


def test_ordinary_regimes_do_not_get_the_fast_path():
    """The pre-emption is specific to EVENT_DRIVEN/HIGH_VOL, not "any high
    confidence" -- otherwise the confirmation delay would mean nothing."""
    rd = RegimeDetector(confirm_bars=3, min_conf=0.34)
    ll = _dominant_ll(Regime.BREAKOUT)
    st = _drive(rd, "X", ll, n=40)
    assert st.current == Regime.LOW_VOL, "a non-shock regime pre-empted the confirm delay"


def test_prior_floor_keeps_every_regime_numerically_reachable():
    """Regression test for a fixed bug: 'the regime filter self-trapped:
    without a prior floor, one state reached 0.999 and no contrary evidence
    could ever climb back' (README, Bugs this build found and fixed, #6).

    TRENDING_UP -> EVENT_DRIVEN and TRENDING_UP -> ILLIQUID are the two
    transition-matrix entries below the 0.004 floor (raw 0.003 each), so they
    are exactly where an unfloored prior would decay fastest. Seed the
    posterior almost entirely on TRENDING_UP, take one step with perfectly
    flat evidence (so the update is driven by the prior alone), and check
    every regime's posterior mass sits at or above what flooring guarantees --
    comfortably above what an unfloored propagation would leave (~0.003,
    normalised down further), and this is the exact numeric guarantee the
    fix provides, not merely "recovery eventually happens."
    """
    rd = RegimeDetector()
    st = rd.get("X")
    st.seen = 100
    n = len(REGIMES)
    st.posterior = np.full(n, 1e-6)
    st.posterior[_IDX[Regime.TRENDING_UP]] = 1.0 - 1e-6 * (n - 1)
    flat_ll = np.zeros(n)

    # Isolate the prior-floor mechanism specifically: _bias_correction also
    # perturbs loglik based on the running predicted marginal vs. the
    # transition matrix's stationary distribution, which would otherwise
    # confound this test's arithmetic with an unrelated (and itself
    # legitimate) correction term.
    with patch.object(RegimeDetector, "_evidence", return_value={"vol_ratio": 1.0}), \
         patch.object(RegimeDetector, "_emissions", staticmethod(lambda z: flat_ll)), \
         patch.object(RegimeDetector, "_bias_correction", staticmethod(lambda st: np.zeros(n))):
        rd.update("X", None, None)

    for regime in (Regime.EVENT_DRIVEN, Regime.ILLIQUID):
        assert st.posterior[_IDX[regime]] >= 0.0035, (
            f"{regime} fell below the floor's guarantee -- "
            "looks like the self-trapping bug"
        )


def test_posterior_is_always_a_valid_distribution():
    rd = RegimeDetector()
    ll = _dominant_ll(Regime.MEAN_REVERTING, weight=7.0)
    st = _drive(rd, "X", ll, n=50)
    assert np.all(st.posterior >= 0.0)
    assert st.posterior.sum() == pytest.approx(1.0, abs=1e-9)


def test_stability_reports_the_fraction_of_recent_bars_in_the_current_regime():
    rd = RegimeDetector()
    st = rd.get("SBIN")
    st.current = Regime.TRENDING_UP
    st.history.extend(
        [Regime.TRENDING_UP] * 6 + [Regime.MEAN_REVERTING] * 4
    )
    assert rd.stability("SBIN", window=10) == pytest.approx(0.6)


def test_stability_is_zero_with_no_history():
    rd = RegimeDetector()
    assert rd.stability("NEWSYM") == 0.0


def test_snapshot_reports_the_expected_shape():
    rd = RegimeDetector()
    ll = _dominant_ll(Regime.TRENDING_UP)
    _drive(rd, "X", ll, n=40)
    snap = rd.snapshot("X")
    assert snap["regime"] in {r.value for r in REGIMES}
    assert snap["initialised"] is True
    assert set(snap["posterior"]) == {r.value for r in REGIMES}
    assert pytest.approx(sum(snap["posterior"].values()), abs=1e-3) == 1.0
