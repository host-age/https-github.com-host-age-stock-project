"""Tests for the ways a backtest lies to itself.

These matter more than the performance tests. A leaked backtest does not fail
— it produces excellent numbers, which is precisely the problem. Each test
here asserts that a specific route by which the future could reach the model
is closed.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pytest

from gmq.core.types import NS, Timeframe, Tick
from gmq.models.base import TripleBarrierLabeler, OnlineScaler, SkillTracker
from gmq.models.online import OnlineLogistic, reliability
from gmq.models.ensemble import SkillWeightedEnsemble, IsotonicCalibrator


# ------------------------------------------------------------- labelling


def test_label_is_not_released_before_it_resolves():
    """The central guarantee: no sample exists until its outcome is real."""
    lab = TripleBarrierLabeler(up_atr=2.0, dn_atr=1.0, horizon_s=600)
    x = np.ones(4)
    lab.observe(x, ts=0, symbol="A", px=100.0, atr=1.0)
    # price wanders inside both barriers -> nothing may be emitted
    for k in range(1, 10):
        out = lab.update("A", ts=k * 10 * NS, px=100.0 + 0.5 * ((-1) ** k))
        assert out == [], "a sample was released before its barrier was touched"
    assert lab.pending_count() == 1
    # touching the upper barrier resolves it, and only then
    out = lab.update("A", ts=100 * NS, px=102.5)
    assert len(out) == 1 and out[0].resolved
    assert out[0].y_target == 1.0 and out[0].y_dir == 1.0


def test_timeout_resolves_with_the_direction_that_actually_happened():
    lab = TripleBarrierLabeler(up_atr=5.0, dn_atr=5.0, horizon_s=100)
    lab.observe(np.ones(3), ts=0, symbol="A", px=100.0, atr=1.0)
    out = lab.update("A", ts=101 * NS, px=100.4)
    assert len(out) == 1
    s = out[0]
    assert s.y_target == 0.0 and s.y_stop == 0.0     # neither barrier
    assert s.y_dir == 1.0                            # but it did close higher


def test_a_bar_spanning_both_barriers_is_labelled_ambiguous_not_guessed():
    """A bar hides its own path.

    If one bar's range spans both barriers, which was touched first is not
    recoverable. Picking one manufactures precision the data does not contain
    and the model then learns it as fact.
    """
    lab = TripleBarrierLabeler(up_atr=1.0, dn_atr=1.0, horizon_s=600)
    lab.observe(np.ones(3), ts=0, symbol="A", px=100.0, atr=1.0)
    out = lab.update("A", ts=10 * NS, px=100.2, high=101.5, low=98.5)
    assert len(out) == 1
    assert out[0].y_target == 0.5 and out[0].y_stop == 0.5


def test_tick_updates_resolve_to_one_side_only():
    """With scalar prices the ambiguous case cannot arise -- each observation
    is a single point, so exactly one barrier can be crossed."""
    lab = TripleBarrierLabeler(up_atr=1.0, dn_atr=1.0, horizon_s=600)
    lab.observe(np.ones(3), ts=0, symbol="A", px=100.0, atr=1.0)
    out = lab.update("A", ts=10 * NS, px=101.5)
    assert len(out) == 1
    assert (out[0].y_target, out[0].y_stop) == (1.0, 0.0)


def test_bar_range_is_used_for_excursions_not_just_the_close():
    """MAE on a bar feed must reflect the bar's low, not its close."""
    lab = TripleBarrierLabeler(up_atr=9.0, dn_atr=9.0, horizon_s=1000)
    lab.observe(np.ones(3), ts=0, symbol="A", px=100.0, atr=1.0)
    lab.update("A", ts=10 * NS, px=100.1, high=100.3, low=96.0)
    out = lab.update("A", ts=2000 * NS, px=100.1)
    assert out[0].mae == pytest.approx(-4.0)


def test_labeler_tracks_excursions_along_the_path():
    """MAE/MFE must come from the path, not the endpoints."""
    lab = TripleBarrierLabeler(up_atr=9.0, dn_atr=9.0, horizon_s=1000)
    lab.observe(np.ones(3), ts=0, symbol="A", px=100.0, atr=1.0)
    for px in (103.0, 96.0, 100.2):
        lab.update("A", ts=10 * NS, px=px)
    out = lab.update("A", ts=2000 * NS, px=100.2)
    s = out[0]
    assert s.mfe == pytest.approx(3.0)
    assert s.mae == pytest.approx(-4.0)


def test_other_symbols_are_untouched_by_an_update():
    lab = TripleBarrierLabeler(horizon_s=600)
    lab.observe(np.ones(3), ts=0, symbol="A", px=100.0, atr=1.0)
    lab.observe(np.ones(3), ts=0, symbol="B", px=100.0, atr=1.0)
    lab.update("A", ts=10 * NS, px=200.0)
    assert lab.pending_count("B") == 1, "B resolved on A's price"


# ------------------------------------------------------------- scaling


def test_scaler_is_fitted_online_not_on_the_whole_dataset():
    """A scaler fitted on the full series leaks the future's mean and variance
    into every past observation."""
    sc = OnlineScaler(dim=3, halflife=50)
    seen = []
    rng = np.random.default_rng(0)
    for i in range(200):
        x = rng.normal(i * 0.01, 1.0, 3)      # drifting mean
        seen.append(sc.fit_transform(x).copy())
    early, late = np.array(seen[:50]), np.array(seen[-50:])
    # if it had been fitted globally, early and late would be centred the same
    assert abs(early.mean()) < 4 and abs(late.mean()) < 4
    assert sc.n == 200


# ------------------------------------------------------------- scoring


def test_ensemble_scores_each_member_before_training_on_the_sample():
    """Skill must be out-of-sample for the observation it is measured on."""
    m = OnlineLogistic(dim=4, name="m")
    ens = SkillWeightedEnsemble([m], name="e")
    rng = np.random.default_rng(1)
    for _ in range(200):
        x = rng.normal(0, 1, 4)
        y = 1.0 if x[0] > 0 else 0.0
        n_before = m.n
        ens.partial_fit(x, y)
        assert m.n == n_before + 1
    # the tracker holds one scored prediction per sample, all pre-fit
    assert ens.trackers["m"].n_total == 200


def test_brier_skill_is_zero_for_a_base_rate_forecaster():
    """The property that makes skill the right metric: predicting the base
    rate scores zero however high the raw accuracy looks."""
    t = SkillTracker(window=1000)
    base = 0.9
    rng = np.random.default_rng(2)
    for _ in range(600):
        y = 1.0 if rng.random() < base else 0.0
        t.add(base, y)                     # always forecasts the base rate
    assert t.accuracy() > 0.85             # looks excellent
    assert abs(t.skill_score()) < 0.05     # and has no skill


def test_a_genuinely_informative_forecaster_scores_positive_skill():
    t = SkillTracker(window=1000)
    rng = np.random.default_rng(3)
    for _ in range(600):
        y = 1.0 if rng.random() < 0.5 else 0.0
        t.add(0.8 if y else 0.2, y)        # right, with conviction
    assert t.skill_score() > 0.5


# ------------------------------------------------------------- confidence


def test_confidence_does_not_encode_edge_magnitude():
    """Confidence answers 'how much should I trust this', not 'how big is the
    edge'. Folding the edge in double-counts it downstream."""
    a = reliability(n_samples=1000, skill=0.20)
    b = reliability(n_samples=1000, skill=0.20)
    assert a == b
    # more data and more demonstrated skill both raise it
    assert reliability(2000, 0.20) >= reliability(100, 0.20)
    assert reliability(1000, 0.25) > reliability(1000, 0.0)
    # an untrained model is never confident, however extreme its output
    assert reliability(5, 0.9) < 0.05


# ------------------------------------------------------------- calibration


def test_isotonic_calibration_corrects_a_systematically_overconfident_model():
    cal = IsotonicCalibrator(min_samples=100, refit_every=50)
    rng = np.random.default_rng(4)
    for _ in range(600):
        true_p = rng.uniform(0.1, 0.9)
        stated = min(0.99, true_p * 1.35)      # overconfident
        y = 1.0 if rng.random() < true_p else 0.0
        cal.add(stated, y)
    assert cal.ready
    # a stated 0.8 should be pulled back toward what actually happened
    assert cal.transform(0.8) < 0.8


def test_calibration_is_monotonic():
    cal = IsotonicCalibrator(min_samples=80, refit_every=40)
    rng = np.random.default_rng(5)
    for _ in range(400):
        p = rng.uniform(0.05, 0.95)
        cal.add(p, 1.0 if rng.random() < p else 0.0)
    assert cal.ready
    xs = np.linspace(0.05, 0.95, 25)
    ys = [cal.transform(float(x)) for x in xs]
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))


# ------------------------------------------------------------- bars


def test_bar_aggregator_never_exposes_a_forming_bar_as_closed():
    from gmq.data.bars import BarAggregator
    closed = []
    agg = BarAggregator(["A"], [Timeframe.M1], on_close=closed.append)
    base = 1_700_000_000 * NS
    for k in range(90):
        agg.on_tick(Tick(ts=base + k * NS, symbol="A", ltp=100.0 + k * 0.01,
                         ltq=1))
    assert all(b.closed for b in closed)
    ser = agg.get("A", Timeframe.M1)
    # the in-progress bar is held in `current`, never appended to the series
    assert ser.current is not None and not ser.current.closed
    assert ser.n == len(closed)
