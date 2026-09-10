from __future__ import annotations

import math

import numpy as np

from gmq.economics import (
    break_even_probability,
    certainty_equivalent,
    distance_time_floor,
    entropy_binary,
    expected_value,
    kelly_fraction,
    beta_posterior,
)


def test_expected_value_and_break_even_probability():
    assert expected_value([0.6, 0.4], [100.0, -50.0]) == 40.0
    assert math.isclose(break_even_probability(100.0, 50.0, 0.0), 1 / 3)
    assert break_even_probability(100.0, 50.0, 100.0) == 1.0


def test_uncertainty_shrunk_kelly_never_exceeds_cap():
    full = kelly_fraction(0.7, 2.0, 1.0, confidence=1.0, cap=0.25)
    weak = kelly_fraction(0.7, 2.0, 1.0, confidence=0.25, cap=0.25)
    assert 0.0 <= weak <= full <= 0.25
    assert kelly_fraction(0.4, 1.0, 1.0, confidence=1.0, cap=0.25) == 0.0


def test_beta_posterior_and_entropy():
    mean, lo, hi = beta_posterior(70, 30)
    assert 0.69 < mean < 0.71
    assert 0.5 < lo < mean < hi < 0.85
    assert entropy_binary(0.5) > entropy_binary(0.9)


def test_certainty_equivalent_penalises_risk():
    safe = certainty_equivalent([10.0, 10.0], [0.5, 0.5], risk_aversion=2.0)
    risky = certainty_equivalent([100.0, -80.0], [0.5, 0.5], risk_aversion=2.0)
    assert safe > risky


def test_speed_of_light_floor():
    # 1 km in vacuum is ~3.33564 microseconds.
    assert math.isclose(
        distance_time_floor(1_000.0),
        1_000.0 / 299_792_458.0,
        rel_tol=1e-12,
    )
