import math

import numpy as np

from gmq.economics import beta_posterior, break_even_probability, certainty_equivalent, expected_value
from gmq.quant_math import (
    acceleration,
    black_scholes,
    covariance_matrix,
    descriptive_stats,
    implementation_shortfall,
    linear_regression,
    marginal_risk_contribution,
    rate_of_change,
    robust_zscore,
    simple_returns,
    twap,
    vwap,
)


def test_descriptive_stats_and_robust_zscore():
    s = descriptive_stats([1, 2, 3, 4, 100])
    assert s["median"] == 3.0
    assert s["q1"] <= s["median"] <= s["q3"]
    assert robust_zscore(100.0, s["median"], s["mad"]) > 10


def test_returns_derivatives_and_regression():
    r = simple_returns([100.0, 101.0, 103.0])
    assert np.allclose(r, [0.01, 103 / 101 - 1])
    assert rate_of_change(110.0, 100.0, 5.0) == 2.0
    assert acceleration(110.0, 105.0, 100.0, 5.0) == 0.0
    slope, intercept, r2, resid = linear_regression([0, 1, 2, 3], [1, 3, 5, 7])
    assert math.isclose(slope, 2.0)
    assert math.isclose(intercept, 1.0)
    assert math.isclose(r2, 1.0)
    assert resid < 1e-12


def test_portfolio_covariance_and_risk_contribution():
    r = np.asarray([[0.01, 0.02], [0.02, 0.03], [-0.01, -0.02]], dtype=float)
    cov = covariance_matrix(r, shrink=0.2)
    var = cov.shape == (2, 2) and np.isfinite(cov).all()
    assert var
    contrib = marginal_risk_contribution([0.5, 0.5], cov)
    assert np.isfinite(contrib).all()
    assert math.isclose(float(contrib.sum()), 1.0, rel_tol=1e-7, abs_tol=1e-7)


def test_execution_and_basic_economic_formulas():
    assert math.isclose(vwap([100, 101], [10, 20]), 100.66666666666667)
    assert math.isclose(twap([100, 102, 104]), 102.0)
    assert math.isclose(implementation_shortfall(100, 101, 1, 10), 10.0)
    assert math.isclose(expected_value([0.6, 0.4], [10, -5]), 4.0)
    assert break_even_probability(10, 5, 1.0) == 0.4
    mean, lo, hi = beta_posterior(60, 40)
    assert lo < mean < hi
    assert certainty_equivalent([100, -100], [0.5, 0.5], 1.0, 1_000_000) < 1.0


def test_black_scholes_reference_case():
    out = black_scholes(100, 100, 1.0, 0.0, 0.2, "call")
    assert 7.0 < out["price"] < 9.0
    assert 0.45 < out["delta"] < 0.55
    assert out["gamma"] > 0
    assert out["vega"] > 0
