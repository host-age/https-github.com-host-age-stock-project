"""Decision-grade economic and statistical primitives for GMQ.

These formulas are deliberately limited to quantities that can affect a trading
 decision or a risk constraint. They are not a promise of profitability.

Core ideas:
  EV       = sum p_i * payoff_i
  p_be     = (loss + cost) / (win + loss)
  Kelly    = (p*b - q) / b, then uncertainty-shrunk and hard-capped
  Bayes    = Beta posterior for binary hit/miss evidence
  CE       = certainty equivalent under exponential utility
  entropy  = residual uncertainty of a probability forecast

All helpers are dependency-light and safe on the hot path.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def expected_value(
    probabilities: Iterable[float],
    payoffs: Iterable[float],
) -> float:
    """Probability-weighted expected payoff in the same units as `payoffs`."""
    p = np.asarray(list(probabilities), dtype=np.float64)
    x = np.asarray(list(payoffs), dtype=np.float64)
    if p.size == 0 or p.size != x.size:
        raise ValueError("probabilities and payoffs must have equal non-zero size")
    if np.any(p < 0):
        raise ValueError("probabilities cannot be negative")
    s = float(p.sum())
    if s <= 0:
        raise ValueError("probabilities must have positive mass")
    return float(np.dot(p / s, x))


def break_even_probability(
    avg_win: float,
    avg_loss: float,
    round_trip_cost: float = 0.0,
) -> float:
    """Win probability needed for non-negative expected value.

    With win W > 0, loss L > 0 and fixed cost C >= 0:
        p*W - (1-p)*L - C >= 0
        p_be = (L + C) / (W + L)
    """
    w = float(avg_win)
    l = float(avg_loss)
    c = max(float(round_trip_cost), 0.0)
    if w <= 0 or l <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (l + c) / (w + l))))


def kelly_fraction(
    p_win: float,
    avg_win: float,
    avg_loss: float,
    round_trip_cost: float = 0.0,
    confidence: float = 1.0,
    cap: float = 0.25,
) -> float:
    """Uncertainty-shrunk fractional Kelly allocation in [0, cap].

    Net payoff is used, then Kelly is shrunk by forecast confidence. A hard cap
    is mandatory because small probability errors can make full Kelly unstable.
    """
    p = min(1.0, max(0.0, float(p_win)))
    w = float(avg_win) - max(float(round_trip_cost), 0.0)
    l = float(avg_loss)
    if w <= 0 or l <= 0:
        return 0.0
    edge = p * w - (1.0 - p) * l
    if edge <= 0:
        return 0.0
    b = w / l
    raw = (p * b - (1.0 - p)) / b
    c = min(1.0, max(0.0, float(confidence)))
    return float(min(max(float(cap), 0.0), max(0.0, raw) * c))


def beta_posterior(
    successes: int,
    failures: int,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Tuple[float, float, float]:
    """Return posterior mean and 5/95% credible bounds for a Bernoulli rate."""
    if successes < 0 or failures < 0:
        raise ValueError("successes/failures cannot be negative")
    a = float(alpha0) + successes
    b = float(beta0) + failures
    if a <= 0 or b <= 0:
        raise ValueError("Beta prior parameters must be positive")
    # Normal approximation is avoided for the point estimate; bounds use the
    # exact scipy-free beta quantile approximation via a Wilson transform.
    n = a + b
    phat = a / n
    z = 1.959963984540054
    den = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / den
    half = z * math.sqrt(max(phat * (1.0 - phat) / n + z * z / (4.0 * n * n), 0.0)) / den
    return float(phat), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def entropy_binary(p: float) -> float:
    """Shannon entropy in nats for a Bernoulli forecast."""
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def information_content(p: float) -> float:
    """Information carried by a probability, relative to an even prior, in nats."""
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    return float(abs(math.log(p / (1.0 - p))))


def certainty_equivalent(
    outcomes: Iterable[float],
    probabilities: Iterable[float] | None = None,
    risk_aversion: float = 1.0,
    wealth: float = 1_000_000.0,
) -> float:
    """Exponential-utility certainty equivalent of a payoff distribution.

        CE = -(W/gamma) * log(E[exp(-gamma * X / W)])

    The wealth scaling keeps the exponent dimensionless and makes the measure
    comparable across account sizes.
    """
    x = np.asarray(list(outcomes), dtype=np.float64)
    if x.size == 0 or wealth <= 0 or risk_aversion < 0:
        return 0.0
    if probabilities is None:
        p = np.full(x.size, 1.0 / x.size)
    else:
        p = np.asarray(list(probabilities), dtype=np.float64)
        if p.size != x.size or np.any(p < 0) or p.sum() <= 0:
            raise ValueError("invalid probability vector")
        p = p / p.sum()
    if risk_aversion == 0:
        return float(np.dot(p, x))
    z = -risk_aversion * x / wealth
    m = float(np.max(z))
    log_mgf = m + math.log(float(np.dot(p, np.exp(z - m))))
    return float(-(wealth / risk_aversion) * log_mgf)


def annualised_volatility(returns: Iterable[float], periods_per_year: float) -> float:
    """Annualised volatility from periodic simple returns."""
    r = np.asarray(list(returns), dtype=np.float64)
    if r.size < 2 or periods_per_year <= 0:
        return 0.0
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def distance_time_floor(distance_m: float, propagation_speed_mps: float = 299_792_458.0) -> float:
    """Physics-only lower bound: t = d / v, useful for latency diagnostics."""
    if distance_m < 0 or propagation_speed_mps <= 0:
        raise ValueError("distance must be >= 0 and speed must be > 0")
    return float(distance_m / propagation_speed_mps)
