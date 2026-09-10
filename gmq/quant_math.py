"""Quantitative mathematics layer for GMQ.

This module maps standard trading mathematics into reusable, numerically-safe
primitives.  It complements the existing online models and risk engine rather
than replacing them.

Coverage:
  * descriptive statistics: mean/median/MAD/variance/std/IQR/robust z-score
  * probability: Gaussian helpers, log-odds, Bayesian Beta updates
  * time-series calculus: return, log-return, slope and acceleration
  * linear algebra: covariance, portfolio variance and marginal risk
  * regression: ordinary least squares with R^2
  * execution economics: VWAP/TWAP and implementation shortfall
  * derivatives: Black-Scholes price and first-order Greeks
  * growth/risk: Kelly and certainty-equivalent utility

All functions are deterministic, side-effect free and safe for use in tests
or decision-time diagnostics.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np

from .economics import (
    beta_posterior,
    break_even_probability,
    certainty_equivalent,
    expected_value,
    kelly_fraction,
)


def _array(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=np.float64)
    if a.size == 0:
        raise ValueError("series must be non-empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("series must contain only finite values")
    return a


def descriptive_stats(values: Iterable[float]) -> dict:
    """Robust descriptive statistics in one pass over a bounded sample."""
    x = _array(values)
    q1, med, q3 = np.quantile(x, [0.25, 0.50, 0.75])
    mean = float(x.mean())
    var = float(x.var(ddof=1)) if x.size > 1 else 0.0
    std = math.sqrt(max(var, 0.0))
    mad = float(np.mean(np.abs(x - med)))
    return {
        "n": int(x.size),
        "mean": mean,
        "median": float(med),
        "mad": mad,
        "variance": var,
        "std": std,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def robust_zscore(x: float, median: float, mad: float) -> float:
    """Median/MAD z-score; substantially less sensitive to price jumps/outliers."""
    scale = 1.4826 * float(mad)
    return 0.0 if scale <= 1e-15 else float((x - median) / scale)


def simple_returns(prices: Iterable[float]) -> np.ndarray:
    p = _array(prices)
    if p.size < 2 or np.any(p <= 0):
        return np.zeros(0, dtype=np.float64)
    return p[1:] / p[:-1] - 1.0


def log_returns(prices: Iterable[float]) -> np.ndarray:
    p = _array(prices)
    if p.size < 2 or np.any(p <= 0):
        return np.zeros(0, dtype=np.float64)
    return np.diff(np.log(p))


def rate_of_change(x_now: float, x_prev: float, dt_s: float) -> float:
    """First derivative dx/dt."""
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    return float((x_now - x_prev) / dt_s)


def acceleration(x_now: float, x_prev: float, x_prev2: float, dt_s: float) -> float:
    """Second discrete derivative d2x/dt2 using equally spaced observations."""
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    return float((x_now - 2.0 * x_prev + x_prev2) / (dt_s * dt_s))


def linear_regression(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float, float, float]:
    """OLS y = beta*x + alpha; returns (slope, intercept, r2, residual_std)."""
    xx = _array(x)
    yy = _array(y)
    if xx.size != yy.size or xx.size < 3:
        raise ValueError("x and y must have equal length >= 3")
    xm, ym = float(xx.mean()), float(yy.mean())
    dx, dy = xx - xm, yy - ym
    sxx = float(np.dot(dx, dx))
    if sxx <= 1e-15:
        raise ValueError("x has no variation")
    slope = float(np.dot(dx, dy) / sxx)
    intercept = ym - slope * xm
    fitted = slope * xx + intercept
    resid = yy - fitted
    sst = float(np.dot(dy, dy))
    sse = float(np.dot(resid, resid))
    r2 = 0.0 if sst <= 1e-15 else max(0.0, 1.0 - sse / sst)
    resid_std = math.sqrt(max(sse / max(xx.size - 2, 1), 0.0))
    return slope, float(intercept), float(r2), float(resid_std)


def covariance_matrix(returns: np.ndarray, shrink: float = 0.10) -> np.ndarray:
    """Sample covariance with diagonal shrinkage for numerical stability.

    Input shape is observations x assets.  Shrinkage reduces estimation noise
    when the portfolio contains many correlated names relative to the amount
    of history available.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 2:
        raise ValueError("returns must be a 2-D array with >=2 observations")
    s = np.asarray(np.cov(r, rowvar=False, ddof=1), dtype=np.float64)
    lam = float(np.clip(shrink, 0.0, 1.0))
    target = np.diag(np.diag(s))
    out = (1.0 - lam) * s + lam * target
    return (out + out.T) * 0.5


def portfolio_variance(weights: Iterable[float], covariance: np.ndarray) -> float:
    w = _array(weights)
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (w.size, w.size):
        raise ValueError("covariance shape must match weights")
    return float(w @ cov @ w)


def marginal_risk_contribution(weights: Iterable[float], covariance: np.ndarray) -> np.ndarray:
    w = _array(weights)
    cov = np.asarray(covariance, dtype=np.float64)
    total = portfolio_variance(w, cov)
    if total <= 1e-18:
        return np.zeros_like(w)
    return (w * (cov @ w)) / total


def gaussian_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    if std <= 0:
        raise ValueError("std must be positive")
    z = (float(x) - mean) / (std * math.sqrt(2.0))
    return float(0.5 * (1.0 + math.erf(z)))


def gaussian_pdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    if std <= 0:
        raise ValueError("std must be positive")
    z = (float(x) - mean) / std
    return float(math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi)))


def vwap(prices: Iterable[float], volumes: Iterable[float]) -> float:
    p = _array(prices)
    v = _array(volumes)
    if p.size != v.size or np.any(v < 0) or float(v.sum()) <= 0:
        raise ValueError("prices and volumes must match and contain positive volume")
    return float(np.dot(p, v) / v.sum())


def twap(prices: Iterable[float]) -> float:
    p = _array(prices)
    return float(p.mean())


def implementation_shortfall(expected_px: float, executed_px: float,
                             side: int, qty: int, fixed_cost: float = 0.0) -> float:
    """Total currency shortfall relative to the decision/reference price.

    side = +1 for BUY, -1 for SELL; positive means execution was worse.
    """
    if qty < 0 or fixed_cost < 0:
        raise ValueError("qty and fixed_cost must be non-negative")
    if side not in (-1, 1):
        raise ValueError("side must be +1 or -1")
    return float(side * (executed_px - expected_px) * qty + fixed_cost)


def black_scholes(drift_px: float, strike: float, time_years: float,
                  rate: float, vol: float, option: str = "call") -> dict:
    """Black-Scholes European option price + delta/gamma/vega/theta.

    `drift_px` is the underlying spot price. This is a reference-value model,
    not a statement that Indian options are frictionless or perfectly European.
    """
    s = float(drift_px)
    k = float(strike)
    t = float(time_years)
    r = float(rate)
    sigma = float(vol)
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("spot, strike, time and vol must be positive")
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    nd1 = gaussian_cdf(d1)
    nd2 = gaussian_cdf(d2)
    pdf1 = gaussian_pdf(d1)
    disc = math.exp(-r * t)
    if option == "call":
        price = s * nd1 - k * disc * nd2
        delta = nd1
        theta = (-s * pdf1 * sigma / (2 * math.sqrt(t)) - r * k * disc * nd2)
    else:
        price = k * disc * gaussian_cdf(-d2) - s * gaussian_cdf(-d1)
        delta = nd1 - 1.0
        theta = (-s * pdf1 * sigma / (2 * math.sqrt(t)) + r * k * disc * gaussian_cdf(-d2))
    gamma = pdf1 / (s * sigma * math.sqrt(t))
    vega = s * pdf1 * math.sqrt(t)
    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta_per_year": float(theta),
        "d1": float(d1),
        "d2": float(d2),
    }


__all__ = [
    "descriptive_stats", "robust_zscore", "simple_returns", "log_returns",
    "rate_of_change", "acceleration", "linear_regression", "covariance_matrix",
    "portfolio_variance", "marginal_risk_contribution", "gaussian_cdf",
    "gaussian_pdf", "vwap", "twap", "implementation_shortfall",
    "black_scholes", "beta_posterior", "break_even_probability",
    "certainty_equivalent", "expected_value", "kelly_fraction",
]
