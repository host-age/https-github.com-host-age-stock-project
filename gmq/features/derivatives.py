"""Futures and options analytics (spec §1: OI, IV, volatility regimes).

Provides Black-76/Black-Scholes pricing, implied-vol inversion, greeks, and
the chain-level aggregates that carry directional information on NSE:
put-call ratio, max-pain, IV skew and term structure, and OI build-up
classification (long build-up / short build-up / long unwinding / short
covering) -- the framing every Indian derivatives desk uses.

If no options chain is supplied the module degrades gracefully: it emits an
IV proxy derived from realised volatility and a variance-risk-premium estimate
of zero, and marks the block as synthetic so the model layer can down-weight
it rather than being fed silent zeros.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.mathx import clamp, safe_div

SQRT2PI = math.sqrt(2.0 * math.pi)


def _n_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def _n_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t: float, vol: float, r: float = 0.065,
             is_call: bool = True) -> float:
    """Black-Scholes on a non-dividend-paying underlying. t in years."""
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        intrinsic = (spot - strike) if is_call else (strike - spot)
        return max(0.0, intrinsic)
    st = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / st
    d2 = d1 - st
    df = math.exp(-r * t)
    if is_call:
        return spot * _n_cdf(d1) - strike * df * _n_cdf(d2)
    return strike * df * _n_cdf(-d2) - spot * _n_cdf(-d1)


def bs_greeks(spot: float, strike: float, t: float, vol: float,
              r: float = 0.065, is_call: bool = True) -> Dict[str, float]:
    if t <= 0 or vol <= 0 or spot <= 0:
        return {"delta": 1.0 if (is_call and spot > strike) else 0.0,
                "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    st = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / st
    d2 = d1 - st
    df = math.exp(-r * t)
    pdf = _n_pdf(d1)
    delta = _n_cdf(d1) if is_call else _n_cdf(d1) - 1.0
    gamma = pdf / (spot * st)
    vega = spot * pdf * math.sqrt(t) / 100.0            # per 1 vol point
    theta_c = (-spot * pdf * vol / (2 * math.sqrt(t))
               - r * strike * df * _n_cdf(d2))
    theta_p = (-spot * pdf * vol / (2 * math.sqrt(t))
               + r * strike * df * _n_cdf(-d2))
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": (theta_c if is_call else theta_p) / 365.0}


def implied_vol(price: float, spot: float, strike: float, t: float,
                r: float = 0.065, is_call: bool = True,
                tol: float = 1e-6, max_iter: int = 60) -> float:
    """Newton with a bisection safety net.

    Pure Newton diverges on deep-ITM/OTM options where vega collapses, which
    is exactly where a naive implementation silently returns garbage and
    poisons the whole IV surface. Bracketing first, Newton inside the bracket.
    """
    if t <= 0 or spot <= 0 or strike <= 0 or price <= 0:
        return 0.0
    intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
    if price <= intrinsic + 1e-9:
        return 0.0
    lo, hi = 1e-4, 5.0
    p_lo = bs_price(spot, strike, t, lo, r, is_call)
    p_hi = bs_price(spot, strike, t, hi, r, is_call)
    if price <= p_lo:
        return lo
    if price >= p_hi:
        return hi
    vol = max(0.05, min(1.5, math.sqrt(2 * math.pi / t) * price / spot))
    for _ in range(max_iter):
        p = bs_price(spot, strike, t, vol, r, is_call)
        diff = p - price
        if abs(diff) < tol:
            return vol
        if diff > 0:
            hi = vol
        else:
            lo = vol
        v = bs_greeks(spot, strike, t, vol, r, is_call)["vega"] * 100.0
        if v > 1e-8:
            step = diff / v
            nxt = vol - step
            if lo < nxt < hi:
                vol = nxt
                continue
        vol = 0.5 * (lo + hi)
    return vol


@dataclass
class OptionQuote:
    strike: float
    expiry_days: float
    is_call: bool
    ltp: float
    oi: int = 0
    oi_change: int = 0
    volume: int = 0
    iv: float = 0.0


@dataclass
class ChainSnapshot:
    symbol: str
    spot: float
    quotes: List[OptionQuote] = field(default_factory=list)
    futures_price: float = 0.0
    futures_oi: int = 0
    futures_oi_change: int = 0
    synthetic: bool = False


class DerivativesEngine:
    """Chain-level analytics + graceful degradation when there is no chain."""

    def __init__(self, r: float = 0.065):
        self.r = r
        self.last: Dict[str, ChainSnapshot] = {}
        self.iv_history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    def update(self, chain: ChainSnapshot) -> None:
        for q in chain.quotes:
            if q.iv <= 0:
                q.iv = implied_vol(q.ltp, chain.spot, q.strike,
                                   max(q.expiry_days, 0.5) / 365.0,
                                   self.r, q.is_call)
        self.last[chain.symbol] = chain
        atm = self.atm_iv(chain)
        if atm > 0:
            h = self.iv_history.setdefault(chain.symbol, [])
            h.append(atm)
            if len(h) > 500:
                del h[:250]

    # ------------------------------------------------------------------
    @staticmethod
    def _nearest(chain: ChainSnapshot, is_call: bool, target: float
                 ) -> Optional[OptionQuote]:
        cands = [q for q in chain.quotes if q.is_call == is_call and q.iv > 0]
        if not cands:
            return None
        return min(cands, key=lambda q: abs(q.strike - target))

    def atm_iv(self, chain: ChainSnapshot) -> float:
        c = self._nearest(chain, True, chain.spot)
        p = self._nearest(chain, False, chain.spot)
        vals = [q.iv for q in (c, p) if q and q.iv > 0]
        return float(np.mean(vals)) if vals else 0.0

    def skew(self, chain: ChainSnapshot, moneyness: float = 0.05) -> float:
        """25-delta-ish skew: OTM put IV minus OTM call IV. Positive = the
        market is paying up for downside protection."""
        put = self._nearest(chain, False, chain.spot * (1 - moneyness))
        call = self._nearest(chain, True, chain.spot * (1 + moneyness))
        if not put or not call:
            return 0.0
        return float(put.iv - call.iv)

    def pcr_oi(self, chain: ChainSnapshot) -> float:
        c = sum(q.oi for q in chain.quotes if q.is_call)
        p = sum(q.oi for q in chain.quotes if not q.is_call)
        return safe_div(float(p), float(c), 1.0)

    def pcr_volume(self, chain: ChainSnapshot) -> float:
        c = sum(q.volume for q in chain.quotes if q.is_call)
        p = sum(q.volume for q in chain.quotes if not q.is_call)
        return safe_div(float(p), float(c), 1.0)

    def max_pain(self, chain: ChainSnapshot) -> float:
        """Strike at which total option-writer payout is minimised."""
        strikes = sorted({q.strike for q in chain.quotes})
        if not strikes:
            return chain.spot
        best, best_pain = strikes[0], float("inf")
        for s in strikes:
            pain = 0.0
            for q in chain.quotes:
                if q.is_call and s > q.strike:
                    pain += (s - q.strike) * q.oi
                elif (not q.is_call) and s < q.strike:
                    pain += (q.strike - s) * q.oi
            if pain < best_pain:
                best_pain, best = pain, s
        return float(best)

    @staticmethod
    def oi_buildup(price_change: float, oi_change: float) -> str:
        """The four-quadrant read used on every Indian derivatives desk."""
        if price_change > 0 and oi_change > 0:
            return "LONG_BUILDUP"
        if price_change < 0 and oi_change > 0:
            return "SHORT_BUILDUP"
        if price_change > 0 and oi_change < 0:
            return "SHORT_COVERING"
        if price_change < 0 and oi_change < 0:
            return "LONG_UNWINDING"
        return "NEUTRAL"

    def iv_rank(self, symbol: str) -> float:
        h = self.iv_history.get(symbol) or []
        if len(h) < 30:
            return 0.5
        cur = h[-1]
        lo, hi = min(h), max(h)
        return float(clamp((cur - lo) / max(hi - lo, 1e-9), 0.0, 1.0))

    # ------------------------------------------------------------------
    def features(self, symbol: str, spot: float, realised_vol_ann: float,
                 price_change_pct: float = 0.0) -> Dict[str, float]:
        chain = self.last.get(symbol)
        f: Dict[str, float] = {}
        if chain is None or not chain.quotes:
            # No chain: emit an honest proxy and flag it.
            f["dv_available"] = 0.0
            f["dv_iv_proxy"] = clamp(realised_vol_ann, 0, 2)
            f["dv_vrp"] = 0.0
            return f
        f["dv_available"] = 1.0
        atm = self.atm_iv(chain)
        f["dv_atm_iv"] = clamp(atm, 0, 3)
        f["dv_iv_rank"] = self.iv_rank(symbol)
        # variance risk premium: implied minus realised. Persistently positive
        # in index options; a collapse toward zero often precedes a vol event.
        f["dv_vrp"] = clamp(atm - realised_vol_ann, -1.5, 1.5)
        f["dv_skew"] = clamp(self.skew(chain), -0.6, 0.6)
        f["dv_pcr_oi"] = clamp(self.pcr_oi(chain), 0, 4)
        f["dv_pcr_vol"] = clamp(self.pcr_volume(chain), 0, 4)
        mp = self.max_pain(chain)
        f["dv_max_pain_dist"] = clamp((spot - mp) / max(spot, 1e-9) * 100,
                                      -15, 15)
        if chain.futures_price > 0 and spot > 0:
            basis = (chain.futures_price - spot) / spot * 100
            f["dv_basis_pct"] = clamp(basis, -5, 5)
            f["dv_fut_oi_chg"] = clamp(
                safe_div(float(chain.futures_oi_change),
                         float(max(chain.futures_oi, 1))) * 100, -50, 50)
            b = self.oi_buildup(price_change_pct, chain.futures_oi_change)
            for k in ("LONG_BUILDUP", "SHORT_BUILDUP", "SHORT_COVERING",
                      "LONG_UNWINDING"):
                f[f"dv_oi_{k.lower()}"] = 1.0 if b == k else 0.0
        # total gamma exposure sign: dealers long gamma damp moves, short
        # gamma amplifies them. A real regime input, not decoration.
        gex = 0.0
        for q in chain.quotes:
            if q.iv <= 0:
                continue
            g = bs_greeks(spot, q.strike, max(q.expiry_days, 0.5) / 365.0,
                          q.iv, self.r, q.is_call)["gamma"]
            gex += g * q.oi * (1 if q.is_call else -1)
        f["dv_gex_sign"] = clamp(math.tanh(gex / 1e6), -1, 1)
        return f


def synthetic_chain(symbol: str, spot: float, realised_vol_ann: float,
                    rng: np.random.Generator, expiry_days: float = 7.0,
                    n_strikes: int = 9) -> ChainSnapshot:
    """Build a plausible chain around spot when no real chain is available.

    Marked `synthetic=True` so nothing downstream can mistake it for observed
    market data. It exists so the derivatives feature block and the options
    code paths are exercised end to end rather than sitting untested.
    """
    base_iv = max(0.08, realised_vol_ann * float(rng.uniform(1.05, 1.35)))
    step = max(spot * 0.01, 0.05)
    atm = round(spot / step) * step
    quotes: List[OptionQuote] = []
    for i in range(-(n_strikes // 2), n_strikes // 2 + 1):
        k = atm + i * step
        if k <= 0:
            continue
        m = math.log(k / spot)
        # smile + downside skew
        iv = base_iv * (1.0 + 0.9 * m * m * 40 - 0.35 * m)
        iv = float(clamp(iv, 0.05, 2.0))
        t = expiry_days / 365.0
        for is_call in (True, False):
            px = bs_price(spot, k, t, iv, 0.065, is_call)
            oi = int(max(0, rng.normal(60000 * math.exp(-abs(i) / 2.5), 9000)))
            quotes.append(OptionQuote(
                strike=float(k), expiry_days=expiry_days, is_call=is_call,
                ltp=round(px, 2), oi=oi,
                oi_change=int(rng.normal(0, oi * 0.08)),
                volume=int(max(0, rng.normal(oi * 0.35, oi * 0.1))),
                iv=iv,
            ))
    fut = spot * (1.0 + 0.065 * expiry_days / 365.0)
    fo = int(rng.uniform(2e6, 9e6))
    return ChainSnapshot(symbol=symbol, spot=spot, quotes=quotes,
                         futures_price=float(fut), futures_oi=fo,
                         futures_oi_change=int(rng.normal(0, fo * 0.05)),
                         synthetic=True)
