"""Cross-sectional context: breadth, correlation, sector strength, beta.

A signal on one name means something different depending on what the rest of
the market is doing. A breakout while breadth is negative and the name's beta
to NIFTY is 1.3 is a very different trade from the same breakout with breadth
strongly positive.

This module also produces the live correlation matrix the portfolio risk
engine needs for §10's correlated-exposure limit, computed incrementally so
it never becomes an O(n^2 * window) recomputation.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.mathx import OnlineCorr, EWMA, Ring, clamp, safe_div, linreg_slope
from ..core.config import sector_of


class CrossSectionalEngine:
    def __init__(self, symbols: List[str], index_symbol: str = "NIFTY",
                 corr_halflife: float = 240.0):
        self.symbols = [s for s in symbols]
        self.index_symbol = index_symbol
        self.tradables = [s for s in symbols if s != index_symbol]
        self.ret: Dict[str, Ring] = {s: Ring(400) for s in symbols}
        self.last_px: Dict[str, float] = {}
        self.corr: Dict[Tuple[str, str], OnlineCorr] = {
            (a, b): OnlineCorr(corr_halflife)
            for a, b in combinations(sorted(self.tradables), 2)
        }
        self.beta_num: Dict[str, EWMA] = {s: EWMA(halflife=corr_halflife)
                                          for s in self.tradables}
        self.beta_den: EWMA = EWMA(halflife=corr_halflife)
        self.index_ret = Ring(400)
        self.n = 0

    # ------------------------------------------------------------------
    def on_bar_close(self, prices: Dict[str, float]) -> None:
        """Call once per common timeframe (1m works well) with every symbol's
        close. Symbols that did not print are skipped, not forward-filled with
        a stale price -- forward-filling manufactures fake zero returns and
        biases correlation toward zero."""
        rets: Dict[str, float] = {}
        for s, px in prices.items():
            if px <= 0:
                continue
            prev = self.last_px.get(s)
            self.last_px[s] = px
            if prev and prev > 0:
                r = math.log(px / prev)
                self.ret[s].push(r) if s in self.ret else None
                rets[s] = r
        if not rets:
            return
        self.n += 1
        idx_r = rets.get(self.index_symbol)
        if idx_r is None:
            # synthetic index: equal-weight mean of the tradable universe
            vals = [v for k, v in rets.items() if k != self.index_symbol]
            idx_r = float(np.mean(vals)) if vals else 0.0
        self.index_ret.push(idx_r)
        self.beta_den.update(idx_r * idx_r)
        for s, r in rets.items():
            if s == self.index_symbol:
                continue
            self.beta_num[s].update(r * idx_r)
        for (a, b), c in self.corr.items():
            ra, rb = rets.get(a), rets.get(b)
            if ra is not None and rb is not None:
                c.update(ra, rb)

    # ------------------------------------------------------------------
    def beta(self, symbol: str) -> float:
        den = self.beta_den.get()
        if den < 1e-14 or symbol not in self.beta_num:
            return 1.0
        return float(clamp(self.beta_num[symbol].get() / den, -3.0, 4.0))

    def correlation(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        key = (a, b) if a < b else (b, a)
        c = self.corr.get(key)
        return c.rho if c is not None else 0.0

    def corr_matrix(self, symbols: Optional[List[str]] = None) -> np.ndarray:
        syms = symbols or self.tradables
        n = len(syms)
        m = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                r = self.correlation(syms[i], syms[j])
                m[i, j] = m[j, i] = r
        return m

    def correlated_cluster(self, symbol: str, threshold: float = 0.65
                           ) -> List[str]:
        """Names whose |rho| with `symbol` exceeds the threshold -- these are
        effectively the same risk and the risk engine treats them as one."""
        return [s for s in self.tradables
                if s != symbol and abs(self.correlation(symbol, s)) >= threshold]

    def avg_pairwise_corr(self) -> float:
        vals = [c.rho for c in self.corr.values() if c.ready]
        return float(np.mean(vals)) if vals else 0.0

    # ------------------------------------------------------------------
    def relative_strength(self, symbol: str, lookback: int = 60) -> float:
        """Cumulative return vs the index over `lookback` bars, in sigma."""
        r = self.ret.get(symbol)
        if r is None or r.n < 20:
            return 0.0
        a = r.last(lookback)
        i = self.index_ret.last(lookback)
        k = min(a.size, i.size)
        if k < 20:
            return 0.0
        excess = a[-k:] - self.beta(symbol) * i[-k:]
        s = float(excess.std())
        if s < 1e-12:
            return 0.0
        return float(clamp(float(excess.sum()) / (s * math.sqrt(k)), -5, 5))

    def sector_strength(self) -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for s in self.tradables:
            r = self.ret.get(s)
            if r is None or r.n < 20:
                continue
            acc.setdefault(sector_of(s), []).append(float(r.last(60).sum()))
        return {k: float(np.mean(v)) for k, v in acc.items() if v}

    def breadth(self) -> Dict[str, float]:
        ups = downs = 0
        cums = []
        for s in self.tradables:
            r = self.ret.get(s)
            if r is None or r.n < 10:
                continue
            cum = float(r.last(60).sum())
            cums.append(cum)
            if cum > 0:
                ups += 1
            elif cum < 0:
                downs += 1
        n = max(1, ups + downs)
        arr = np.asarray(cums) if cums else np.zeros(1)
        return {
            "xs_breadth": (ups - downs) / n,
            "xs_dispersion": float(arr.std()) * 100,
            "xs_avg_corr": self.avg_pairwise_corr(),
        }

    # ------------------------------------------------------------------
    def features(self, symbol: str) -> Dict[str, float]:
        f = dict(self.breadth())
        f["xs_beta"] = self.beta(symbol)
        f["xs_rel_strength"] = self.relative_strength(symbol)
        ss = self.sector_strength()
        sec = sector_of(symbol)
        f["xs_sector_ret"] = float(clamp(ss.get(sec, 0.0) * 100, -10, 10))
        if ss:
            vals = sorted(ss.values(), reverse=True)
            rank = vals.index(ss.get(sec, 0.0)) if sec in ss else len(vals) // 2
            f["xs_sector_rank"] = 1.0 - rank / max(len(vals) - 1, 1)
        f["xs_cluster_size"] = float(len(self.correlated_cluster(symbol)))
        ir = self.index_ret.last(60)
        if ir.size > 20:
            s, r2 = linreg_slope(np.cumsum(ir))
            f["xs_index_trend"] = float(clamp(s * 1e4 * 20, -10, 10))
            f["xs_index_trend_r2"] = r2
        return f
