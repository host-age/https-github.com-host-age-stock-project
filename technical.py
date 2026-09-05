"""Technical and statistical features, computed per timeframe on bar close.

Two rules govern everything in this module:

1. **No lookahead.** Every function receives only closed bars. The forming bar
   is never passed in. This is enforced upstream by only calling the feature
   layer from the BAR_CLOSE event.
2. **Everything is normalised.** Raw prices are useless as model inputs across
   instruments trading at 148 and 12450 rupees. Every feature is either a
   ratio, a z-score, an ATR multiple or bounded to [-1, 1], so one model can
   be trained across the whole universe.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.mathx import (
    linreg_slope, hurst, half_life_ou, clamp, safe_div, zscore,
)


# --------------------------------------------------------------------------
# primitive indicators (vectorised, operate on ordered numpy arrays)
# --------------------------------------------------------------------------


def ema(x: np.ndarray, span: int) -> np.ndarray:
    if x.size == 0:
        return x
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def ema_last(x: np.ndarray, span: int) -> float:
    """Only the final value -- avoids allocating the whole series."""
    if x.size == 0:
        return 0.0
    a = 2.0 / (span + 1.0)
    v = x[0]
    for i in range(1, x.size):
        v += a * (x[i] - v)
    return float(v)


def sma(x: np.ndarray, n: int) -> float:
    return float(x[-n:].mean()) if x.size else 0.0


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    if c.size < 2:
        return np.maximum(h - l, 1e-9)
    pc = np.concatenate(([c[0]], c[:-1]))
    return np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> float:
    tr = true_range(h, l, c)
    if tr.size == 0:
        return 0.0
    return float(ema_last(tr[-(n * 3):], n))


def rsi(c: np.ndarray, n: int = 14) -> float:
    if c.size < n + 1:
        return 50.0
    d = np.diff(c[-(n * 4):])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = ema_last(up, n)
    ad = ema_last(dn, n)
    if ad < 1e-12:
        return 100.0 if au > 0 else 50.0
    rs = au / ad
    return float(100.0 - 100.0 / (1.0 + rs))


def macd(c: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9
         ) -> Tuple[float, float, float]:
    if c.size < slow + sig:
        return 0.0, 0.0, 0.0
    f = ema(c, fast)
    s = ema(c, slow)
    line = f - s
    signal = ema(line, sig)
    hist = line - signal
    scale = max(abs(c[-1]), 1e-9)
    return (float(line[-1]) / scale, float(signal[-1]) / scale,
            float(hist[-1]) / scale)


def adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14
        ) -> Tuple[float, float, float]:
    """Returns (ADX, +DI, -DI). ADX > 25 conventionally means 'trending'."""
    if c.size < n * 2 + 2:
        return 0.0, 0.0, 0.0
    k = min(c.size, n * 5)
    h, l, c = h[-k:], l[-k:], c[-k:]
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)[1:]
    atr_ = ema_last(tr, n)
    if atr_ < 1e-12:
        return 0.0, 0.0, 0.0
    pdi = 100.0 * ema_last(plus, n) / atr_
    mdi = 100.0 * ema_last(minus, n) / atr_
    denom = pdi + mdi
    dx = 100.0 * abs(pdi - mdi) / denom if denom > 1e-12 else 0.0
    return float(dx), float(pdi), float(mdi)


def bollinger(c: np.ndarray, n: int = 20, k: float = 2.0
              ) -> Tuple[float, float, float]:
    """Returns (%B, bandwidth as fraction of mid, z-score of close)."""
    if c.size < n:
        return 0.5, 0.0, 0.0
    w = c[-n:]
    m = float(w.mean())
    s = float(w.std())
    if s < 1e-12 or m < 1e-12:
        return 0.5, 0.0, 0.0
    upper, lower = m + k * s, m - k * s
    pb = (c[-1] - lower) / (upper - lower)
    return float(clamp(pb, -0.5, 1.5)), float((upper - lower) / m), \
        float((c[-1] - m) / s)


def keltner_position(h, l, c, n: int = 20, mult: float = 2.0) -> float:
    if c.size < n:
        return 0.0
    mid = ema_last(c[-n * 3:], n)
    a = atr(h, l, c, n)
    if a < 1e-12:
        return 0.0
    return float(clamp((c[-1] - mid) / (mult * a), -3.0, 3.0))


def stochastic(h, l, c, n: int = 14) -> float:
    if c.size < n:
        return 50.0
    hh = float(h[-n:].max())
    ll = float(l[-n:].min())
    if hh - ll < 1e-12:
        return 50.0
    return float(100.0 * (c[-1] - ll) / (hh - ll))


def cci(h, l, c, n: int = 20) -> float:
    if c.size < n:
        return 0.0
    tp = (h[-n:] + l[-n:] + c[-n:]) / 3.0
    m = tp.mean()
    md = np.abs(tp - m).mean()
    if md < 1e-12:
        return 0.0
    return float(clamp((tp[-1] - m) / (0.015 * md), -400, 400))


def obv_slope(c: np.ndarray, v: np.ndarray, n: int = 30) -> float:
    if c.size < 5:
        return 0.0
    k = min(n, c.size - 1)
    d = np.sign(np.diff(c[-(k + 1):]))
    o = np.cumsum(d * v[-k:])
    s, _ = linreg_slope(o)
    scale = max(abs(o).max(), 1.0)
    return float(clamp(s * k / scale, -3, 3))


def vwap_distance(h, l, c, v, n: int = 60) -> float:
    """Distance from rolling VWAP in ATR units -- the intraday anchor most
    Indian desks actually trade around."""
    if c.size < 5:
        return 0.0
    k = min(n, c.size)
    tp = (h[-k:] + l[-k:] + c[-k:]) / 3.0
    vv = v[-k:]
    tot = vv.sum()
    vw = float((tp * vv).sum() / tot) if tot > 0 else float(tp.mean())
    a = atr(h, l, c, 14)
    if a < 1e-12:
        return 0.0
    return float(clamp((c[-1] - vw) / a, -6, 6))


def donchian_position(h, l, c, n: int = 20) -> float:
    if c.size < n:
        return 0.0
    hh, ll = float(h[-n:].max()), float(l[-n:].min())
    if hh - ll < 1e-12:
        return 0.0
    return float(2.0 * (c[-1] - ll) / (hh - ll) - 1.0)


def realised_vol(c: np.ndarray, n: int = 30, ann: float = 0.0) -> float:
    if c.size < 3:
        return 0.0
    r = np.diff(np.log(np.maximum(c[-(n + 1):], 1e-9)))
    s = float(r.std())
    return s * math.sqrt(ann) if ann else s


def parkinson_vol(h, l, n: int = 30) -> float:
    """Range-based vol -- much lower variance than close-to-close, which is
    why it is the better input to a stop-distance calculation."""
    k = min(n, h.size)
    if k < 3:
        return 0.0
    lr = np.log(np.maximum(h[-k:], 1e-9) / np.maximum(l[-k:], 1e-9))
    return float(math.sqrt(float((lr ** 2).mean()) / (4.0 * math.log(2.0))))


def volume_zscore(v: np.ndarray, n: int = 40) -> float:
    if v.size < 8:
        return 0.0
    w = v[-n:]
    m, s = float(w.mean()), float(w.std())
    if s < 1e-9:
        return 0.0
    return float(clamp((v[-1] - m) / s, -5, 5))


def support_resistance(h, l, c, n: int = 120, tol: float = 0.0025
                       ) -> Tuple[float, float, float]:
    """Nearest swing support/resistance and how far price sits between them.

    Levels are swing pivots confirmed by two bars either side, clustered by
    proximity and weighted by how many times they were touched -- a level
    price has respected four times matters more than a single spike high.
    """
    k = min(n, c.size)
    if k < 12:
        px = float(c[-1]) if c.size else 0.0
        return px * 0.99, px * 1.01, 0.5
    hh, ll, cc = h[-k:], l[-k:], c[-k:]
    px = float(cc[-1])
    highs, lows = [], []
    for i in range(2, k - 2):
        if hh[i] >= hh[i - 1] and hh[i] >= hh[i - 2] and \
           hh[i] >= hh[i + 1] and hh[i] >= hh[i + 2]:
            highs.append(float(hh[i]))
        if ll[i] <= ll[i - 1] and ll[i] <= ll[i - 2] and \
           ll[i] <= ll[i + 1] and ll[i] <= ll[i + 2]:
            lows.append(float(ll[i]))

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        out, grp = [], [levels[0]]
        for x in levels[1:]:
            if abs(x - grp[-1]) / max(grp[-1], 1e-9) <= tol:
                grp.append(x)
            else:
                out.append((float(np.mean(grp)), len(grp)))
                grp = [x]
        out.append((float(np.mean(grp)), len(grp)))
        return out

    res = [(lv, w) for lv, w in cluster(highs) if lv > px]
    sup = [(lv, w) for lv, w in cluster(lows) if lv < px]
    r = min(res, key=lambda t: (t[0] - px) / max(t[1], 1))[0] if res else float(hh.max())
    s = max(sup, key=lambda t: -(px - t[0]) / max(t[1], 1))[0] if sup else float(ll.min())
    if r <= s:
        r, s = float(hh.max()), float(ll.min())
    span = max(r - s, 1e-9)
    return s, r, float(clamp((px - s) / span, 0.0, 1.0))


# --------------------------------------------------------------------------
# per-timeframe feature block
# --------------------------------------------------------------------------


def timeframe_features(o, h, l, c, v, prefix: str = "") -> Dict[str, float]:
    """~30 normalised features for one timeframe."""
    f: Dict[str, float] = {}
    if c.size < 5:
        return f
    px = float(c[-1])
    a = atr(h, l, c, 14)
    atr_pct = safe_div(a, px)

    # -- trend
    for span in (9, 21, 50, 200):
        if c.size >= max(5, span // 4):
            e = ema_last(c[-min(c.size, span * 4):], span)
            f[f"{prefix}ema{span}_dist_atr"] = clamp(safe_div(px - e, a), -8, 8)
    slope, r2 = linreg_slope(c[-min(c.size, 60):])
    f[f"{prefix}slope_atr"] = clamp(safe_div(slope * 20, a), -8, 8)
    f[f"{prefix}slope_r2"] = r2
    adx_, pdi, mdi = adx(h, l, c)
    f[f"{prefix}adx"] = adx_ / 100.0
    f[f"{prefix}di_diff"] = (pdi - mdi) / 100.0

    # -- momentum / oscillators
    f[f"{prefix}rsi"] = (rsi(c) - 50.0) / 50.0
    f[f"{prefix}stoch"] = (stochastic(h, l, c) - 50.0) / 50.0
    f[f"{prefix}cci"] = cci(h, l, c) / 200.0
    ml, ms, mh = macd(c)
    f[f"{prefix}macd_hist"] = clamp(mh * 1e4, -50, 50) / 50.0
    f[f"{prefix}macd_line"] = clamp(ml * 1e4, -80, 80) / 80.0

    # -- returns over several lookbacks, in ATR units so they are comparable
    for k in (1, 3, 5, 10, 20):
        if c.size > k:
            f[f"{prefix}ret{k}_atr"] = clamp(safe_div(px - float(c[-k - 1]), a),
                                             -10, 10)

    # -- volatility structure
    rv_s = realised_vol(c, 10)
    rv_l = realised_vol(c, 60)
    f[f"{prefix}atr_pct"] = clamp(atr_pct * 100, 0, 20)
    f[f"{prefix}vol_ratio"] = clamp(safe_div(rv_s, rv_l, 1.0), 0, 5)
    f[f"{prefix}parkinson"] = clamp(parkinson_vol(h, l) * 100, 0, 20)

    # -- bands / channels
    pb, bw, bz = bollinger(c)
    f[f"{prefix}bb_pctb"] = pb
    f[f"{prefix}bb_width"] = clamp(bw * 100, 0, 30)
    f[f"{prefix}bb_z"] = clamp(bz, -4, 4)
    f[f"{prefix}keltner"] = keltner_position(h, l, c)
    f[f"{prefix}donchian"] = donchian_position(h, l, c)

    # -- volume
    f[f"{prefix}vol_z"] = volume_zscore(v)
    f[f"{prefix}obv_slope"] = obv_slope(c, v)
    f[f"{prefix}vwap_dist"] = vwap_distance(h, l, c, v)

    # -- statistical character: is this series trending or reverting?
    lc = np.log(np.maximum(c[-min(c.size, 250):], 1e-9))
    f[f"{prefix}hurst"] = hurst(lc)
    hl = half_life_ou(lc)
    f[f"{prefix}mr_halflife"] = clamp(1.0 / hl if math.isfinite(hl) and hl > 0
                                      else 0.0, 0, 1)

    # -- structure
    s_lvl, r_lvl, pos = support_resistance(h, l, c)
    f[f"{prefix}sr_position"] = pos
    f[f"{prefix}dist_support_atr"] = clamp(safe_div(px - s_lvl, a), 0, 15)
    f[f"{prefix}dist_resist_atr"] = clamp(safe_div(r_lvl - px, a), 0, 15)

    # -- candle shape of the last closed bar
    rng = float(h[-1] - l[-1])
    if rng > 1e-12:
        f[f"{prefix}body_frac"] = float((c[-1] - o[-1]) / rng)
        f[f"{prefix}upper_wick"] = float((h[-1] - max(c[-1], o[-1])) / rng)
        f[f"{prefix}lower_wick"] = float((min(c[-1], o[-1]) - l[-1]) / rng)
    return f
