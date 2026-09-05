"""Multi-timeframe bar aggregation (spec §4).

One tick in, up to eight timeframes updated, O(1) each. Bars are emitted twice:
`BAR` on every update (so intrabar logic can see the forming candle) and
`BAR_CLOSE` exactly once when the interval rolls -- which is the only event
models are allowed to train on, because using a forming bar's close is one of
the classic ways to leak the future into a backtest.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Callable, Deque, Iterable
from collections import deque

import numpy as np

from ..core.types import Bar, Tick, Timeframe, NS


class BarSeries:
    """Ring of closed bars for one (symbol, timeframe) with vectorised access."""

    __slots__ = ("symbol", "tf", "cap", "_ts", "_o", "_h", "_l", "_c", "_v",
                 "_bv", "_sv", "n", "i", "current")

    def __init__(self, symbol: str, tf: Timeframe, cap: int = 1500):
        self.symbol = symbol
        self.tf = tf
        self.cap = cap
        self._ts = np.zeros(cap, dtype=np.int64)
        self._o = np.zeros(cap)
        self._h = np.zeros(cap)
        self._l = np.zeros(cap)
        self._c = np.zeros(cap)
        self._v = np.zeros(cap)
        self._bv = np.zeros(cap)
        self._sv = np.zeros(cap)
        self.n = 0
        self.i = 0
        self.current: Optional[Bar] = None

    def append(self, b: Bar) -> None:
        i = self.i
        self._ts[i] = b.ts
        self._o[i] = b.o
        self._h[i] = b.h
        self._l[i] = b.l
        self._c[i] = b.c
        self._v[i] = b.v
        self._bv[i] = b.buy_v
        self._sv[i] = b.sell_v
        self.i = (i + 1) % self.cap
        if self.n < self.cap:
            self.n += 1

    def _ordered(self, arr: np.ndarray) -> np.ndarray:
        if self.n < self.cap:
            return arr[: self.n]
        return np.concatenate((arr[self.i:], arr[: self.i]))

    @property
    def ts(self) -> np.ndarray: return self._ordered(self._ts)
    @property
    def open(self) -> np.ndarray: return self._ordered(self._o)
    @property
    def high(self) -> np.ndarray: return self._ordered(self._h)
    @property
    def low(self) -> np.ndarray: return self._ordered(self._l)
    @property
    def close(self) -> np.ndarray: return self._ordered(self._c)
    @property
    def volume(self) -> np.ndarray: return self._ordered(self._v)
    @property
    def buy_vol(self) -> np.ndarray: return self._ordered(self._bv)
    @property
    def sell_vol(self) -> np.ndarray: return self._ordered(self._sv)

    def closes(self, k: int) -> np.ndarray:
        c = self.close
        return c[-k:] if c.size >= k else c

    def returns(self, k: int = 250) -> np.ndarray:
        c = self.closes(k + 1)
        if c.size < 2:
            return np.zeros(0)
        return np.diff(np.log(np.maximum(c, 1e-9)))

    def last_close(self) -> float:
        return float(self._c[(self.i - 1) % self.cap]) if self.n else 0.0

    @property
    def ready(self) -> bool:
        return self.n >= 30

    def __len__(self) -> int:
        return self.n


class BarAggregator:
    """Tick -> N timeframes. Handles gaps, session boundaries and warm start."""

    def __init__(self, symbols: Iterable[str],
                 timeframes: Iterable[Timeframe],
                 on_close: Optional[Callable[[Bar], None]] = None,
                 on_update: Optional[Callable[[Bar], None]] = None,
                 cap: int = 1500):
        self.timeframes: List[Timeframe] = list(timeframes)
        self.series: Dict[str, Dict[Timeframe, BarSeries]] = {}
        for s in symbols:
            self.series[s] = {tf: BarSeries(s, tf, cap) for tf in self.timeframes}
        self.on_close = on_close
        self.on_update = on_update
        self._last_vol: Dict[str, int] = defaultdict(int)

    def ensure(self, symbol: str) -> None:
        if symbol not in self.series:
            self.series[symbol] = {
                tf: BarSeries(symbol, tf) for tf in self.timeframes
            }

    @staticmethod
    def _bucket(ts: int, tf: Timeframe) -> int:
        """Floor a nanosecond timestamp to the start of its interval.

        Weekly buckets are anchored to Monday; daily to the IST calendar day.
        Everything else is a plain modulo, which is what the exchange's own
        1-minute bars do.
        """
        sec = ts // NS
        step = tf.seconds
        if step == 0:
            return ts
        if tf is Timeframe.D1:
            ist = sec + 19800                      # shift to IST for day edges
            return ((ist // 86400) * 86400 - 19800) * NS
        if tf is Timeframe.W1:
            ist = sec + 19800
            days = ist // 86400
            # unix epoch (1970-01-01) was a Thursday -> +3 aligns to Monday
            monday = ((days + 3) // 7) * 7 - 3
            return (monday * 86400 - 19800) * NS
        return ((sec // step) * step) * NS

    def on_tick(self, t: Tick) -> List[Bar]:
        """Returns the list of bars that CLOSED on this tick."""
        self.ensure(t.symbol)
        closed: List[Bar] = []
        # per-tick traded quantity, derived from cumulative volume if needed
        q = t.ltq
        if q <= 0 and t.volume:
            prev = self._last_vol[t.symbol]
            q = max(0, t.volume - prev)
            self._last_vol[t.symbol] = t.volume
        px = t.ltp
        for tf in self.timeframes:
            ser = self.series[t.symbol][tf]
            b = ser.current
            bucket = self._bucket(t.ts, tf)
            if b is None:
                ser.current = self._new_bar(t.symbol, tf, bucket, px, q, t)
                continue
            if bucket > b.ts:
                b.closed = True
                ser.append(b)
                closed.append(b)
                if self.on_close:
                    self.on_close(b)
                ser.current = self._new_bar(t.symbol, tf, bucket, px, q, t)
            else:
                if px > b.h:
                    b.h = px
                if px < b.l:
                    b.l = px
                b.c = px
                b.v += q
                b.vwap_num += px * q
                b.n_trades += 1
                if t.aggressor > 0:
                    b.buy_v += q
                elif t.aggressor < 0:
                    b.sell_v += q
                if self.on_update:
                    self.on_update(b)
        return closed

    @staticmethod
    def _new_bar(symbol: str, tf: Timeframe, bucket: int, px: float, q: int,
                 t: Tick) -> Bar:
        return Bar(ts=bucket, symbol=symbol, tf=tf, o=px, h=px, l=px, c=px,
                   v=q, n_trades=1, vwap_num=px * q,
                   buy_v=q if t.aggressor > 0 else 0,
                   sell_v=q if t.aggressor < 0 else 0)

    def force_close(self, symbol: Optional[str] = None) -> List[Bar]:
        """Close all forming bars -- used at session end so the daily/weekly
        candle is complete before end-of-day analytics run."""
        out: List[Bar] = []
        syms = [symbol] if symbol else list(self.series)
        for s in syms:
            for tf, ser in self.series[s].items():
                if ser.current is not None:
                    ser.current.closed = True
                    ser.append(ser.current)
                    out.append(ser.current)
                    if self.on_close:
                        self.on_close(ser.current)
                    ser.current = None
        return out

    def get(self, symbol: str, tf: Timeframe) -> Optional[BarSeries]:
        return self.series.get(symbol, {}).get(tf)

    def seed(self, symbol: str, tf: Timeframe, bars: Iterable[Bar]) -> None:
        """Warm start from history so the system is not blind at open."""
        self.ensure(symbol)
        ser = self.series[symbol][tf]
        for b in bars:
            b.closed = True
            ser.append(b)

    def snapshot(self, symbol: str) -> Dict[str, Dict[str, float]]:
        out = {}
        for tf, ser in self.series.get(symbol, {}).items():
            b = ser.current
            if b is None and ser.n == 0:
                continue
            src = b if b is not None else None
            out[tf.value] = {
                "o": src.o if src else float(ser.open[-1]),
                "h": src.h if src else float(ser.high[-1]),
                "l": src.l if src else float(ser.low[-1]),
                "c": src.c if src else float(ser.close[-1]),
                "v": float(src.v if src else ser.volume[-1]),
                "n": float(ser.n),
            }
        return out
