"""Clocks and the NSE trading calendar.

`SimClock` drives backtests and the simulated exchange; `LiveClock` wraps the
wall clock. Both expose the same interface so no downstream component knows or
cares which one it is running against.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta, date, time as dtime
from typing import Optional

from .types import NS

IST = timezone(timedelta(hours=5, minutes=30))

# NSE equity cash market session (IST)
PREOPEN_START = dtime(9, 0)
PREOPEN_END = dtime(9, 8)
OPEN_TIME = dtime(9, 15)
CLOSE_TIME = dtime(15, 30)
POSTCLOSE_END = dtime(16, 0)
# MIS auto square-off
SQUARE_OFF = dtime(15, 15)


class SessionPhase:
    CLOSED = "CLOSED"
    PREOPEN = "PREOPEN"
    OPEN = "OPEN"
    SQUARE_OFF = "SQUARE_OFF"
    POSTCLOSE = "POSTCLOSE"


class Clock:
    def now_ns(self) -> int:                      # pragma: no cover - iface
        raise NotImplementedError

    def now_dt(self) -> datetime:
        return datetime.fromtimestamp(self.now_ns() / NS, IST)

    @property
    def ist(self) -> dtime:
        return self.now_dt().timetz().replace(tzinfo=None)

    def phase(self) -> str:
        dt = self.now_dt()
        if dt.weekday() >= 5 or is_holiday(dt.date()):
            return SessionPhase.CLOSED
        t = dt.time()
        if PREOPEN_START <= t < PREOPEN_END:
            return SessionPhase.PREOPEN
        if OPEN_TIME <= t < SQUARE_OFF:
            return SessionPhase.OPEN
        if SQUARE_OFF <= t < CLOSE_TIME:
            return SessionPhase.SQUARE_OFF
        if CLOSE_TIME <= t < POSTCLOSE_END:
            return SessionPhase.POSTCLOSE
        return SessionPhase.CLOSED

    def is_open(self) -> bool:
        return self.phase() in (SessionPhase.OPEN, SessionPhase.SQUARE_OFF)

    def seconds_to_close(self) -> float:
        dt = self.now_dt()
        close = dt.replace(hour=CLOSE_TIME.hour, minute=CLOSE_TIME.minute,
                           second=0, microsecond=0)
        return max(0.0, (close - dt).total_seconds())


class SimClock(Clock):
    """Deterministic clock advanced explicitly by the event loop."""

    __slots__ = ("_ns",)

    def __init__(self, start: Optional[datetime] = None):
        if start is None:
            start = datetime.now(IST).replace(hour=9, minute=15, second=0,
                                              microsecond=0)
        self._ns = int(start.timestamp() * NS)

    def now_ns(self) -> int:
        return self._ns

    def set_ns(self, ns: int) -> None:
        # never travel backwards -- ordering guarantees depend on it
        if ns > self._ns:
            self._ns = ns

    def advance_ns(self, delta: int) -> int:
        self._ns += delta
        return self._ns

    def advance_s(self, secs: float) -> int:
        return self.advance_ns(int(secs * NS))


class LiveClock(Clock):
    __slots__ = ()

    def now_ns(self) -> int:
        return time.time_ns()


# --------------------------------------------------------------------------
# holiday calendar
# --------------------------------------------------------------------------
# Trading holidays are published annually by the exchange. This is a static
# default; `load_holidays()` lets an operator point at the official list.
_HOLIDAYS: set[date] = set()


def load_holidays(dates) -> None:
    _HOLIDAYS.clear()
    for d in dates:
        if isinstance(d, str):
            d = date.fromisoformat(d)
        _HOLIDAYS.add(d)


def is_holiday(d: date) -> bool:
    return d in _HOLIDAYS


def session_bounds(d: date):
    """(open_ns, close_ns) for a given trading date."""
    o = datetime.combine(d, OPEN_TIME, IST)
    c = datetime.combine(d, CLOSE_TIME, IST)
    return int(o.timestamp() * NS), int(c.timestamp() * NS)
