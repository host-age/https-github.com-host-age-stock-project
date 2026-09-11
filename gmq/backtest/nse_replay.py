"""NSE-compatible historical replay utilities.

This module feeds real historical OHLCV rows into GMQ's existing ReplayFeed
rather than creating a second market-data model. It is deliberately strict
about chronological order so historical validation cannot silently introduce
look-ahead through unsorted rows.

Expected CSV columns:
    timestamp,symbol,open,high,low,close,volume

`timestamp` may be an integer nanosecond epoch or an ISO-8601 datetime.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from ..core.bus import EventBus
from ..core.clock import IST, SimClock
from ..core.types import NS
from ..data.feed import ReplayFeed, ReplayRow


@dataclass(frozen=True)
class ReplayDataset:
    rows: List[ReplayRow]
    symbols: List[str]
    first_ts: int
    last_ts: int

    @property
    def row_count(self) -> int:
        return len(self.rows)


class ReplayDataError(ValueError):
    pass


def _parse_ts(value: str) -> int:
    text = str(value).strip()
    if not text:
        raise ReplayDataError("empty timestamp")
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReplayDataError(f"invalid timestamp: {text!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return int(dt.timestamp() * 1_000_000_000)


def load_nse_csv(path: str | Path, *, require_ascending: bool = True) -> ReplayDataset:
    """Load and validate a date-ordered OHLCV CSV for ReplayFeed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[ReplayRow] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"timestamp", "symbol", "open", "high", "low", "close"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ReplayDataError(
                "CSV must contain: " + ", ".join(sorted(required))
            )
        prev_ts: Optional[int] = None
        for line_no, raw in enumerate(reader, start=2):
            try:
                ts = _parse_ts(raw["timestamp"])
                symbol = str(raw["symbol"]).strip().upper()
                if not symbol:
                    raise ReplayDataError("empty symbol")
                o = float(raw["open"])
                h = float(raw["high"])
                l = float(raw["low"])
                c = float(raw["close"])
                v = int(float(raw.get("volume") or 0))
                if not all(np.isfinite(x) for x in (o, h, l, c)):
                    raise ReplayDataError("non-finite OHLC")
                if min(o, h, l, c) <= 0 or h < max(o, c) or l > min(o, c) or h < l:
                    raise ReplayDataError("invalid OHLC relationship")
                if v < 0:
                    raise ReplayDataError("negative volume")
                if require_ascending and prev_ts is not None and ts < prev_ts:
                    raise ReplayDataError(
                        f"timestamps out of order at CSV line {line_no}"
                    )
                prev_ts = ts
                rows.append(ReplayRow(ts=ts, symbol=symbol, o=o, h=h, l=l, c=c, v=v))
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayDataError(f"invalid CSV line {line_no}: {exc}") from exc

    if not rows:
        raise ReplayDataError("CSV contains no data rows")
    if not require_ascending:
        rows.sort(key=lambda r: r.ts)
    return ReplayDataset(
        rows=rows,
        symbols=sorted({r.symbol for r in rows}),
        first_ts=rows[0].ts,
        last_ts=rows[-1].ts,
    )


def replay_rows(
    dataset: ReplayDataset,
    *,
    sub_steps: int = 12,
    seed: int = 7,
    spread_bps: float = 2.0,
) -> dict:
    """Run the common ReplayFeed against a real SimClock and EventBus.

    This function validates the data path only; strategy/risk decisions remain
    owned by TradingEngine. It is useful as a deterministic ingestion smoke
    test before invoking a full historical strategy run.
    """
    bus = EventBus(swallow_errors=False)
    clock = SimClock(start=datetime.fromtimestamp(dataset.first_ts / 1e9, tz=IST))
    received: list[tuple[int, str, float]] = []
    from ..core.bus import Topic

    bus.subscribe(
        Topic.TICK,
        lambda t: received.append((t.ts, t.symbol, t.ltp)),
        priority=100,
        name="replay.validation",
    )
    feed = ReplayFeed(
        bus, clock, dataset.rows,
        sub_steps=sub_steps, seed=seed, spread_bps=spread_bps,
    )
    while not feed.exhausted:
        feed.step()
        bus.drain(100000)

    tick_times = [x[0] for x in received]
    monotonic = all(b >= a for a, b in zip(tick_times, tick_times[1:]))
    return {
        "rows": dataset.row_count,
        "symbols": dataset.symbols,
        "ticks_emitted": len(received),
        "monotonic_timestamps": monotonic,
        "first_tick_ns": tick_times[0] if tick_times else 0,
        "last_tick_ns": tick_times[-1] if tick_times else 0,
        "duration_seconds": (tick_times[-1] - tick_times[0]) / NS if len(tick_times) > 1 else 0.0,
    }


def replay_csv(path: str | Path, **kwargs) -> dict:
    return replay_rows(load_nse_csv(path), **kwargs)
