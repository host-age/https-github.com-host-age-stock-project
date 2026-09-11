"""Normalize NSE-compatible tabular files into GMQ replay rows."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

from ...backtest.nse_replay import ReplayDataError, ReplayDataset
from ..feed import ReplayRow

ALIASES = {
    "timestamp": ("timestamp", "ts", "date", "datetime", "trade_date"),
    "symbol": ("symbol", "ticker", "tradingsymbol", "series_symbol"),
    "open": ("open", "open_price", "openprice"),
    "high": ("high", "high_price", "highprice"),
    "low": ("low", "low_price", "lowprice"),
    "close": ("close", "close_price", "closeprice", "last_price", "ltp"),
    "volume": ("volume", "volume_traded", "tottrdqty", "total_traded_quantity"),
}


def _pick(fieldnames: Iterable[str], names: tuple[str, ...], required: bool = True) -> str | None:
    lookup = {str(x).strip().lower(): x for x in fieldnames}
    for name in names:
        if name in lookup:
            return lookup[name]
    if required:
        raise ReplayDataError(f"missing required field; accepted aliases: {names}")
    return None


def normalize_csv_file(path: str | Path) -> ReplayDataset:
    """Normalize one CSV into the canonical GMQ replay dataset.

    Timestamps must already be non-decreasing in the source file. We reject
    out-of-order data rather than sorting it, because silently reordering a
    partially corrupt or mis-exported dataset can hide a point-in-time error in
    historical validation.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        cols = {k: _pick(fields, v, k != "volume") for k, v in ALIASES.items()}
        rows: list[ReplayRow] = []
        prev_ts: int | None = None
        for line_no, raw in enumerate(reader, start=2):
            try:
                ts = raw[cols["timestamp"]]
                from ...backtest.nse_replay import _parse_ts
                ts_ns = _parse_ts(ts)
                if prev_ts is not None and ts_ns < prev_ts:
                    raise ReplayDataError("timestamps out of order")
                prev_ts = ts_ns

                symbol = raw[cols["symbol"]].strip().upper()
                o = float(raw[cols["open"]])
                h = float(raw[cols["high"]])
                l = float(raw[cols["low"]])
                c = float(raw[cols["close"]])
                v = int(float(raw[cols["volume"]])) if cols["volume"] else 0

                if not symbol or not all(math.isfinite(x) for x in (o, h, l, c)):
                    raise ReplayDataError("invalid security row")
                if min(o, h, l, c) <= 0 or h < max(o, c) or l > min(o, c) or h < l or v < 0:
                    raise ReplayDataError("invalid security row")
                rows.append(ReplayRow(ts_ns, symbol, o, h, l, c, v))
            except Exception as exc:
                raise ReplayDataError(f"line {line_no}: {exc}") from exc

    if not rows:
        raise ReplayDataError("no data rows")
    return ReplayDataset(
        rows,
        sorted({r.symbol for r in rows}),
        rows[0].ts,
        rows[-1].ts,
    )
