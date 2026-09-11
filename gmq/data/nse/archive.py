"""SQLite-backed normalized NSE archive with point-in-time reads."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from ..feed import ReplayRow


class NSEArchive:
    def __init__(self, path: str = "data/archive/nse.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS bars (ts_ns INTEGER NOT NULL, symbol TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume INTEGER NOT NULL, PRIMARY KEY(ts_ns, symbol))"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON bars(symbol, ts_ns)")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def upsert(self, rows: Iterable[ReplayRow]) -> int:
        values = [(r.ts, r.symbol, r.o, r.h, r.l, r.c, r.v) for r in rows]
        self.db.executemany(
            "INSERT OR REPLACE INTO bars(ts_ns,symbol,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
            values,
        )
        self.db.commit()
        return len(values)

    def snapshot(self, symbol: str, at_ts_ns: int, limit: int = 500) -> list[ReplayRow]:
        cur = self.db.execute(
            "SELECT ts_ns,symbol,open,high,low,close,volume FROM bars WHERE symbol=? AND ts_ns<=? ORDER BY ts_ns DESC LIMIT ?",
            (symbol.upper(), at_ts_ns, limit),
        )
        rows = [ReplayRow(*r) for r in cur.fetchall()]
        rows.reverse()
        return rows

    def range(self, start_ts_ns: int, end_ts_ns: int, symbol: str | None = None) -> list[ReplayRow]:
        if symbol:
            cur = self.db.execute(
                "SELECT ts_ns,symbol,open,high,low,close,volume FROM bars WHERE symbol=? AND ts_ns BETWEEN ? AND ? ORDER BY ts_ns,symbol",
                (symbol.upper(), start_ts_ns, end_ts_ns),
            )
        else:
            cur = self.db.execute(
                "SELECT ts_ns,symbol,open,high,low,close,volume FROM bars WHERE ts_ns BETWEEN ? AND ? ORDER BY ts_ns,symbol",
                (start_ts_ns, end_ts_ns),
            )
        return [ReplayRow(*r) for r in cur.fetchall()]

    def symbols(self) -> list[str]:
        cur = self.db.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol")
        return [r[0] for r in cur.fetchall()]
