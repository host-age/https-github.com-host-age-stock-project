"""The explainable decision log (spec §7, §13).

Every decision the engine makes is written as a structured record:

    Market State -> Data Signals -> Model Predictions -> Alternative Scenarios
                 -> Risk Assessment -> Trade Decision -> Execution -> Outcome

Two stores, deliberately:

* **JSONL** -- append-only, one record per line, never rewritten. This is the
  audit trail. An append-only file cannot be quietly amended after a bad day,
  which is the entire point of an audit trail.
* **SQLite** -- indexed, queryable, for the analytics layer and dashboard.

The outcome section is filled in later, when the position finally closes, and
is linked back by `decision_id`. That link is what makes it possible to ask the
question that actually matters: *when the model said 70%, how often was it
right, and did the trades it was right about make money?*
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from ..core.types import DecisionRecord, NS


SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    ts          INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    regime      TEXT,
    regime_conf REAL,
    move        TEXT,
    qty         INTEGER,
    price       REAL,
    stop        REAL,
    target      REAL,
    ev          REAL,
    score       REAL,
    p_win       REAL,
    confidence  REAL,
    p_up        REAL,
    risk_ok     INTEGER,
    risk_reason TEXT,
    search_ms   REAL,
    nodes       INTEGER,
    rationale   TEXT,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dec_ts     ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_dec_symbol ON decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_dec_move   ON decisions(move);

CREATE TABLE IF NOT EXISTS outcomes (
    decision_id  TEXT PRIMARY KEY,
    closed_ts    INTEGER,
    pnl          REAL,
    r_multiple   REAL,
    mfe          REAL,
    mae          REAL,
    hold_s       REAL,
    exit_reason  TEXT,
    entry_px     REAL,
    exit_px      REAL,
    slippage_bps REAL,
    thesis_correct INTEGER,
    loss_cause   TEXT,
    payload      TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      INTEGER,
    kind    TEXT,
    symbol  TEXT,
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS idx_ev_ts   ON events(ts);
CREATE INDEX IF NOT EXISTS idx_ev_kind ON events(kind);
"""


def new_decision_id() -> str:
    return uuid.uuid4().hex[:16]


class DecisionJournal:
    def __init__(self, run_dir: str, run_id: str = "run",
                 write_jsonl: bool = True, flush_every: int = 50):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.run_id = run_id
        self.jsonl_path = os.path.join(run_dir, f"{run_id}_decisions.jsonl")
        self.db_path = os.path.join(run_dir, f"{run_id}.sqlite")
        self._fh = open(self.jsonl_path, "a", buffering=1 << 16) \
            if write_jsonl else None
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._lock = threading.Lock()
        self._since_flush = 0
        self.flush_every = flush_every
        self.n_decisions = 0
        self.n_outcomes = 0

    # ------------------------------------------------------------------
    def record(self, rec: DecisionRecord) -> None:
        payload = rec.to_dict()
        line = json.dumps(payload, separators=(",", ":"), default=str)
        chosen = rec.chosen or {}
        risk = rec.risk or {}
        preds = rec.predictions or {}
        with self._lock:
            if self._fh:
                self._fh.write(line + "\n")
            self._db.execute(
                "INSERT OR REPLACE INTO decisions VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.decision_id, rec.ts, rec.symbol, rec.regime,
                 rec.regime_conf, chosen.get("move"), chosen.get("qty"),
                 chosen.get("price"), chosen.get("stop"), chosen.get("target"),
                 chosen.get("ev"), chosen.get("score"), chosen.get("p_win"),
                 preds.get("confidence"), preds.get("p_up"),
                 1 if risk.get("approved") else 0, risk.get("reason"),
                 rec.search_ms, rec.nodes, rec.rationale, line))
            self.n_decisions += 1
            self._since_flush += 1
            if self._since_flush >= self.flush_every:
                self._db.commit()
                self._since_flush = 0

    # ------------------------------------------------------------------
    def record_outcome(self, decision_id: str, outcome: Dict[str, Any]) -> None:
        if not decision_id:
            return
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (decision_id, outcome.get("closed_ts", 0),
                 outcome.get("pnl", 0.0), outcome.get("r_multiple", 0.0),
                 outcome.get("mfe", 0.0), outcome.get("mae", 0.0),
                 outcome.get("hold_s", 0.0), outcome.get("exit_reason", ""),
                 outcome.get("entry_px", 0.0), outcome.get("exit_px", 0.0),
                 outcome.get("slippage_bps", 0.0),
                 1 if outcome.get("thesis_correct") else 0,
                 outcome.get("loss_cause", ""),
                 json.dumps(outcome, default=str)))
            self.n_outcomes += 1
            if self._fh:
                self._fh.write(json.dumps(
                    {"type": "outcome", "decision_id": decision_id,
                     **outcome}, default=str) + "\n")

    def event(self, ts: int, kind: str, symbol: str = "",
              detail: Any = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO events (ts, kind, symbol, detail) VALUES (?,?,?,?)",
                (ts, kind, symbol,
                 json.dumps(detail, default=str) if detail is not None else ""))

    # ------------------------------------------------------------------
    def flush(self) -> None:
        with self._lock:
            self._db.commit()
            if self._fh:
                self._fh.flush()
            self._since_flush = 0

    def close(self) -> None:
        self.flush()
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None
            self._db.close()

    # ------------------------------------------------------------------
    def query(self, sql: str, args: Iterable = ()) -> List[tuple]:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    def decisions_for(self, symbol: str, limit: int = 50) -> List[dict]:
        rows = self.query(
            "SELECT payload FROM decisions WHERE symbol=? "
            "ORDER BY ts DESC LIMIT ?", (symbol, limit))
        return [json.loads(r[0]) for r in rows]

    def recent(self, limit: int = 50, acted_only: bool = False) -> List[dict]:
        sql = "SELECT payload FROM decisions"
        if acted_only:
            sql += " WHERE move IS NOT NULL AND move != 'WAIT'"
        sql += " ORDER BY ts DESC LIMIT ?"
        return [json.loads(r[0]) for r in self.query(sql, (limit,))]

    def calibration_report(self, bins: int = 10) -> List[dict]:
        """The question the journal exists to answer: when the engine said
        70%, how often was it right?"""
        rows = self.query(
            "SELECT d.p_win, o.pnl, o.r_multiple FROM decisions d "
            "JOIN outcomes o ON d.decision_id = o.decision_id "
            "WHERE d.p_win IS NOT NULL AND d.move NOT IN ('WAIT','MOVE_STOP')")
        if not rows:
            return []
        out = []
        for i in range(bins):
            lo, hi = i / bins, (i + 1) / bins
            sel = [r for r in rows if r[0] is not None and lo <= r[0] < hi]
            if len(sel) < 3:
                continue
            wins = sum(1 for r in sel if (r[1] or 0) > 0)
            out.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "predicted": round(sum(r[0] for r in sel) / len(sel), 3),
                "observed": round(wins / len(sel), 3),
                "n": len(sel),
                "avg_r": round(sum(r[2] or 0 for r in sel) / len(sel), 3),
            })
        return out

    def summary(self) -> dict:
        d = self.query("SELECT COUNT(*), SUM(CASE WHEN move!='WAIT' THEN 1 ELSE 0 END) "
                       "FROM decisions")[0]
        o = self.query("SELECT COUNT(*), SUM(pnl), AVG(r_multiple) FROM outcomes")[0]
        return {
            "decisions": d[0] or 0,
            "actions": d[1] or 0,
            "outcomes": o[0] or 0,
            "total_pnl": round(o[1] or 0.0, 2),
            "avg_r": round(o[2] or 0.0, 3),
            "jsonl": self.jsonl_path,
            "db": self.db_path,
        }
