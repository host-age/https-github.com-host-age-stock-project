from pathlib import Path

import pytest

from gmq.backtest.nse_replay import ReplayDataError, load_nse_csv, replay_csv


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_nse_csv_accepts_iso_and_preserves_order(tmp_path):
    p = tmp_path / "nse.csv"
    _write(
        p,
        "timestamp,symbol,open,high,low,close,volume\n"
        "2026-09-10T09:15:00+05:30,TCS,100,102,99,101,1000\n"
        "2026-09-10T09:16:00+05:30,TCS,101,103,100,102,1200\n",
    )
    ds = load_nse_csv(p)
    assert ds.row_count == 2
    assert ds.symbols == ["TCS"]
    assert ds.first_ts < ds.last_ts


def test_rejects_out_of_order_rows(tmp_path):
    p = tmp_path / "bad.csv"
    _write(
        p,
        "timestamp,symbol,open,high,low,close,volume\n"
        "2026-09-10T09:16:00+05:30,TCS,101,103,100,102,1200\n"
        "2026-09-10T09:15:00+05:30,TCS,100,102,99,101,1000\n",
    )
    with pytest.raises(ReplayDataError, match="out of order"):
        load_nse_csv(p)


def test_replay_emits_monotonic_ticks(tmp_path):
    p = tmp_path / "nse.csv"
    _write(
        p,
        "timestamp,symbol,open,high,low,close,volume\n"
        "2026-09-10T09:15:00+05:30,TCS,100,102,99,101,1000\n"
        "2026-09-10T09:16:00+05:30,TCS,101,103,100,102,1200\n"
        "2026-09-10T09:17:00+05:30,TCS,102,104,101,103,1400\n",
    )
    rep = replay_csv(p, sub_steps=4, seed=1, spread_bps=2.0)
    assert rep["rows"] == 3
    assert rep["ticks_emitted"] == 12
    assert rep["monotonic_timestamps"] is True
