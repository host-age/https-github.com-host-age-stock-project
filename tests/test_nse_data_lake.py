from pathlib import Path

from gmq.data.nse.archive import NSEArchive
from gmq.data.nse.client import NSEDataClient
from gmq.data.nse.normalize import normalize_csv_file
from gmq.knowledge.point_in_time import PointInTimeKnowledge


def write_csv(path: Path, rows: str) -> None:
    path.write_text(
        "timestamp,symbol,open,high,low,close,volume\n" + rows,
        encoding="utf-8",
    )


def test_normalize_archive_and_point_in_time(tmp_path):
    src = tmp_path / "nse.csv"
    write_csv(
        src,
        "2026-09-10T09:15:00+05:30,TCS,100,101,99,100.5,1000\n"
        "2026-09-10T09:16:00+05:30,TCS,100.5,102,100,101.5,1200\n"
        "2026-09-10T09:17:00+05:30,TCS,101.5,103,101,102.5,1400\n",
    )
    ds = normalize_csv_file(src)
    db = NSEArchive(str(tmp_path / "nse.sqlite3"))
    try:
        assert db.upsert(ds.rows) == 3
        snap = PointInTimeKnowledge(db).market("TCS", ds.rows[1].ts)
        assert len(snap.bars) == 2
        assert snap.bars[-1].c == 101.5
        assert all(r.ts <= ds.rows[1].ts for r in snap.bars)
    finally:
        db.close()


def test_client_uses_cache_and_hash(tmp_path):
    client = NSEDataClient(cache_dir=str(tmp_path))
    result = client.save_bytes("fixture://nse", "2026-09-10.bin", b"abc")
    assert result.bytes == 3
    assert len(result.sha256) == 64


def test_normalizer_maps_common_nse_aliases(tmp_path):
    src = tmp_path / "alias.csv"
    src.write_text(
        "TRADE_DATE,SERIES_SYMBOL,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TOTTRDQTY\n"
        "2026-09-10T09:15:00+05:30,TCS,100,101,99,100.5,1000\n",
        encoding="utf-8",
    )
    ds = normalize_csv_file(src)
    assert ds.symbols == ["TCS"]
    assert ds.rows[0].v == 1000
