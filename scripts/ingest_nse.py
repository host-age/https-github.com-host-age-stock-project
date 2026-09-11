#!/usr/bin/env python3
"""Normalize one or more NSE-compatible files into the GMQ archive."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmq.data.nse.archive import NSEArchive
from gmq.data.nse.normalize import normalize_csv_file


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="NSE-compatible CSV files")
    p.add_argument("--archive", default="data/archive/nse.sqlite3")
    args = p.parse_args()
    archive = NSEArchive(args.archive)
    total = 0
    summaries = []
    try:
        for path in args.files:
            ds = normalize_csv_file(path)
            count = archive.upsert(ds.rows)
            total += count
            summaries.append({"file": path, "rows": ds.row_count, "symbols": len(ds.symbols)})
    finally:
        archive.close()
    print(json.dumps({"total_rows": total, "files": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
