#!/usr/bin/env python3
"""Summarize an NSE replay JSON result plus optional trade metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("result", help="JSON emitted by scripts/replay_market.py")
    args = p.parse_args()
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    print("GMQ NSE REPLAY REPORT")
    print(f"Rows: {data.get('rows', 0)}")
    print(f"Symbols: {', '.join(data.get('symbols', []))}")
    print(f"Ticks emitted: {data.get('ticks_emitted', 0)}")
    print(f"Monotonic timestamps: {data.get('monotonic_timestamps', False)}")
    print(f"Replay duration (s): {data.get('duration_seconds', 0.0):.3f}")
    print("\nThis report is a data-path validation. Trading performance metrics are added when the replay is wired to the full TradingEngine strategy loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
