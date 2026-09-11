#!/usr/bin/env python3
"""Run the GMQ historical NSE replay data-path validation.

Usage:
  python scripts/replay_nse.py path/to/nse.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow execution as `python scripts/replay_nse.py ...` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmq.backtest.nse_replay import replay_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay NSE OHLCV through GMQ's ReplayFeed")
    parser.add_argument("csv", help="CSV with timestamp,symbol,open,high,low,close,volume")
    parser.add_argument("--sub-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spread-bps", type=float, default=2.0)
    args = parser.parse_args()

    result = replay_csv(
        args.csv,
        sub_steps=max(4, args.sub_steps),
        seed=args.seed,
        spread_bps=args.spread_bps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
