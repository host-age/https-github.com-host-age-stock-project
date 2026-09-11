#!/usr/bin/env python3
"""Run the validated NSE historical replay data path and emit JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmq.backtest.nse_replay import replay_csv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--sub-steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--spread-bps", type=float, default=2.0)
    p.add_argument("--output", default="runs/nse_replay.json")
    args = p.parse_args()
    result = replay_csv(args.csv, sub_steps=max(4, args.sub_steps), seed=args.seed, spread_bps=args.spread_bps)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
