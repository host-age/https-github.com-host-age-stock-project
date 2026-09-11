#!/usr/bin/env python3
"""Run GMQ against live Kite market data with paper-only execution.

Required environment:
  KITE_API_KEY
  KITE_ACCESS_TOKEN
Optional:
  SYMBOLS=RELIANCE,TCS,INFY
  START_CAPITAL=1000000
  PORT=10000

This launcher never enables the real Kite order adapter. It starts the existing
server, which uses KiteTicker for live quotes and the GMQ paper broker for
simulated fills against those quotes.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    required = ["KITE_API_KEY", "KITE_ACCESS_TOKEN"]
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        print("Missing required environment variables: " + ", ".join(missing), file=sys.stderr)
        return 2

    os.environ["LIVE_TRADING_ENABLED"] = "0"
    os.environ["KITE_LIVE_ORDERS"] = "0"

    from gmq.app.server import main as server_main
    server_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
