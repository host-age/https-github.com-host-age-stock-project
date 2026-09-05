"""A/B the churn fixes against the build they replaced.

The rule this repository has held itself to is that a change is not an
improvement because it is well argued. The walk-forward and Monte Carlo
machinery exists to test whether a change helps or is another curve fit, and
the four changes made to stop the engine churning have to face the same test
as anything else.

Design:

* **Paired on seed.** Both arms see the identical simulated market -- same
  seed, same start, same universe, same number of days. Comparing arms across
  different market draws would confound the change with the weather, and the
  seed-to-seed variance here is far larger than the effect being measured.
* **Both arms are the same code.** The baseline is not an old checkout; it is
  the current build with the four switches turned off. That removes "something
  else also changed" as an explanation.
* **Every arm reports the diagnosis, not just the P&L.** The fixes were aimed
  at a specific pathology -- fifteen-second holds and profits handed back --
  so the report has to show whether that pathology moved, independently of
  whether the P&L did. A change that fixes the mechanism and loses money is a
  different situation from one that never fixed the mechanism at all.

Usage:  python3 tools/ab_test.py [days] [n_seeds] [n_symbols]
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np

sys.path.insert(0, ".")
from gmq.core.config import Config
from gmq.core.clock import IST
from gmq.app.engine import TradingEngine
from gmq.analytics import metrics


# The four switches under test, and what each one does.
ARMS: Dict[str, dict] = {
    # the build as it was: no fee charging path in the evaluator, no
    # hysteresis, fresh dice every re-evaluation, instant re-entry, no
    # excursion-fitted exit
    "before": dict(liquidation_in_eval=False, exit_hysteresis=0.0,
                   common_random_numbers=False, reentry_cooldown_s=0.0,
                   giveback_enabled=False),
    "after": dict(liquidation_in_eval=True, exit_hysteresis=1.0,
                  common_random_numbers=True, reentry_cooldown_s=120.0,
                  giveback_enabled=True),
}


def run_one(arm: str, seed: int, days: float, nsym: int) -> dict:
    cfg = Config(run_id=f"ab_{arm}_{seed}", seed=seed)
    cfg.data.symbols = cfg.data.symbols[:nsym]
    for k, v in ARMS[arm].items():
        setattr(cfg.search, k, v)
    # Deterministic search: the wall clock would let a busy machine explore a
    # different tree in one arm than the other, which is precisely the
    # confound this harness exists to remove.
    cfg.search.node_budget = 900
    eng = TradingEngine(cfg, run_dir=f"runs/ab/{arm}_{seed}",
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                        journal=False)
    t0 = time.time()
    rep = eng.run(days=days, dt=0.5)
    trades = list(eng.portfolio.trades)
    st = rep["trades"]
    exc = eng.excursions.stats()

    # holding-period P&L buckets: the shape that showed short holds losing
    buckets = {"0-10s": [], "10-30s": [], "30-120s": [], "120s+": []}
    for t in trades:
        net = t.pnl - t.fees
        h = t.hold_s
        key = ("0-10s" if h < 10 else "10-30s" if h < 30
               else "30-120s" if h < 120 else "120s+")
        buckets[key].append(net)

    return {
        "arm": arm, "seed": seed, "wall_s": round(time.time() - t0, 1),
        "return_pct": rep["equity"].get("total_return_pct", 0.0),
        "net_pnl": round(sum(t.pnl - t.fees for t in trades), 2),
        "trades": st.get("trades", 0),
        "win_rate": st.get("win_rate", 0.0),
        "expectancy_r": st.get("expectancy_r"),
        "profit_factor": st.get("profit_factor", 0.0),
        "median_hold_s": st.get("median_hold_s", 0.0),
        "max_dd_pct": rep["equity"].get("max_drawdown_pct", 0.0),
        "sharpe": rep["equity"].get("sharpe", 0.0),
        "fees": round(eng.portfolio.total_fees, 2),
        "decisions": rep["decisions"],
        "actions": rep["actions"],
        "losers_in_profit_first": round(exc.get("losers_in_profit_first", 0.0), 3),
        "capture_ratio": round(exc.get("capture_ratio", 0.0), 3),
        "hysteresis_holds": eng.search.stats.hysteresis_holds,
        "giveback": eng.excursions.policy.as_dict(),
        "pnl_by_hold": {k: [len(v), round(float(np.sum(v)), 1)]
                        for k, v in buckets.items()},
    }


def paired_bootstrap(before: List[float], after: List[float],
                     n_boot: int = 20000, seed: int = 3) -> float:
    """P(the improvement is <= 0), resampling seeds in pairs.

    With a handful of seeds this will almost never reach significance, and
    that is the honest answer rather than a defect of the test: four paired
    observations cannot establish a small effect. It is reported so the size
    of the claim matches the size of the evidence.
    """
    d = np.array(after) - np.array(before)
    if d.size < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    return float((d[idx].mean(axis=1) <= 0).mean())


def main(days: float = 1.5, n_seeds: int = 4, nsym: int = 6) -> None:
    seeds = [11, 23, 41, 67, 89, 103][:n_seeds]
    rows: List[dict] = []
    for seed in seeds:
        for arm in ("before", "after"):
            r = run_one(arm, seed, days, nsym)
            rows.append(r)
            print(f"  {arm:6s} seed={seed:3d} "
                  f"ret={r['return_pct']:+6.2f}% trades={r['trades']:3d} "
                  f"hold={r['median_hold_s']:7.1f}s "
                  f"pf={r['profit_factor']:5.2f} "
                  f"giveback_in_profit={r['losers_in_profit_first']:.2f} "
                  f"({r['wall_s']}s)", flush=True)

    os.makedirs("runs/ab", exist_ok=True)
    with open("runs/ab/results.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)

    def col(arm, key):
        return [r[key] for r in rows if r["arm"] == arm]

    print("\n" + "=" * 74)
    print(f"A/B  |  {days} days x {len(seeds)} paired seeds x {nsym} symbols")
    print("=" * 74)
    hdr = f"{'metric':<26}{'before':>13}{'after':>13}{'delta':>13}"
    print(hdr); print("-" * 74)
    for key, fmt, better in (
            ("return_pct", "{:+.2f}%", "up"),
            ("net_pnl", "{:+,.0f}", "up"),
            ("trades", "{:.1f}", ""),
            ("win_rate", "{:.3f}", "up"),
            ("expectancy_r", "{:+.3f}", "up"),
            ("profit_factor", "{:.2f}", "up"),
            ("median_hold_s", "{:.1f}s", "up"),
            ("max_dd_pct", "{:.2f}%", "down"),
            ("fees", "{:,.0f}", ""),
            ("actions", "{:.0f}", ""),
            ("losers_in_profit_first", "{:.3f}", "down"),
            ("capture_ratio", "{:.3f}", "up"),
    ):
        b = [v for v in col("before", key) if v is not None]
        a = [v for v in col("after", key) if v is not None]
        if not b or not a:
            continue
        mb, ma = float(np.mean(b)), float(np.mean(a))
        mark = ""
        if better:
            good = (ma > mb) if better == "up" else (ma < mb)
            mark = "  ok" if good else "  --"
        print(f"{key:<26}{fmt.format(mb):>13}{fmt.format(ma):>13}"
              f"{fmt.format(ma - mb):>13}{mark}")

    p = paired_bootstrap(col("before", "net_pnl"), col("after", "net_pnl"))
    print("-" * 74)
    print(f"paired bootstrap on net P&L: P(improvement <= 0) = {p:.3f} "
          f"over {len(seeds)} seeds")
    print("(with this few paired seeds, treat anything short of a very large "
          "effect as unproven.)")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(days=float(a[0]) if a else 1.5,
         n_seeds=int(a[1]) if len(a) > 1 else 4,
         nsym=int(a[2]) if len(a) > 2 else 6)
