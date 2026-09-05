"""Run the self-improvement loop against the real trading engine.

Demonstrates the whole ABULIDN cycle on live substrate:

  1. a proposal to change a strategy knob is written down;
  2. it is checked against the immutable risk boundaries;
  3. it is experimented on paired seeds through the trading engine;
  4. the statistical gate decides VALIDATED / REJECTED / INCONCLUSIVE;
  5. a VALIDATED change is promoted and its rollback recorded;
  6. a fresh set of seeds re-checks it, rolling back on regression.

It also shows the boundary in action: a second proposal that would loosen a
hard risk limit is refused before it can be measured.

Usage:  python3 tools/evolve_demo.py [days] [n_seeds] [nsym]
"""
from __future__ import annotations

import itertools
import json
import sys

sys.path.insert(0, ".")
from gmq.core.config import Config
from gmq.evolve import (Proposal, KnobChange, EvolutionController, ChangeJournal)
from gmq.evolve.trader import TradingEvaluator


def main(days: float = 0.6, n_seeds: int = 8, nsym: int = 6) -> None:
    cfg = Config()
    ev = TradingEvaluator(cfg, days=days, nsym=nsym)
    journal = ChangeJournal("runs/evolve/journal.jsonl")
    ctl = EvolutionController(ev, journal=journal, min_samples=6,
                             clock_fn=lambda: "")

    seeds = list(range(11, 11 + n_seeds))
    fresh = list(range(101, 101 + n_seeds))

    print("=" * 74)
    print("ABULIDN self-improvement loop  |  live trading substrate")
    print(f"{days} days x {n_seeds} paired seeds x {nsym} symbols")
    print("=" * 74)

    # ---- 1. a boundary violation is refused, never measured -------------
    print("\n[1] Proposal that would LOOSEN a hard risk limit:")
    bad = Proposal(
        id="E-BOUND", problem="drawdown cap feels tight",
        root_cause="(none -- this is the controller trying to relax safety)",
        changes=[KnobChange("risk.max_drawdown_pct", None, 12.0)],
        expected_benefit="more room to run", risk="catastrophic")
    ctl.submit(bad, current_values={"risk.max_drawdown_pct":
                                    cfg.risk.max_drawdown_pct})
    print(f"    state={bad.state.value}")
    print(f"    decision: {bad.decision}")

    # ---- 2. a real strategy proposal goes through the full loop ---------
    print("\n[2] Proposal to change a STRATEGY knob (exit hysteresis):")
    p = Proposal(
        id="E-101", problem="engine may still churn at the margin",
        root_cause="hysteresis bar may be too low to hold theses",
        changes=[KnobChange("search.exit_hysteresis", None, 1.5)],
        expected_benefit="fewer noise-driven exits, longer holds",
        risk="may hold losers slightly longer",
        metrics=["objective"], rollback_condition="objective regresses on fresh seeds",
        affected_components=["strategy.search"])
    ctl.submit(p, current_values={"search.exit_hysteresis":
                                  cfg.search.exit_hysteresis})
    print(f"    submitted, state={p.state.value}")
    print(f"    experimenting on seeds {seeds} ...", flush=True)
    res = ctl.experiment(p, seeds=seeds)
    print(f"    baseline  objective: "
          f"{[round(x, 2) for x in res.baseline]}")
    print(f"    candidate objective: "
          f"{[round(x, 2) for x in res.candidate]}")
    print(f"    verdict: {res.sig.verdict}  ({res.sig.note})")
    print(f"    {res.sig.as_dict()}")

    if p.state.value == "VALIDATED":
        ctl.promote(p)
        print(f"    -> PROMOTED. active config now: {ctl.active}")
        print(f"    monitoring on fresh seeds {fresh} ...", flush=True)
        rolled = ctl.check_rollback(p, seeds=fresh, tolerance=0.0)
        print(f"    rollback triggered: {rolled}  (state={p.state.value})")
    else:
        print("    -> not promoted; the evidence did not support it. "
              "That is the loop working, not failing.")

    print("\n" + "=" * 74)
    print("change journal:")
    for e in journal.entries:
        print(f"  {e['id']:8s} {e['event']:22s} state={e['state']}")
    print("\njournal persisted to runs/evolve/journal.jsonl")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(days=float(a[0]) if a else 0.6,
         n_seeds=int(a[1]) if len(a) > 1 else 8,
         nsym=int(a[2]) if len(a) > 2 else 6)
