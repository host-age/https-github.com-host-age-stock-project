"""Autonomous self-improvement run (the ABULIDN loop, unattended).

Given a batch of candidate improvements, this drives each one through the full
loop without a human in between: boundary check -> paired experiment ->
statistical gate -> promote or reject -> journal. Promoted changes accumulate
into a live config, and each later proposal is tested on top of what has
already been promoted, so the run compounds improvements rather than testing
each in isolation against the original baseline.

It is deliberately conservative in exactly the ways the charter demands:

  * a proposal that would loosen a hard risk limit is refused, not tested;
  * a proposal is promoted only if it beats the current live config on paired
    seeds at the configured significance;
  * an INCONCLUSIVE result promotes nothing -- the honest answer when the
    evidence is too thin, not a coin flip;
  * everything, including every rejection, is written to the journal.

The candidate list is where judgement still lives: a human (or a smarter
proposer) decides *what* to try. The loop decides, with evidence, what to
keep. That division is the whole point -- the machine is trusted to measure,
not to imagine.

Usage:  python3 tools/automate.py [days] [n_seeds] [nsym]
"""
from __future__ import annotations

import json
import sys
import time
from typing import List

sys.path.insert(0, ".")
from gmq.core.config import Config
from gmq.evolve import (Proposal, KnobChange, EvolutionController, ChangeJournal)
from gmq.evolve.trader import TradingEvaluator


# The batch. Each is a hypothesis about how to convert the models' measured
# directional edge into more profit, expressed as a change to a tunable knob.
# None of these can touch a hard risk limit -- the registry forbids it.
def candidates() -> List[Proposal]:
    return [
        Proposal(
            id="A-01", problem="entries may be too timid to express the edge",
            root_cause="min_edge_bps set above where the 5m model's skill pays",
            changes=[KnobChange("search.min_edge_bps", None, 4.0)],
            expected_benefit="more of the real edge is acted on",
            risk="more marginal trades; costs may eat them",
            metrics=["objective"], rollback_condition="objective regresses",
            affected_components=["strategy.moves"]),
        Proposal(
            id="A-02", problem="risk aversion may be suppressing good bets",
            root_cause="gamma on variance too high for the account size",
            changes=[KnobChange("search.risk_aversion", None, 1.2)],
            expected_benefit="edge-positive trades sized closer to Kelly",
            risk="higher variance of returns",
            metrics=["objective"], rollback_condition="drawdown worsens"),
        Proposal(
            id="A-03", problem="holding winners too briefly",
            root_cause="give-back keep-fraction protects too little of the run",
            changes=[KnobChange("search.giveback_keep_fraction", None, 0.65)],
            expected_benefit="more of each winner's excursion is captured",
            risk="gives back more before stopping out",
            metrics=["objective"], rollback_condition="capture ratio falls"),
        Proposal(
            id="A-04", problem="deeper look-ahead may find better lines",
            root_cause="max_depth=3 may miss option value at the 900s horizon",
            changes=[KnobChange("search.max_depth", None, 4)],
            expected_benefit="better exits via more plies of look-ahead",
            risk="slower; may overfit the scenario model",
            metrics=["objective"], rollback_condition="no objective gain"),
        Proposal(
            id="A-05", problem="drawdown penalty may be over-braking",
            root_cause="dd_penalty tuned for churn, not for a real edge",
            changes=[KnobChange("search.dd_penalty", None, 0.05)],
            expected_benefit="less suppression of edge-positive risk",
            risk="deeper drawdowns",
            metrics=["objective"], rollback_condition="max_dd worsens materially"),
    ]


def main(days: float = 0.6, n_seeds: int = 8, nsym: int = 6) -> None:
    cfg = Config()
    ev = TradingEvaluator(cfg, days=days, nsym=nsym)
    journal = ChangeJournal("runs/evolve/auto_journal.jsonl")
    ctl = EvolutionController(ev, journal=journal, min_samples=6,
                             clock_fn=lambda: "")

    fit_seeds = list(range(11, 11 + n_seeds))
    oos_seeds = list(range(201, 201 + n_seeds))

    print("=" * 74)
    print("AUTONOMOUS SELF-IMPROVEMENT RUN")
    print(f"{days} days x {n_seeds} paired seeds x {nsym} symbols")
    print(f"fit seeds {fit_seeds[0]}..{fit_seeds[-1]}  |  "
          f"oos seeds {oos_seeds[0]}..{oos_seeds[-1]}")
    print("=" * 74)

    promoted, rejected, inconclusive = [], [], []
    t0 = time.time()

    for p in candidates():
        cur = cfg
        # current live value of the knob, so a guarded change can be checked
        path = p.changes[0].path
        obj = cfg
        for part in path.split(".")[:-1]:
            obj = getattr(obj, part)
        current = getattr(obj, path.split(".")[-1])
        ctl.submit(p, current_values={path: current})
        if p.state.value == "REJECTED":
            print(f"\n[{p.id}] {path} -> {p.changes[0].after}: REFUSED "
                  f"({p.decision})")
            rejected.append(p.id)
            continue
        print(f"\n[{p.id}] {path}: {current} -> {p.changes[0].after}   "
              f"experimenting...", flush=True)
        res = ctl.experiment(p, seeds=fit_seeds)
        print(f"        {res.sig.verdict}: {res.sig.note}")
        if p.state.value == "VALIDATED":
            ctl.promote(p)
            # out-of-sample confirmation before we keep it
            rolled = ctl.check_rollback(p, seeds=oos_seeds, tolerance=0.0)
            if rolled:
                print(f"        promoted, then ROLLED BACK on fresh seeds "
                      f"(curve fit)")
                rejected.append(p.id)
            else:
                print(f"        PROMOTED and confirmed OOS. live config: "
                      f"{ctl.active}")
                promoted.append(p.id)
        elif p.state.value == "REJECTED":
            rejected.append(p.id)
        else:
            inconclusive.append(p.id)

    wall = time.time() - t0
    print("\n" + "=" * 74)
    print(f"RESULT after {wall:.0f}s")
    print(f"  promoted     : {promoted or '(none)'}")
    print(f"  rejected     : {rejected or '(none)'}")
    print(f"  inconclusive : {inconclusive or '(none)'}")
    print(f"\n  final live config: {json.dumps(ctl.active, indent=2)}")
    print(f"\n  journal: runs/evolve/auto_journal.jsonl "
          f"({len(journal.entries)} entries)")

    # persist the winning config so a later run / the engine can load it
    if ctl.active:
        with open("runs/evolve/promoted_config.json", "w") as fh:
            json.dump(ctl.active, fh, indent=2)
        print("  winning knobs saved to runs/evolve/promoted_config.json")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(days=float(a[0]) if a else 0.6,
         n_seeds=int(a[1]) if len(a) > 1 else 8,
         nsym=int(a[2]) if len(a) > 2 else 6)
