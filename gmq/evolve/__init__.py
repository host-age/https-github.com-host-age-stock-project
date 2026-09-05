"""Controlled, measurable, reversible self-improvement (the ABULIDN loop).

This subpackage is the trading engine turned on itself. Everywhere else in
the system measures whether a *trade* was good; this measures whether a
*change to the system* was good, and holds that judgement to the same standard
the rest of the codebase holds a backtest to: nothing is believed because it
is newer or because it was well argued, only because paired, out-of-sample
evidence says it beat what it replaced.

The loop, and where each stage lives:

    OBSERVE / DIAGNOSE   analytics.metrics, the run reports (upstream)
    PROPOSE              proposal.Proposal
    boundary check       boundaries.KnobRegistry  (immutable risk layer)
    EXPERIMENT           controller.EvolutionController.experiment
    VERIFY (statistics)  stats.significance_gate
    PROMOTE / REJECT     controller.EvolutionController
    MONITOR / ROLLBACK   controller.EvolutionController.check_rollback
    journal              proposal.ChangeJournal  (every step, persisted)

The one rule that makes it safe rather than merely clever: the controller can
propose and test changes to the *strategy* -- search depth, objective weights,
exit rules, model routing -- but it has no path to weaken the *hard risk
limits*. That boundary is enforced in code here, mirroring the risk engine the
model layer already cannot override. A self-improving system that can relax its
own safety limits is not self-improving, it is self-endangering.
"""
from .proposal import (Proposal, ProposalState, ChangeJournal, KnobChange)
from .boundaries import KnobRegistry, BoundaryError, KnobSpec
from .stats import significance_gate, paired_bootstrap, SigResult
from .controller import EvolutionController, ExperimentResult

__all__ = [
    "Proposal", "ProposalState", "ChangeJournal", "KnobChange",
    "KnobRegistry", "BoundaryError", "KnobSpec",
    "significance_gate", "paired_bootstrap", "SigResult",
    "EvolutionController", "ExperimentResult",
]
