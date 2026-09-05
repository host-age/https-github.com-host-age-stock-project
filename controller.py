"""The evolution controller -- the state machine that runs the loop.

It ties the pieces together: a proposal is checked against the immutable
boundaries, experimented on paired seeds through a pluggable evaluator, judged
by the statistical gate, and either promoted (with its rollback recorded) or
rejected. Every step is journalled.

The controller does not know anything about trading. It knows about knobs,
paired experiments, and evidence. The trading-specific part -- actually
running the engine over seeds and returning a number -- is injected as an
`evaluator`, so the exact same controller can later drive any other system
that exposes the same tiny interface. That separation is deliberate: the
governance is the reusable asset (spec's whole point), the substrate is not.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .boundaries import KnobRegistry, BoundaryError
from .proposal import (Proposal, ProposalState, ChangeJournal, KnobChange,
                       can_transition)
from .stats import significance_gate, SigResult


class Evaluator(Protocol):
    """The one thing a substrate must provide.

    Given a config patch (a {knob_path: value} mapping, empty for the
    baseline) and a list of seeds, run the system once per seed and return the
    objective value for each -- higher meaning better. The controller supplies
    the *same* seeds to both arms so the comparison is paired.
    """
    def __call__(self, patch: Dict[str, Any],
                 seeds: Sequence[int]) -> List[float]: ...


@dataclass
class ExperimentResult:
    proposal_id: str
    baseline: List[float]
    candidate: List[float]
    sig: SigResult
    guarded_metrics: Dict[str, Any] = field(default_factory=dict)


def _clock_ts(clock_fn: Optional[Callable[[], str]]) -> str:
    return clock_fn() if clock_fn else ""


class EvolutionController:
    """Drives PROPOSED -> EXPERIMENT -> VALIDATED -> PROMOTED / REJECTED.

    `clock_fn` returns a timestamp string; it is injected rather than called
    directly so the controller stays deterministic under replay (the trading
    codebase already forbids wall-clock calls in reproducible paths).
    """

    def __init__(self, evaluator: Evaluator,
                 registry: Optional[KnobRegistry] = None,
                 journal: Optional[ChangeJournal] = None,
                 min_samples: int = 8, max_p_value: float = 0.10,
                 min_effect: float = 0.0,
                 clock_fn: Optional[Callable[[], str]] = None):
        self.evaluator = evaluator
        self.registry = registry or KnobRegistry()
        self.journal = journal or ChangeJournal()
        self.min_samples = min_samples
        self.max_p_value = max_p_value
        self.min_effect = min_effect
        self.clock_fn = clock_fn
        # what is live now: {path: value} of every promoted change
        self.active: Dict[str, Any] = {}
        self.results: Dict[str, ExperimentResult] = {}

    # ------------------------------------------------------------------
    def _move(self, p: Proposal, to: ProposalState, event: str,
              extra: Optional[dict] = None) -> None:
        assert can_transition(p.state, to), \
            f"illegal transition {p.state} -> {to}"
        p.state = to
        self.journal.record(p, event, _clock_ts(self.clock_fn), extra)

    # ------------------------------------------------------------------
    def submit(self, p: Proposal,
               current_values: Optional[Dict[str, float]] = None) -> Proposal:
        """Register a proposal after checking it against the boundaries.

        A proposal that would cross an immutable boundary is REJECTED here and
        never reaches experimentation -- it is not a candidate to be measured,
        it is a request that is not permitted.
        """
        current_values = current_values or {}
        try:
            for c in p.changes:
                cur = current_values.get(c.path)
                coerced = self.registry.validate(c.path, c.after, cur)
                c.after = coerced
                if cur is not None:
                    c.before = cur
        except BoundaryError as e:
            p.decision = str(e)
            self._move(p, ProposalState.REJECTED, "REJECT_BOUNDARY",
                       {"reason": str(e)})
            return p
        self.journal.record(p, "PROPOSE", _clock_ts(self.clock_fn))
        return p

    # ------------------------------------------------------------------
    def experiment(self, p: Proposal, seeds: Sequence[int],
                   metric: str = "objective") -> ExperimentResult:
        """Run the paired A/B and judge it. Sets state to VALIDATED /
        REJECTED / INCONCLUSIVE."""
        if p.state is not ProposalState.PROPOSED and \
                p.state is not ProposalState.INCONCLUSIVE:
            raise ValueError(f"cannot experiment on a proposal in {p.state}")
        self._move(p, ProposalState.EXPERIMENT, "EXPERIMENT_START",
                   {"seeds": list(seeds), "metric": metric})

        # Baseline = whatever is already live; candidate = live + this patch.
        base_patch = dict(self.active)
        cand_patch = {**self.active, **p.patch()}
        baseline = self.evaluator(base_patch, seeds)
        candidate = self.evaluator(cand_patch, seeds)

        sig = significance_gate(
            baseline, candidate, min_samples=self.min_samples,
            max_p_value=self.max_p_value, min_effect=self.min_effect,
            higher_is_better=True)
        p.metrics_before = {metric: sig.mean_before}
        p.metrics_after = {metric: sig.mean_after}
        p.evidence.append({"metric": metric, **sig.as_dict()})
        p.decision = sig.note

        res = ExperimentResult(p.id, list(baseline), list(candidate), sig)
        self.results[p.id] = res

        to = {"VALIDATED": ProposalState.VALIDATED,
              "REJECTED": ProposalState.REJECTED,
              "INCONCLUSIVE": ProposalState.INCONCLUSIVE}[sig.verdict]
        self._move(p, to, f"EXPERIMENT_{sig.verdict}", {"sig": sig.as_dict()})
        return res

    # ------------------------------------------------------------------
    def promote(self, p: Proposal) -> None:
        """Make a validated change live. Only VALIDATED proposals qualify."""
        if p.state is not ProposalState.VALIDATED:
            raise ValueError(f"only VALIDATED proposals may be promoted, "
                             f"not {p.state}")
        self.active.update(p.patch())
        p.decision = f"promoted: {p.decision}"
        self._move(p, ProposalState.PROMOTED, "PROMOTE",
                   {"active_after": dict(self.active)})

    # ------------------------------------------------------------------
    def rollback(self, p: Proposal, reason: str,
                 metrics_now: Optional[Dict[str, float]] = None) -> None:
        """Revert a promoted change and record why (spec §10).

        Never preserves a bad change because it is newer. The inverse patch
        restores the exact prior values, and any *other* promoted change that
        did not touch these knobs is left untouched.
        """
        if p.state is not ProposalState.PROMOTED:
            raise ValueError(f"only PROMOTED proposals can be rolled back, "
                             f"not {p.state}")
        for path, before in p.inverse_patch().items():
            if before is None:
                self.active.pop(path, None)
            else:
                self.active[path] = before
        self._move(p, ProposalState.ROLLED_BACK, "ROLLBACK",
                   {"reason": reason, "metrics_now": metrics_now or {},
                    "active_after": dict(self.active)})

    # ------------------------------------------------------------------
    def check_rollback(self, p: Proposal, seeds: Sequence[int],
                       tolerance: float = 0.0) -> bool:
        """Re-measure a promoted change on fresh seeds; roll back on regression.

        This is the MONITOR stage (spec §10). A change earns promotion on one
        set of seeds; this asks whether it still beats the pre-change baseline
        on a *different* set. A change that only worked on the seeds it was
        fitted to is a curve fit that slipped through, and monitoring on new
        draws is how it gets caught after the fact.
        """
        if p.state is not ProposalState.PROMOTED:
            return False
        without = {k: v for k, v in self.active.items()
                   if k not in p.patch()}
        with_change = dict(self.active)
        base = self.evaluator(without, seeds)
        cur = self.evaluator(with_change, seeds)
        import numpy as np
        delta = float(np.mean(cur) - np.mean(base))
        if delta < -abs(tolerance):
            self.rollback(p, f"post-promotion regression: delta={delta:+.4f} "
                             f"on {len(seeds)} fresh seeds",
                          metrics_now={"delta": delta})
            return True
        return False
