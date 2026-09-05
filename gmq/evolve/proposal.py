r"""The proposal, its lifecycle, and the change journal (spec §4, §5, §17).

A change to the system is never applied directly. It is written down as a
Proposal, moves through a fixed sequence of states, and every transition is
recorded in the journal with the evidence that justified it. The states are
the ones the spec mandates, and they may not be skipped:

    PROPOSED  -> EXPERIMENT -> VALIDATED -> PROMOTED
                            \-> REJECTED
                            \-> INCONCLUSIVE

The value of writing it down is not bureaucracy. It is that six months later,
when a config is behaving oddly, there is a record of exactly which change
introduced it, what it was measured against, and what the rollback is. A
system that improves itself without a journal cannot be debugged; it can only
be distrusted.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ProposalState(str, Enum):
    PROPOSED = "PROPOSED"
    EXPERIMENT = "EXPERIMENT"
    VALIDATED = "VALIDATED"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    ROLLED_BACK = "ROLLED_BACK"


# The only transitions the controller is allowed to make. Anything else is a
# bug in the controller, and asserting it here catches that bug loudly.
_ALLOWED: Dict[ProposalState, set] = {
    ProposalState.PROPOSED: {ProposalState.EXPERIMENT, ProposalState.REJECTED},
    ProposalState.EXPERIMENT: {ProposalState.VALIDATED, ProposalState.REJECTED,
                               ProposalState.INCONCLUSIVE},
    ProposalState.VALIDATED: {ProposalState.PROMOTED, ProposalState.REJECTED},
    ProposalState.PROMOTED: {ProposalState.ROLLED_BACK},
    ProposalState.REJECTED: set(),
    ProposalState.INCONCLUSIVE: {ProposalState.EXPERIMENT},  # re-test w/ more data
    ProposalState.ROLLED_BACK: set(),
}


def can_transition(a: ProposalState, b: ProposalState) -> bool:
    return b in _ALLOWED.get(a, set())


@dataclass
class KnobChange:
    """One parameter moving from one value to another."""
    path: str
    before: Any
    after: Any

    def as_dict(self) -> dict:
        return {"path": self.path, "before": self.before, "after": self.after}


@dataclass
class Proposal:
    """A candidate improvement and everything known about it (spec §4/§17)."""
    id: str
    problem: str
    root_cause: str
    changes: List[KnobChange]
    expected_benefit: str = ""
    risk: str = ""
    metrics: List[str] = field(default_factory=list)
    rollback_condition: str = ""
    affected_components: List[str] = field(default_factory=list)
    state: ProposalState = ProposalState.PROPOSED
    # filled in as the proposal moves through the loop
    evidence: List[dict] = field(default_factory=list)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    decision: str = ""
    created_ts: str = ""
    history: List[dict] = field(default_factory=list)

    def patch(self) -> Dict[str, Any]:
        """The change as a flat {path: new_value} mapping."""
        return {c.path: c.after for c in self.changes}

    def inverse_patch(self) -> Dict[str, Any]:
        """The rollback: {path: original_value}."""
        return {c.path: c.before for c in self.changes}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d


class ChangeJournal:
    """Append-only record of every proposal transition (spec §17).

    Append-only is the point. A journal you can rewrite is a journal that can
    be made to agree with whatever the system currently believes, which is
    exactly the failure the whole loop exists to prevent. Failure history is
    kept even when -- especially when -- it is inconvenient (spec §14).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.entries: List[dict] = []
        if path and os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))

    def record(self, proposal: Proposal, event: str, ts: str,
               extra: Optional[dict] = None) -> None:
        entry = {
            "ts": ts,
            "id": proposal.id,
            "event": event,
            "state": proposal.state.value,
            "changes": [c.as_dict() for c in proposal.changes],
            "decision": proposal.decision,
            "metrics_before": proposal.metrics_before,
            "metrics_after": proposal.metrics_after,
        }
        if extra:
            entry.update(extra)
        self.entries.append(entry)
        proposal.history.append({"event": event, "state": proposal.state.value,
                                 "ts": ts})
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")

    def for_id(self, pid: str) -> List[dict]:
        return [e for e in self.entries if e["id"] == pid]

    def promoted(self) -> List[dict]:
        return [e for e in self.entries if e["event"] == "PROMOTE"]
