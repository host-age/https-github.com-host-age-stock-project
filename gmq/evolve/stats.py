"""The statistical gate (spec §7, §20).

The rule the whole system is built on: a change is not an improvement because
its output looks better. It is an improvement when paired, out-of-sample
evidence says so at a stated confidence, with a sample large enough to support
the claim. This module is where that rule is enforced for changes to the
system itself, using the same machinery -- paired resampling, effect size,
minimum n -- the trading side already uses for changes to strategy.

Everything here is paired on seed. The baseline arm and the experiment arm see
the *identical* simulated markets; only the config differs. Comparing arms
across different market draws would drown the effect of the change in the
seed-to-seed variance, which for this system is far larger than any single
tuning change. Pairing removes that variance by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class SigResult:
    n: int
    mean_before: float
    mean_after: float
    delta: float
    p_value: float          # P(the true improvement is <= 0)
    ci_lo: float            # bootstrap CI on the paired delta
    ci_hi: float
    verdict: str            # VALIDATED | REJECTED | INCONCLUSIVE
    note: str = ""

    def as_dict(self) -> dict:
        return {"n": self.n, "mean_before": round(self.mean_before, 4),
                "mean_after": round(self.mean_after, 4),
                "delta": round(self.delta, 4),
                "p_value": round(self.p_value, 4),
                "ci": [round(self.ci_lo, 4), round(self.ci_hi, 4)],
                "verdict": self.verdict, "note": self.note}


def paired_bootstrap(before: Sequence[float], after: Sequence[float],
                     n_boot: int = 20000, seed: int = 17):
    """Resample the paired differences. Returns (p_value, ci_lo, ci_hi).

    p_value is P(mean improvement <= 0) under the resampling distribution.
    The CI is the 5th/95th percentile of the bootstrapped mean delta.
    """
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    d = a - b
    if d.size == 0:
        return 1.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    p = float((means <= 0.0).mean())
    return p, float(np.percentile(means, 5)), float(np.percentile(means, 95))


def significance_gate(before: Sequence[float], after: Sequence[float],
                      min_samples: int = 8, max_p_value: float = 0.10,
                      min_effect: float = 0.0,
                      higher_is_better: bool = True) -> SigResult:
    """Decide VALIDATED / REJECTED / INCONCLUSIVE for a paired experiment.

    * Fewer than `min_samples` pairs -> INCONCLUSIVE. This is not a failure of
      the change; it is a statement that the experiment did not gather enough
      evidence to have an opinion. Promoting on thin evidence is exactly the
      overfitting the loop exists to prevent, so the honest answer is to run
      more seeds, not to lower the bar.
    * A significant move in the *wrong* direction -> REJECTED.
    * A significant move in the right direction, past the effect floor ->
      VALIDATED.
    * Anything else (real but too small, or not significant) -> INCONCLUSIVE.
    """
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    n = min(b.size, a.size)
    if n < min_samples:
        return SigResult(n, float(b.mean()) if b.size else 0.0,
                         float(a.mean()) if a.size else 0.0, 0.0, 1.0, 0.0, 0.0,
                         "INCONCLUSIVE",
                         f"only {n} paired samples (<{min_samples}); "
                         "gather more before deciding")
    # orient so that "after - before" positive always means "better"
    sign = 1.0 if higher_is_better else -1.0
    d = sign * (a - b)
    p, lo, hi = paired_bootstrap(sign * b, sign * a)
    mean_delta = float(d.mean())
    raw_before, raw_after = float(b.mean()), float(a.mean())

    if p <= max_p_value and mean_delta > min_effect:
        return SigResult(n, raw_before, raw_after, float((a - b).mean()),
                         p, lo, hi, "VALIDATED",
                         f"improvement significant at p={p:.3f}")
    # significant, but the wrong way: the change is actively harmful
    p_worse, _, _ = paired_bootstrap(sign * a, sign * b)
    if p_worse <= max_p_value and mean_delta < 0:
        return SigResult(n, raw_before, raw_after, float((a - b).mean()),
                         p, lo, hi, "REJECTED",
                         f"significantly worse at p={p_worse:.3f}")
    return SigResult(n, raw_before, raw_after, float((a - b).mean()),
                     p, lo, hi, "INCONCLUSIVE",
                     f"no significant effect (p={p:.3f}, "
                     f"delta={mean_delta:+.4f})")
