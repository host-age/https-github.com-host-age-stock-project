"""Excursion-calibrated exit policy (spec §6/§7/§17).

The measurement that motivated this module: on a full run, **61% of losing
trades had been in profit at some point before they lost**. That is not a
prediction failure. The entry thesis was right often enough to put the trade
in the money; the exit policy then handed the money back. A fixed
"trail at 1.5 ATR once the trade is up 1.4R" rule is a guess about where the
favourable excursion of a *winner* differs from the favourable excursion of a
*loser*, and there is no reason for that guess to be right on this market, in
this regime, at this holding period.

So measure it instead. Every closed trade contributes three numbers -- peak
favourable excursion, worst adverse excursion, and where it finally settled,
all in units of the risk taken at entry -- and the policy parameters are fitted
to that distribution.

**The counterfactual, and why it is not the obvious one.** The obvious design
stores three summary numbers per trade and replays a trailing stop against
them: peaked at 2.1R, closed at 0.3R, so a stop protecting 55% of the peak
would have exited at 1.15R. That is wrong, and wrong in the flattering
direction. A trailing stop trails the *running* peak, so it would have fired
on the first 45% retracement -- and the 2.1R peak, which is the number the
whole calculation is built on, would then never have happened. The peak is not
invariant to the policy being tested. Fitted against that replay, pure noise
produces a "policy" with a large positive edge and a bootstrap p-value of
zero, which is how this was caught.

So the rule is stated in a form that *can* be replayed exactly, and the data
needed to replay it is recorded live instead of reconstructed afterwards.
Once the running peak first reaches the activation level `a`, the protected
level is fixed at `k * a` and does not move again. `PathExcursion` watches
each open position and records, for every activation level on the grid,
whether the trade ever reached it and the worst R seen *after* it did. Those
two numbers answer the counterfactual exactly: the stop fired if and only if
the worst-after-arming fell through the fixed level.

One honest limitation remains, and no amount of bookkeeping removes it: the
paths on record were produced under whatever exit policy was live at the time,
so a policy *looser* than the incumbent is evaluated on trades the incumbent
may have already truncated. That censoring is why the fitted policy is not
trusted on the strength of its in-sample edge -- it has to survive the
out-of-sample A/B.

**Guarding against fitting the noise.** A grid search over two parameters on
a few hundred trades will always find something that looks better. Three
defences, all of which must pass before the fitted parameters are used at all:

  1. a minimum sample count, below which the defaults stand;
  2. a paired bootstrap against the incumbent policy on the same trades, so
     the comparison controls for which trades happened to be in the sample;
  3. a requirement that the winning cell is not an isolated spike -- its
     neighbours in the grid must also beat the incumbent, because a parameter
     pair that only works at exactly one setting is a coincidence.

That last one is the cheapest overfitting test there is and it rejects most
of what the first two let through.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Activation levels the live recorder watches. The fitted policy can only
# choose from these, because these are the only ones for which the exact
# counterfactual was recorded.
_ACTIVATIONS = (0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5)
_KEEPS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)


@dataclass(slots=True)
class Excursion:
    """One closed trade, in units of entry risk.

    `min_after` is the exact-counterfactual payload: `min_after[i]` is the
    worst R this trade reached after its running peak first touched
    `_ACTIVATIONS[i]`, or None if it never got there.
    """
    mfe_r: float
    mae_r: float
    final_r: float
    min_after: Tuple[Optional[float], ...] = ()
    #: mean |change in R| between consecutive observations. A stop is not
    #: filled at its level, it is filled at the first print through it, so the
    #: expected fill is about half a step past the level.
    step_r: float = 0.0
    hold_s: float = 0.0
    regime: str = ""
    won: bool = False


class PathExcursion:
    """Live per-position recorder for the exact give-back counterfactual.

    Cheap on purpose -- seven floats and seven flags per open position, updated
    on the same tick that marks the position. Reconstructing this after the
    fact from summary statistics is precisely the mistake this class exists to
    avoid.
    """
    __slots__ = ("peak_r", "trough_r", "last_r", "reached", "min_after",
                 "_steps", "_n_steps", "_started")

    def __init__(self) -> None:
        self.peak_r = 0.0
        self.trough_r = 0.0
        self.last_r = 0.0
        self.reached = [False] * len(_ACTIVATIONS)
        self.min_after: List[Optional[float]] = [None] * len(_ACTIVATIONS)
        self._steps = 0.0
        self._n_steps = 0
        self._started = False

    @property
    def step_r(self) -> float:
        """Typical move in R between observations -- the granularity at which
        a stop can actually be acted on."""
        return self._steps / self._n_steps if self._n_steps else 0.0

    def update(self, r: float) -> None:
        if not math.isfinite(r):
            return
        if self._started:
            self._steps += abs(r - self.last_r)
            self._n_steps += 1
        self._started = True
        self.last_r = r
        self.peak_r = max(self.peak_r, r)
        self.trough_r = min(self.trough_r, r)
        for i, a in enumerate(_ACTIVATIONS):
            if self.reached[i]:
                cur = self.min_after[i]
                if cur is None or r < cur:
                    self.min_after[i] = r
            elif self.peak_r >= a:
                # Arms this tick; the worst-after window starts now.
                self.reached[i] = True
                self.min_after[i] = r

    def snapshot(self) -> Tuple[Optional[float], ...]:
        return tuple(self.min_after)


@dataclass
class GivebackPolicy:
    """Where to arm the give-back stop, and how much of the peak to keep."""
    activation_r: float = 1.4      # arm once peak favourable excursion >= this
    keep_fraction: float = 0.55    # protect this share of the activation level
    fitted: bool = False           # False -> these are the defaults
    n_samples: int = 0
    edge_r: float = 0.0            # mean R improvement vs the incumbent
    p_value: float = 1.0           # bootstrap P(improvement <= 0)
    note: str = "defaults"

    def stop_r(self, peak_r: float) -> Optional[float]:
        """The R level to protect, or None if the stop is not armed yet.

        Deliberately a function of the *activation* level, not of the running
        peak. A level that trails the peak cannot be back-tested from anything
        short of a full path, and it moves the peak it is measured against.
        This one is fixed the moment it arms, which makes it both replayable
        and, from the trader's side, a promise rather than a moving target.
        """
        if peak_r < self.activation_r:
            return None
        return self.activation_r * self.keep_fraction

    def as_dict(self) -> dict:
        return {"activation_r": round(self.activation_r, 3),
                "keep_fraction": round(self.keep_fraction, 3),
                "fitted": self.fitted, "n": self.n_samples,
                "edge_r": round(self.edge_r, 4),
                "p_value": round(self.p_value, 4), "note": self.note}


def _level_index(activation_r: float) -> Optional[int]:
    for i, a in enumerate(_ACTIVATIONS):
        if abs(a - activation_r) < 1e-9:
            return i
    return None


def replay(ex: Sequence[Excursion], pol: GivebackPolicy,
           slippage_r: float = 0.04) -> np.ndarray:
    """R-multiples the give-back policy would have produced on these trades.

    Exact, not approximate, provided the trade carries the recorded
    `min_after` payload: the stop fired if and only if the worst R seen after
    the level armed fell through the protected level. A trade recorded without
    that payload -- one from an older run -- falls back to the summary test,
    which is why `ExcursionBook.fit` refuses to fit unless the payload is
    present on essentially every row.
    """
    if not len(ex):
        return np.zeros(0)
    idx = _level_index(pol.activation_r)
    out = np.empty(len(ex))
    for i, e in enumerate(ex):
        lvl = pol.stop_r(e.mfe_r)
        if lvl is None:
            out[i] = e.final_r
            continue
        worst = None
        if idx is not None and idx < len(e.min_after):
            worst = e.min_after[idx]
        if worst is None:
            # no exact record: fall back to the summary (peak precedes close)
            worst = min(e.final_r, e.mae_r)
        # A stop fills at the first print *through* the level, not at it. The
        # expected overshoot is about half the observed step size, and leaving
        # it out is not a rounding error: on a driftless random walk sampled
        # coarsely, the un-charged overshoot alone manufactures an apparent
        # edge of 0.35R per trade at a bootstrap p-value of zero. Charging it
        # makes the same noise correctly fit nothing.
        slip = slippage_r + 0.5 * max(e.step_r, 0.0)
        out[i] = (lvl - slip) if worst <= lvl else e.final_r
    return out


def _reality_check(deltas: np.ndarray, n_boot: int = 1500,
                   seed: int = 17) -> float:
    """White's Reality Check p-value for the best of many candidate policies.

    `deltas` is (n_trades, n_policies): each column is one grid cell's paired
    improvement over the do-nothing baseline, trade by trade.

    The naive test -- bootstrap the winning cell on its own -- answers the
    wrong question. It asks "is *this* policy better than nothing", having
    already chosen it *because* it looked best out of forty-two. On a driftless
    random walk that procedure returns p = 0.096 for a cell with no edge
    whatsoever, which is exactly the kind of number that gets a curve fit
    promoted to a strategy.

    The right question is "could the *best of forty-two* look this good by
    chance", and it is answered by bootstrapping the maximum statistic with
    each column recentred to have zero mean -- imposing the null that no
    policy has any edge, while preserving the correlation between cells (they
    are highly correlated: neighbouring cells trade almost the same trades).
    """
    n, m = deltas.shape
    if n < 20 or m == 0:
        return 1.0
    means = deltas.mean(axis=0)
    observed = float(means.max())
    if observed <= 0:
        return 1.0
    centred = deltas - means                   # impose the null
    rng = np.random.default_rng(seed)
    worse = 0
    batch = 250
    for start in range(0, n_boot, batch):
        b = min(batch, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        # (b, n) @ ... -> take column means per resample, then the max
        boot_max = centred[idx].mean(axis=1).max(axis=1)
        worse += int((boot_max >= observed).sum())
    return float(worse / n_boot)


class ExcursionBook:
    """Running record of trade excursions, and the policy fitted to them."""

    def __init__(self, min_samples: int = 120, max_keep: int = 4000,
                 slippage_r: float = 0.04, max_p_value: float = 0.10,
                 default: Optional[GivebackPolicy] = None):
        self.min_samples = min_samples
        self.max_keep = max_keep
        self.slippage_r = slippage_r
        self.max_p_value = max_p_value
        self.default = default or GivebackPolicy()
        self.policy = GivebackPolicy(
            activation_r=self.default.activation_r,
            keep_fraction=self.default.keep_fraction)
        self.rows: List[Excursion] = []
        self.fits: int = 0

    # ------------------------------------------------------------------
    def add(self, mfe_r: float, mae_r: float, final_r: float,
            min_after: Sequence[Optional[float]] = (),
            step_r: float = 0.0,
            hold_s: float = 0.0, regime: str = "") -> None:
        if not all(math.isfinite(v) for v in (mfe_r, mae_r, final_r)):
            return
        self.rows.append(Excursion(mfe_r=float(mfe_r), mae_r=float(mae_r),
                                   final_r=float(final_r),
                                   min_after=tuple(min_after),
                                   step_r=float(step_r), hold_s=hold_s,
                                   regime=regime, won=final_r > 0))
        if len(self.rows) > self.max_keep:
            self.rows = self.rows[-self.max_keep:]

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """The diagnosis this module exists to answer, as numbers."""
        if not self.rows:
            return {"n": 0}
        mfe = np.array([r.mfe_r for r in self.rows])
        mae = np.array([r.mae_r for r in self.rows])
        fin = np.array([r.final_r for r in self.rows])
        lost = fin <= 0
        gave_back = lost & (mfe > 0.25)
        won = fin > 0
        return {
            "n": int(fin.size),
            "win_rate": float(won.mean()),
            "mean_r": float(fin.mean()),
            "median_mfe_r": float(np.median(mfe)),
            "median_mae_r": float(np.median(mae)),
            "mfe_r_winners": float(np.median(mfe[won])) if won.any() else 0.0,
            "mfe_r_losers": float(np.median(mfe[lost])) if lost.any() else 0.0,
            # the headline: losers that were in profit first
            "losers_in_profit_first": float(gave_back.sum() / max(lost.sum(), 1)),
            "capture_ratio": float(fin[won].mean() / mfe[won].mean())
            if won.any() and mfe[won].mean() > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    def fit(self) -> GivebackPolicy:
        """Refit the give-back policy. Returns the policy actually in force.

        The benchmark is **no give-back rule at all**, not the incumbent one.
        That distinction is the whole difference between a calibration and a
        rationalisation. Scored against the incumbent, a driftless random walk
        produces a "significant" improvement of 0.35R per trade at p=0.000 --
        not because the winning cell is any good, but because the incumbent is
        actively harmful on that data and almost anything beats it. Measured
        against doing nothing, the same data correctly fits nothing at all.

        So the winner has to beat the do-nothing baseline, and if none does,
        the answer is to switch the rule off rather than to install the least
        bad version of it.
        """
        n = len(self.rows)
        if n < self.min_samples:
            self.policy.n_samples = n
            self.policy.note = f"insufficient sample ({n}/{self.min_samples})"
            return self.policy
        rows = self.rows
        # Refuse to fit on rows that cannot be replayed exactly. The fallback
        # inside `replay` exists so an old row does not crash the fit; it is
        # not good enough to *base* a fit on, because it is biased in the
        # direction that makes the give-back rule look free.
        exact = sum(1 for r in rows if len(r.min_after) == len(_ACTIVATIONS))
        if exact < 0.95 * n:
            self.policy.n_samples = n
            self.policy.note = f"only {exact}/{n} rows exactly replayable"
            return self.policy
        # The baseline is the trade as it actually finished -- what a book with
        # no give-back rule would have earned.
        base = np.array([r.final_r for r in rows])

        grid: Dict[Tuple[float, float], float] = {}
        cols: List[np.ndarray] = []
        for a in _ACTIVATIONS:
            for k in _KEEPS:
                cand = GivebackPolicy(activation_r=a, keep_fraction=k)
                d = replay(rows, cand, self.slippage_r) - base
                cols.append(d)
                grid[(a, k)] = float(d.mean())
        deltas = np.column_stack(cols)

        (a_best, k_best), edge = max(grid.items(), key=lambda kv: kv[1])
        self.fits += 1

        if edge <= 0:
            # Nothing beats doing nothing. Switch the rule off -- an unbounded
            # activation level never arms -- rather than keeping a default that
            # this evidence says is costing money.
            self.policy = GivebackPolicy(
                activation_r=float("inf"),
                keep_fraction=self.default.keep_fraction,
                fitted=False, n_samples=n, edge_r=edge, p_value=1.0,
                note="disabled: no give-back level beat holding")
            return self.policy

        # Guard 3: the winner's neighbours must agree. An isolated peak in a
        # 2-D grid is what fitting noise looks like.
        ai = _ACTIVATIONS.index(a_best)
        ki = _KEEPS.index(k_best)
        neigh = [grid[(_ACTIVATIONS[i], _KEEPS[j])]
                 for i in (ai - 1, ai, ai + 1) for j in (ki - 1, ki, ki + 1)
                 if 0 <= i < len(_ACTIVATIONS) and 0 <= j < len(_KEEPS)
                 and not (i == ai and j == ki)]
        robust = bool(neigh) and float(np.mean([v > 0 for v in neigh])) >= 0.6

        best = GivebackPolicy(activation_r=a_best, keep_fraction=k_best)
        p = _reality_check(deltas)

        if p > self.max_p_value or not robust:
            why = (f"p={p:.3f}>{self.max_p_value}" if p > self.max_p_value
                   else "isolated grid cell")
            # Falling back to the defaults is only right if the defaults are
            # themselves harmless. If this sample says the incumbent rule is
            # losing money, "no evidence for a better rule" is not a reason to
            # keep running the one that is costing something.
            inc = float((replay(rows, self.default, self.slippage_r)
                         - base).mean())
            keep_default = inc >= 0.0
            self.policy = GivebackPolicy(
                activation_r=(self.default.activation_r if keep_default
                              else float("inf")),
                keep_fraction=self.default.keep_fraction,
                fitted=False, n_samples=n, edge_r=edge, p_value=p,
                note=f"rejected: {why}" +
                     ("" if keep_default else "; incumbent disabled"))
            return self.policy

        best.fitted = True
        best.n_samples = n
        best.edge_r = edge
        best.p_value = p
        best.note = "fitted"
        self.policy = best
        return self.policy
