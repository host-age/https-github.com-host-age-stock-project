"""Scenario generation -- the market's replies in the search tree.

In chess the opponent has a finite move list. Here the "opponent" is the
market, whose reply set is a continuum, so the search samples it: each ply
generates a bundle of price paths representing what could plausibly happen
next, and the move evaluation averages over them.

Three properties matter more than realism-in-general:

1. **Paths, not endpoints.** A trade with a stop is path-dependent. Sampling
   only the terminal price would price a stop-out and a round-trip identically.
   Every scenario is a full path, and barrier touches are checked along it.

2. **Bootstrapped, not Gaussian.** The block bootstrap resamples the symbol's
   own recent returns in contiguous blocks, so fat tails, volatility clustering
   and short-range autocorrelation survive into the scenarios. A Gaussian path
   generator would systematically under-price exactly the tail the stop exists
   to survive.

3. **Antithetic pairing.** Paths are generated in mirrored pairs around the
   drift. With a few dozen paths per node, plain sampling leaves enough Monte
   Carlo noise that the search will happily pick a move whose apparent edge is
   pure sampling error. Antithetic pairing cancels the first-order noise.

The regime conditions the shape: a trending regime gets persistent drift and
momentum in the block draw, mean-reverting gets an OU pull, event-driven gets
a jump component and fatter tails.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Regime, Prediction
from ..core.mathx import clamp


@dataclass
class PathBundle:
    """A set of sampled future price paths for one symbol at one node."""
    paths: np.ndarray          # (n_paths, n_steps) prices
    probs: np.ndarray          # (n_paths,) weights, sum to 1
    dt_s: float                # seconds per step
    horizon_s: float
    labels: List[str]

    @property
    def n(self) -> int:
        return self.paths.shape[0]

    def terminal(self) -> np.ndarray:
        return self.paths[:, -1]

    def max_up(self, entry: float) -> np.ndarray:
        return self.paths.max(axis=1) - entry

    def max_down(self, entry: float) -> np.ndarray:
        return self.paths.min(axis=1) - entry


# Per-regime path shaping. `persist` is the AR(1) coefficient applied to the
# bootstrapped increments; `jump_p` the per-path probability of a shock.
REGIME_SHAPE: Dict[Regime, dict] = {
    Regime.TRENDING_UP:    dict(persist=+0.20, vol_mult=1.00, jump_p=0.01, ou=0.00, tail=1.0),
    Regime.TRENDING_DOWN:  dict(persist=+0.20, vol_mult=1.10, jump_p=0.02, ou=0.00, tail=1.1),
    Regime.MEAN_REVERTING: dict(persist=-0.15, vol_mult=0.90, jump_p=0.01, ou=0.10, tail=1.0),
    Regime.BREAKOUT:       dict(persist=+0.30, vol_mult=1.70, jump_p=0.06, ou=0.00, tail=1.3),
    Regime.HIGH_VOL:       dict(persist=+0.05, vol_mult=2.10, jump_p=0.05, ou=0.02, tail=1.4),
    Regime.LOW_VOL:        dict(persist=+0.00, vol_mult=0.55, jump_p=0.00, ou=0.05, tail=0.9),
    Regime.EVENT_DRIVEN:   dict(persist=+0.10, vol_mult=2.80, jump_p=0.18, ou=0.00, tail=1.8),
    Regime.ILLIQUID:       dict(persist=+0.05, vol_mult=1.30, jump_p=0.04, ou=0.03, tail=1.5),
}


class ScenarioGenerator:
    def __init__(self, seed: int = 11, block: int = 6):
        self.rng = np.random.default_rng(seed)
        self.block = block
        self._hist: Dict[str, np.ndarray] = {}

    def set_history(self, symbol: str, log_returns: np.ndarray) -> None:
        """Recent per-bar log returns used as the bootstrap source."""
        if log_returns.size >= 30:
            self._hist[symbol] = log_returns[-600:].astype(np.float64)

    # ------------------------------------------------------------------
    def common_random_seed(self, symbol: str, spot: float, atr: float,
                           regime: Regime, qty: int) -> int:
        """A seed that changes only when the *situation* changes.

        Without this, two searches over an identical state draw different
        scenarios and can reach different conclusions -- so a position exits
        not because anything happened, but because the dice landed differently
        on the fourth re-evaluation. Re-deciding every five seconds turns that
        into a near-certainty: at a 23% per-decision chance of a noise-driven
        exit, the expected holding time collapses to about twenty seconds
        regardless of the thesis.

        Common random numbers are the standard fix. The seed is quantised on
        price (in quarter-ATR steps), regime and position sign, so an unchanged
        situation reuses the same draws and only *genuine* new information can
        flip the decision. When the situation really does move, the seed moves
        with it and fresh scenarios are drawn.
        """
        step = max(atr * 0.25, spot * 0.0005, 1e-9)
        bucket = int(spot / step)
        sign = (qty > 0) - (qty < 0)
        # NOT the builtin hash(). Python randomises string hashing per process
        # (PYTHONHASHSEED), so `hash(("RELIANCE", ...))` returns a different
        # value in every run -- which means the same backtest, replayed, draws
        # different scenarios and can reach different decisions. That defeats
        # the entire purpose of common random numbers, and worse, it defeats
        # it *invisibly*: nothing errors, the numbers are merely never quite
        # reproducible. It was caught here only because a behavioural test
        # passed alone and failed in the suite. crc32 is stable across
        # processes and machines.
        key = f"{symbol}|{bucket}|{regime.value}|{sign}".encode()
        return zlib.crc32(key) & 0x7FFFFFFF

    def generate(self, symbol: str, spot: float, pred: Prediction,
                 regime: Regime, horizon_s: float, n_paths: int = 24,
                 n_steps: int = 12, atr: float = 0.0,
                 antithetic: bool = True,
                 seed_key: Optional[int] = None) -> PathBundle:
        if spot <= 0 or n_paths < 2:
            return PathBundle(np.full((1, 1), max(spot, 1e-9)),
                              np.ones(1), horizon_s, horizon_s, ["flat"])
        shape = REGIME_SHAPE.get(regime, REGIME_SHAPE[Regime.LOW_VOL])
        dt_s = horizon_s / n_steps
        # Common random numbers: an unchanged situation reuses its draws.
        saved_rng = None
        if seed_key is not None:
            saved_rng = self.rng
            self.rng = np.random.default_rng(seed_key)
        try:
            return self._generate(symbol, spot, pred, regime, horizon_s,
                                  n_paths, n_steps, atr, antithetic, shape,
                                  dt_s)
        finally:
            if saved_rng is not None:
                self.rng = saved_rng

    def _generate(self, symbol, spot, pred, regime, horizon_s, n_paths,
                  n_steps, atr, antithetic, shape, dt_s) -> PathBundle:

        # -- per-step volatility -------------------------------------
        # Prefer the model's own expected dispersion; fall back to ATR, which
        # is a range measure and needs the ~1.2 conversion to a close-to-close
        # standard deviation before it can be used as one.
        if pred.exp_vol > 1e-9:
            sigma_h = pred.exp_vol
        elif atr > 0:
            sigma_h = (atr / spot) / 1.2 * math.sqrt(horizon_s / 300.0)
        else:
            sigma_h = 0.002
        sigma_step = sigma_h / math.sqrt(n_steps) * shape["vol_mult"]

        # -- drift ----------------------------------------------------
        # The directional edge, converted to a drift over the horizon. Scaled
        # by confidence: an unconfident 0.7 must not produce the same drift as
        # a well-evidenced 0.7, or the search will bet the same size on both.
        edge = (2.0 * pred.p_up - 1.0) * clamp(pred.confidence, 0.0, 1.0)
        mu_h = edge * sigma_h * 0.85
        if abs(pred.exp_return) > 1e-9:
            mu_h = 0.5 * mu_h + 0.5 * clamp(pred.exp_return, -0.05, 0.05) * \
                clamp(pred.confidence, 0.0, 1.0)
        mu_step = mu_h / n_steps

        # -- innovations ---------------------------------------------
        half = (n_paths + 1) // 2 if antithetic else n_paths
        eps = self._draw(symbol, half, n_steps, shape["tail"])
        # normalise to unit variance so sigma_step means what it says
        s = eps.std()
        if s > 1e-12:
            eps = eps / s
        if antithetic:
            eps = np.concatenate([eps, -eps], axis=0)[:n_paths]

        # AR(1) persistence / anti-persistence
        rho = shape["persist"]
        if abs(rho) > 1e-6:
            out = np.empty_like(eps)
            out[:, 0] = eps[:, 0]
            for k in range(1, n_steps):
                out[:, k] = rho * out[:, k - 1] + math.sqrt(1 - rho * rho) * eps[:, k]
            eps = out

        inc = mu_step + sigma_step * eps

        # OU pull back toward spot for reverting regimes
        ou = shape["ou"]
        if ou > 0:
            lvl = np.zeros(n_paths)
            for k in range(n_steps):
                inc[:, k] -= ou * lvl
                lvl = lvl + inc[:, k]

        # jumps
        if shape["jump_p"] > 0:
            hit = self.rng.random(n_paths) < shape["jump_p"]
            if hit.any():
                where = self.rng.integers(0, n_steps, size=int(hit.sum()))
                mag = self.rng.standard_normal(int(hit.sum())) * sigma_h * 3.0
                inc[np.where(hit)[0], where] += mag

        logp = np.log(spot) + np.cumsum(inc, axis=1)
        paths = np.exp(logp)

        probs = np.full(n_paths, 1.0 / n_paths)
        labels = self._label(paths, spot)
        return PathBundle(paths=paths, probs=probs, dt_s=dt_s,
                          horizon_s=horizon_s, labels=labels)

    # ------------------------------------------------------------------
    def _draw(self, symbol: str, n_paths: int, n_steps: int,
              tail: float) -> np.ndarray:
        """Block bootstrap from the symbol's own returns, Gaussian fallback."""
        hist = self._hist.get(symbol)
        if hist is None or hist.size < self.block * 4:
            z = self.rng.standard_normal((n_paths, n_steps))
            if tail > 1.05:
                # Student-t-ish tails via a variance mixture
                scale = np.sqrt(self.rng.chisquare(4, size=(n_paths, 1)) / 4.0)
                z = z / np.maximum(scale, 0.3)
            return z
        h = hist - hist.mean()
        sd = h.std()
        if sd > 1e-12:
            h = h / sd
        n_blocks = int(math.ceil(n_steps / self.block))
        starts = self.rng.integers(0, max(1, h.size - self.block),
                                   size=(n_paths, n_blocks))
        out = np.empty((n_paths, n_blocks * self.block))
        for i in range(n_paths):
            out[i] = np.concatenate([h[s: s + self.block] for s in starts[i]])
        return out[:, :n_steps]

    @staticmethod
    def _label(paths: np.ndarray, spot: float) -> List[str]:
        """Human-readable tags so the decision log says what each branch was."""
        term = paths[:, -1] / spot - 1.0
        hi = paths.max(axis=1) / spot - 1.0
        lo = paths.min(axis=1) / spot - 1.0
        out = []
        for t, h, l in zip(term, hi, lo):
            if t > 0.004:
                out.append("rally" if l > -0.002 else "dip_then_rally")
            elif t < -0.004:
                out.append("selloff" if h < 0.002 else "pop_then_fade")
            elif h > 0.004 and l < -0.004:
                out.append("whipsaw")
            else:
                out.append("chop")
        return out

    def summarise(self, bundle: PathBundle, spot: float,
                  top: int = 5) -> List[dict]:
        """Group paths into named scenarios with probabilities, for the log."""
        from collections import Counter
        term = bundle.terminal() / spot - 1.0
        cnt = Counter(bundle.labels)
        out = []
        for name, c in cnt.most_common(top):
            m = np.array([i for i, l in enumerate(bundle.labels) if l == name])
            out.append({
                "name": name,
                "prob": round(c / bundle.n, 3),
                "mean_ret_bps": round(float(term[m].mean()) * 1e4, 1),
                "worst_bps": round(float((bundle.paths[m].min(axis=1) / spot - 1).min()) * 1e4, 1),
                "best_bps": round(float((bundle.paths[m].max(axis=1) / spot - 1).max()) * 1e4, 1),
            })
        return out
