"""The trading-engine substrate for the evolution controller.

This is the only file in the subpackage that knows what a trade is. It turns a
{knob_path: value} patch into a Config, runs the engine over a list of seeds,
and returns one objective number per seed for the controller to compare. Keep
everything trading-specific here so the controller stays a general governance
engine.
"""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, List, Sequence

import numpy as np

from ..core.config import Config
from ..core.clock import IST
from ..app.engine import TradingEngine


def apply_patch(cfg: Config, patch: Dict[str, Any]) -> Config:
    """Return a deep copy of cfg with the dotted-path knobs set."""
    c = copy.deepcopy(cfg)
    for path, value in patch.items():
        obj = c
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise AttributeError(f"config has no knob {path!r}")
        setattr(obj, parts[-1], value)
    return c


def objective_of(rep: dict, trades: list, fees: float) -> float:
    """Collapse a run's report into one number, higher = better.

    Deliberately risk-adjusted, not raw P&L (spec §15): the whole point of the
    objective is that a strategy which makes slightly more money by carrying a
    catastrophic tail is *worse*, not better. This uses net return penalised by
    drawdown, so a change that lifts return by widening the worst day does not
    read as an improvement. It mirrors the search's own objective philosophy so
    the controller optimises the same thing the engine does.
    """
    eq = rep.get("equity", {})
    ret = float(eq.get("total_return_pct", 0.0))
    dd = float(eq.get("max_drawdown_pct", 0.0))
    # net return minus a linear drawdown penalty; both already in percent
    return ret - 0.5 * dd


class TradingEvaluator:
    """Runs the engine over seeds and returns the per-seed objective.

    A single instance is reused across baseline and candidate arms so the only
    thing that differs between arms is the patch. Runs are deliberately short
    and node-budgeted (not wall-clock) so the whole A/B is deterministic and
    replayable, which is the same discipline the backtest side already keeps.
    """

    def __init__(self, base_cfg: Config, days: float = 0.8, nsym: int = 6,
                 dt: float = 0.5, node_budget: int = 900,
                 start: datetime = None):
        self.base_cfg = base_cfg
        self.days = days
        self.nsym = nsym
        self.dt = dt
        self.node_budget = node_budget
        self.start = start or datetime(2026, 8, 17, 9, 15, tzinfo=IST)
        self._cache: Dict[tuple, float] = {}

    def _run_one(self, patch: Dict[str, Any], seed: int) -> float:
        key = (tuple(sorted(patch.items())), seed)
        if key in self._cache:
            return self._cache[key]
        cfg = apply_patch(self.base_cfg, patch)
        cfg.seed = seed
        cfg.run_id = f"evolve_{seed}"
        cfg.data.symbols = cfg.data.symbols[:self.nsym]
        cfg.search.node_budget = self.node_budget
        eng = TradingEngine(cfg, run_dir=f"runs/evolve/{seed}",
                            start=self.start, journal=False)
        rep = eng.run(days=self.days, dt=self.dt)
        val = objective_of(rep, list(eng.portfolio.trades),
                           eng.portfolio.total_fees)
        self._cache[key] = val
        return val

    def __call__(self, patch: Dict[str, Any],
                 seeds: Sequence[int]) -> List[float]:
        return [self._run_one(dict(patch), s) for s in seeds]
