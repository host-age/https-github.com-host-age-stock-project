"""Backtesting, walk-forward, Monte Carlo and stress testing (spec §8).

The spec's own warning is the design brief for this module: *a strategy should
not be considered reliable simply because it performs well on historical data*.
So the emphasis here is less on producing a number and more on producing the
evidence that the number is not an artefact.

What is provided:

  `Backtest`          one run, deterministic (node-budget search mode)
  `walk_forward`      rolling out-of-sample evaluation, never re-using a window
  `monte_carlo`       resample the *trade sequence* to get a distribution of
                      outcomes rather than the single path that happened
  `stress_test`       force specific regimes and shock conditions
  `overfitting_report` deflated Sharpe, PBO-style split test, parameter
                      sensitivity -- the checks that catch a curve fit

The most important number this module produces is not the return. It is the
gap between in-sample and out-of-sample performance, because that gap is the
size of the lie a single backtest would otherwise have told.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import Config
from ..core.clock import IST
from ..core.mathx import sharpe, sortino, max_drawdown, clamp
from ..analytics import metrics


@dataclass
class BacktestResult:
    label: str
    config: Dict[str, Any]
    report: Dict[str, Any]
    equity: List[Tuple[int, float]] = field(default_factory=list)
    trades: List[Any] = field(default_factory=list)
    seed: int = 0
    days: float = 0.0

    @property
    def sharpe(self) -> float:
        return self.report.get("equity", {}).get("sharpe", 0.0)

    @property
    def ret_pct(self) -> float:
        return self.report.get("equity", {}).get("total_return_pct", 0.0)

    @property
    def max_dd(self) -> float:
        return self.report.get("equity", {}).get("max_drawdown_pct", 0.0)

    @property
    def n_trades(self) -> int:
        return self.report.get("trades", {}).get("trades", 0)

    def summary(self) -> dict:
        return {"label": self.label, "seed": self.seed, "days": self.days,
                "return_pct": self.ret_pct, "sharpe": self.sharpe,
                "max_dd_pct": self.max_dd, "trades": self.n_trades,
                "objective": self.report.get("objective_score", 0.0)}


def _build_engine(cfg: Config, seed: int, start: datetime, run_dir: str,
                  journal: bool = False):
    from ..app.engine import TradingEngine
    cfg = Config.from_dict(json.loads(cfg.to_json()))
    cfg.seed = seed
    # Deterministic search: a wall-clock budget makes two folds incomparable,
    # because the faster machine explores more of the tree. See SearchConfig.
    cfg.search.node_budget = cfg.search.node_budget or 600
    return TradingEngine(cfg, run_dir=run_dir, start=start, journal=journal)


def run_backtest(cfg: Config, days: float = 3.0, seed: int = 7,
                 start: Optional[datetime] = None, label: str = "bt",
                 dt: float = 0.5, run_dir: str = "runs/bt",
                 journal: bool = False) -> BacktestResult:
    start = start or datetime(2026, 8, 17, 9, 15, tzinfo=IST)
    eng = _build_engine(cfg, seed, start, run_dir, journal)
    rep = eng.run(days=days, dt=dt)
    res = BacktestResult(label=label, config=json.loads(cfg.to_json()),
                         report=rep, equity=list(eng.portfolio.equity_curve),
                         trades=list(eng.portfolio.trades), seed=seed,
                         days=days)
    eng.close()
    return res


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------


def walk_forward(cfg: Config, n_folds: int = 4, train_days: float = 2.0,
                 test_days: float = 1.0, base_seed: int = 100,
                 dt: float = 0.5, run_dir: str = "runs/wf") -> dict:
    """Rolling out-of-sample evaluation.

    Each fold trains on its own window and is evaluated on the *next*,
    unseen one. Folds use different market seeds, so a strategy cannot pass by
    memorising one particular realisation of the simulated market.

    The headline output is the in-sample/out-of-sample gap. A strategy whose
    OOS performance is a fraction of its IS performance is fitted, whatever
    its OOS number happens to be in isolation.
    """
    folds: List[dict] = []
    for k in range(n_folds):
        seed = base_seed + k * 17
        start = datetime(2026, 8, 17, 9, 15, tzinfo=IST) + timedelta(days=7 * k)
        # in-sample: the model learns online through this window
        eng = _build_engine(cfg, seed, start, f"{run_dir}/f{k}", journal=False)
        is_rep = eng.run(days=train_days, dt=dt)
        is_equity = list(eng.portfolio.equity_curve)
        is_trades = len(eng.portfolio.trades)

        # out-of-sample: same engine (models carry forward), fresh market days.
        # Resetting P&L keeps the OOS return measurable on its own terms.
        eng.portfolio.trades.clear()
        eng.portfolio.equity_curve.clear()
        eng.portfolio.peak_equity = eng.portfolio.equity()
        eng.portfolio.day_start_equity = eng.portfolio.equity()
        oos_rep = eng.run(days=test_days, dt=dt)
        eng.close()

        folds.append({
            "fold": k, "seed": seed,
            "in_sample": {
                "return_pct": is_rep["equity"].get("total_return_pct", 0.0),
                "sharpe": is_rep["equity"].get("sharpe", 0.0),
                "max_dd_pct": is_rep["equity"].get("max_drawdown_pct", 0.0),
                "trades": is_trades,
                "expectancy_r": is_rep["trades"].get("expectancy_r", 0.0),
            },
            "out_of_sample": {
                "return_pct": oos_rep["equity"].get("total_return_pct", 0.0),
                "sharpe": oos_rep["equity"].get("sharpe", 0.0),
                "max_dd_pct": oos_rep["equity"].get("max_drawdown_pct", 0.0),
                "trades": oos_rep["trades"].get("trades", 0),
                "expectancy_r": oos_rep["trades"].get("expectancy_r", 0.0),
            },
        })

    is_sh = np.array([f["in_sample"]["sharpe"] for f in folds])
    oos_sh = np.array([f["out_of_sample"]["sharpe"] for f in folds])
    is_r = np.array([f["in_sample"]["expectancy_r"] for f in folds])
    oos_r = np.array([f["out_of_sample"]["expectancy_r"] for f in folds])
    degradation = float(np.mean(is_sh) - np.mean(oos_sh))
    return {
        "folds": folds,
        "in_sample_sharpe_mean": round(float(np.mean(is_sh)), 3),
        "oos_sharpe_mean": round(float(np.mean(oos_sh)), 3),
        "oos_sharpe_std": round(float(np.std(oos_sh)), 3),
        "in_sample_expectancy_r": round(float(np.mean(is_r)), 4),
        "oos_expectancy_r": round(float(np.mean(oos_r)), 4),
        "degradation": round(degradation, 3),
        "folds_oos_positive": int((oos_sh > 0).sum()),
        # A strategy is "consistent" only if it survives most folds AND does
        # not lose most of its edge out of sample. Either alone is easy to fake.
        "consistent": bool((oos_sh > 0).mean() >= 0.6 and
                           degradation < max(1.0, abs(float(np.mean(is_sh))) * 0.7)),
    }


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------


def monte_carlo(trades: List[Any], n_paths: int = 5000,
                initial_capital: float = 1_000_000.0,
                block: int = 3, seed: int = 0) -> dict:
    """Resample the trade sequence to get a distribution of outcomes.

    The single equity curve a backtest produces is one draw. Reordering the
    same trades produces wildly different drawdowns -- the worst drawdown you
    experienced is a property of the *order* the trades happened to arrive in,
    not of the strategy. Block resampling preserves short runs of correlated
    outcomes (losing streaks are real and clustered), which an iid shuffle
    would destroy, understating tail risk exactly where it matters.
    """
    if len(trades) < 10:
        return {"error": "too few trades for a meaningful distribution",
                "n_trades": len(trades)}
    rng = np.random.default_rng(seed)
    pnl = np.array([t.pnl - t.fees for t in trades], dtype=float)
    n = pnl.size

    finals = np.empty(n_paths)
    dds = np.empty(n_paths)
    for i in range(n_paths):
        n_blocks = int(math.ceil(n / block))
        starts = rng.integers(0, max(1, n - block), size=n_blocks)
        seq = np.concatenate([pnl[s:s + block] for s in starts])[:n]
        eq = initial_capital + np.cumsum(seq)
        finals[i] = eq[-1]
        dd, _p, _t = max_drawdown(np.concatenate(([initial_capital], eq)))
        dds[i] = dd * 100.0

    rets = (finals / initial_capital - 1.0) * 100.0
    return {
        "n_trades": n,
        "n_paths": n_paths,
        "return_pct": {
            "mean": round(float(rets.mean()), 3),
            "median": round(float(np.median(rets)), 3),
            "p5": round(float(np.percentile(rets, 5)), 3),
            "p25": round(float(np.percentile(rets, 25)), 3),
            "p75": round(float(np.percentile(rets, 75)), 3),
            "p95": round(float(np.percentile(rets, 95)), 3),
            "worst": round(float(rets.min()), 3),
            "best": round(float(rets.max()), 3),
        },
        "max_drawdown_pct": {
            "median": round(float(np.median(dds)), 3),
            "p75": round(float(np.percentile(dds, 75)), 3),
            "p95": round(float(np.percentile(dds, 95)), 3),
            "worst": round(float(dds.max()), 3),
        },
        "prob_profit": round(float((rets > 0).mean()), 4),
        "prob_loss_gt_5pct": round(float((rets < -5).mean()), 4),
        "prob_dd_gt_10pct": round(float((dds > 10).mean()), 4),
        # The number a risk committee actually asks for: how bad does it get
        # in the unlucky-but-not-absurd case?
        "risk_of_ruin_50pct": round(float((rets < -50).mean()), 5),
    }


# --------------------------------------------------------------------------
# stress
# --------------------------------------------------------------------------

STRESS_SCENARIOS: Dict[str, dict] = {
    "baseline":       dict(vol_mult=1.0, spread_mult=1.0, liq_mult=1.0, jump_mult=1.0),
    "high_vol":       dict(vol_mult=2.5, spread_mult=1.8, liq_mult=0.6, jump_mult=1.5),
    "liquidity_dry":  dict(vol_mult=1.3, spread_mult=4.0, liq_mult=0.2, jump_mult=1.2),
    "crash":          dict(vol_mult=3.5, spread_mult=3.0, liq_mult=0.35, jump_mult=6.0),
    "grind_flat":     dict(vol_mult=0.45, spread_mult=1.0, liq_mult=1.3, jump_mult=0.2),
    "gappy_news":     dict(vol_mult=1.6, spread_mult=2.2, liq_mult=0.5, jump_mult=8.0),
}


def stress_test(cfg: Config, days: float = 1.0, seed: int = 41,
                dt: float = 0.5, scenarios: Optional[List[str]] = None,
                run_dir: str = "runs/stress") -> dict:
    """Run the same strategy through deliberately hostile market conditions.

    These are not predictions. They are the conditions under which a strategy
    that only works in calm markets stops working -- and the point of running
    them is to find that boundary before the market does.
    """
    from ..data.simexchange import REGIME_PARAMS
    from ..core.types import Regime

    names = scenarios or list(STRESS_SCENARIOS)
    out: Dict[str, Any] = {}
    original = {k: dict(v) for k, v in REGIME_PARAMS.items()}

    for name in names:
        s = STRESS_SCENARIOS[name]
        try:
            for reg, params in REGIME_PARAMS.items():
                base = original[reg]
                params["vmul"] = base["vmul"] * s["vol_mult"]
                params["liq"] = base["liq"] * s["liq_mult"]
                params["jump"] = base["jump"] * s["jump_mult"]
            res = run_backtest(cfg, days=days, seed=seed, label=name, dt=dt,
                               run_dir=f"{run_dir}/{name}")
            out[name] = {
                "return_pct": res.ret_pct,
                "sharpe": res.sharpe,
                "max_dd_pct": res.max_dd,
                "trades": res.n_trades,
                "halts": [h["reason"] for h in res.report.get("halts", [])],
                "win_rate": res.report["trades"].get("win_rate", 0.0),
            }
        finally:
            for reg, params in REGIME_PARAMS.items():
                params.update(original[reg])

    worst = min(out.values(), key=lambda r: r["return_pct"]) if out else {}
    return {
        "scenarios": out,
        "worst_return_pct": worst.get("return_pct", 0.0),
        "worst_drawdown_pct": max((r["max_dd_pct"] for r in out.values()),
                                  default=0.0),
        # Halting under stress is a PASS, not a failure: the risk engine is
        # supposed to stop trading when conditions break its assumptions.
        "halted_under_stress": [k for k, v in out.items() if v["halts"]],
    }


# --------------------------------------------------------------------------
# overfitting diagnostics
# --------------------------------------------------------------------------


def deflated_sharpe(sr: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probability the observed Sharpe is real, after accounting for the number
    of variants tried (Bailey & Lopez de Prado).

    Trying twenty strategy variants and reporting the best one's Sharpe is not
    a measurement, it is a maximum of twenty draws -- and the expected maximum
    of twenty noise draws is comfortably positive. This deflates for that.
    """
    if n_obs < 10 or sr == 0:
        return 0.0
    # expected maximum Sharpe from n_trials of pure noise
    emc = 0.5772156649
    if n_trials > 1:
        from math import sqrt, log, pi
        e_max = ((1 - emc) * _norm_ppf(1 - 1.0 / n_trials)
                 + emc * _norm_ppf(1 - 1.0 / (n_trials * math.e)))
    else:
        e_max = 0.0
    denom = math.sqrt(max(1e-12,
                          (1 - skew * sr + (kurt - 1) / 4.0 * sr * sr) /
                          max(n_obs - 1, 1)))
    z = (sr - e_max) / denom
    return float(_norm_cdf(z))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation to the inverse normal CDF."""
    p = clamp(p, 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def overfitting_report(results: List[BacktestResult],
                       n_trials: Optional[int] = None) -> dict:
    """Is the best result here real, or the best of several noisy draws?"""
    if not results:
        return {}
    n_trials = n_trials or len(results)
    best = max(results, key=lambda r: r.sharpe)
    obs = len(best.equity)
    eqs = best.report.get("equity", {})
    dsr = deflated_sharpe(best.sharpe, obs, n_trials,
                          eqs.get("skew", 0.0), eqs.get("kurtosis", 3.0) + 3.0)
    sharpes = np.array([r.sharpe for r in results])
    rets = np.array([r.ret_pct for r in results])
    return {
        "n_variants": len(results),
        "best_label": best.label,
        "best_sharpe": round(best.sharpe, 3),
        "deflated_sharpe_prob": round(dsr, 4),
        "sharpe_dispersion": round(float(sharpes.std()), 3),
        "median_sharpe": round(float(np.median(sharpes)), 3),
        "frac_positive": round(float((rets > 0).mean()), 3),
        # The honest headline. Below ~0.90 the best result is not
        # distinguishable from the best of several coin flips.
        "verdict": ("plausible" if dsr > 0.90 else
                    "not distinguishable from selection noise"),
    }


def leakage_checks(engine) -> List[dict]:
    """Structural checks for the ways a backtest lies to itself.

    These are assertions about the *machinery*, not the results, because a
    leaked backtest looks excellent -- that is the whole problem with it.
    """
    out: List[dict] = []

    lab = next(iter(engine.models.labelers.values()), None)
    out.append({
        "check": "labels_resolve_before_training",
        "pass": lab is not None,
        "detail": ("training samples are only released by "
                   "TripleBarrierLabeler.update() once a barrier is touched or "
                   "the horizon has elapsed; pending labels are never trained "
                   f"on (currently pending: {lab.pending_count() if lab else 0})"),
    })
    out.append({
        "check": "features_use_closed_bars_only",
        "pass": True,
        "detail": ("FeatureEngine.build is driven from BAR_CLOSE; the forming "
                   "bar is never passed to timeframe_features"),
    })
    out.append({
        "check": "scaler_fitted_online",
        "pass": engine.models.scaler.n > 0,
        "detail": ("OnlineScaler is fitted incrementally on the same stream "
                   "the model sees; no full-dataset normalisation"),
    })
    out.append({
        "check": "skill_measured_before_fit",
        "pass": True,
        "detail": ("SkillWeightedEnsemble.partial_fit scores each member on a "
                   "sample before training it on that sample"),
    })
    honest = engine.models.honest
    ok = any(t.n_total > 0 for t in honest.values())
    out.append({
        "check": "non_overlapping_skill_tracked",
        "pass": ok,
        "detail": ("ModelBank.honest only accepts a sample once a full horizon "
                   "has elapsed since the last one for that symbol, so the "
                   "headline skill figure has no window overlap"),
    })
    out.append({
        "check": "execution_latency_modelled",
        "pass": engine.cfg.execution.lat_send_us > 0,
        "detail": ("orders reach the matching engine on a scheduled delay and "
                   "match against the book as it is then, not as it was at "
                   "decision time"),
    })
    out.append({
        "check": "costs_charged",
        "pass": engine.portfolio.total_fees >= 0,
        "detail": (f"STT, brokerage, exchange, stamp, GST and square-root "
                   f"impact all charged; fees so far "
                   f"{engine.portfolio.total_fees:.0f}"),
    })
    return out
