"""Performance analytics (spec §17).

The headline number is deliberately not the win rate. Per §15, a strategy with
a 90% win rate and one catastrophic loss is worse than a lower-win-rate
strategy with better risk-adjusted returns, so the report leads with Sharpe,
Sortino, Calmar and the tail, and reports win rate as a descriptive statistic
rather than a measure of quality.

Also included, because they are what actually diagnose a strategy:
  * **expectancy in R** -- the only per-trade number that is comparable across
    instruments and position sizes
  * **MAE/MFE distributions** -- how much heat winners took, and how much
    profit losers gave back. This is what tells you whether the stop is in the
    wrong place, which no aggregate P&L number will ever reveal.
  * **the tail ratio and the worst-trade contribution** -- whether the equity
    curve is a strategy or a lottery ticket
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.mathx import sharpe, sortino, max_drawdown, var_es, clamp, safe_div
from ..risk.portfolio import TradeRecord


def trade_stats(trades: Sequence[TradeRecord]) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = np.array([t.pnl - t.fees for t in trades])
    # Trades whose entry risk was never established cannot be expressed in R.
    # They are excluded and counted, not coerced to zero -- a zero would drag
    # expectancy toward the middle and hide that the figure is incomplete.
    r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]
    n_unmeasurable = len(trades) - len(r_vals)
    r = np.array(r_vals) if r_vals else np.zeros(0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    hold = np.array([t.hold_s for t in trades])

    # How much of the total P&L came from the single best trade? If one trade
    # is most of the result, there is no strategy here, only a sample of one.
    total = float(pnl.sum())
    best_share = float(pnl.max() / total) if total > 0 and pnl.size else 0.0

    return {
        "trades": len(trades),
        "win_rate": round(float((pnl > 0).mean()), 4),
        "total_pnl": round(total, 2),
        "gross_profit": round(gross_win, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(safe_div(gross_win, gross_loss, float("inf")
                                        if gross_win > 0 else 0.0), 3),
        "expectancy": round(float(pnl.mean()), 2),
        "expectancy_r": round(float(r.mean()), 4) if r.size else None,
        "r_measurable": len(r_vals),
        "r_unmeasurable": n_unmeasurable,
        "avg_win": round(float(wins.mean()), 2) if wins.size else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if losses.size else 0.0,
        "payoff_ratio": round(safe_div(
            float(wins.mean()) if wins.size else 0.0,
            abs(float(losses.mean())) if losses.size else 0.0), 3),
        "best": round(float(pnl.max()), 2),
        "worst": round(float(pnl.min()), 2),
        "best_trade_share": round(best_share, 3),
        "std_r": round(float(r.std()), 4) if r.size else None,
        "median_hold_s": round(float(np.median(hold)), 1),
        "avg_hold_s": round(float(hold.mean()), 1),
        "max_consecutive_losses": _max_streak(pnl <= 0),
        "max_consecutive_wins": _max_streak(pnl > 0),
    }


def _max_streak(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return int(best)


def excursion_stats(trades: Sequence[TradeRecord]) -> dict:
    """MAE/MFE analysis -- where the stop and target actually belong.

    If winners routinely take more heat than the stop allows, the stop is too
    tight and is converting winners into losers. If losers routinely show
    large favourable excursion first, the exit is too slow and is handing
    profits back.
    """
    if not trades:
        return {}
    win = [t for t in trades if (t.pnl - t.fees) > 0]
    lose = [t for t in trades if (t.pnl - t.fees) <= 0]
    out: Dict[str, float] = {}

    def q(vals, p):
        return round(float(np.percentile(np.abs(vals), p)), 2) if len(vals) else 0.0

    if win:
        mae_w = np.array([t.mae for t in win])
        out["winner_mae_median"] = q(mae_w, 50)
        out["winner_mae_p80"] = q(mae_w, 80)
        out["winner_mae_p95"] = q(mae_w, 95)
    if lose:
        mfe_l = np.array([t.mfe for t in lose])
        out["loser_mfe_median"] = q(mfe_l, 50)
        out["loser_mfe_p80"] = q(mfe_l, 80)
        # Fraction of losing trades that were meaningfully in profit at some
        # point -- the clearest single sign of an exit policy that is too slow.
        risks = np.array([abs(t.entry_px - t.exit_px) or 1e-9 for t in lose])
        out["losers_that_were_winners"] = round(
            float((mfe_l > risks).mean()), 3)
    return out


def equity_stats(equity: Sequence[Tuple[int, float]],
                 periods_per_year: float = 252.0,
                 bars_per_day: float = 375.0) -> dict:
    if len(equity) < 3:
        return {}
    e = np.array([v for _ts, v in equity], dtype=float)
    rets = np.diff(e) / np.maximum(e[:-1], 1e-9)
    dd, peak_i, trough_i = max_drawdown(e)
    ann_factor = periods_per_year * bars_per_day
    total_ret = e[-1] / e[0] - 1.0
    n = len(rets)
    years = n / max(ann_factor, 1.0)
    cagr = ((e[-1] / e[0]) ** (1 / years) - 1.0) if years > 0.01 and e[0] > 0 else 0.0
    v99, es99 = var_es(rets, 0.99)
    pos, neg = rets[rets > 0], rets[rets < 0]
    return {
        "total_return_pct": round(total_ret * 100, 4),
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe(rets, ann_factor), 3),
        "sortino": round(sortino(rets, ann_factor), 3),
        "calmar": round(safe_div(cagr, dd, 0.0), 3),
        "max_drawdown_pct": round(dd * 100, 3),
        "volatility_ann_pct": round(float(rets.std()) * math.sqrt(ann_factor) * 100, 3),
        "var_99_pct": round(v99 * 100, 4),
        "es_99_pct": round(es99 * 100, 4),
        "skew": round(float(_skew(rets)), 3),
        "kurtosis": round(float(_kurt(rets)), 3),
        "tail_ratio": round(safe_div(
            float(np.percentile(rets, 95)),
            abs(float(np.percentile(rets, 5)))), 3),
        "up_periods": int(pos.size),
        "down_periods": int(neg.size),
        "recovery_periods": int(len(e) - trough_i - 1),
    }


def _skew(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    s = x.std()
    return float(((x - x.mean()) ** 3).mean() / (s ** 3)) if s > 1e-15 else 0.0


def _kurt(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    s = x.std()
    return float(((x - x.mean()) ** 4).mean() / (s ** 4) - 3.0) if s > 1e-15 else 0.0


def by_regime(trades: Sequence[TradeRecord]) -> Dict[str, dict]:
    """Per-regime performance -- feeds the §9 requirement that strategy adapts
    to regime. A strategy that only works in one regime is fine; a strategy
    that is allowed to trade in all of them is not."""
    groups: Dict[str, List[TradeRecord]] = defaultdict(list)
    for t in trades:
        groups[t.entry_regime or "UNKNOWN"].append(t)
    return {k: trade_stats(v) for k, v in groups.items()}


def by_symbol(trades: Sequence[TradeRecord]) -> Dict[str, dict]:
    groups: Dict[str, List[TradeRecord]] = defaultdict(list)
    for t in trades:
        groups[t.symbol].append(t)
    return {k: trade_stats(v) for k, v in groups.items()}


def by_exit_reason(trades: Sequence[TradeRecord]) -> Dict[str, dict]:
    groups: Dict[str, List[TradeRecord]] = defaultdict(list)
    for t in trades:
        groups[t.exit_reason or "unknown"].append(t)
    return {k: trade_stats(v) for k, v in groups.items()}


def objective_score(eq: dict, tr: dict,
                    w_ret: float = 1.0, w_dd: float = 1.6, w_vol: float = 0.8,
                    w_ploss: float = 0.5, w_cost: float = 1.0) -> float:
    """The §15 objective, applied to a whole run rather than a single move.

    Used to rank strategies and walk-forward folds. Note the drawdown term is
    weighted above the return term, which is what makes a high-win-rate
    strategy with one catastrophic loss rank below a steadier one -- exactly
    the ordering the specification asks for.
    """
    if not eq or not tr:
        return 0.0
    ret = eq.get("total_return_pct", 0.0)
    dd = eq.get("max_drawdown_pct", 0.0)
    vol = eq.get("volatility_ann_pct", 0.0)
    p_loss = 1.0 - tr.get("win_rate", 0.5)
    worst_share = tr.get("best_trade_share", 0.0)
    return float(
        w_ret * ret
        - w_dd * dd
        - w_vol * vol * 0.1
        - w_ploss * p_loss * abs(ret) * 0.2
        # a result that hangs off one trade is penalised as the lottery it is
        - 8.0 * max(0.0, worst_share - 0.35) * abs(ret)
    )


def full_report(trades: Sequence[TradeRecord],
                equity: Sequence[Tuple[int, float]],
                extra: Optional[dict] = None) -> dict:
    tr = trade_stats(trades)
    eq = equity_stats(equity)
    rep = {
        "trades": tr,
        "equity": eq,
        "excursions": excursion_stats(trades),
        "by_regime": by_regime(trades),
        "by_symbol": by_symbol(trades),
        "by_exit_reason": by_exit_reason(trades),
        "objective_score": round(objective_score(eq, tr), 3),
    }
    if extra:
        rep.update(extra)
    return rep
