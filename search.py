"""The Grandmaster scenario search (spec §2).

Observe -> Analyse -> Generate Moves -> Evaluate -> Simulate -> Select.

The engine is an **expectimax search over a stochastic opponent**. Our nodes
are choice nodes (we pick the move that maximises the objective); the market's
nodes are chance nodes (we average over sampled paths, weighted by
probability). That is the correct structure for this problem -- minimax would
be wrong, because the market is not adversarial, it is random. Assuming an
adversary would make the engine pathologically timid; assuming the mean would
make it blind to the tail. Expectimax with a risk-penalised objective sits
where it should.

  depth 1   what happens if I act now
  depth 2   ... and then what will I want to do after the market replies
  depth 3   ... and after that

What each ply buys is not prediction accuracy -- it is *option value*. A move
that looks slightly worse now but leaves a good reply available in every
branch beats a move that looks better now but leaves nothing to do if the
market moves against it. That is the whole reason to search rather than score.

Cost control, because this must fit inside a latency budget:
  * beam search -- only the top-K moves at each ply are expanded
  * scenario count decays with depth (deep nodes are noisier anyway)
  * a branch whose optimistic bound cannot beat the incumbent is cut
  * a hard wall-clock budget, checked between expansions, with the best
    completed depth returned (iterative deepening, so there is always an answer)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import (
    MoveType, Side, Regime, Prediction, MoveEval, Scenario,
)
from ..core.config import SearchConfig
from ..core.mathx import clamp
from .moves import MoveGenerator, MoveContext, Candidate
from .scenarios import ScenarioGenerator, PathBundle
from .evaluator import CostModel, evaluate, objective, fill_probability


@dataclass(slots=True)
class SimState:
    """Position state inside one simulated line of play."""
    px: float
    qty: int
    avg: float
    stop: float
    target: float
    realised: float = 0.0
    fees: float = 0.0
    t_s: float = 0.0
    partials: int = 0
    scale_ins: int = 0
    max_dd: float = 0.0
    peak: float = 0.0
    closed_reason: str = ""

    def equity_delta(self) -> float:
        return self.realised + (self.px - self.avg) * self.qty - self.fees


@dataclass
class SearchStats:
    nodes: int = 0
    leaves: int = 0
    pruned: int = 0
    depth_reached: int = 0
    ms: float = 0.0
    timed_out: bool = False
    hysteresis_holds: int = 0


class GrandmasterSearch:
    def __init__(self, cfg: SearchConfig, cost: Optional[CostModel] = None,
                 seed: int = 13):
        self.cfg = cfg
        self._out_of_budget = lambda: False
        self.cost = cost or CostModel()
        self.scen = ScenarioGenerator(seed=seed)
        self.moves = MoveGenerator(
            reentry_cooldown_s=getattr(cfg, "reentry_cooldown_s", 120.0))
        self.stats = SearchStats()

    # ------------------------------------------------------------------
    def search(self, ctx: MoveContext, equity: float,
               market: dict) -> Tuple[MoveEval, List[MoveEval], SearchStats]:
        """Returns (best move, all root candidates scored, stats)."""
        t0 = time.perf_counter()
        self.stats = SearchStats()
        # Reserve a margin. A deadline check can only fire *between* units of
        # work, so the last unit always overruns it; budgeting the full number
        # therefore reliably overshoots it. Aiming at 82% lands the measured
        # wall time at or under the configured budget.
        deadline = t0 + 0.82 * self.cfg.time_budget_ms / 1000.0
        node_cap = int(self.cfg.node_budget or 0)

        def out_of_budget() -> bool:
            if node_cap:
                return self.stats.nodes >= node_cap
            return time.perf_counter() > deadline
        self._out_of_budget = out_of_budget

        root = SimState(px=ctx.px, qty=ctx.qty, avg=ctx.avg_price,
                        stop=ctx.stop, target=ctx.target,
                        partials=ctx.partials, scale_ins=ctx.scale_ins)
        root.peak = root.equity_delta()

        cands = self.moves.generate(ctx)
        scored: List[MoveEval] = []

        # Iterative deepening.
        #
        # Results are accumulated per candidate, keyed by identity, and each
        # deeper pass *overwrites* a candidate's entry. That matters: if the
        # budget runs out partway through depth 2, the candidates already
        # re-evaluated keep their depth-2 score and the rest keep their
        # complete depth-1 score. Returning only the partial depth-2 level --
        # the obvious implementation -- silently throws away every candidate
        # the deeper pass had not reached yet, so a timeout could hand back a
        # move the engine had already scored as worse. That failure is
        # invisible in the output: the answer just quietly gets worse under
        # load, which is exactly when it matters most.
        results: Dict[tuple, MoveEval] = {}

        def key(c: Candidate) -> tuple:
            return (c.move, c.qty, round(c.stop, 4), round(c.limit_px, 4),
                    c.passive, c.tag)

        for depth in range(1, self.cfg.max_depth + 1):
            if out_of_budget() and results:
                break
            complete = True
            for cand in cands:
                if out_of_budget() and results:
                    self.stats.timed_out = True
                    complete = False
                    break
                results[key(cand)] = self._eval_move(
                    root, cand, ctx, equity, market, depth=depth,
                    deadline=deadline)
            if results:
                self.stats.depth_reached = max(self.stats.depth_reached, depth)
            if not complete:
                break
            # Narrow to the beam for the next, more expensive ply.
            level = sorted(results.values(), key=lambda m: -m.score)
            top = level[: max(self.cfg.beam_width, 2)]
            keep = {(m.move, m.qty, round(m.stop, 4)) for m in top}
            nxt = [c for c in cands
                   if (c.move, c.qty, round(c.stop, 4)) in keep]
            cands = nxt or cands[:2]

        self.stats.ms = (time.perf_counter() - t0) * 1000.0
        if not results:
            wait = MoveEval(move=MoveType.WAIT, rationale="no candidates")
            return wait, [wait], self.stats
        final = sorted(results.values(), key=lambda m: -m.score)
        final = self._apply_hysteresis(final, ctx, market)
        return final[0], final, self.stats

    # ------------------------------------------------------------------
    def _apply_hysteresis(self, ranked: List[MoveEval], ctx: MoveContext,
                          market: dict) -> List[MoveEval]:
        """Require a decision to *change* the position to beat holding by more
        than the round trip it costs.

        A marginal preference is not a reason to trade. Re-evaluating an open
        position every few seconds with a noisy estimator means that whenever
        exiting and holding are close, the noise decides -- and it decides
        again a few seconds later. Measured on a real run: 29% of hold-or-exit
        decisions were separated by under five rupees, exit won 23% of them,
        and the median holding period collapsed to fifteen seconds against a
        round-trip cost of 4.3bp. The strategy was paying a spread to express
        a coin flip.

        The economically correct bar is the cost of being wrong about the
        change: closing now and re-opening later costs a full round trip, so
        closing must be worth at least that much more than holding. This is
        the same reason a human trader does not flatten because the screen
        looks marginally worse than it did ten seconds ago.
        """
        if ctx.qty == 0 or len(ranked) < 2:
            return ranked
        best = ranked[0]
        if best.move in (MoveType.WAIT, MoveType.MOVE_STOP):
            return ranked
        hold = next((m for m in ranked if m.move is MoveType.WAIT), None)
        if hold is None:
            return ranked

        notional = abs(ctx.qty) * ctx.px
        spread_bps = float(market.get("spread_bps", 3.0))
        # a full round trip: out now, back in later
        round_trip = (self.cost.fees(notional, Side.SELL)
                      + self.cost.fees(notional, Side.BUY)
                      + notional * spread_bps / 1e4)
        margin = round_trip * self.cfg.exit_hysteresis
        if best.score - hold.score >= margin:
            return ranked
        hold.rationale = (f"hold | exit edge {best.score - hold.score:+.0f} "
                          f"< round-trip bar {margin:.0f}")
        self.stats.hysteresis_holds += 1
        return [hold] + [m for m in ranked if m is not hold]

    # ------------------------------------------------------------------
    def _eval_move(self, state: SimState, cand: Candidate, ctx: MoveContext,
                   equity: float, market: dict, depth: int,
                   deadline: float) -> MoveEval:
        """Expected value of playing `cand` now, searched `depth` plies deep."""
        self.stats.nodes += 1
        after, cost, filled_p = self._apply(state, cand, ctx, market)
        n_paths = max(4, int(self.cfg.scenarios_per_node /
                             (1 + 0.6 * (self.cfg.max_depth - depth))))
        horizon = self.cfg.horizon_s

        # Common random numbers across re-evaluations of the same situation:
        # every candidate move at this node is scored against the *same*
        # sampled futures, and an unchanged situation re-uses them next time.
        # Comparing moves on different draws is comparing them on luck.
        seed_key = self.scen.common_random_seed(
            ctx.symbol, ctx.px, ctx.atr, ctx.regime, ctx.qty) \
            if self.cfg.common_random_numbers else None
        bundle = self.scen.generate(
            ctx.symbol, after.px, ctx.pred, ctx.regime, horizon,
            n_paths=n_paths, n_steps=12, atr=ctx.atr,
            antithetic=self.cfg.use_antithetic, seed_key=seed_key)

        outcomes = np.empty(bundle.n)
        dds = np.empty(bundle.n)
        scen_records: List[Scenario] = []
        children: List[MoveEval] = []

        checked = 0
        for i in range(bundle.n):
            # The budget has to be checked inside this loop too. Checking only
            # between candidates lets a single deep candidate overrun by the
            # cost of a whole scenario bundle, which is where the measured
            # overshoot past the budget was coming from.
            if (i & 1) == 1 and i >= 4 and self._out_of_budget():
                outcomes = outcomes[:i]
                dds = dds[:i]
                w_trunc = True
                break
            path = bundle.paths[i]
            end, dd, hit_stop, hit_tgt = self._roll(after, path, ctx)
            val = evaluate(end.qty, end.px, end.avg, end.stop, end.target,
                           end.realised, end.fees, ctx.atr,
                           exit_cost=self._liquidation_cost(end, ctx, market))
            # Recurse: what is the best thing we could do from there? This is
            # where option value comes from -- a line is worth more if good
            # replies remain available in it.
            if depth > 1 and end.qty != 0 and not self._out_of_budget():
                sub_ctx = self._child_ctx(ctx, end, horizon)
                sub_cands = self.moves.generate(sub_ctx)[: self.cfg.beam_width]
                best_sub = -1e18
                for sc in sub_cands:
                    sub = self._eval_move(end, sc, sub_ctx, equity, market,
                                          depth - 1, deadline)
                    if sub.score > best_sub:
                        best_sub = sub.score
                        if depth == self.cfg.max_depth:
                            children.append(sub)
                if best_sub > -1e17:
                    val = val + self.cfg.discount * (best_sub - val) * 0.5
            else:
                self.stats.leaves += 1
            outcomes[i] = val
            dds[i] = dd
            if i < 6:
                scen_records.append(Scenario(
                    name=bundle.labels[i], prob=float(bundle.probs[i]),
                    ret=float(path[-1] / max(after.px, 1e-9) - 1.0),
                    max_dd=float(dd), max_run=float(path.max() / max(after.px, 1e-9) - 1),
                    hit_stop=hit_stop, hit_target=hit_tgt,
                    hold_s=float(end.t_s),
                ))

        n_done = outcomes.size
        w = bundle.probs[:n_done]
        w = w / w.sum() if w.sum() > 0 else np.ones(max(n_done, 1)) / max(n_done, 1)
        if n_done == 0:
            outcomes = np.zeros(1); dds = np.zeros(1); w = np.ones(1)
        # A passive order that does not fill leaves us in the prior state --
        # that branch has to be priced, not assumed away.
        if filled_p < 0.999:
            base = evaluate(state.qty, state.px, state.avg, state.stop,
                            state.target, state.realised, state.fees, ctx.atr,
                            exit_cost=self._liquidation_cost(state, ctx,
                                                             market))
            outcomes = filled_p * outcomes + (1 - filled_p) * base
            cost *= filled_p

        ev = float(np.dot(w, outcomes))
        var = float(np.dot(w, (outcomes - ev) ** 2))
        exp_dd = float(np.dot(w, dds))
        k = max(1, int(0.15 * outcomes.size))
        worst = np.sort(outcomes)[:k]
        cvar = float(max(0.0, -worst.mean()))
        # "No position, nothing happens" is a draw, not a loss -- counting an
        # exactly-zero outcome as a loss made WAIT report P(win)=0%.
        p_win = float((outcomes > 1e-9).mean() +
                      0.5 * (np.abs(outcomes) <= 1e-9).mean())
        risk_unit = max(abs(ctx.atr * max(abs(cand.qty), 1)), 1e-9)

        capital = abs(after.qty) * after.px
        # `cost` is not passed: it is already inside `ev` via the simulated
        # line's fees. It is still carried on the MoveEval for the audit log.
        score = objective(ev, var, exp_dd, cvar, capital, equity,
                          self.cfg.risk_aversion, self.cfg.dd_penalty,
                          self.cfg.cvar_penalty)

        me = MoveEval(
            move=cand.move, qty=cand.qty, price=cand.limit_px or ctx.px,
            stop=cand.stop or state.stop, target=cand.target or state.target,
            ev=ev, ev_r=ev / risk_unit, score=score, exp_dd=exp_dd,
            p_win=p_win, p_loss=1.0 - p_win, variance=var, cvar=cvar,
            cost=cost, depth=depth, scenarios=scen_records,
            children=children[:3],
            rationale=self._rationale(cand, ev, p_win, exp_dd, cost, ctx,
                                      filled_p),
        )
        return me

    # ------------------------------------------------------------------
    def _liquidation_cost(self, s: SimState, ctx: MoveContext,
                          market: dict) -> float:
        """What it would cost to close `s` right now, in rupees."""
        if s.qty == 0 or not getattr(self.cfg, "liquidation_in_eval", True):
            return 0.0
        side = Side.SELL if s.qty > 0 else Side.BUY
        return self.cost.cost_rupees(
            abs(s.qty), s.px, side,
            float(market.get("spread_bps", 3.0)),
            int(market.get("top_qty", max(abs(s.qty), 1))),
            False, ctx.liquidity)

    # ------------------------------------------------------------------
    def _apply(self, s: SimState, c: Candidate, ctx: MoveContext,
               market: dict) -> Tuple[SimState, float, float]:
        """Apply a move, returning (new state, cost in rupees, P(fill))."""
        n = replace(s)
        if c.move is MoveType.WAIT:
            return n, 0.0, 1.0
        if c.move is MoveType.MOVE_STOP:
            n.stop = c.stop
            return n, 0.0, 1.0

        qty = c.qty
        if qty == 0:
            return n, 0.0, 1.0
        side = Side.BUY if qty > 0 else Side.SELL
        spread_bps = float(market.get("spread_bps", 3.0))
        top_qty = int(market.get("top_qty", max(abs(qty), 1)))
        liq = float(market.get("liquidity", 1.0))

        p_fill = fill_probability(
            c.passive, spread_bps,
            int(market.get("queue_ahead", 0)), top_qty,
            self.cfg.horizon_s, float(market.get("trade_rate", 1.0)),
            float(market.get("imbalance", 0.0)), side)

        exec_px = c.limit_px or (ctx.ask if side is Side.BUY else ctx.bid) or s.px
        cost = self.cost.cost_rupees(abs(qty), exec_px, side, spread_bps,
                                     top_qty, c.passive, liq)

        # position accounting, mirroring Position.apply_fill
        if n.qty == 0 or (n.qty > 0) == (qty > 0):
            new_qty = n.qty + qty
            if new_qty != 0:
                n.avg = (n.avg * n.qty + exec_px * qty) / new_qty
            n.qty = new_qty
            if c.move is MoveType.INCREASE:
                n.scale_ins += 1
        else:
            closing = min(abs(qty), abs(n.qty))
            n.realised += (exec_px - n.avg) * closing * (1 if n.qty > 0 else -1)
            rem = n.qty + qty
            if rem != 0 and (rem > 0) != (n.qty > 0):
                n.avg = exec_px
            n.qty = rem
            if n.qty == 0:
                n.avg = 0.0
                n.stop = n.target = 0.0
            if c.move is MoveType.TAKE_PARTIAL:
                n.partials += 1
        if c.stop:
            n.stop = c.stop
        if c.target:
            n.target = c.target
        n.fees += cost
        return n, cost, p_fill

    # ------------------------------------------------------------------
    def _roll(self, s: SimState, path: np.ndarray, ctx: MoveContext
              ) -> Tuple[SimState, float, bool, bool]:
        """Walk one sampled path, honouring stop and target along the way.

        Checking barriers step by step rather than at the endpoint is the
        difference between a search that understands stops and one that does
        not: at the endpoint, a path that dipped through the stop and recovered
        looks identical to one that never dipped at all.
        """
        n = replace(s)
        dd = 0.0
        peak = n.equity_delta()
        hit_stop = hit_tgt = False
        steps = path.size
        dt = self.cfg.horizon_s / max(steps, 1)

        for k in range(steps):
            px = float(path[k])
            n.px = px
            n.t_s += dt
            eq = n.equity_delta()
            if eq > peak:
                peak = eq
            dd = max(dd, peak - eq)
            if n.qty == 0:
                continue
            long_ = n.qty > 0
            if n.stop > 0 and ((long_ and px <= n.stop) or
                               (not long_ and px >= n.stop)):
                # Assume the stop fills at the stop level plus slippage. A stop
                # is a market order once triggered; pretending it fills exactly
                # at the trigger is the most common way a backtest flatters
                # itself, and the error is largest precisely in the fast moves
                # where stops actually get hit.
                slip = ctx.atr * 0.10 + px * 0.0002
                fill = n.stop - (slip if long_ else -slip)
                n.realised += (fill - n.avg) * n.qty
                n.fees += self.cost.cost_rupees(
                    abs(n.qty), fill, Side.SELL if long_ else Side.BUY,
                    3.0, max(abs(n.qty), 1), False, ctx.liquidity)
                n.qty = 0
                n.avg = 0.0
                n.closed_reason = "stop"
                hit_stop = True
                eq = n.equity_delta()
                dd = max(dd, peak - eq)
                continue
            if n.target > 0 and ((long_ and px >= n.target) or
                                 (not long_ and px <= n.target)):
                fill = n.target
                n.realised += (fill - n.avg) * n.qty
                n.fees += self.cost.cost_rupees(
                    abs(n.qty), fill, Side.SELL if long_ else Side.BUY,
                    3.0, max(abs(n.qty), 1), False, ctx.liquidity)
                n.qty = 0
                n.avg = 0.0
                n.closed_reason = "target"
                hit_tgt = True
                continue
        # Carry the peak forward. It was a local until the give-back rule
        # needed it: a child node has to know how far the line had run before
        # it got there, or the deeper plies evaluate a stop policy the shallow
        # ones would have armed.
        n.peak = peak
        n.max_dd = max(n.max_dd, dd)
        return n, dd, hit_stop, hit_tgt

    # ------------------------------------------------------------------
    def _child_ctx(self, ctx: MoveContext, s: SimState,
                   elapsed: float) -> MoveContext:
        """Context for a node one ply deeper.

        The prediction is *decayed toward neutral* with depth. Believing the
        current model output still applies three plies into the future is the
        classic way a deep search convinces itself of a fantasy: the edge is a
        short-horizon estimate, and pretending it persists makes deep lines
        look better than shallow ones for no reason other than that they are
        deeper.
        """
        decay = 0.55
        p = Prediction(
            p_up=0.5 + (ctx.pred.p_up - 0.5) * decay,
            exp_return=ctx.pred.exp_return * decay,
            exp_vol=ctx.pred.exp_vol,
            p_target=ctx.pred.p_target, p_stop=ctx.pred.p_stop,
            exp_hold_s=ctx.pred.exp_hold_s,
            p_continuation=ctx.pred.p_continuation,
            confidence=ctx.pred.confidence * decay,
            horizon_s=ctx.pred.horizon_s, source=ctx.pred.source,
        )
        r_mult = 0.0
        if s.qty and s.stop > 0 and s.avg > 0:
            risk = abs(s.avg - s.stop) * abs(s.qty)
            if risk > 0:
                r_mult = (s.px - s.avg) * s.qty / risk
        rps = ctx.risk_per_share or (abs(s.avg - s.stop) if s.stop > 0 else 0.0)
        peak_r = 0.0
        if s.qty and rps > 0:
            peak_r = max(ctx.peak_r, s.peak / (rps * abs(s.qty)))
        # A line that closed inside the search starts its own cooldown. Without
        # this, the search prices exit-and-re-enter as if the re-entry were
        # available immediately, which is exactly the move the cooldown exists
        # to forbid -- and a search that plans a move it cannot play is worse
        # than one that never considered it.
        s_since_exit = 0.0 if (s.qty == 0 and ctx.qty != 0) \
            else ctx.s_since_exit + s.t_s
        return MoveContext(
            symbol=ctx.symbol, px=s.px,
            bid=s.px * (1 - 0.5 * ctx.pred.exp_vol * 0.0 - 0.00005),
            ask=s.px * (1 + 0.00005),
            atr=ctx.atr, qty=s.qty, avg_price=s.avg, stop=s.stop,
            target=s.target, max_qty=ctx.max_qty, unit_qty=ctx.unit_qty,
            regime=ctx.regime, alignment=ctx.alignment * decay,
            alignment_conflict=ctx.alignment_conflict, pred=p,
            hold_s=ctx.hold_s + s.t_s, partials=s.partials,
            scale_ins=s.scale_ins, liquidity=ctx.liquidity,
            r_multiple=r_mult,
            seconds_to_close=max(0.0, ctx.seconds_to_close - s.t_s),
            can_open=ctx.can_open,
            expected_mae_pct=ctx.expected_mae_pct,
            peak_r=peak_r, risk_per_share=rps, s_since_exit=s_since_exit,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _rationale(c: Candidate, ev: float, p_win: float, dd: float,
                   cost: float, ctx: MoveContext, p_fill: float) -> str:
        bits = [c.describe()]
        bits.append(f"EV={ev:+.0f}")
        bits.append(f"P(win)={p_win:.0%}")
        bits.append(f"E[dd]={dd:.0f}")
        if cost > 0:
            bits.append(f"cost={cost:.0f}")
        if c.passive:
            bits.append(f"P(fill)={p_fill:.0%}")
        if c.tag:
            bits.append(c.tag)
        return " | ".join(bits)
