"""The evaluation function and the cost model.

`evaluate` is the search's leaf heuristic -- the analogue of a chess engine's
static evaluation. It answers: given that we end a line holding this position
at this price with this stop, how good is that, in rupees?

`CostModel` is the other half of honesty. A search that ignores costs will
happily discover that flipping the position every thirty seconds has positive
expected value. Every move is charged:

  * half the spread on a taking order (a passive order pays none, but risks
    not filling, which the fill-probability model handles)
  * brokerage and statutory charges, which in India are not negligible: STT
    on the sell side, exchange transaction charges, stamp duty, GST
  * square-root market impact, scaled by the visible depth

The objective (spec §15) deliberately does not maximise expected return. It
maximises expected return minus penalties on drawdown, variance and tail loss,
so a line with a 90% chance of a small gain and a 10% chance of a catastrophic
one scores below a steadier line with a lower mean.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..core.types import Side, Instrument, DepthSnapshot
from ..core.mathx import clamp

# Fraction of the quoted spread a passive fill gives back to adverse selection.
# Empirically around a third of the spread for liquid large-caps and worse in
# thin names; 0.35 is a deliberately unflattering central estimate. Without it,
# "post on the bid" evaluates as free money and the engine will discover that.
ADVERSE_SELECTION = 0.35


@dataclass
class CostModel:
    """India-specific round-trip costs, in fractions of turnover."""
    brokerage_bps: float = 0.30          # discount broker, capped per order
    stt_sell_bps: float = 2.50           # securities transaction tax, sell side
    exchange_bps: float = 0.32
    stamp_buy_bps: float = 0.30
    gst_on_charges: float = 0.18
    sebi_bps: float = 0.01
    impact_coef: float = 0.55            # sqrt-impact coefficient
    max_impact_bps: float = 60.0

    def fees(self, notional: float, side: Side) -> float:
        """Statutory + brokerage, in rupees. Not symmetric: STT is sell-side."""
        if notional <= 0:
            return 0.0
        bro = notional * self.brokerage_bps / 1e4
        exch = notional * self.exchange_bps / 1e4
        sebi = notional * self.sebi_bps / 1e4
        gst = (bro + exch) * self.gst_on_charges
        stt = notional * self.stt_sell_bps / 1e4 if side is Side.SELL else 0.0
        stamp = notional * self.stamp_buy_bps / 1e4 if side is Side.BUY else 0.0
        return bro + exch + sebi + gst + stt + stamp

    def impact_bps(self, qty: int, top_qty: int, spread_bps: float,
                   liquidity: float = 1.0) -> float:
        """Square-root impact: cost grows with the square root of size
        relative to available depth, not linearly."""
        if qty <= 0:
            return 0.0
        depth = max(float(top_qty), 1.0)
        participation = qty / depth
        imp = self.impact_coef * spread_bps * math.sqrt(max(participation, 0.0))
        imp /= max(liquidity, 0.15)
        return float(min(imp, self.max_impact_bps))

    def total_bps(self, qty: int, px: float, side: Side, spread_bps: float,
                  top_qty: int, passive: bool = False,
                  liquidity: float = 1.0) -> float:
        """All-in cost of one order, in bps of its own notional."""
        notional = qty * px
        if notional <= 0:
            return 0.0
        fee_bps = self.fees(notional, side) / notional * 1e4
        if passive:
            # A resting order earns the spread instead of paying it -- but the
            # spread credit is not the whole story, and treating it as such is
            # how a backtest talks itself into "just post on the bid" as a
            # strategy. A limit order fills when the market comes to it, which
            # is disproportionately when the market is about to keep going
            # through it. That adverse selection is charged here as a fraction
            # of the spread; the net is still cheaper than crossing, but it is
            # not free, and in a wide-spread name it is not close to free.
            return fee_bps - spread_bps * 0.25 + spread_bps * ADVERSE_SELECTION
        cross = spread_bps * 0.5
        return fee_bps + cross + self.impact_bps(qty, top_qty, spread_bps,
                                                 liquidity)

    def cost_rupees(self, qty: int, px: float, side: Side, spread_bps: float,
                    top_qty: int, passive: bool = False,
                    liquidity: float = 1.0) -> float:
        bps = self.total_bps(qty, px, side, spread_bps, top_qty, passive,
                             liquidity)
        return abs(qty) * px * bps / 1e4


def fill_probability(passive: bool, spread_bps: float, queue_ahead: int,
                     top_qty: int, horizon_s: float, trade_rate: float,
                     imbalance: float, side: Side) -> float:
    """Probability a passive order fills within the horizon.

    A limit order at the touch is not free money: it fills when the market
    comes to it, which is disproportionately when the market is about to keep
    going through it. The imbalance term captures that adverse selection --
    a bid fills easily when sellers are pressing, which is exactly when being
    long is worst.
    """
    if not passive:
        return 1.0
    if top_qty <= 0:
        return 0.3
    ahead = max(queue_ahead, 0)
    # expected volume traded at this level over the horizon
    expected = max(trade_rate * horizon_s, 1.0)
    base = clamp(expected / max(ahead + top_qty * 0.5, 1.0), 0.0, 1.0)
    # pressure from the other side helps a resting order fill
    press = imbalance * (-side.sign)
    return float(clamp(base * (1.0 + 0.5 * press), 0.02, 0.98))


# --------------------------------------------------------------------------


def evaluate(qty: int, px: float, avg: float, stop: float, target: float,
             realised: float, fees: float, atr: float,
             unrealised_only: bool = False, exit_cost: float = 0.0) -> float:
    """Static evaluation of a terminal node, in rupees.

    Open risk is charged against the position: holding a live position is not
    the same as holding cash of equal mark-to-market value, because the
    distance to the stop is a real liability. Without this term the search
    prefers to stay in every position forever, since an open position's
    expected value never gets debited for the risk it carries.

    `exit_cost` values an open position at what it is *worth*, which is what
    it would fetch on liquidation, not what the screen says. The distinction
    is not academic. Without it, entering charges the entry cost while the
    matching exit cost is only ever charged on the paths where a stop or
    target happens to trigger -- so every path that simply runs out of horizon
    gets out for free, and the search sees a round trip as costing roughly
    half of one. At a 4.3bp round trip against an average gross win of about
    the same size, that mispricing is the entire difference between a strategy
    and a fee generator.
    """
    unreal = (px - avg) * qty if qty else 0.0
    v = realised + unreal - fees
    if qty:
        v -= max(exit_cost, 0.0)
    if qty and stop > 0:
        # Carry charge on open risk. Holding a live position is not equivalent
        # to holding cash of the same mark-to-market value: the distance to the
        # stop is a real liability that has not been settled yet. Kept small
        # (4% of open risk) because expected drawdown and tail loss are priced
        # separately in the objective -- charging the full risk here as well
        # would double-count it and, once again, stop the engine trading.
        open_risk = abs(px - stop) * abs(qty)
        v -= 0.04 * open_risk
    return float(v)


def objective(ev: float, variance: float, exp_dd: float, cvar: float,
              capital_used: float, equity: float,
              risk_aversion: float = 2.0, dd_penalty: float = 0.08,
              cvar_penalty: float = 0.06, util_penalty: float = 0.02) -> float:
    """Risk-adjusted score (spec §15), in rupees. Higher is better.

    Calibration note, because getting this wrong silently disables the whole
    engine. The tempting form is `EV - k * stdev`, but a standard deviation is
    the same order of magnitude as the *position*, while EV is a small fraction
    of it. On a typical trade EV might be 180 rupees against a 600-rupee
    standard deviation, so `EV - 1.25 * sd` demands a per-trade Sharpe above
    1.25 before any trade is worth taking -- and the engine simply never
    trades. Nothing looks broken; it just sits there.

    The correct form for variance is quadratic utility over *wealth*: the
    penalty is gamma/2 * Var / equity, which is the Arrow-Pratt form and has
    the right property that a bet small relative to the account is judged
    almost purely on its mean. Drawdown and tail loss then carry the risk
    control, as linear penalties with coefficients that read directly: at
    dd_penalty=0.16, one rupee of expected drawdown cancels sixteen paise of
    expected profit.

    Transaction costs are deliberately NOT a parameter here. They are already
    booked into each simulated line's fees and therefore already inside `ev`;
    subtracting them again halved every entry's score and was enough on its
    own to keep the engine permanently flat.

    Drawdown and tail loss are both downside measures of the same outcome
    distribution, so they are weighted as a pair rather than each at "what
    feels right" individually -- otherwise the same risk is paid for twice and,
    again, nothing ever clears the bar.

    Hard limits are not represented here at all. They belong in the risk
    engine, which can veto; an objective that merely dislikes a breach can
    always be talked into one by a large enough number.
    """
    util = (capital_used / equity) if equity > 0 else 0.0
    var_pen = (risk_aversion * max(variance, 0.0) / (2.0 * equity)) \
        if equity > 0 else 0.0
    return float(
        ev
        - var_pen
        - dd_penalty * max(exp_dd, 0.0)
        - cvar_penalty * max(cvar, 0.0)
        - util_penalty * util * abs(ev)
    )
