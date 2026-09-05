"""The hard risk layer (spec §14).

This is the component the AI cannot argue with.

Structurally, `RiskEngine` holds no reference to the model bank, the search or
the strategy layer, and exposes no method that relaxes a limit. The decision
engine can only *ask*; the answer is a verdict. There is deliberately no
`override`, no `force`, and no confidence level high enough to change a `no`
into a `yes`. That asymmetry is the entire point: a limit that a sufficiently
optimistic model can talk its way past is not a limit, it is a suggestion, and
the day it matters is precisely the day the model is most confident and most
wrong.

The engine also watches the AI itself. If forecast quality collapses, if the
order rate spikes, if rejects pile up, or if realised slippage runs far past
what was modelled, it halts trading on its own -- without needing anything
upstream to notice or agree.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import (
    Side, Order, Position, RiskVerdict, MoveType, NS,
)
from ..core.config import RiskLimits, sector_of
from ..core.mathx import clamp, safe_div
from .portfolio import Portfolio


class HaltReason:
    NONE = ""
    DAILY_LOSS = "DAILY_LOSS_LIMIT"
    DRAWDOWN = "MAX_DRAWDOWN"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    LOSS_VELOCITY = "LOSS_VELOCITY"
    MODEL_DEGRADED = "MODEL_DEGRADED"
    ORDER_RATE = "ORDER_RATE_ANOMALY"
    REJECT_RATE = "REJECT_RATE"
    SLIPPAGE = "EXCESS_SLIPPAGE"
    VAR_BREACH = "VAR_BREACH"
    MANUAL = "MANUAL"
    DATA_STALE = "DATA_STALE"


@dataclass
class TradeIntent:
    """What the decision engine wants to do. The only thing risk ever sees."""
    symbol: str
    move: MoveType
    qty: int                        # signed change in position
    price: float
    stop: float = 0.0
    target: float = 0.0
    decision_id: str = ""
    confidence: float = 0.0
    sector: str = ""
    is_reducing: bool = False       # closing/reducing is always allowed

    def __post_init__(self):
        if not self.sector:
            self.sector = sector_of(self.symbol)


class RiskEngine:
    def __init__(self, limits: RiskLimits, portfolio: Portfolio,
                 cross=None, n_symbols: int = 1):
        self.limits = limits
        self.pf = portfolio
        self.cross = cross
        self.n_symbols = max(1, int(n_symbols))
        self.halted = False
        self.halt_reason = HaltReason.NONE
        self.halt_ns = 0
        self.breaches: List[dict] = []
        self.vetoes = 0
        self.approvals = 0
        self.scale_downs = 0
        self._order_times: Deque[int] = deque(maxlen=600)
        self._rejects: Deque[int] = deque(maxlen=400)
        self._sent: Deque[int] = deque(maxlen=400)
        self._slippage: Deque[float] = deque(maxlen=300)
        self._model_ok = True
        self._model_note = ""
        self._stale: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # state the engine watches on its own
    # ------------------------------------------------------------------
    def on_order_sent(self, ts: int) -> None:
        self._order_times.append(ts)
        self._sent.append(ts)

    def on_order_rejected(self, ts: int) -> None:
        self._rejects.append(ts)

    def on_slippage(self, bps: float) -> None:
        self._slippage.append(float(bps))

    def on_model_health(self, ok: bool, note: str = "") -> None:
        self._model_ok = ok
        self._model_note = note
        # Model quality is a *condition*, not an event. A daily-loss halt is a
        # decision about a day and should stand until the day ends; a halt
        # because forecast skill collapsed should lift when skill recovers,
        # or the first noisy patch permanently disables the system. This is
        # the only automatic un-halt in the engine, and it can only clear the
        # one reason it owns.
        if ok and self.halted and self.halt_reason == HaltReason.MODEL_DEGRADED:
            self.halted = False
            self.halt_reason = HaltReason.NONE
            self.breaches.append({"ts": self.halt_ns, "reason": "MODEL_RECOVERED",
                                  "detail": note,
                                  "equity": round(self.pf.equity(), 2)})

    def on_data_stale(self, symbol: str, stale: bool) -> None:
        self._stale[symbol] = stale

    # ------------------------------------------------------------------
    def _loss_streak_threshold(self, alpha: float = 0.05) -> int:
        """How long a losing streak has to be before it means anything.

        A fixed "halt after 5 losses in a row" sounds prudent and measures
        nothing. At a 50% win rate, a run of five appears somewhere in sixty
        trades roughly 83% of the time -- so on any active day the limit fires
        on chance alone, halts a working strategy, and teaches its operator to
        ignore it.

        The threshold that means something scales with the strategy's own
        observed win rate and how many trades it has actually taken today:
        the shortest streak whose expected number of occurrences in that many
        trades is below `alpha`. The configured constant is kept as a floor so
        a very low-frequency strategy still has a hard backstop.
        """
        trades = self.pf.trades
        n_today = max(self.pf.day_trades, 1)
        if len(trades) >= 30:
            wins = sum(1 for t in trades[-200:] if (t.pnl - t.fees) > 0)
            p_win = clamp(wins / len(trades[-200:]), 0.05, 0.95)
        else:
            p_win = 0.5
        p_loss = 1.0 - p_win
        if p_loss <= 0.01:
            return self.limits.consecutive_loss_halt
        # expected runs of length k in n trades ~ n * p_loss**k
        k = math.log(alpha / n_today) / math.log(p_loss)
        return max(self.limits.consecutive_loss_halt, int(math.ceil(k)))

    def halt(self, reason: str, ts: int = 0, detail: str = "") -> None:
        if self.halted:
            return
        self.halted = True
        self.halt_reason = reason
        self.halt_ns = ts
        self.breaches.append({"ts": ts, "reason": reason, "detail": detail,
                              "equity": round(self.pf.equity(), 2)})

    def resume(self, ts: int = 0) -> None:
        """Manual resume only. Nothing in the automated path calls this."""
        self.halted = False
        self.halt_reason = HaltReason.NONE

    # ------------------------------------------------------------------
    def monitor(self, ts: int) -> Optional[str]:
        """Autonomous circuit breakers. Called on a timer, not by the strategy.

        This runs whether or not the decision engine asks for anything, which
        is what makes it independent: a model that has stopped submitting
        orders because it is confused still gets caught here.
        """
        L = self.limits
        if self.halted:
            return self.halt_reason

        if self.pf.day_pnl_pct() <= -L.max_daily_loss_pct:
            self.halt(HaltReason.DAILY_LOSS, ts,
                      f"day P&L {self.pf.day_pnl_pct():.2f}%")
            return self.halt_reason

        if self.pf.drawdown_pct() >= L.max_drawdown_pct:
            self.halt(HaltReason.DRAWDOWN, ts,
                      f"drawdown {self.pf.drawdown_pct():.2f}%")
            return self.halt_reason

        need = self._loss_streak_threshold()
        if self.pf.consecutive_losses >= need:
            self.halt(HaltReason.CONSECUTIVE_LOSSES, ts,
                      f"{self.pf.consecutive_losses} in a row "
                      f"(surprising above {need})")
            return self.halt_reason

        vel = self.pf.loss_velocity(ts, L.loss_velocity_window_s)
        if vel <= -L.loss_velocity_halt_pct:
            self.halt(HaltReason.LOSS_VELOCITY, ts,
                      f"{vel:.2f}% in {L.loss_velocity_window_s:.0f}s")
            return self.halt_reason

        if not self._model_ok:
            self.halt(HaltReason.MODEL_DEGRADED, ts, self._model_note)
            return self.halt_reason

        # order-rate anomaly: a model in a feedback loop submits far more
        # orders than a working one, and it does it before the P&L notices
        # Sustained rate over five minutes, not an instantaneous burst.
        #
        # A one-minute window cannot tell a runaway loop from a legitimate
        # burst, and legitimate bursts are routine here: a correlated book
        # stops out together, a session squares off eight positions at once.
        # What actually distinguishes a malfunctioning model is that it keeps
        # going, so the detector should measure a sustained rate.
        window_s = 300.0
        cutoff = ts - int(window_s * NS)
        recent = sum(1 for t in self._order_times if t >= cutoff)
        per_min = max(L.max_orders_per_min_per_symbol * self.n_symbols,
                      float(L.max_orders_per_min))
        cap = int(per_min * window_s / 60.0)
        if recent > cap:
            self.halt(HaltReason.ORDER_RATE, ts,
                      f"{recent} decisions in {window_s:.0f}s vs cap {cap}")
            return self.halt_reason

        if len(self._sent) >= 40:
            rr = len(self._rejects) / max(len(self._sent), 1)
            if rr > L.max_reject_rate:
                self.halt(HaltReason.REJECT_RATE, ts, f"reject rate {rr:.2f}")
                return self.halt_reason

        if len(self._slippage) >= 25:
            avg = float(np.mean(list(self._slippage)[-25:]))
            if avg > L.max_slippage_bps_halt:
                self.halt(HaltReason.SLIPPAGE, ts, f"avg slippage {avg:.1f}bps")
                return self.halt_reason

        if self.cross is not None and self.pf.open_positions:
            var, es = self.pf.var_es(self.cross)
            if var > L.max_var_pct or es > L.max_es_pct:
                self.halt(HaltReason.VAR_BREACH, ts,
                          f"VaR {var:.2f}% ES {es:.2f}%")
                return self.halt_reason
        return None

    # ------------------------------------------------------------------
    def check(self, intent: TradeIntent, ts: int = 0) -> RiskVerdict:
        """The gate. Every order in the system passes through here."""
        L = self.limits
        breached: List[str] = []

        # Reducing risk is always permitted -- including while halted. A halt
        # must never trap the book in a position it cannot exit; that would
        # turn a risk control into the risk.
        if intent.is_reducing or intent.move in (
                MoveType.EXIT, MoveType.REDUCE, MoveType.TAKE_PARTIAL):
            self.approvals += 1
            return RiskVerdict(True, "reducing", max_qty=abs(intent.qty))

        if self.halted:
            self.vetoes += 1
            return RiskVerdict(False, f"halted:{self.halt_reason}",
                               breached=[self.halt_reason])

        if intent.move is MoveType.MOVE_STOP:
            self.approvals += 1
            return RiskVerdict(True, "stop_move", max_qty=0)

        if self._stale.get(intent.symbol):
            self.vetoes += 1
            return RiskVerdict(False, "stale_data", breached=["DATA_STALE"])

        qty = abs(intent.qty)
        if qty <= 0:
            return RiskVerdict(True, "no_size", max_qty=0)

        equity = self.pf.equity()
        if equity <= 0:
            self.vetoes += 1
            return RiskVerdict(False, "no_equity", breached=["NO_EQUITY"])

        px = intent.price
        allowed = qty

        # -- per-trade risk
        if intent.stop > 0:
            dist = abs(px - intent.stop)
            if dist > 0:
                budget = equity * L.max_risk_per_trade_pct / 100.0
                cap = int(budget / dist)
                if cap < allowed:
                    allowed = cap
                    breached.append("MAX_RISK_PER_TRADE")
        else:
            # An entry with no stop has unbounded loss. There is no size at
            # which that is acceptable, so it is refused outright rather than
            # scaled down.
            self.vetoes += 1
            return RiskVerdict(False, "no_stop_on_entry",
                               breached=["NO_STOP"])

        # -- single-name concentration
        pos = self.pf.positions.get(intent.symbol)
        cur_notional = abs(pos.qty) * px if pos and pos.qty else 0.0
        room = equity * L.max_position_pct / 100.0 - cur_notional
        cap = int(max(0.0, room) / px) if px > 0 else 0
        if cap < allowed:
            allowed = cap
            breached.append("MAX_POSITION_PCT")

        # -- open position count
        if (not pos or pos.qty == 0) and \
                len(self.pf.open_positions) >= L.max_open_positions:
            self.vetoes += 1
            return RiskVerdict(False, "max_open_positions",
                               breached=["MAX_OPEN_POSITIONS"])

        # -- daily trade count
        if self.pf.day_trades >= L.max_daily_trades:
            self.vetoes += 1
            return RiskVerdict(False, "max_daily_trades",
                               breached=["MAX_DAILY_TRADES"])

        # -- gross / net exposure and leverage
        add_notional = allowed * px
        gross = self.pf.gross_exposure() + add_notional
        if gross / equity * 100.0 > L.max_gross_exposure_pct:
            room = equity * L.max_gross_exposure_pct / 100.0 - \
                self.pf.gross_exposure()
            cap = int(max(0.0, room) / px)
            if cap < allowed:
                allowed = cap
                breached.append("MAX_GROSS_EXPOSURE")

        sign = 1 if intent.qty > 0 else -1
        net = self.pf.net_exposure() + sign * allowed * px
        if abs(net) / equity * 100.0 > L.max_net_exposure_pct:
            room = equity * L.max_net_exposure_pct / 100.0 - \
                abs(self.pf.net_exposure())
            cap = int(max(0.0, room) / px)
            if cap < allowed:
                allowed = cap
                breached.append("MAX_NET_EXPOSURE")

        if (self.pf.gross_exposure() + allowed * px) / equity > L.max_leverage:
            room = equity * L.max_leverage - self.pf.gross_exposure()
            cap = int(max(0.0, room) / px)
            if cap < allowed:
                allowed = cap
                breached.append("MAX_LEVERAGE")

        # -- sector concentration
        sec_exp = self.pf.sector_exposure().get(intent.sector, 0.0)
        sec_room_pct = L.max_sector_exposure_pct - sec_exp
        cap = int(max(0.0, sec_room_pct) / 100.0 * equity / px)
        if cap < allowed:
            allowed = cap
            breached.append("MAX_SECTOR_EXPOSURE")

        # -- correlated cluster: the limit that actually matters
        if self.cross is not None:
            cluster = self.cross.correlated_cluster(intent.symbol,
                                                    L.corr_threshold)
            if cluster:
                expo = 0.0
                for s in cluster:
                    p = self.pf.positions.get(s)
                    if p and p.qty and (p.direction == sign):
                        expo += abs(p.qty) * self.pf.last(s)
                cur_pct = safe_div(expo, equity) * 100.0
                room_pct = L.max_correlated_exposure_pct - cur_pct
                cap = int(max(0.0, room_pct) / 100.0 * equity / px)
                if cap < allowed:
                    allowed = cap
                    breached.append("MAX_CORRELATED_EXPOSURE")

        allowed = max(0, allowed)
        if allowed < L.min_qty:
            self.vetoes += 1
            return RiskVerdict(False, "size_reduced_to_zero: " +
                               ",".join(breached) if breached else "too_small",
                               breached=breached, scaled_from=qty)

        if allowed < qty:
            self.scale_downs += 1
            return RiskVerdict(True, "scaled: " + ",".join(breached),
                               max_qty=allowed, breached=breached,
                               scaled_from=qty)
        self.approvals += 1
        return RiskVerdict(True, "ok", max_qty=allowed)

    # ------------------------------------------------------------------
    def emergency_liquidation_required(self, ts: int) -> Tuple[bool, str]:
        """Conditions that demand flattening now, not merely stopping."""
        L = self.limits
        dd = self.pf.drawdown_pct()
        if dd >= L.max_drawdown_pct * 1.25:
            return True, f"drawdown {dd:.2f}% past hard floor"
        if self.pf.day_pnl_pct() <= -L.max_daily_loss_pct * 1.5:
            return True, f"day loss {self.pf.day_pnl_pct():.2f}%"
        if self.cross is not None and self.pf.open_positions:
            var, es = self.pf.var_es(self.cross)
            if es > L.max_es_pct * 1.5:
                return True, f"expected shortfall {es:.2f}%"
        return False, ""

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        cutoff_rate = len(self._rejects) / max(len(self._sent), 1)
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "approvals": self.approvals,
            "vetoes": self.vetoes,
            "scale_downs": self.scale_downs,
            "breaches": self.breaches[-10:],
            "reject_rate": round(cutoff_rate, 3),
            "avg_slippage_bps": round(float(np.mean(self._slippage)), 2)
            if self._slippage else 0.0,
            "model_ok": self._model_ok,
            "limits": {
                "risk_per_trade_pct": self.limits.max_risk_per_trade_pct,
                "daily_loss_pct": self.limits.max_daily_loss_pct,
                "max_drawdown_pct": self.limits.max_drawdown_pct,
                "max_positions": self.limits.max_open_positions,
            },
        }
