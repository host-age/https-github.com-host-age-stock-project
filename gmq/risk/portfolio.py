"""Portfolio state and portfolio-level risk (spec §10).

The central idea: **positions are not independent bets.** Five long positions
in HDFCBANK, ICICIBANK, SBIN, AXISBANK and KOTAKBANK are not five trades at
0.5% risk each; on any day that matters they are one trade at roughly 2.5%
risk, because they will all be wrong together. This module computes the
correlation structure and reports effective (correlation-adjusted) exposure,
which is what the risk engine actually limits.

Everything here is measurement. Enforcement lives in `engine.py`.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from ..core.types import Position, Fill, Side, NS
from ..core.config import sector_of
from ..core.mathx import clamp, var_es, max_drawdown, safe_div


@dataclass
class TradeRecord:
    """One completed round trip -- the unit of performance analysis."""
    symbol: str
    side: int
    qty: int
    entry_px: float
    exit_px: float
    entry_ns: int
    exit_ns: int
    pnl: float
    fees: float
    #: None when the entry risk was never established -- an unmeasurable
    #: trade, excluded from R-based aggregates rather than faked
    r_multiple: Optional[float]
    mfe: float
    mae: float
    exit_reason: str
    #: 1R in rupees, as fixed at entry. 0.0 when it was never established --
    #: the same unmeasurable case r_multiple reports as None.
    initial_risk: float = 0.0
    entry_regime: str = ""
    exit_regime: str = ""
    decision_id: str = ""
    thesis: str = ""
    confidence: float = 0.0
    predicted_p_up: float = 0.5
    slippage_bps: float = 0.0
    hold_s: float = 0.0
    thesis_correct: Optional[bool] = None


class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[int, float]] = []
        self.peak_equity = float(initial_capital)
        self.day_start_equity = float(initial_capital)
        self.day_trades = 0
        self.total_fees = 0.0
        self.realised_today = 0.0
        self.consecutive_losses = 0
        self.recent_pnl: Deque[Tuple[int, float]] = deque(maxlen=4000)
        self._last_marks: Dict[str, float] = {}
        self._ret_ring: Deque[float] = deque(maxlen=2000)
        self._last_equity = float(initial_capital)

    # ------------------------------------------------------------------
    def position(self, symbol: str) -> Position:
        p = self.positions.get(symbol)
        if p is None:
            p = Position(symbol=symbol)
            self.positions[symbol] = p
        return p

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.qty != 0]

    def mark(self, symbol: str, px: float) -> None:
        if px > 0:
            self._last_marks[symbol] = px
            p = self.positions.get(symbol)
            if p is not None and p.qty:
                p.update_excursions(px)

    def last(self, symbol: str) -> float:
        return self._last_marks.get(symbol, 0.0)

    # ------------------------------------------------------------------
    def equity(self) -> float:
        """Cash plus the mark-to-market value of open positions.

        Note this is `qty * price`, not `unrealised`. Cash was already debited
        by the full purchase price when the fill was applied, so adding only
        the unrealised gain would count the principal as spent and gone --
        which understates equity by the entire notional of the book and makes
        every percentage-of-equity risk limit fire far too early.
        """
        e = self.cash
        for p in self.positions.values():
            if p.qty:
                px = self._last_marks.get(p.symbol, p.avg_price)
                e += p.qty * px
        return e

    def record_equity(self, ts: int) -> float:
        e = self.equity()
        self.equity_curve.append((ts, e))
        if e > self.peak_equity:
            self.peak_equity = e
        if self._last_equity > 0:
            self._ret_ring.append(e / self._last_equity - 1.0)
        self._last_equity = e
        return e

    def drawdown_pct(self) -> float:
        e = self.equity()
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - e) / self.peak_equity * 100.0)

    def day_pnl(self) -> float:
        return self.equity() - self.day_start_equity

    def day_pnl_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return self.day_pnl() / self.day_start_equity * 100.0

    # ------------------------------------------------------------------
    def apply_fill(self, f: Fill, decision_id: str = "",
                   regime: str = "") -> Optional[TradeRecord]:
        p = self.position(f.symbol)
        was_qty = p.qty
        entry_px, entry_ns = p.avg_price, p.opened_ns
        mfe, mae = p.mfe, p.mae
        booked = p.apply_fill(f)
        self.cash -= f.qty * f.side.sign * f.price
        self.cash -= f.fee
        self.total_fees += f.fee
        self.realised_today += booked
        self.recent_pnl.append((f.ts, booked))
        if decision_id and p.decision_id == "":
            p.decision_id = decision_id
        if regime and not p.entry_regime:
            p.entry_regime = regime

        rec = None
        if was_qty != 0 and p.qty == 0:
            # An unknown risk is reported as unknown, never as a sentinel.
            # Dividing by 1e-9 to "avoid a crash" turns one unmeasurable trade
            # into an expectancy of 1.7e7 R and destroys the whole report --
            # loudly enough to notice here, quietly enough to believe if the
            # number had been merely large.
            risk = p.initial_risk if p.initial_risk > 0 else None
            rec = TradeRecord(
                symbol=f.symbol, side=1 if was_qty > 0 else -1,
                qty=abs(was_qty), entry_px=entry_px, exit_px=f.price,
                entry_ns=entry_ns, exit_ns=f.ts,
                pnl=p.realised, fees=p.fees,
                # Net of fees, like every other performance number in the
                # system. Reporting R gross while reporting rupees net is how
                # a run shows a 42% win rate and +1.80R expectancy in the same
                # table -- two different trades being described. It only became
                # visible once fees were actually being charged.
                r_multiple=((p.realised - p.fees) / risk) if risk else None,
                mfe=mfe, mae=mae, exit_reason="",
                initial_risk=float(risk or 0.0),
                entry_regime=p.entry_regime, exit_regime=regime,
                decision_id=p.decision_id, thesis=p.thesis,
                confidence=p.entry_confidence,
                hold_s=(f.ts - entry_ns) / NS if entry_ns else 0.0,
            )
            self.trades.append(rec)
            self.day_trades += 1
            if p.realised - p.fees < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            # reset for the next life of this symbol
            self.positions[f.symbol] = Position(symbol=f.symbol)
        return rec

    # ------------------------------------------------------------------
    def gross_exposure(self) -> float:
        return sum(abs(p.qty) * self._last_marks.get(p.symbol, p.avg_price)
                   for p in self.positions.values() if p.qty)

    def net_exposure(self) -> float:
        return sum(p.qty * self._last_marks.get(p.symbol, p.avg_price)
                   for p in self.positions.values() if p.qty)

    def gross_pct(self) -> float:
        e = self.equity()
        return safe_div(self.gross_exposure(), e) * 100.0

    def net_pct(self) -> float:
        e = self.equity()
        return safe_div(self.net_exposure(), e) * 100.0

    def leverage(self) -> float:
        e = self.equity()
        return safe_div(self.gross_exposure(), e)

    def sector_exposure(self) -> Dict[str, float]:
        out: Dict[str, float] = defaultdict(float)
        for p in self.positions.values():
            if not p.qty:
                continue
            px = self._last_marks.get(p.symbol, p.avg_price)
            out[sector_of(p.symbol)] += abs(p.qty) * px
        e = self.equity()
        return {k: v / e * 100.0 for k, v in out.items()} if e > 0 else {}

    def open_risk(self) -> float:
        """Total rupees between here and every stop. The real exposure number:
        gross notional is what you own, this is what you can lose."""
        tot = 0.0
        for p in self.positions.values():
            if not p.qty or p.stop <= 0:
                continue
            px = self._last_marks.get(p.symbol, p.avg_price)
            tot += abs(px - p.stop) * abs(p.qty)
        return tot

    def open_risk_pct(self) -> float:
        return safe_div(self.open_risk(), self.equity()) * 100.0

    # ------------------------------------------------------------------
    def correlated_exposure(self, cross, threshold: float = 0.65
                            ) -> Tuple[float, List[List[str]]]:
        """Largest correlation-adjusted cluster exposure, as % of equity.

        Positions are grouped so that any two names correlated above the
        threshold *and pointing the same way* land in the same cluster. Signs
        matter: long HDFCBANK and short ICICIBANK at rho=0.8 is a spread, not a
        concentration, and a limit that treated it as one would forbid the
        least risky thing in the book.
        """
        open_ = [p for p in self.positions.values() if p.qty]
        if not open_:
            return 0.0, []
        n = len(open_)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(n):
            for j in range(i + 1, n):
                rho = cross.correlation(open_[i].symbol, open_[j].symbol)
                same_dir = (open_[i].direction == open_[j].direction)
                # a positive correlation with the same sign, or a negative
                # correlation with opposite signs, is the same underlying bet
                if (rho >= threshold and same_dir) or \
                   (rho <= -threshold and not same_dir):
                    union(i, j)

        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        e = self.equity()
        worst = 0.0
        clusters: List[List[str]] = []
        for members in groups.values():
            expo = sum(abs(open_[i].qty) *
                       self._last_marks.get(open_[i].symbol, open_[i].avg_price)
                       for i in members)
            pct = safe_div(expo, e) * 100.0
            worst = max(worst, pct)
            if len(members) > 1:
                clusters.append([open_[i].symbol for i in members])
        return worst, clusters

    # ------------------------------------------------------------------
    def var_es(self, cross, horizon_days: float = 1.0,
               alpha: float = 0.99) -> Tuple[float, float]:
        """Parametric VaR / Expected Shortfall as % of equity.

        Uses the live correlation matrix rather than summing per-position VaR.
        Summing them assumes perfect correlation, which overstates a diversified
        book and -- far worse -- understates nothing, so it feels safe while
        making the limit meaningless.
        """
        open_ = [p for p in self.positions.values() if p.qty]
        e = self.equity()
        if not open_ or e <= 0:
            return 0.0, 0.0
        syms = [p.symbol for p in open_]
        w = np.array([p.qty * self._last_marks.get(p.symbol, p.avg_price) / e
                      for p in open_])
        vols = []
        for s in syms:
            r = cross.ret.get(s)
            if r is not None and r.n > 30:
                # per-bar vol -> daily (375 one-minute bars in an NSE session)
                vols.append(float(r.view().std()) * math.sqrt(375.0))
            else:
                vols.append(0.015)
        v = np.array(vols)
        C = cross.corr_matrix(syms)
        cov = np.outer(v, v) * C
        port_var = float(w @ cov @ w) * horizon_days
        sd = math.sqrt(max(port_var, 0.0))
        z = 2.326 if alpha >= 0.99 else 1.645
        var = z * sd
        # ES for a normal: sigma * phi(z) / (1 - alpha)
        phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        es = sd * phi / (1.0 - alpha)
        return var * 100.0, es * 100.0

    def historical_var_es(self, alpha: float = 0.99) -> Tuple[float, float]:
        if len(self._ret_ring) < 50:
            return 0.0, 0.0
        v, e = var_es(np.asarray(self._ret_ring), alpha)
        return v * 100.0, e * 100.0

    # ------------------------------------------------------------------
    def loss_velocity(self, now_ns: int, window_s: float) -> float:
        """Realised loss inside a trailing window, as % of equity.

        A slow bleed and a fast one need different responses. Daily-loss limits
        alone let an account lose its whole budget in ninety seconds and call
        it compliant.
        """
        cutoff = now_ns - int(window_s * NS)
        tot = sum(p for ts, p in self.recent_pnl if ts >= cutoff)
        e = self.equity()
        return safe_div(tot, e) * 100.0 if e > 0 else 0.0

    def roll_day(self, ts: int) -> None:
        self.day_start_equity = self.equity()
        self.day_trades = 0
        self.realised_today = 0.0

    def snapshot(self, cross=None) -> dict:
        e = self.equity()
        d = {
            "equity": round(e, 2),
            "cash": round(self.cash, 2),
            "initial": self.initial,
            "return_pct": round((e / self.initial - 1) * 100, 4),
            "day_pnl": round(self.day_pnl(), 2),
            "day_pnl_pct": round(self.day_pnl_pct(), 4),
            "drawdown_pct": round(self.drawdown_pct(), 4),
            "peak_equity": round(self.peak_equity, 2),
            "gross_pct": round(self.gross_pct(), 2),
            "net_pct": round(self.net_pct(), 2),
            "leverage": round(self.leverage(), 3),
            "open_positions": len(self.open_positions),
            "open_risk_pct": round(self.open_risk_pct(), 3),
            "sector_exposure": {k: round(v, 2)
                                for k, v in self.sector_exposure().items()},
            "trades": len(self.trades),
            "day_trades": self.day_trades,
            "fees": round(self.total_fees, 2),
            "consecutive_losses": self.consecutive_losses,
        }
        if cross is not None:
            worst, clusters = self.correlated_exposure(cross)
            v, es = self.var_es(cross)
            d.update({"max_cluster_pct": round(worst, 2),
                      "clusters": clusters,
                      "var_99_pct": round(v, 3),
                      "es_99_pct": round(es, 3)})
        return d
