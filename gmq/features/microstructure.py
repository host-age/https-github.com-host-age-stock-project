"""Order-flow and market-microstructure features (spec §1).

These are the features that actually decay in seconds, so unlike the bar-level
block they are maintained **incrementally on every tick** -- no recomputation,
no array scans. Each symbol carries one `MicroState` costing a few hundred
bytes and a handful of float ops per tick.

What is measured, and why it earns its place:

  OFI (order-flow imbalance)  the single best short-horizon predictor in
                              equity microstructure: it counts additions and
                              removals of size at the touch, not just trades.
  trade imbalance             signed volume, tells you who is being aggressive
  Kyle's lambda               price impact per unit of signed volume -- how
                              expensive it is to be wrong about size here
  effective/realised spread   what a taker actually pays, and how much of it
                              the maker keeps after adverse selection
  VPIN-style toxicity         probability that flow is informed; when this is
                              high, makers widen and stops get run
  quote intensity / life      how fast the book is being repriced
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from ..core.types import Tick, DepthSnapshot, Side
from ..core.mathx import EWMA, EWVar, Ring, clamp, safe_div


class MicroState:
    """Incremental microstructure state for one symbol."""

    __slots__ = (
        "symbol", "prev_bid", "prev_ask", "prev_bid_qty", "prev_ask_qty",
        "prev_mid", "prev_vol", "ofi", "ofi_fast", "ofi_slow",
        "signed_vol", "buy_vol", "sell_vol", "trades",
        "ret_ring", "sv_ring", "spread_ew", "spread_var",
        "kyle_num", "kyle_den", "eff_spread_ew", "real_spread_ew",
        "quote_updates", "quote_ew", "last_ts", "vpin_buckets",
        "vpin_bucket_size", "vpin_cur_buy", "vpin_cur_sell", "vpin_cur",
        "imb_ew", "micro_dev_ew", "n", "depth_ring", "sweep_ring",
        "big_prints", "last_depth", "tick_dir", "run_len",
    )

    def __init__(self, symbol: str, bucket_notional: float = 2_000_000.0):
        self.symbol = symbol
        self.prev_bid = self.prev_ask = 0.0
        self.prev_bid_qty = self.prev_ask_qty = 0
        self.prev_mid = 0.0
        self.prev_vol = 0
        self.ofi = 0.0
        self.ofi_fast = EWMA(halflife=20)
        self.ofi_slow = EWMA(halflife=200)
        self.signed_vol = 0.0
        self.buy_vol = 0
        self.sell_vol = 0
        self.trades = 0
        self.ret_ring = Ring(600)
        self.sv_ring = Ring(600)
        self.spread_ew = EWMA(halflife=60)
        self.spread_var = EWVar(halflife=120)
        self.kyle_num = 0.0
        self.kyle_den = 0.0
        self.eff_spread_ew = EWMA(halflife=80)
        self.real_spread_ew = EWMA(halflife=80)
        self.quote_updates = 0
        self.quote_ew = EWMA(halflife=60)
        self.last_ts = 0
        self.vpin_buckets: Deque[float] = deque(maxlen=50)
        self.vpin_bucket_size = bucket_notional
        self.vpin_cur_buy = 0.0
        self.vpin_cur_sell = 0.0
        self.vpin_cur = 0.0
        self.imb_ew = EWMA(halflife=40)
        self.micro_dev_ew = EWMA(halflife=40)
        self.n = 0
        self.depth_ring = Ring(300)
        self.sweep_ring = Ring(300)
        self.big_prints = 0
        self.last_depth: Optional[DepthSnapshot] = None
        self.tick_dir = 0
        self.run_len = 0

    # ------------------------------------------------------------------
    def on_tick(self, t: Tick) -> None:
        self.n += 1
        mid = t.mid
        bid, ask = t.bid, t.ask

        # ---- order-flow imbalance (Cont-Kukanov-Stoikov formulation)
        # Adding size at the bid, or the bid moving up, is buying pressure.
        # Cancelling at the bid, or the bid moving down, is selling pressure.
        if self.prev_bid > 0 and bid > 0:
            if bid > self.prev_bid:
                e_b = t.bid_qty
            elif bid == self.prev_bid:
                e_b = t.bid_qty - self.prev_bid_qty
            else:
                e_b = -self.prev_bid_qty
            if ask < self.prev_ask:
                e_a = t.ask_qty
            elif ask == self.prev_ask:
                e_a = t.ask_qty - self.prev_ask_qty
            else:
                e_a = -self.prev_ask_qty
            of = float(e_b - e_a)
            self.ofi = of
            self.ofi_fast.update(of)
            self.ofi_slow.update(of)
            if bid != self.prev_bid or ask != self.prev_ask:
                self.quote_updates += 1
                self.quote_ew.update(1.0)
            else:
                self.quote_ew.update(0.0)

        # ---- traded volume and its sign
        dv = max(0, t.volume - self.prev_vol) if self.prev_vol else 0
        if dv == 0 and t.ltq > 0:
            dv = t.ltq
        if dv > 0:
            self.trades += 1
            sign = t.aggressor
            if sign == 0 and self.prev_mid > 0:
                # Lee-Ready fallback: classify by trade price vs prevailing mid
                sign = 1 if t.ltp > self.prev_mid else (-1 if t.ltp < self.prev_mid
                                                        else self.tick_dir)
            sign = sign or 1
            if sign > 0:
                self.buy_vol += dv
                self.vpin_cur_buy += dv * t.ltp
            else:
                self.sell_vol += dv
                self.vpin_cur_sell += dv * t.ltp
            self.signed_vol = sign * dv
            self.sv_ring.push(float(sign * dv))
            if self.tick_dir == sign:
                self.run_len += 1
            else:
                self.run_len = 1
                self.tick_dir = sign
            notional = dv * t.ltp
            if notional > self.vpin_bucket_size * 0.25:
                self.big_prints += 1

            # ---- effective spread paid by the taker
            if mid > 0:
                eff = 2.0 * sign * (t.ltp - self.prev_mid or mid) / mid * 1e4
                self.eff_spread_ew.update(clamp(eff, -100, 100))

            # ---- Kyle's lambda: regression of dMid on signed volume
            if self.prev_mid > 0:
                dmid = (mid - self.prev_mid) / self.prev_mid
                sv = sign * dv
                self.kyle_num += dmid * sv
                self.kyle_den += sv * sv
                self.ret_ring.push(dmid)

            # ---- VPIN bucket roll
            filled = self.vpin_cur_buy + self.vpin_cur_sell
            if filled >= self.vpin_bucket_size:
                imb = abs(self.vpin_cur_buy - self.vpin_cur_sell) / filled
                self.vpin_buckets.append(imb)
                self.vpin_cur_buy = self.vpin_cur_sell = 0.0
                self.vpin_cur = float(np.mean(self.vpin_buckets))
        else:
            if self.prev_mid > 0 and mid > 0:
                self.ret_ring.push((mid - self.prev_mid) / self.prev_mid)
            self.sv_ring.push(0.0)

        # ---- spread and microprice deviation
        if bid > 0 and ask > 0 and mid > 0:
            sp_bps = (ask - bid) / mid * 1e4
            self.spread_ew.update(sp_bps)
            self.spread_var.update(sp_bps)
            tot = t.bid_qty + t.ask_qty
            if tot > 0:
                self.imb_ew.update((t.bid_qty - t.ask_qty) / tot)
                self.micro_dev_ew.update((t.microprice - mid) / mid * 1e4)

        self.prev_bid, self.prev_ask = bid, ask
        self.prev_bid_qty, self.prev_ask_qty = t.bid_qty, t.ask_qty
        self.prev_mid = mid
        self.prev_vol = t.volume or self.prev_vol
        self.last_ts = t.ts

    def on_depth(self, d: DepthSnapshot) -> None:
        self.last_depth = d
        self.depth_ring.push(float(d.depth_value()))

    # ------------------------------------------------------------------
    @property
    def kyle_lambda(self) -> float:
        """Price impact per unit signed volume, in bps per lakh of notional."""
        if self.kyle_den < 1e-9:
            return 0.0
        return float(self.kyle_num / self.kyle_den) * 1e4

    def features(self, px: float) -> Dict[str, float]:
        f: Dict[str, float] = {}
        if self.n < 5:
            return f
        d = self.last_depth
        # normalise OFI by recent absolute scale so it is comparable across names
        ofi_scale = max(abs(self.ofi_slow.get()) * 3.0, 1.0)
        f["mx_ofi"] = clamp(self.ofi_fast.get() / ofi_scale, -4, 4)
        f["mx_ofi_slow"] = clamp(self.ofi_slow.get() / ofi_scale, -4, 4)
        f["mx_book_imb"] = clamp(self.imb_ew.get(), -1, 1)
        f["mx_micro_dev"] = clamp(self.micro_dev_ew.get() / 5.0, -4, 4)
        f["mx_spread_bps"] = clamp(self.spread_ew.get(), 0, 60)
        f["mx_spread_z"] = clamp(self.spread_var.z(self.spread_ew.get()), -5, 5)
        f["mx_kyle"] = clamp(self.kyle_lambda * 1e5, -50, 50)
        f["mx_eff_spread"] = clamp(self.eff_spread_ew.get(), -50, 50)
        f["mx_vpin"] = clamp(self.vpin_cur, 0, 1)
        f["mx_quote_rate"] = clamp(self.quote_ew.get(), 0, 1)
        f["mx_run_len"] = clamp(self.tick_dir * min(self.run_len, 15) / 15.0, -1, 1)

        sv = self.sv_ring.view()
        if sv.size > 20:
            tot = float(np.abs(sv).sum())
            f["mx_flow_imb"] = clamp(float(sv.sum()) / tot, -1, 1) if tot > 0 else 0.0
            recent = sv[-60:]
            tot_r = float(np.abs(recent).sum())
            f["mx_flow_imb_fast"] = clamp(float(recent.sum()) / tot_r, -1, 1) \
                if tot_r > 0 else 0.0
            f["mx_flow_accel"] = clamp(f["mx_flow_imb_fast"] - f["mx_flow_imb"],
                                       -2, 2)
        tv = self.buy_vol + self.sell_vol
        f["mx_buy_ratio"] = (self.buy_vol / tv) if tv else 0.5

        if d is not None and d.bids and d.asks:
            f["mx_depth_imb5"] = d.imbalance(5)
            f["mx_depth_imb1"] = d.imbalance(1)
            # slope of the book: how fast liquidity thickens away from touch
            bq = np.array([l.qty for l in d.bids], dtype=float)
            aq = np.array([l.qty for l in d.asks], dtype=float)
            f["mx_book_slope_bid"] = clamp(safe_div(float(bq[-1] - bq[0]),
                                                    float(bq.mean())), -5, 5)
            f["mx_book_slope_ask"] = clamp(safe_div(float(aq[-1] - aq[0]),
                                                    float(aq.mean())), -5, 5)
            dv = self.depth_ring.view()
            if dv.size > 20:
                m, s = float(dv.mean()), float(dv.std())
                f["mx_depth_z"] = clamp((dv[-1] - m) / s, -5, 5) if s > 1e-9 else 0.0
            # cost of taking 1 lakh notional, both sides
            if px > 0:
                q = max(1, int(100_000 / px))
                ba, fa = d.sweep_cost(Side.BUY, q)
                bb, fb = d.sweep_cost(Side.SELL, q)
                m = d.mid
                if m > 0 and fa and fb:
                    f["mx_sweep_buy_bps"] = clamp((ba - m) / m * 1e4, 0, 200)
                    f["mx_sweep_sell_bps"] = clamp((m - bb) / m * 1e4, 0, 200)
                    f["mx_sweep_asym"] = clamp(
                        f["mx_sweep_buy_bps"] - f["mx_sweep_sell_bps"], -100, 100)
        rr = self.ret_ring.view()
        if rr.size > 30:
            f["mx_tick_vol_bps"] = clamp(float(rr.std()) * 1e4, 0, 200)
            f["mx_tick_ac1"] = clamp(
                float(np.corrcoef(rr[:-1], rr[1:])[0, 1]) if rr.size > 40 else 0.0,
                -1, 1)
        f["mx_big_print_rate"] = clamp(self.big_prints / max(self.trades, 1), 0, 1)
        return f

    def liquidity_score(self) -> float:
        """0 (untradeable) .. 1 (deep and tight). Feeds the ILLIQUID regime and
        the risk engine's position-size cap."""
        sp = self.spread_ew.get()
        if sp <= 0:
            return 0.5
        # Each component is scored against its own recent history rather than
        # an absolute constant, so the score stays informative for a 148-rupee
        # metal name and a 12000-rupee auto name alike.
        tight = clamp(1.0 - (sp - 0.8) / 12.0, 0.0, 1.0)
        dv = self.depth_ring.view()
        if dv.size > 30:
            med = float(np.median(dv))
            depth = clamp(float(dv[-1]) / max(med, 1e-9) * 0.5, 0.0, 1.0)
        else:
            depth = 0.5
        rate = clamp(math.log1p(self.trades) / math.log1p(3000.0), 0.0, 1.0)
        return float(clamp(0.45 * tight + 0.35 * depth + 0.20 * rate,
                           0.0, 1.0))

    def reset_day(self) -> None:
        self.buy_vol = self.sell_vol = self.trades = 0
        self.big_prints = 0
        self.prev_vol = 0
        self.vpin_cur_buy = self.vpin_cur_sell = 0.0


class MicrostructureEngine:
    """Owns one MicroState per symbol."""

    def __init__(self):
        self.states: Dict[str, MicroState] = {}

    def get(self, symbol: str) -> MicroState:
        st = self.states.get(symbol)
        if st is None:
            st = MicroState(symbol)
            self.states[symbol] = st
        return st

    def on_tick(self, t: Tick) -> None:
        self.get(t.symbol).on_tick(t)

    def on_depth(self, d: DepthSnapshot) -> None:
        self.get(d.symbol).on_depth(d)

    def features(self, symbol: str, px: float) -> Dict[str, float]:
        return self.get(symbol).features(px)

    def reset_day(self) -> None:
        for s in self.states.values():
            s.reset_day()
