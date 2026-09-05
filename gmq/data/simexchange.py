"""A simulated NSE.

This is not a tick *playback* -- it is a market. Prices are not drawn from a
distribution and printed; they *emerge* from order flow hitting a real matching
engine:

  latent fair value  ->  market makers quote around it (with inventory skew)
                     ->  informed traders lift/hit when value drifts away
                     ->  noise traders trade randomly
                     ->  the book matches, prints, and moves

That matters for this project specifically, because the agent's edge is
supposed to come from microstructure (order-flow imbalance, queue position,
sweep cost, spread dynamics). If ticks were synthesised directly, those
features would be decoration. Here they are causal: an imbalance really does
precede a move, because the flow that creates the imbalance is the flow that
creates the move.

Latent value model per symbol
-----------------------------
  d log V = mu_regime dt + beta_mkt * dM + beta_sec * dS + sigma_t dW + jumps

  * sigma_t follows a GARCH(1,1)-style recursion  -> volatility clustering
  * mu_regime is driven by a Markov chain over the 8 regimes in the spec
  * dM is a common market factor -> correlation between names is real, so the
    portfolio correlation limits in §10 have something to actually bite on
  * jumps model news/events, and fire the EVENT_DRIVEN regime
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np

from ..core.types import (
    Side, Tick, Instrument, InstrumentType, Regime, Fill, DepthSnapshot, NS,
)
from ..core.config import sector_of
from .orderbook import OrderBook


# Markov transition matrix over regimes. Rows/cols follow REGIME_ORDER.
REGIME_ORDER = [
    Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.MEAN_REVERTING,
    Regime.BREAKOUT, Regime.HIGH_VOL, Regime.LOW_VOL,
    Regime.EVENT_DRIVEN, Regime.ILLIQUID,
]

# Persistent by design: real regimes last, they do not flicker every bar.
_P = np.array([
    #  TU     TD     MR     BO     HV     LV     EV     IL
    [0.960, 0.004, 0.014, 0.008, 0.006, 0.006, 0.001, 0.001],  # TRENDING_UP
    [0.004, 0.958, 0.014, 0.008, 0.009, 0.005, 0.001, 0.001],  # TRENDING_DOWN
    [0.012, 0.012, 0.940, 0.020, 0.006, 0.008, 0.001, 0.001],  # MEAN_REVERTING
    [0.070, 0.060, 0.020, 0.820, 0.024, 0.004, 0.001, 0.001],  # BREAKOUT
    [0.020, 0.022, 0.030, 0.030, 0.880, 0.014, 0.003, 0.001],  # HIGH_VOL
    [0.014, 0.010, 0.040, 0.010, 0.006, 0.918, 0.001, 0.001],  # LOW_VOL
    [0.100, 0.090, 0.040, 0.090, 0.130, 0.010, 0.520, 0.020],  # EVENT_DRIVEN
    [0.020, 0.020, 0.060, 0.010, 0.020, 0.060, 0.010, 0.800],  # ILLIQUID
])

# per-regime (drift per second, vol multiplier, jump intensity, liquidity mult)
# Drift is calibrated so a trending regime is actually *detectable*: a drift
# far below the one-minute noise floor is not a trend, it is a rounding error,
# and no detector -- ours or a human's -- could ever confirm it. +/-1e-5 per
# second is ~6bp/min against ~20bp/min noise, i.e. a run of roughly 1.5-2% over
# a half-hour regime. That is what an intraday trend in a large-cap looks like.
REGIME_PARAMS = {
    Regime.TRENDING_UP:    dict(mu=+1.05e-5, vmul=0.95, jump=0.00004, liq=1.05, mr=0.000),
    Regime.TRENDING_DOWN:  dict(mu=-1.10e-5, vmul=1.10, jump=0.00006, liq=0.95, mr=0.000),
    Regime.MEAN_REVERTING: dict(mu=0.0,      vmul=0.85, jump=0.00002, liq=1.15, mr=0.005),
    Regime.BREAKOUT:       dict(mu=0.0,      vmul=1.85, jump=0.00030, liq=0.80, mr=0.000),
    Regime.HIGH_VOL:       dict(mu=0.0,      vmul=2.40, jump=0.00025, liq=0.62, mr=0.001),
    Regime.LOW_VOL:        dict(mu=+0.2e-6,  vmul=0.45, jump=0.00001, liq=1.35, mr=0.003),
    Regime.EVENT_DRIVEN:   dict(mu=0.0,      vmul=3.30, jump=0.00220, liq=0.42, mr=0.000),
    Regime.ILLIQUID:       dict(mu=0.0,      vmul=1.15, jump=0.00008, liq=0.18, mr=0.002),
}


@dataclass
class SymbolState:
    inst: Instrument
    fair: float
    prev_close: float
    regime: Regime = Regime.LOW_VOL
    sigma: float = 0.00009          # per-second vol of log value
    sigma_lr: float = 0.00009       # long-run level for the GARCH pull
    beta_mkt: float = 1.0
    beta_sec: float = 0.5
    book: Optional[OrderBook] = None
    # market-maker inventory -> quote skew (creates real short-term MR)
    mm_inv: int = 0
    mm_target: int = 0
    mm_centre: float = 0.0
    mm_ts: int = 0
    base_qty: int = 400
    day_open: float = 0.0
    anchor: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    oi: int = 0
    last_news: str = ""
    news_until: int = 0
    regime_since: int = 0


class SimExchange:
    """Owns the books, the background participants and the tape."""

    def __init__(self, symbols: List[str], seed: int = 7,
                 index_symbol: str = "NIFTY",
                 on_tick: Optional[Callable[[Tick], None]] = None,
                 on_depth: Optional[Callable[[DepthSnapshot], None]] = None,
                 on_agent_fill: Optional[Callable[[Fill], None]] = None,
                 on_news: Optional[Callable[[dict], None]] = None,
                 depth_levels: int = 5):
        self.rng = np.random.default_rng(seed)
        self.symbols = list(symbols)
        self.index_symbol = index_symbol
        self.on_tick = on_tick
        self.on_depth = on_depth
        self.on_agent_fill = on_agent_fill
        self.on_news = on_news
        self.depth_levels = depth_levels
        self.state: Dict[str, SymbolState] = {}
        self.ts: int = 0
        self._mkt_factor = 0.0
        self._sector_factor: Dict[str, float] = defaultdict(float)
        self._mkt_regime = Regime.LOW_VOL
        self._tape: List[Tuple[int, str, float, int, int]] = []
        self._build(symbols)

    # ------------------------------------------------------------------
    def _build(self, symbols: List[str]) -> None:
        # plausible NSE price levels for the default universe
        seeds = {
            "RELIANCE": 2890.0, "TCS": 3950.0, "HDFCBANK": 1685.0,
            "INFY": 1560.0, "ICICIBANK": 1210.0, "SBIN": 815.0,
            "BHARTIARTL": 1495.0, "ITC": 438.0, "LT": 3620.0,
            "AXISBANK": 1145.0, "KOTAKBANK": 1790.0, "MARUTI": 12450.0,
            "TATAMOTORS": 985.0, "SUNPHARMA": 1720.0, "TATASTEEL": 148.0,
            "WIPRO": 545.0, "HCLTECH": 1680.0, "HINDUNILVR": 2460.0,
            "NIFTY": 24350.0,
        }
        betas = {
            "RELIANCE": 1.10, "TCS": 0.80, "HDFCBANK": 1.05, "INFY": 0.88,
            "ICICIBANK": 1.15, "SBIN": 1.30, "BHARTIARTL": 0.92, "ITC": 0.70,
            "LT": 1.20, "AXISBANK": 1.25, "NIFTY": 1.00,
        }
        for sym in symbols:
            px = seeds.get(sym, float(self.rng.uniform(200, 3000)))
            itype = InstrumentType.INDEX if sym == self.index_symbol else InstrumentType.EQ
            inst = Instrument(
                symbol=sym, token=abs(hash(sym)) % 900000 + 100,
                itype=itype, sector=sector_of(sym),
                tick_size=0.05, lot_size=1,
                circuit_pct=0.10 if itype is InstrumentType.EQ else 0.15,
                cost_bps=3.5,
            )
            st = SymbolState(
                inst=inst, fair=px, prev_close=px,
                beta_mkt=betas.get(sym, float(self.rng.uniform(0.7, 1.4))),
                beta_sec=float(self.rng.uniform(0.25, 0.75)),
                sigma=float(self.rng.uniform(0.000035, 0.00008)),
                regime=REGIME_ORDER[int(self.rng.integers(0, 6))],
                base_qty=int(max(80, 6_000_000 / px / 12)),
            )
            st.sigma_lr = st.sigma
            st.day_open = st.day_high = st.day_low = px
            st.book = OrderBook(inst, px, on_trade=self._agent_maker_fill)
            self.state[sym] = st
            self._seed_book(st)

    def _agent_maker_fill(self, f: Fill) -> None:
        if self.on_agent_fill:
            self.on_agent_fill(f)

    # ------------------------------------------------------------------
    def _seed_book(self, st: SymbolState) -> None:
        """Lay an initial ladder so the first tick has depth on both sides."""
        bk = st.book
        tick = st.inst.tick_size
        spread_ticks = max(1, int(round(st.fair * 0.00012 / tick)))
        for i in range(self.depth_levels + 3):
            bpx = st.fair - (spread_ticks / 2 + i) * tick
            apx = st.fair + (spread_ticks / 2 + i) * tick
            q = int(st.base_qty * (1.0 + 0.45 * i) * self.rng.uniform(0.6, 1.4))
            bk.add_limit(Side.BUY, bpx, q, self.ts)
            bk.add_limit(Side.SELL, apx, q, self.ts)

    # ------------------------------------------------------------------
    def step(self, ts: int, dt_s: float) -> None:
        """Advance the whole market by dt_s seconds and publish."""
        self.ts = ts
        self._step_factors(dt_s)
        for sym, st in self.state.items():
            self._step_symbol(st, dt_s)
            self._run_participants(st, dt_s)
            self._publish(st)

    def _step_factors(self, dt: float) -> None:
        # market factor gets its own regime, so 'everything sells off together'
        p = REGIME_PARAMS[self._mkt_regime]
        row = _P[REGIME_ORDER.index(self._mkt_regime)]
        if self.rng.random() < 0.02:
            self._mkt_regime = REGIME_ORDER[int(self.rng.choice(8, p=row))]
        sig = 0.000075 * p["vmul"] * math.sqrt(dt)
        self._mkt_factor = p["mu"] * dt + sig * float(self.rng.standard_normal())
        for sec in set(s.inst.sector for s in self.state.values()):
            self._sector_factor[sec] = 0.000035 * math.sqrt(dt) * \
                float(self.rng.standard_normal())

    def _step_symbol(self, st: SymbolState, dt: float) -> None:
        # regime transition
        row = _P[REGIME_ORDER.index(st.regime)]
        if self.rng.random() < 0.004:
            new = REGIME_ORDER[int(self.rng.choice(8, p=row))]
            if new is not st.regime:
                st.regime = new
                st.regime_since = self.ts
        p = REGIME_PARAMS[st.regime]

        # GARCH(1,1)-ish variance recursion -> vol clusters, as it does live
        target = st.sigma_lr * p["vmul"]
        st.sigma = math.sqrt(max(1e-12,
                                 0.94 * st.sigma ** 2 + 0.06 * target ** 2))

        drift = p["mu"] * dt
        # Mean reversion pulls toward a slowly-moving anchor, not the day's
        # open. Anchoring to the open would glue price to it all session, which
        # is not what mean reversion looks like -- the level itself drifts.
        if st.anchor <= 0:
            st.anchor = st.fair
        st.anchor += (st.fair - st.anchor) * (1.0 - math.exp(-dt / 1800.0))
        if p["mr"] > 0:
            drift += -p["mr"] * math.log(max(st.fair, 1e-9) / st.anchor) * dt

        shock = st.sigma * math.sqrt(dt) * float(self.rng.standard_normal())
        common = st.beta_mkt * self._mkt_factor + \
            st.beta_sec * self._sector_factor[st.inst.sector]

        jump = 0.0
        if self.rng.random() < p["jump"] * dt:
            mag = float(self.rng.standard_normal()) * 0.0035 + \
                0.002 * np.sign(self.rng.standard_normal())
            jump = mag
            st.last_news = self._news_headline(st.inst.symbol, mag)
            st.news_until = self.ts + 300 * NS
            if self.on_news:
                self.on_news({
                    "ts": self.ts, "symbol": st.inst.symbol,
                    "headline": st.last_news,
                    "impact": float(mag), "regime": st.regime.value,
                })
            if abs(mag) > 0.005:
                st.regime = Regime.EVENT_DRIVEN
                st.regime_since = self.ts

        st.fair *= math.exp(drift + shock + common + jump)
        # circuit clamp on the latent value too, else the book detaches
        hi = st.prev_close * (1 + st.inst.circuit_pct * 0.98)
        lo = st.prev_close * (1 - st.inst.circuit_pct * 0.98)
        st.fair = min(max(st.fair, lo), hi)
        st.day_high = max(st.day_high, st.fair)
        st.day_low = min(st.day_low, st.fair)

    # ------------------------------------------------------------------
    def _run_participants(self, st: SymbolState, dt: float) -> None:
        bk = st.book
        if bk.halted:
            return
        p = REGIME_PARAMS[st.regime]
        tick = st.inst.tick_size
        liq = p["liq"]

        # ---- market makers: refresh quotes around fair, skewed by inventory
        # inventory skew is what makes short-horizon mean reversion real:
        # a maker who got run over on the offer marks his quotes down.
        #
        # Quotes are NOT cancel-replaced every step. Real makers leave orders
        # resting and only reprice when value has moved past their edge; doing
        # it that way is both faster and the reason queue position and
        # order-flow imbalance carry information here.
        half = max(0.5 * tick, st.fair * 0.00006 / max(liq, 0.1) *
                   (0.5 + 0.5 * st.sigma / max(st.sigma_lr, 1e-9)))
        # Skew is measured in half-spreads, not in percent: a maker shades his
        # quote by a fraction of his own edge, never by whole percent moves.
        inv_frac = max(-1.0, min(1.0, -st.mm_inv / max(st.base_qty * 25.0, 1.0)))
        centre = st.fair + inv_frac * 1.5 * half

        drift = abs(centre - st.mm_centre) / max(centre, 1e-9) if st.mm_centre else 1.0
        stale = (self.ts - st.mm_ts) > int(1.5 * NS)
        depleted = (bk.best_bid <= 0 or bk.best_ask <= 0 or
                    min(bk.level_qty(Side.BUY, bk.best_bid_ticks),
                        bk.level_qty(Side.SELL, bk.best_ask_ticks)) < st.base_qty * 0.45)
        if drift * centre > half * 0.30 or stale or depleted:
            bk.cancel_all("MM")
            st.mm_centre = centre
            st.mm_ts = self.ts
            for i in range(self.depth_levels + 3):
                step = half + i * max(tick, centre * 0.00005)
                q = int(st.base_qty * liq * (1.0 + 0.7 * i) *
                        self.rng.uniform(0.55, 1.5))
                if q <= 0:
                    continue
                b_px = st.inst.round_price(centre - step)
                a_px = st.inst.round_price(centre + step)
                if b_px > 0:
                    _f, eid, _ = bk.add_limit(Side.BUY, b_px, q, self.ts, owner="MM")
                    if eid:
                        bk.orders[eid].owner = "MM"
                _f, eid, _ = bk.add_limit(Side.SELL, a_px, q, self.ts, owner="MM")
                if eid:
                    bk.orders[eid].owner = "MM"

        # ---- informed flow: trades when the book is stale vs fair value.
        # Sized as a fraction of visible depth, so it moves price by pushing
        # through levels rather than by teleporting it.
        mid = bk.mid or st.fair
        edge = (st.fair - mid) / max(mid, 1e-9)
        thresh = 0.00008
        if abs(edge) > thresh:
            intensity = min(1.0, (abs(edge) - thresh) / 0.00045)
            if self.rng.random() < 0.55 * intensity * dt * 4:
                side = Side.BUY if edge > 0 else Side.SELL
                avail = bk.level_qty(side.opposite,
                                     bk.best_ask_ticks if side is Side.BUY
                                     else bk.best_bid_ticks)
                qty = int(max(1, avail * intensity * self.rng.uniform(0.25, 0.95)))
                if qty > 0:
                    fills, rem, _ = bk.add_market(side, qty, self.ts,
                                                  owner="INFORMED")
                    self._absorb(st, fills, side)

        # ---- noise flow: uninformed, roughly balanced, fattens the tape
        lam = 2.2 * dt * liq
        n_noise = int(self.rng.poisson(max(lam, 0.0)))
        for _ in range(n_noise):
            side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
            qty = int(max(1, st.base_qty * self.rng.uniform(0.02, 0.18)))
            if self.rng.random() < 0.55:
                fills, rem, _ = bk.add_market(side, qty, self.ts, owner="NOISE")
                self._absorb(st, fills, side)
            else:
                off = self.rng.integers(1, 5) * tick
                px = (bk.best_bid - off) if side is Side.BUY else (bk.best_ask + off)
                if px > 0:
                    bk.add_limit(side, st.inst.round_price(px), qty, self.ts,
                                 owner="NOISE")

        # ---- momentum herd: piles in after a strong print run
        if st.regime in (Regime.BREAKOUT, Regime.TRENDING_UP, Regime.TRENDING_DOWN):
            if self.rng.random() < 0.05 * dt:
                side = Side.BUY if st.regime is not Regime.TRENDING_DOWN else Side.SELL
                if st.regime is Regime.BREAKOUT:
                    side = Side.BUY if edge >= 0 else Side.SELL
                qty = int(st.base_qty * self.rng.uniform(0.5, 1.6))
                fills, rem, _ = bk.add_market(side, qty, self.ts, owner="MOMO")
                self._absorb(st, fills, side)

        st.mm_inv = int(st.mm_inv * 0.985)          # makers hedge off inventory
        if st.inst.is_derivative or st.inst.itype is InstrumentType.INDEX:
            st.oi = max(0, st.oi + int(self.rng.normal(0, st.base_qty * 2)))

    _HEADLINES_POS = (
        "{sym}: block deal reported at a premium",
        "{sym}: quarterly revenue beats street estimates",
        "{sym}: brokerage upgrades to BUY, raises target",
        "{sym}: large order win announced",
        "{sym}: promoter raises stake",
        "{sym}: margin guidance revised upward",
    )
    _HEADLINES_NEG = (
        "{sym}: management flags demand softness",
        "{sym}: margin pressure guidance cut",
        "{sym}: regulator opens review",
        "{sym}: brokerage downgrades, trims target",
        "{sym}: promoter pledge disclosure",
        "{sym}: order book slippage reported",
    )

    def _news_headline(self, sym: str, mag: float) -> str:
        pool = self._HEADLINES_POS if mag > 0 else self._HEADLINES_NEG
        return pool[int(self.rng.integers(0, len(pool)))].format(sym=sym)

    def _absorb(self, st: SymbolState, fills: List[Fill], aggressor: Side) -> None:
        """Makers take the other side of aggressive flow -> inventory builds."""
        for f in fills:
            st.mm_inv -= f.qty * aggressor.sign
            self._tape.append((self.ts, st.inst.symbol, f.price, f.qty,
                               aggressor.sign))
        if len(self._tape) > 200_000:
            del self._tape[:100_000]

    # ------------------------------------------------------------------
    def _publish(self, st: SymbolState) -> None:
        bk = st.book
        d = bk.depth(self.ts, self.depth_levels)
        # aggressor of the most recent print
        agg = 0
        if self._tape and self._tape[-1][1] == st.inst.symbol:
            agg = self._tape[-1][4]
        t = Tick(
            ts=self.ts, symbol=st.inst.symbol, ltp=bk.ltp,
            ltq=self._tape[-1][3] if (self._tape and self._tape[-1][1] == st.inst.symbol) else 0,
            bid=d.best_bid, ask=d.best_ask,
            bid_qty=d.bids[0].qty if d.bids else 0,
            ask_qty=d.asks[0].qty if d.asks else 0,
            volume=bk.volume, oi=st.oi, aggressor=agg,
        )
        if self.on_tick:
            self.on_tick(t)
        if self.on_depth:
            self.on_depth(d)

    # ------------------------------------------------------------------
    # agent-facing API
    # ------------------------------------------------------------------
    def submit_limit(self, symbol: str, side: Side, price: float, qty: int,
                     coid: str, ioc: bool = False):
        bk = self.state[symbol].book
        fills, eid, err = bk.add_limit(side, price, qty, self.ts,
                                       owner="AGENT", client_order_id=coid,
                                       ioc=ioc)
        if fills:
            self._absorb(self.state[symbol], fills, side)
        return fills, eid, err

    def submit_market(self, symbol: str, side: Side, qty: int, coid: str):
        bk = self.state[symbol].book
        fills, rem, err = bk.add_market(side, qty, self.ts, owner="AGENT",
                                        client_order_id=coid)
        if fills:
            self._absorb(self.state[symbol], fills, side)
        return fills, rem, err

    def cancel(self, symbol: str, eid: int) -> bool:
        return self.state[symbol].book.cancel(eid)

    def modify(self, symbol: str, eid: int, qty=None, price=None):
        return self.state[symbol].book.modify(eid, qty, price, self.ts)

    def book_of(self, symbol: str) -> OrderBook:
        return self.state[symbol].book

    def true_regime(self, symbol: str) -> Regime:
        """Ground truth -- only the evaluation harness may read this. The
        trading agent never sees it; it has to infer the regime like anyone
        else. Used to score the regime detector honestly."""
        return self.state[symbol].regime

    def roll_session(self) -> None:
        """End of day: settle closes, reset intraday state, re-band circuits."""
        for st in self.state.values():
            st.prev_close = st.book.ltp or st.fair
            st.book = OrderBook(st.inst, st.prev_close,
                                on_trade=self._agent_maker_fill)
            st.day_open = st.day_high = st.day_low = st.prev_close
            st.mm_inv = 0
            self._seed_book(st)

    def tape(self, symbol: str, n: int = 500):
        out = [t for t in reversed(self._tape) if t[1] == symbol][:n]
        return list(reversed(out))
