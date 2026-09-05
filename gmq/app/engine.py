"""The orchestrator: every block from spec §16, wired into one event loop.

  Market Data -> Features -> Regime -> Models -> Strategy/Search
              -> Portfolio Risk -> Execution -> Trade Monitor
              -> Performance Analytics -> Model Evaluation

Ordering matters and is enforced by bus priorities, not by luck:

  10  market data normalisation and book maintenance
  15  feature engineering (microstructure, per tick)
  35  label resolution and model training
  50  risk engine sees every fill before execution acts again
  60  execution
  90  monitors
 200  journalling

Decision cadence is separated from data cadence on purpose. Data arrives at
several hundred events per second; decisions are taken on a fixed cadence per
symbol. Re-deciding on every tick would not make the system faster, it would
make it noisier -- the same features, resampled, produce a different answer by
chance, and the engine would trade its own sampling error.
"""
from __future__ import annotations

import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..core.bus import EventBus, Topic, TimerWheel
from ..core.clock import SimClock, LiveClock, Clock, SessionPhase, IST
from ..core.config import Config, sector_of
from ..core.types import (
    Tick, Bar, Fill, Order, OrderType, Side, Product, Timeframe, Regime,
    MoveType, Prediction, DecisionRecord, Position, NS,
)
from ..core.mathx import clamp, safe_div
from ..data.feed import SimFeed, MarketDataEngine
from ..features.registry import FeatureEngine
from ..features.derivatives import synthetic_chain
from ..regime.detector import RegimeDetector
from ..models.bank import ModelBank
from ..strategy.search import GrandmasterSearch
from ..strategy.moves import MoveContext
from ..strategy.evaluator import CostModel
from ..risk.portfolio import Portfolio
from ..risk.engine import RiskEngine, TradeIntent, HaltReason
from ..risk.sizing import PositionSizer
from ..risk.stops import StopPolicy
from ..execution.simbroker import SimBroker
from ..execution.router import OrderRouter
from ..monitor.trade_monitor import TradeMonitor, Thesis
from ..monitor.journal import DecisionJournal, new_decision_id
from ..analytics import metrics
from ..analytics.excursion import ExcursionBook, GivebackPolicy, PathExcursion


class TradingEngine:
    def __init__(self, cfg: Config, run_dir: Optional[str] = None,
                 start: Optional[datetime] = None, journal: bool = True,
                 synthetic_options: bool = True, live: bool = False):
        self.cfg = cfg
        self.live = live
        self.symbols = list(cfg.data.symbols)
        self.bus = EventBus(swallow_errors=True)
        # Live mode runs on the wall clock and takes its ticks from an external
        # feed (KiteFeed) pushed onto the bus; sim mode owns a SimClock and a
        # SimFeed it steps itself. Everything between the feed and the broker is
        # byte-for-byte identical either way -- that identity is the whole point
        # of validating on the simulator and then flipping this one flag.
        self.clock = LiveClock() if live else \
            SimClock(start or datetime(2026, 8, 17, 9, 15, tzinfo=IST))
        self.timers = TimerWheel()
        self.rng = np.random.default_rng(cfg.seed)

        # ---- block 1: market data
        self.mde = MarketDataEngine(self.bus, self.clock, self.symbols,
                                    index_symbol=cfg.data.index_symbol)
        # In live mode the feed is external (set via attach_live_feed); the
        # engine never steps it, ticks arrive from the websocket.
        self.feed = None if live else \
            SimFeed(self.bus, self.clock, self.symbols, seed=cfg.seed,
                    depth_levels=cfg.data.depth_levels,
                    index_symbol=cfg.data.index_symbol)

        # ---- blocks 2-4: features, regime, models
        self.features = FeatureEngine(self.symbols, cfg.data.index_symbol)
        self.regime = RegimeDetector()
        self.models = ModelBank(cfg.models, dim=220)

        # ---- blocks 5-6: strategy and search
        self.cost = CostModel()
        self.search = GrandmasterSearch(cfg.search, self.cost, seed=cfg.seed + 1)

        # ---- block 7: portfolio risk
        self.portfolio = Portfolio(cfg.initial_capital)
        self.risk = RiskEngine(cfg.risk, self.portfolio, self.features.cross,
                               n_symbols=len(self.symbols))
        self.sizer = PositionSizer(cfg.risk)
        # The exit policy is fitted, not assumed. The book collects every
        # closed trade's excursions and periodically refits the give-back rule
        # to them; until it has evidence, the defaults below stand.
        self.excursions = ExcursionBook(
            min_samples=cfg.search.giveback_min_samples,
            default=GivebackPolicy(
                activation_r=cfg.search.giveback_activation_r,
                keep_fraction=cfg.search.giveback_keep_fraction))
        gb = self.excursions.policy if cfg.search.giveback_enabled else None
        self.stops = StopPolicy(giveback=gb)
        self.search.moves.giveback = gb
        self._closes_since_fit = 0

        # ---- block 8: execution
        # Live: a paper broker filling against the real quotes (no real orders).
        # Sim: the simulated exchange. Same SimBroker class, same accounting.
        if live:
            from ..execution.paper import make_paper_broker
            self.broker = make_paper_broker(
                self.symbols, cfg.execution, self.bus, self.clock,
                self.timers, seed=cfg.seed + 2)
        else:
            self.broker = SimBroker(self.feed.exchange, self.clock,
                                    cfg.execution, self.bus, self.timers,
                                    seed=cfg.seed + 2)
        self.broker.connect()
        self.router = OrderRouter(self.broker, self.clock, cfg.execution,
                                  self.bus, on_fill=self._on_fill,
                                  risk_hook=self._risk_hook)

        # ---- blocks 9-11: monitor, journal, analytics
        self.monitor = TradeMonitor()
        self.journal = DecisionJournal(run_dir or cfg.journal_dir,
                                       cfg.run_id) if journal else None

        # ---- wiring
        self.bus.subscribe(Topic.TICK, self.features.on_tick, 15, "feat.tick")
        self.bus.subscribe(Topic.DEPTH, self.features.on_depth, 15, "feat.depth")
        self.bus.subscribe(Topic.TICK, self._on_tick, 20, "engine.tick")
        self.bus.subscribe(Topic.TICK, self.broker.on_tick, 25, "broker.stops")
        self.bus.subscribe(Topic.TICK, self._train_on_tick, 35, "models.train")
        self.bus.subscribe(Topic.NEWS, self._on_news, 30, "engine.news")

        # ---- runtime state
        self.decision_every_s = 5.0
        self.idle_decision_every_s = 20.0
        # No trading until the models have seen enough resolved labels to have
        # an opinion worth acting on. Without this the engine spends its first
        # session trading a randomly-initialised model, and those trades are
        # not merely unprofitable -- they train the model on the consequences
        # of its own noise.
        self.warmup_labels = 600
        self.warmup_minutes = 45
        self._last_decision: Dict[str, float] = defaultdict(float)
        self._last_minute = -1
        self._news_until: Dict[str, int] = defaultdict(int)
        self.decisions = 0
        self.actions = 0
        self.vetoed = 0
        self.search_ms: Deque[float] = deque(maxlen=2000)
        self.loop_ms: Deque[float] = deque(maxlen=2000)
        self.tick_to_decision_us: Deque[float] = deque(maxlen=2000)
        self.recent_decisions: Deque[dict] = deque(maxlen=200)
        self.synthetic_options = synthetic_options
        self._opt_refresh = 0
        self.day_index = 0
        self.halt_events: List[dict] = []
        self.started_ns = 0
        self._pending_decision: Dict[str, str] = {}
        # resting stop-loss order per symbol
        self._stop_orders: Dict[str, str] = {}
        self.stop_outs = 0
        # when each symbol was last flattened, for the re-entry cooldown
        self._last_exit_ns: Dict[str, int] = {}
        # live R-path recorder per open position, for the exit calibration
        self._paths: Dict[str, PathExcursion] = {}

    # ==================================================================
    # data path
    # ==================================================================
    def _on_tick(self, t: Tick) -> None:
        # Live: the paper broker fills against real quotes, so it must see every
        # tick -- both to refresh its book and to trigger resting stops on the
        # real tape. In sim the exchange already has the book; this is a no-op.
        if self.live:
            self.broker.ex.update(t)
            self.broker.on_tick(t)
        self.portfolio.mark(t.symbol, t.ltp)
        self.risk.on_data_stale(t.symbol, False)
        self._track_excursion(t)
        self._enforce_stops(t)

    def _track_excursion(self, t: Tick) -> None:
        """Record the live R-path of every open position.

        This has to happen on the tick, not on the decision: the give-back
        calibration needs the worst R reached after a level armed, and the
        decision loop only looks every five seconds. Sampling the path at the
        decision cadence would systematically miss the excursions that matter
        and quietly bias the fit toward a looser stop.
        """
        pos = self.portfolio.positions.get(t.symbol)
        if pos is None or pos.qty == 0 or pos.initial_risk <= 0:
            return
        tr = self._paths.get(t.symbol)
        if tr is None:
            tr = self._paths[t.symbol] = PathExcursion()
        tr.update(pos.unrealised(t.ltp) / pos.initial_risk)

    def _train_on_tick(self, t: Tick) -> None:
        self.models.on_price(t.symbol, t.ts, t.ltp)

    def _on_news(self, n: dict) -> None:
        self._news_until[n["symbol"]] = n["ts"] + int(300 * NS)
        if self.journal:
            self.journal.event(n["ts"], "news", n["symbol"], n)

    def _on_fill(self, f: Fill) -> None:
        did = self._pending_decision.get(f.symbol, "")
        st = self.regime.get(f.symbol)
        rec = self.portfolio.apply_fill(f, did, st.current.value)
        pos = self.portfolio.positions.get(f.symbol)
        if pos and pos.qty:
            th = self.monitor.get(f.symbol)
            if th:
                pos.stop = th.stop
                pos.target = th.target
                pos.thesis = f"{th.regime} edge={2*th.p_up-1:+.2f}"
                pos.entry_confidence = th.confidence
                # 1R is fixed at entry and never recomputed.
                #
                # Recomputing it on every fill looks harmless and destroys the
                # metric: after two partial exits the position is a fraction of
                # its original size, so `risk` shrinks while realised P&L still
                # reflects the whole trade -- and the R-multiple explodes. That
                # is how a run can report +1.67 rupees of expectancy and -0.72R
                # at the same time, which is not a rounding difference, it is
                # two different trades being described.
                if pos.initial_risk <= 0:
                    pos.initial_stop = th.stop
                    pos.initial_risk = abs(th.entry_px - th.stop) * abs(pos.qty)
            self._sync_stop_order(f.symbol, pos)
        if rec is not None:
            self._cancel_stop_order(f.symbol)
            self._close_out(rec, f)

    def _close_out(self, rec, f: Fill) -> None:
        th = self.monitor.close_thesis(f.symbol)
        diag = self.monitor.latest(f.symbol)
        rec.exit_reason = rec.exit_reason or "signal"
        rec.thesis_correct = None
        if th is not None:
            realised = (f.price - th.entry_px) * th.side
            predicted_up = th.p_up > 0.5
            went_up = f.price > th.entry_px
            rec.thesis_correct = (predicted_up == went_up)
            rec.decision_id = th.decision_id
            rec.confidence = th.confidence
            rec.predicted_p_up = th.p_up
        if self.journal and rec.decision_id:
            self.journal.record_outcome(rec.decision_id, {
                "closed_ts": f.ts, "pnl": rec.pnl - rec.fees,
                "r_multiple": rec.r_multiple, "mfe": rec.mfe, "mae": rec.mae,
                "hold_s": rec.hold_s, "exit_reason": rec.exit_reason,
                "entry_px": rec.entry_px, "exit_px": rec.exit_px,
                "thesis_correct": rec.thesis_correct,
                "loss_cause": diag.cause.value if diag else "",
                "entry_regime": rec.entry_regime,
            })
        self._pending_decision.pop(f.symbol, None)
        # Start the re-entry cooldown for this name.
        self._last_exit_ns[f.symbol] = f.ts
        self._feed_excursions(rec)

    def _feed_excursions(self, rec) -> None:
        """Record the closed trade's excursions and refit the exit rule.

        Only trades whose 1R is actually known contribute. A trade with no
        measurable risk unit has no R to report, and filling one in with a
        placeholder is how a metric turns into fiction -- an earlier build did
        exactly that and reported an expectancy of sixteen million R.
        """
        risk = getattr(rec, "initial_risk", 0.0) or 0.0
        tr = self._paths.pop(rec.symbol, None)
        if risk <= 0 or rec.r_multiple is None:
            return
        # Gross R here, deliberately. The excursion path (`min_after`) is
        # built from unrealised mark-to-market, which carries no fees, so the
        # settlement point has to be measured the same way or the replay
        # compares a net endpoint against a gross path. Fees are constant
        # across the policies being compared, so they cancel in the fit; they
        # do not cancel in the report, which is why TradeRecord keeps net.
        self.excursions.add(mfe_r=rec.mfe / risk, mae_r=rec.mae / risk,
                            final_r=rec.pnl / risk,
                            min_after=tr.snapshot() if tr else (),
                            hold_s=rec.hold_s, regime=rec.entry_regime or "")
        self._closes_since_fit += 1
        every = max(int(self.cfg.search.giveback_refit_every), 1)
        if self._closes_since_fit >= every:
            self._closes_since_fit = 0
            pol = self.excursions.fit()
            if self.cfg.search.giveback_enabled:
                # fit() returns a fresh object, so both consumers have to be
                # repointed -- holding the old one is a silent no-op.
                self.stops.giveback = pol
                self.search.moves.giveback = pol
                if self.journal and pol.fitted:
                    self.journal.event(self.clock.now_ns(), "policy_fit", "",
                                       pol.as_dict())

    # ------------------------------------------------------------------
    # stop-loss orders
    #
    # The engine computed stops, sized every position against them and told
    # the risk engine about them -- and, in an earlier build, never actually
    # placed one. Exits happened only when the search chose to exit, which
    # means the number the whole risk framework is denominated in was not
    # enforced by anything. A stop that exists only in a variable is not a
    # stop.
    #
    # There are two layers here on purpose:
    #   1. a real SL-M order resting at the broker, which is what protects the
    #      position when the process is not looking at it;
    #   2. a local check on every tick, because a broker-held stop can be
    #      rejected, cancelled or lost on a reconnect, and discovering that
    #      after the fact is how small losses become large ones.
    # ------------------------------------------------------------------
    def _sync_stop_order(self, symbol: str, pos: Position) -> None:
        """Ensure a resting stop order matches the position's current stop."""
        if pos.qty == 0 or pos.stop <= 0:
            self._cancel_stop_order(symbol)
            return
        side = Side.SELL if pos.qty > 0 else Side.BUY
        want_qty = abs(pos.qty)
        cur = self._stop_orders.get(symbol)
        if cur is not None:
            o = self.router.live_orders.get(cur)
            if o is not None and o.is_live:
                if (abs(o.trigger - pos.stop) < 1e-6 and o.qty == want_qty
                        and o.side is side):
                    return                      # already correct
                self.router.cancel(cur)
            self._stop_orders.pop(symbol, None)
        orders = self.router.submit(
            symbol, side, want_qty, OrderType.SL_M, trigger=pos.stop,
            tag=f"stop:{pos.decision_id}", intent_price=pos.stop,
            product=Product.MIS, counts_as_decision=False)
        if orders:
            self._stop_orders[symbol] = orders[0].order_id

    def _cancel_stop_order(self, symbol: str) -> None:
        oid = self._stop_orders.pop(symbol, None)
        if oid:
            self.router.cancel(oid)

    def _enforce_stops(self, t: Tick) -> None:
        """Local safety net, checked on every tick regardless of halt state."""
        pos = self.portfolio.positions.get(t.symbol)
        if pos is None or pos.qty == 0 or pos.stop <= 0:
            return
        hit = (pos.qty > 0 and t.ltp <= pos.stop) or \
              (pos.qty < 0 and t.ltp >= pos.stop)
        if not hit:
            return
        # If the broker-held stop is doing its job it will fill momentarily;
        # only step in once price is clearly through the level, so the two
        # layers cannot double-close the same position.
        slack = max(pos.stop * 0.0006, 0.05)
        through = (pos.qty > 0 and t.ltp <= pos.stop - slack) or \
                  (pos.qty < 0 and t.ltp >= pos.stop + slack)
        if not through:
            return
        self._cancel_stop_order(t.symbol)
        self.router.flatten(t.symbol, pos.qty, t.ltp)
        self.stop_outs += 1
        if self.journal:
            self.journal.event(t.ts, "stop_out", t.symbol,
                               {"stop": pos.stop, "ltp": t.ltp, "qty": pos.qty})

    def _risk_hook(self, kind: str, value) -> None:
        if kind == "sent":
            self.risk.on_order_sent(int(value))
        elif kind == "reject":
            self.risk.on_order_rejected(int(value))
        elif kind == "slippage":
            self.risk.on_slippage(float(value))

    # ==================================================================
    # main loop
    # ==================================================================
    def run(self, days: float = 1.0, dt: float = 0.5,
            progress: Optional[Any] = None) -> dict:
        self.started_ns = self.clock.now_ns()
        steps_per_day = int(22500 / dt)
        total = int(steps_per_day * days)
        per_min = max(1, int(60 / dt))
        t_wall = time.time()

        for i in range(total):
            t0 = time.perf_counter()
            self.feed.step(dt)
            self.broker.pump()

            now = self.clock.now_ns()
            if i % per_min == per_min - 1:
                self._on_minute(now)
            self._maybe_decide(now)

            # risk monitors run on their own clock, independent of anything
            # the decision engine does or fails to do
            if i % 20 == 0:
                self._risk_sweep(now)
            if i % 40 == 0:
                self.router.reconcile(now)
                self.portfolio.record_equity(now)
            self.bus.drain(200)
            self.loop_ms.append((time.perf_counter() - t0) * 1000.0)

            if i and i % steps_per_day == 0:
                self._roll_day(now)
            if progress and i % (steps_per_day // 4) == 0 and i:
                progress(i / steps_per_day, self.snapshot())

        self._square_off(self.clock.now_ns(), "run_end")
        self.portfolio.record_equity(self.clock.now_ns())
        if self.journal:
            self.journal.flush()
        return self.report(wall_s=time.time() - t_wall)

    # ------------------------------------------------------------------
    # live path -- real ticks in, paper fills, wall-clock cadence
    # ------------------------------------------------------------------
    def attach_live_feed(self, feed) -> None:
        """Register the live feed (e.g. KiteFeed). It pushes ticks onto the
        same bus; the engine only reads. Call before run_live()."""
        if not self.live:
            raise RuntimeError("attach_live_feed requires live=True")
        self.live_feed = feed

    def run_live(self, poll_s: float = 0.2, max_seconds: float = 0.0,
                 on_beat=None) -> dict:
        """Trade the real market on the wall clock until the session closes.

        The inverse of run(): there, the loop pulls ticks from a sim feed;
        here, ticks arrive asynchronously from the websocket onto the bus and
        the loop's job is to drain them, run the per-minute and decision
        cadences on real time, and keep the risk monitors turning. It never
        touches a real order -- fills come from the paper broker against the
        live quotes.

        `max_seconds` caps the run for a smoke test; 0 means "until close".
        """
        if not self.live:
            raise RuntimeError("run_live requires live=True")
        self.started_ns = self.clock.now_ns()
        feed = getattr(self, "live_feed", None)
        if feed is not None and hasattr(feed, "start"):
            feed.start()
        t_wall = time.time()
        last_min = -1
        last_risk = 0.0
        last_recon = 0.0
        try:
            while True:
                now = self.clock.now_ns()
                now_s = now / NS
                if max_seconds and (time.time() - t_wall) >= max_seconds:
                    break
                phase = self.clock.phase()
                if phase is SessionPhase.CLOSED and (time.time() - t_wall) > 1:
                    break
                # deliver any latency-delayed broker events, then all ticks
                self.broker.pump()
                self.bus.drain(1000)

                minute = int(now_s // 60)
                if minute != last_min:
                    last_min = minute
                    self._on_minute(now)
                self._maybe_decide(now)

                if now_s - last_risk >= 1.0:
                    last_risk = now_s
                    self._risk_sweep(now)
                if now_s - last_recon >= self.cfg.execution.reconcile_every_s:
                    last_recon = now_s
                    self.router.reconcile(now)
                    self.portfolio.record_equity(now)
                if on_beat:
                    on_beat(self.snapshot())
                time.sleep(poll_s)
        finally:
            if feed is not None and hasattr(feed, "stop"):
                feed.stop()
            self._square_off(self.clock.now_ns(), "session_close")
            self.portfolio.record_equity(self.clock.now_ns())
            if self.journal:
                self.journal.flush()
        return self.report(wall_s=time.time() - t_wall)

    # ------------------------------------------------------------------
    def _on_minute(self, now: int) -> None:
        prices = {s: self.mde.ltp(s) for s in self.symbols}
        self.features.on_minute(prices)
        if not self.models.feature_names and self.features.feature_names():
            self.models.feature_names = self.features.feature_names()

        # refresh the synthetic options chain every 15 minutes so the
        # derivatives feature block is exercised end to end
        if self.synthetic_options and now - self._opt_refresh > 900 * NS:
            self._opt_refresh = now
            for s in self.symbols:
                px = prices.get(s, 0.0)
                if px <= 0:
                    continue
                ser = self.mde.series(s, Timeframe.M1)
                if ser is None or ser.n < 30:
                    continue
                rv = float(np.diff(np.log(np.maximum(ser.closes(60), 1e-9))).std())
                self.features.derivs.update(
                    synthetic_chain(s, px, rv * math.sqrt(375 * 252), self.rng))

        for s in self.symbols:
            fv = self.features.build(s, now, self.mde)
            if not fv.ready:
                continue
            st = self.regime.update(s, fv, self.mde)
            self.models.observe(s, now, fv,
                                st.current.value if st.initialised else "")
            ser = self.mde.series(s, Timeframe.M1)
            if ser is not None and ser.n > 40:
                self.search.scen.set_history(s, ser.returns(300))

        ok, note = self.models.health()
        self.risk.on_model_health(ok, note)

    # ------------------------------------------------------------------
    def _maybe_decide(self, now: int) -> None:
        for s in self.symbols:
            last = self._last_decision.get(s, 0.0)
            # Adaptive cadence: an open position is re-evaluated often, because
            # that is where the risk is. A flat symbol with nothing happening
            # is checked less often -- deciding more frequently on the same
            # unchanged evidence does not find more opportunities, it just
            # gives sampling noise more chances to look like one.
            pos = self.portfolio.positions.get(s)
            period = self.decision_every_s if (pos and pos.qty) \
                else self.idle_decision_every_s
            if (now - last) < period * NS:
                continue
            if self.mde.is_stale(s):
                self.risk.on_data_stale(s, True)
                continue
            self._last_decision[s] = now
            self._decide(s, now)

    # ------------------------------------------------------------------
    def _decide(self, symbol: str, now: int) -> None:
        t_start = time.perf_counter()
        # Reuse the bar-level features computed on the last minute close and
        # refresh only the microstructure block. Bar indicators cannot have
        # changed since; recomputing them here was ~90% of decision latency.
        fv = self.features.refresh(symbol, now, self.mde)
        if fv is None or not fv.ready or fv.atr <= 0:
            return
        tick = self.mde.last_tick.get(symbol)
        if tick is None or tick.ltp <= 0:
            return

        x = self.models.transform(fv)
        if x.size == 0:
            return
        pred = self.models.consensus(symbol, x)
        rs = self.regime.get(symbol)
        regime = rs.current if rs.initialised else Regime.LOW_VOL

        pos = self.portfolio.positions.get(symbol) or Position(symbol=symbol)
        equity = self.portfolio.equity()

        # ---- sizing (proposed, before the risk engine has its say)
        ann_vol = fv.atr_pct * math.sqrt(375 * 252) / 1.2
        hm = self.models.models[self.models.horizons[1]]
        exp_mae = max(0.0, hm.mae_q.value(x)) if hm.mae_q.n > 100 else 0.0
        depth = self.mde.depth.get(symbol)
        top_qty = (depth.bids[0].qty if depth and depth.bids else 0)
        skill = self.models.honest[self.models.horizons[1]].skill_score()

        plan = self.stops.initial(
            Side.BUY, tick.ltp, fv.atr, regime,
            support=fv.values.get("5m_dist_support_atr", 0) and
            tick.ltp - fv.values["5m_dist_support_atr"] * fv.atr or 0.0,
            depth=depth, expected_mae_pct=exp_mae,
            realised_vol_bar=fv.atr_pct / 1.2)
        size = self.sizer.size(
            equity=equity, px=tick.ltp, stop=plan.stop, pred=pred,
            atr=fv.atr, ann_vol=ann_vol, top_depth_qty=top_qty,
            model_skill=skill,
            existing_exposure_pct=safe_div(abs(pos.qty) * tick.ltp, equity) * 100)
        heat = self.sizer.heat_scale(self.portfolio.open_risk_pct(),
                                     self.portfolio.drawdown_pct())
        unit_qty = max(0, int(size.qty * heat))

        # ---- monitor an open position before deciding what to do with it
        diag = None
        if pos.qty:
            diag = self.monitor.diagnose(
                symbol, tick.ltp, fv.atr, pred, regime, fv.alignment,
                tick.spread_bps, fv.liquidity, now,
                recent_news=now < self._news_until.get(symbol, 0),
                realised_slippage_bps=float(np.mean(self.router.slippages))
                if self.router.slippages else 0.0,
                bar_return_z=fv.values.get("1m_ret1_atr", 0.0))

        # ---- the search
        warm = (self.models.samples_seen >= self.warmup_labels and
                (now - self.started_ns) / NS >= self.warmup_minutes * 60)
        can_open = (not self.risk.halted) and warm
        ctx = MoveContext(
            symbol=symbol, px=tick.ltp, bid=tick.bid or tick.ltp,
            ask=tick.ask or tick.ltp, atr=fv.atr, qty=pos.qty,
            avg_price=pos.avg_price, stop=pos.stop, target=pos.target,
            max_qty=int(equity * self.cfg.risk.max_position_pct / 100 / tick.ltp),
            unit_qty=unit_qty, regime=regime, alignment=fv.alignment,
            alignment_conflict=fv.alignment_conflict, pred=pred,
            hold_s=(now - pos.opened_ns) / NS if pos.opened_ns else 0.0,
            partials=pos.partials, scale_ins=pos.scale_ins,
            liquidity=fv.liquidity,
            r_multiple=pos.r_multiple(tick.ltp),
            seconds_to_close=self.clock.seconds_to_close(),
            can_open=can_open, expected_mae_pct=exp_mae,
            peak_r=(pos.mfe / pos.initial_risk) if pos.initial_risk > 0 else 0.0,
            risk_per_share=(pos.initial_risk / abs(pos.qty))
            if (pos.initial_risk > 0 and pos.qty) else 0.0,
            s_since_exit=((now - self._last_exit_ns[symbol]) / NS
                          if symbol in self._last_exit_ns else 1e9))

        market = {
            "spread_bps": tick.spread_bps or 3.0,
            "top_qty": top_qty or max(unit_qty, 1),
            "liquidity": fv.liquidity,
            "trade_rate": max(0.2, self.features.micro.get(symbol).trades /
                              max((now - self.started_ns) / NS, 1.0)),
            "imbalance": fv.values.get("mx_depth_imb5", 0.0),
            "queue_ahead": int((top_qty or 0) * 0.5),
        }
        best, all_moves, stats = self.search.search(ctx, equity, market)
        self.search_ms.append(stats.ms)
        self.decisions += 1

        # ---- the monitor can override the search toward safety, never away
        forced = self._safety_override(pos, diag, best, ctx, now, fv)
        if forced is not None:
            best = forced

        # ---- risk gate
        did = new_decision_id()
        reducing = best.move in (MoveType.EXIT, MoveType.REDUCE,
                                 MoveType.TAKE_PARTIAL) or \
            (pos.qty != 0 and best.qty != 0 and
             (best.qty > 0) != (pos.qty > 0) and abs(best.qty) <= abs(pos.qty))
        intent = TradeIntent(
            symbol=symbol, move=best.move, qty=best.qty, price=tick.ltp,
            stop=best.stop or plan.stop, target=best.target or plan.target,
            decision_id=did, confidence=pred.confidence, is_reducing=reducing)
        verdict = self.risk.check(intent, now)
        if not verdict.approved:
            self.vetoed += 1

        # ---- execute
        exec_info = self._execute(best, verdict, ctx, intent, symbol, now, fv,
                                  pred, did, plan)

        # ---- record
        self._journal(did, symbol, now, tick, fv, pred, rs, all_moves, best,
                      verdict, exec_info, stats, size, plan, diag)
        self.tick_to_decision_us.append((time.perf_counter() - t_start) * 1e6)

    # ------------------------------------------------------------------
    def _safety_override(self, pos, diag, best, ctx, now, fv):
        """The monitor may make a decision safer; it may never make it riskier.

        Asymmetry is the whole design. A diagnosis is evidence, and evidence
        is allowed to talk the engine out of holding risk. It is never allowed
        to talk it into more -- because a component that can escalate risk on
        its own is a second decision engine, and the risk limits were written
        for one.
        """
        if pos.qty == 0 or diag is None:
            return None
        from ..core.types import MoveEval
        if diag.recommendation == "EXIT" and best.move not in (
                MoveType.EXIT, MoveType.REVERSE):
            return MoveEval(move=MoveType.EXIT, qty=-pos.qty,
                            price=ctx.px, rationale=f"monitor: {diag.note}")
        if diag.recommendation == "REDUCE" and best.move in (
                MoveType.WAIT, MoveType.INCREASE, MoveType.MOVE_STOP):
            q = -int(pos.qty * 0.5)
            if q != 0:
                return MoveEval(move=MoveType.REDUCE, qty=q, price=ctx.px,
                                rationale=f"monitor: {diag.note}")
        if diag.recommendation == "TIGHTEN" and best.move is MoveType.WAIT:
            side = Side.BUY if pos.qty > 0 else Side.SELL
            new_stop, why = self.stops.update(
                side, ctx.px, pos.avg_price, pos.stop, fv.atr, ctx.regime,
                pos.r_multiple(ctx.px), thesis_weakening=True,
                regime_changed=(diag.cause.value == "REGIME_CHANGE"),
                peak_r=ctx.peak_r, risk_per_share=ctx.risk_per_share)
            if new_stop != pos.stop:
                return MoveEval(move=MoveType.MOVE_STOP, qty=0, stop=new_stop,
                                price=ctx.px,
                                rationale=f"monitor/{why}: {diag.note}")
        return None

    # ------------------------------------------------------------------
    def _execute(self, best, verdict, ctx, intent, symbol, now, fv, pred,
                 did, plan) -> dict:
        info: dict = {"submitted": False, "reason": verdict.reason}
        if best.move is MoveType.WAIT:
            return info
        if best.move is MoveType.MOVE_STOP:
            pos = self.portfolio.positions.get(symbol)
            if pos and pos.qty and best.stop:
                side = Side.BUY if pos.qty > 0 else Side.SELL
                # the ratchet is enforced in StopPolicy; this call cannot widen
                new_stop, why = self.stops.update(
                    side, ctx.px, pos.avg_price, pos.stop, fv.atr, ctx.regime,
                    pos.r_multiple(ctx.px),
                    peak_r=ctx.peak_r, risk_per_share=ctx.risk_per_share)
                tighter = max(new_stop, best.stop) if pos.qty > 0 \
                    else min(new_stop, best.stop)
                if (pos.qty > 0 and tighter > pos.stop) or \
                        (pos.qty < 0 and (tighter < pos.stop or pos.stop <= 0)):
                    pos.stop = tighter
                    th = self.monitor.get(symbol)
                    if th:
                        th.stop = tighter
                    # the resting order has to follow, or the ratchet only
                    # tightens a number and not the actual protection
                    self._sync_stop_order(symbol, pos)
                    info.update({"stop_moved_to": round(tighter, 2),
                                 "reason": why})
            return info
        if not verdict.approved:
            return info

        qty = min(abs(best.qty), verdict.max_qty) if verdict.max_qty else abs(best.qty)
        if qty <= 0:
            info["reason"] = "zero_after_risk"
            return info
        side = Side.BUY if best.qty > 0 else Side.SELL
        depth = self.mde.depth.get(symbol)
        top = depth.bids[0].qty if depth and depth.bids else 0

        orders = self.router.submit(
            symbol, side, int(qty), OrderType.MARKET, tag=did,
            intent_price=ctx.px, top_depth=top, product=Product.MIS)
        self.actions += 1
        self._pending_decision[symbol] = did
        info.update({"submitted": True, "qty": int(qty), "side": side.name,
                     "orders": [o.order_id for o in orders],
                     "scaled_from": verdict.scaled_from})

        # open a thesis on a new position so the monitor has something to check
        if best.move in (MoveType.ENTER_LONG, MoveType.ENTER_SHORT,
                         MoveType.REVERSE):
            sgn = 1 if side is Side.BUY else -1
            stop = best.stop or (ctx.px - sgn * fv.atr)
            target = best.target or (ctx.px + sgn * fv.atr * 2)
            self.monitor.open_thesis(Thesis(
                decision_id=did, symbol=symbol, side=sgn, entry_px=ctx.px,
                entry_ns=now, p_up=pred.p_up, confidence=pred.confidence,
                regime=ctx.regime.value, alignment=fv.alignment,
                expected_hold_s=pred.exp_hold_s,
                expected_move=pred.exp_return,
                key_signals={k: v for k, v in list(fv.values.items())[:0]},
                stop=stop, target=target,
                initial_risk=abs(ctx.px - stop) * qty))
        return info

    # ------------------------------------------------------------------
    def _journal(self, did, symbol, now, tick, fv, pred, rs, all_moves, best,
                 verdict, exec_info, stats, size, plan, diag) -> None:
        if self.journal is None:
            return
        # WAIT decisions are recorded but not fanned out to the dashboard, or
        # the interesting ones drown in them.
        rec = DecisionRecord(
            decision_id=did, ts=now, symbol=symbol,
            market_state={
                "ltp": tick.ltp, "bid": tick.bid, "ask": tick.ask,
                "spread_bps": round(tick.spread_bps, 2),
                "volume": tick.volume,
                "day_change_pct": round(self.mde.day_change_pct(symbol), 3),
                "atr": round(fv.atr, 3), "atr_pct": round(fv.atr_pct * 100, 3),
                "liquidity": round(fv.liquidity, 3),
                "session": self.clock.phase(),
            },
            signals={k: round(float(v), 5) for k, v in fv.values.items()
                     if k.startswith(("mx_", "mtf_", "xs_", "dv_"))},
            predictions={
                "p_up": round(pred.p_up, 4),
                "confidence": round(pred.confidence, 4),
                "p_target": round(pred.p_target, 4),
                "p_stop": round(pred.p_stop, 4),
                "exp_return": round(pred.exp_return, 6),
                "exp_hold_s": round(pred.exp_hold_s, 1),
                "p_continuation": round(pred.p_continuation, 3),
                "horizons": {str(int(h)): round(p.p_up, 4) for h, p
                             in self.models.predict_all(symbol,
                                                        self.models.transform(fv)).items()},
            },
            regime=rs.current.value, regime_conf=round(rs.confidence, 4),
            candidates=[{
                "move": m.move.value, "qty": m.qty,
                "score": round(m.score, 2), "ev": round(m.ev, 2),
                "p_win": round(m.p_win, 3), "exp_dd": round(m.exp_dd, 2),
                "cvar": round(m.cvar, 2), "cost": round(m.cost, 2),
                "rationale": m.rationale,
                "scenarios": [{"name": s.name, "prob": s.prob,
                               "ret_bps": round(s.ret * 1e4, 1),
                               "hit_stop": s.hit_stop,
                               "hit_target": s.hit_target}
                              for s in m.scenarios[:4]],
            } for m in all_moves[:6]],
            risk={
                "approved": verdict.approved, "reason": verdict.reason,
                "max_qty": verdict.max_qty, "breached": verdict.breached,
                "scaled_from": verdict.scaled_from,
                "sizing": size.as_dict(), "stop_plan": plan.as_dict(),
                "equity": round(self.portfolio.equity(), 2),
                "open_risk_pct": round(self.portfolio.open_risk_pct(), 3),
                "drawdown_pct": round(self.portfolio.drawdown_pct(), 3),
                "halted": self.risk.halted,
                "diagnosis": diag.as_dict() if diag else None,
            },
            chosen={
                "move": best.move.value, "qty": best.qty,
                "price": round(best.price, 2), "stop": round(best.stop, 2),
                "target": round(best.target, 2), "ev": round(best.ev, 2),
                "score": round(best.score, 2), "p_win": round(best.p_win, 3),
            },
            rationale=best.rationale,
            execution=exec_info,
            search_ms=round(stats.ms, 3), nodes=stats.nodes,
        )
        self.journal.record(rec)
        if best.move is not MoveType.WAIT:
            self.recent_decisions.append({
                "ts": now, "symbol": symbol, "move": best.move.value,
                "qty": best.qty, "price": round(best.price, 2),
                "ev": round(best.ev, 1), "score": round(best.score, 1),
                "regime": rs.current.value, "p_up": round(pred.p_up, 3),
                "approved": verdict.approved, "reason": verdict.reason,
                "rationale": best.rationale,
            })

    # ==================================================================
    # session / risk management
    # ==================================================================
    def _risk_sweep(self, now: int) -> None:
        reason = self.risk.monitor(now)
        if reason and not any(h["reason"] == reason for h in self.halt_events):
            self.halt_events.append({"ts": now, "reason": reason,
                                     "equity": self.portfolio.equity()})
            if self.journal:
                self.journal.event(now, "halt", "", {"reason": reason})
        must, why = self.risk.emergency_liquidation_required(now)
        if must:
            self._square_off(now, f"emergency:{why}")
            return
        if self.risk.halted:
            # Halted means take no NEW risk. Existing positions are still
            # managed -- abandoning them unmanaged would be its own risk.
            self._manage_open_only(now)
        if self.cfg.risk.square_off_intraday and \
                self.clock.phase() == SessionPhase.SQUARE_OFF:
            self._square_off(now, "intraday_square_off")

    def _manage_open_only(self, now: int) -> None:
        for pos in list(self.portfolio.open_positions):
            px = self.portfolio.last(pos.symbol)
            if px <= 0 or pos.stop <= 0:
                continue
            hit = (pos.qty > 0 and px <= pos.stop) or \
                  (pos.qty < 0 and px >= pos.stop)
            if hit:
                self.router.flatten(pos.symbol, pos.qty, px)

    def _square_off(self, now: int, reason: str) -> None:
        for pos in list(self.portfolio.open_positions):
            px = self.portfolio.last(pos.symbol) or pos.avg_price
            self.router.flatten(pos.symbol, pos.qty, px)
            if self.journal:
                self.journal.event(now, "square_off", pos.symbol,
                                   {"reason": reason, "qty": pos.qty})
        # let the flattening orders reach the exchange. In sim we advance the
        # clock and step the feed; live runs on the wall clock and the paper
        # broker fills against the last live quote, so we just pump and wait.
        for _ in range(6):
            if self.live:
                self.broker.pump()
                self.bus.drain(500)
                time.sleep(0.05)
            else:
                self.clock.advance_s(0.2)
                self.broker.pump()
                self.feed.step(0.2)

    def _roll_day(self, now: int) -> None:
        self._square_off(now, "eod")
        self.day_index += 1
        self.portfolio.roll_day(now)
        self.mde.bars.force_close()
        self.features.reset_day()
        self.feed.roll_session()
        for s in self.symbols:
            self.mde.prev_close[s] = self.mde.ltp(s)
            self.mde.day_open.pop(s, None)

        # Advance the clock to the next session's open.
        #
        # Without this the simulated clock simply keeps running past 15:30
        # into the evening and overnight. Nothing errors; the engine just
        # stops opening positions forever, because the move generator quite
        # correctly refuses to take new intraday risk minutes before a close
        # -- and from its point of view the close is always minutes away. A
        # multi-day run silently becomes a one-day run.
        self._advance_to_next_session()

    def _advance_to_next_session(self) -> None:
        from ..core.clock import OPEN_TIME, is_holiday
        from datetime import timedelta
        dt_now = self.clock.now_dt()
        nxt = (dt_now + timedelta(days=1)).replace(
            hour=OPEN_TIME.hour, minute=OPEN_TIME.minute,
            second=0, microsecond=0)
        # skip weekends and exchange holidays
        for _ in range(10):
            if nxt.weekday() < 5 and not is_holiday(nxt.date()):
                break
            nxt = nxt + timedelta(days=1)
        self.clock.set_ns(int(nxt.timestamp() * NS))
        self.started_ns = min(self.started_ns, self.clock.now_ns())
        # every symbol's last tick is now hours stale by construction; clear
        # the staleness flags so the first ticks of the new session are not
        # rejected as a data outage
        for s in self.symbols:
            self.risk.on_data_stale(s, False)
            self._last_decision[s] = 0.0

        # A halt is a decision about a day. It does not silently expire; only
        # limits that are themselves daily start the new session clean.
        if self.risk.halt_reason in (HaltReason.DAILY_LOSS,
                                     HaltReason.LOSS_VELOCITY,
                                     HaltReason.CONSECUTIVE_LOSSES):
            self.risk.resume(self.clock.now_ns())
            self.portfolio.consecutive_losses = 0

    # ==================================================================
    # reporting
    # ==================================================================
    def snapshot(self) -> dict:
        return {
            "ts": self.clock.now_ns(),
            "time": self.clock.now_dt().strftime("%Y-%m-%d %H:%M:%S"),
            "session": self.clock.phase(),
            "portfolio": self.portfolio.snapshot(self.features.cross),
            "risk": self.risk.snapshot(),
            "execution": self.router.stats(),
            "broker": self.broker.stats(),
            "decisions": self.decisions,
            "actions": self.actions,
            "vetoed": self.vetoed,
            "warm": self.models.samples_seen >= self.warmup_labels,
            "stop_outs": self.stop_outs,
            "labels": self.models.samples_seen,
            "mean_search_ms": round(float(np.mean(self.search_ms)), 3)
            if self.search_ms else 0.0,
            "p99_search_ms": round(float(np.percentile(self.search_ms, 99)), 3)
            if len(self.search_ms) > 50 else 0.0,
            "mean_decision_us": round(float(np.mean(self.tick_to_decision_us)), 1)
            if self.tick_to_decision_us else 0.0,
            "positions": [
                {"symbol": p.symbol, "qty": p.qty,
                 "avg": round(p.avg_price, 2),
                 "ltp": round(self.portfolio.last(p.symbol), 2),
                 "stop": round(p.stop, 2), "target": round(p.target, 2),
                 "pnl": round(p.unrealised(self.portfolio.last(p.symbol)), 2),
                 "r": round(p.r_multiple(self.portfolio.last(p.symbol)), 2),
                 "regime": p.entry_regime}
                for p in self.portfolio.open_positions],
            "regimes": {s: self.regime.snapshot(s) for s in self.symbols},
            "loss_causes": self.monitor.cause_summary(),
        }

    def report(self, wall_s: float = 0.0) -> dict:
        rep = metrics.full_report(
            self.portfolio.trades, self.portfolio.equity_curve,
            extra={
                "run_id": self.cfg.run_id,
                "wall_s": round(wall_s, 1),
                "days": self.day_index + 1,
                "decisions": self.decisions,
                "actions": self.actions,
                "vetoed": self.vetoed,
                "stop_outs": self.stop_outs,
                "action_rate": round(self.actions / max(self.decisions, 1), 4),
                "latency": {
                    "mean_search_ms": round(float(np.mean(self.search_ms)), 3)
                    if self.search_ms else 0.0,
                    "p99_search_ms": round(float(np.percentile(self.search_ms, 99)), 3)
                    if len(self.search_ms) > 50 else 0.0,
                    "mean_loop_ms": round(float(np.mean(self.loop_ms)), 4)
                    if self.loop_ms else 0.0,
                    "mean_decision_us": round(float(np.mean(self.tick_to_decision_us)), 1)
                    if self.tick_to_decision_us else 0.0,
                },
                "execution": self.router.stats(),
                "broker": self.broker.stats(),
                "risk": self.risk.snapshot(),
                "models": self.models.stats(),
                "loss_causes": self.monitor.cause_summary(),
                "halts": self.halt_events,
                "regime_detection": {s: self.regime.snapshot(s)
                                     for s in self.symbols[:3]},
            })
        if self.journal:
            rep["journal"] = self.journal.summary()
            rep["decision_calibration"] = self.journal.calibration_report()
        return rep

    def close(self) -> None:
        if self.journal:
            self.journal.close()
