"""Event-driven backbone (spec §12) with live-path latency instrumentation.

Everything downstream of market data is push-based: the feed emits, the bus
dispatches synchronously to subscribers in priority order. No polling loops
anywhere on the hot path.

Two dispatch modes:
  * ``emit()`` -- synchronous, in-line. Lowest latency, used on the hot path.
  * ``post()`` -- queued, drained by ``drain()``. Used for slow/side-effecting
    consumers so they cannot add latency to a trading decision.

The bus records per-handler latency and, for market-data events, feeds the
central LatencyMonitor. Exchange timestamps are used for data-age measurement;
monotonic clocks are used for elapsed durations.
"""
from __future__ import annotations

import time
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, DefaultDict, Deque, Dict, List, Any, Tuple

from .latency import LatencyMonitor, LatencyStage


class Topic:
    TICK = "tick"
    DEPTH = "depth"
    BAR = "bar"
    BAR_CLOSE = "bar_close"
    FEATURES = "features"
    REGIME = "regime"
    SIGNAL = "signal"
    DECISION = "decision"
    ORDER_NEW = "order_new"
    ORDER_UPDATE = "order_update"
    FILL = "fill"
    POSITION = "position"
    RISK_EVENT = "risk_event"
    NEWS = "news"
    HALT = "halt"
    SESSION = "session"
    HEARTBEAT = "heartbeat"


@dataclass(order=True)
class _Sub:
    priority: int
    seq: int
    fn: Callable = field(compare=False)
    name: str = field(default="", compare=False)
    calls: int = field(default=0, compare=False)
    total_ns: int = field(default=0, compare=False)
    max_ns: int = field(default=0, compare=False)
    errors: int = field(default=0, compare=False)


class EventBus:
    """Deterministic, single-threaded, priority-ordered dispatch."""

    __slots__ = (
        "_subs", "_queue", "_seq", "_counts", "_swallow", "_log",
        "latency",
    )

    def __init__(self, swallow_errors: bool = True,
                 latency: LatencyMonitor | None = None):
        self._subs: DefaultDict[str, List[_Sub]] = defaultdict(list)
        self._queue: Deque[Tuple[str, Any]] = deque()
        self._seq = 0
        self._counts: DefaultDict[str, int] = defaultdict(int)
        self._swallow = swallow_errors
        self._log: Deque[str] = deque(maxlen=200)
        self.latency = latency or LatencyMonitor()

    # -- wiring ---------------------------------------------------------
    def subscribe(self, topic: str, fn: Callable, priority: int = 100,
                  name: str = "") -> None:
        """Lower priority number == called earlier.

        Convention used across the system:
            10  data normalisation / book maintenance
            15  feature engineering
            30  regime + models
            40  strategy / search
            50  risk engine
            60  execution
            90  monitors
           200  journalling, dashboards (usually via post())
        """
        self._seq += 1
        s = _Sub(priority, self._seq, fn,
                 name or getattr(fn, "__qualname__", "?"))
        subs = self._subs[topic]
        subs.append(s)
        subs.sort()

    def unsubscribe(self, topic: str, fn: Callable) -> None:
        self._subs[topic] = [s for s in self._subs[topic] if s.fn is not fn]

    # -- dispatch -------------------------------------------------------
    def emit(self, topic: str, payload: Any) -> None:
        """Synchronous dispatch. Hot path.

        For market-data payloads, a latency trace starts at dispatch entry.
        Handler timings are recorded into the trace using the same monotonic
        clock already used by the bus' per-handler statistics. This adds only
        a small amount of bookkeeping and keeps the actual decision path
        synchronous and deterministic.
        """
        self._counts[topic] += 1

        trace_id = None
        trace = None
        if topic in (Topic.TICK, Topic.DEPTH):
            symbol = getattr(payload, "symbol", "")
            exchange_ts = int(getattr(payload, "ts", 0) or 0)
            if symbol:
                trace_id = self.latency.new_trace(symbol, exchange_ts)
                self.latency.finish_ingest(trace_id, exchange_ts)
                trace = self.latency.traces[trace_id]

        dispatch_start = time.perf_counter_ns()
        for s in self._subs.get(topic, ()):
            t0 = time.perf_counter_ns()
            try:
                s.fn(payload)
            except Exception as e:                     # noqa: BLE001
                s.errors += 1
                self._log.append(
                    f"{topic}/{s.name}: {type(e).__name__}: {e}"
                )
                if not self._swallow:
                    raise
            dt = time.perf_counter_ns() - t0
            s.calls += 1
            s.total_ns += dt
            if dt > s.max_ns:
                s.max_ns = dt
            if trace_id:
                self.latency.mark_stage(
                    trace_id, f"{topic}:{s.name}", dt
                )

        if trace_id and trace is not None:
            dispatch_ns = time.perf_counter_ns() - dispatch_start
            self.latency.mark_stage(
                trace_id,
                LatencyStage.TOTAL,
                dispatch_ns,
            )
            # A data event has completed its current synchronous processing.
            # This is an observation metric, not an order admission decision.
            self.latency.mark_decision(trace_id)

        # Bound per-event trace memory even during high-tick-rate sessions.
        if trace_id and (self._counts[topic] & 0x3FF) == 0:
            self.latency.prune()

    def post(self, topic: str, payload: Any) -> None:
        """Deferred dispatch -- kept off the critical path."""
        self._queue.append((topic, payload))

    def drain(self, budget: int = 10_000) -> int:
        n = 0
        while self._queue and n < budget:
            topic, payload = self._queue.popleft()
            self.emit(topic, payload)
            n += 1
        return n

    # -- introspection --------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"counts": dict(self._counts), "handlers": []}
        for topic, subs in self._subs.items():
            for s in subs:
                if not s.calls:
                    continue
                out["handlers"].append({
                    "topic": topic,
                    "name": s.name,
                    "priority": s.priority,
                    "calls": s.calls,
                    "mean_us": s.total_ns / s.calls / 1000.0,
                    "max_us": s.max_ns / 1000.0,
                    "errors": s.errors,
                })
        out["handlers"].sort(key=lambda h: -h["mean_us"] * h["calls"])
        out["errors"] = list(self._log)[-20:]
        out["latency"] = self.latency.snapshot()
        return out

    def reset_stats(self) -> None:
        self._counts.clear()
        for subs in self._subs.values():
            for s in subs:
                s.calls = s.total_ns = s.max_ns = s.errors = 0


class TimerWheel:
    """Simulation-clock timers so nothing has to poll.

    Used for latency modelling (an order 'arrives' at the exchange N micros
    after it was sent), periodic risk sweeps and session transitions.
    """

    __slots__ = ("_heap", "_seq")

    def __init__(self):
        self._heap: List[Tuple[int, int, Callable, Any]] = []
        self._seq = 0

    def schedule(self, at_ns: int, fn: Callable, payload: Any = None) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (at_ns, self._seq, fn, payload))

    def next_ts(self):
        return self._heap[0][0] if self._heap else None

    def run_until(self, now_ns: int) -> int:
        n = 0
        while self._heap and self._heap[0][0] <= now_ns:
            _, _, fn, payload = heapq.heappop(self._heap)
            if payload is None:
                fn()
            else:
                fn(payload)
            n += 1
        return n

    def __len__(self):
        return len(self._heap)
