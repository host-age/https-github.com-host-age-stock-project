"""Latency measurement and admission control for live trading.

Separates the physical propagation lower bound (distance / speed) from the
actual observed trading latency measured with monotonic clocks and exchange
timestamps. The physical calculation is informational only; live admission is
based on measured data age and processing time.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

NS = 1_000_000_000
C_MPS = 299_792_458.0


class LatencyStage:
    INGEST = "ingest"
    FEATURES = "features"
    MODELS = "models"
    SEARCH = "search"
    RISK = "risk"
    ROUTING = "routing"
    BROKER = "broker"
    TOTAL = "total"


@dataclass(slots=True)
class LatencyTrace:
    trace_id: str
    symbol: str
    exchange_ts_ns: int
    received_mono_ns: int
    stages: Dict[str, int] = field(default_factory=dict)
    decision_ts_mono_ns: int = 0
    order_ts_mono_ns: int = 0
    accepted: bool = True
    reject_reason: str = ""

    @property
    def data_age_ns(self) -> int:
        return int(self.stages.get("data_age_ns", 0))

    @property
    def decision_latency_ns(self) -> int:
        if not self.decision_ts_mono_ns:
            return 0
        return max(0, self.decision_ts_mono_ns - self.received_mono_ns)

    @property
    def order_latency_ns(self) -> int:
        if not self.order_ts_mono_ns:
            return 0
        return max(0, self.order_ts_mono_ns - self.received_mono_ns)


@dataclass(slots=True)
class LatencyBudget:
    max_data_age_ms: float = 250.0
    max_decision_ms: float = 100.0
    max_order_path_ms: float = 250.0
    reserve_ms: float = 10.0

    @property
    def usable_decision_ms(self) -> float:
        return max(0.0, self.max_decision_ms - self.reserve_ms)


class RollingLatency:
    __slots__ = ("maxlen", "values")

    def __init__(self, maxlen: int = 5000):
        self.maxlen = maxlen
        self.values: List[float] = []

    def add(self, value_ms: float) -> None:
        self.values.append(float(max(0.0, value_ms)))
        if len(self.values) > self.maxlen:
            del self.values[: len(self.values) - self.maxlen]

    def percentile(self, q: float) -> float:
        if not self.values:
            return 0.0
        xs = sorted(self.values)
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        frac = pos - lo
        return xs[lo] + (xs[hi] - xs[lo]) * frac


class LatencyMonitor:
    """Track market-data age and decision/order-path latency."""

    def __init__(self, budget: Optional[LatencyBudget] = None):
        self.budget = budget or LatencyBudget()
        self._seq = 0
        self.traces: Dict[str, LatencyTrace] = {}
        self.data_age_ms = RollingLatency()
        self.decision_ms = RollingLatency()
        self.order_path_ms = RollingLatency()
        self.accepted = 0
        self.rejections = 0

    @staticmethod
    def propagation_seconds(distance_m: float,
                            speed_mps: float = C_MPS) -> float:
        """Theoretical propagation time: distance / speed."""
        if not math.isfinite(distance_m) or distance_m < 0:
            raise ValueError("distance_m must be finite and non-negative")
        if not math.isfinite(speed_mps) or speed_mps <= 0:
            raise ValueError("speed_mps must be finite and positive")
        return distance_m / speed_mps

    def new_trace(self, symbol: str, exchange_ts_ns: int) -> str:
        self._seq += 1
        trace_id = f"L{self._seq:09d}"
        self.traces[trace_id] = LatencyTrace(
            trace_id=trace_id,
            symbol=symbol,
            exchange_ts_ns=int(exchange_ts_ns or 0),
            received_mono_ns=time.perf_counter_ns(),
        )
        return trace_id

    def finish_ingest(self, trace_id: str, exchange_ts_ns: int,
                      receive_wall_ns: Optional[int] = None) -> float:
        """Measure exchange-event age when the event reaches the process."""
        tr = self.traces[trace_id]
        wall_ns = int(receive_wall_ns or time.time_ns())
        ex_ns = int(exchange_ts_ns or tr.exchange_ts_ns or 0)
        age_ns = max(0, wall_ns - ex_ns) if ex_ns else 0
        tr.stages["data_age_ns"] = age_ns
        age_ms = age_ns / 1e6
        self.data_age_ms.add(age_ms)
        return age_ms

    def mark_stage(self, trace_id: str, stage: str, elapsed_ns: int) -> int:
        tr = self.traces[trace_id]
        ns = max(0, int(elapsed_ns))
        tr.stages[stage] = ns
        return ns

    def mark_decision(self, trace_id: str) -> float:
        tr = self.traces[trace_id]
        tr.decision_ts_mono_ns = time.perf_counter_ns()
        ms = tr.decision_latency_ns / 1e6
        self.decision_ms.add(ms)
        return ms

    def mark_order(self, trace_id: str) -> float:
        tr = self.traces[trace_id]
        tr.order_ts_mono_ns = time.perf_counter_ns()
        ms = tr.order_latency_ns / 1e6
        self.order_path_ms.add(ms)
        return ms

    def admit(self, trace_id: str) -> bool:
        """Reject stale or overly-late decisions before order submission."""
        tr = self.traces[trace_id]
        if tr.data_age_ns:
            age_ms = tr.data_age_ns / 1e6
            if age_ms > self.budget.max_data_age_ms:
                tr.accepted = False
                tr.reject_reason = "STALE_MARKET_DATA"
                self.rejections += 1
                return False
        if tr.decision_ts_mono_ns:
            decision_ms = tr.decision_latency_ns / 1e6
            if decision_ms > self.budget.usable_decision_ms:
                tr.accepted = False
                tr.reject_reason = "DECISION_LATENCY_BUDGET"
                self.rejections += 1
                return False
        self.accepted += 1
        return True

    def snapshot(self) -> dict:
        return {
            "samples": len(self.decision_ms.values),
            "data_age_ms": self._percentiles(self.data_age_ms),
            "decision_ms": self._percentiles(self.decision_ms),
            "order_path_ms": self._percentiles(self.order_path_ms),
            "accepted": self.accepted,
            "rejections": self.rejections,
            "budget": {
                "max_data_age_ms": self.budget.max_data_age_ms,
                "max_decision_ms": self.budget.max_decision_ms,
                "usable_decision_ms": self.budget.usable_decision_ms,
                "max_order_path_ms": self.budget.max_order_path_ms,
            },
        }

    @staticmethod
    def _percentiles(store: RollingLatency) -> dict:
        return {
            "p50": round(store.percentile(0.50), 3),
            "p95": round(store.percentile(0.95), 3),
            "p99": round(store.percentile(0.99), 3),
            "p999": round(store.percentile(0.999), 3),
            "max": round(max(store.values), 3) if store.values else 0.0,
        }

    def prune(self, keep: int = 1000) -> None:
        """Bound trace memory; rolling latency statistics are retained."""
        if len(self.traces) <= keep:
            return
        ids = list(self.traces)
        for key in ids[: len(ids) - keep]:
            self.traces.pop(key, None)
