from gmq.core.bus import EventBus, Topic
from gmq.core.latency import C_MPS, LatencyBudget, LatencyMonitor
from gmq.core.types import Tick


def test_propagation_formula_uses_distance_over_speed():
    lm = LatencyMonitor()
    assert lm.propagation_seconds(C_MPS) == 1.0
    assert lm.propagation_seconds(0.0) == 0.0


def test_stale_market_data_is_rejected():
    lm = LatencyMonitor(LatencyBudget(max_data_age_ms=10.0))
    trace_id = lm.new_trace("RELIANCE", 1_700_000_000_000_000_000)
    lm.finish_ingest(
        trace_id,
        1_700_000_000_000_000_000,
        receive_wall_ns=1_700_000_000_020_000_000,
    )
    assert lm.traces[trace_id].data_age_ns == 20_000_000
    assert not lm.admit(trace_id)
    assert lm.traces[trace_id].reject_reason == "STALE_MARKET_DATA"


def test_event_bus_records_live_tick_latency():
    bus = EventBus(swallow_errors=False)
    seen = []
    bus.subscribe(Topic.TICK, lambda t: seen.append(t.symbol), 10, "test")
    bus.emit(
        Topic.TICK,
        Tick(
            ts=1_700_000_000_000_000_000,
            symbol="RELIANCE",
            ltp=3000.0,
            bid=2999.95,
            ask=3000.05,
        ),
    )
    assert seen == ["RELIANCE"]
    stats = bus.stats()["latency"]
    assert stats["samples"] == 1
    assert stats["data_age_ms"]["p50"] >= 0.0
    assert stats["decision_ms"]["p50"] >= 0.0
    assert bus.latency.traces
