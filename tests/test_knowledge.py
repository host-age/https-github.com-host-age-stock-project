from pathlib import Path

from gmq.knowledge import FutureEvent, KnowledgeStore, PreTradeKnowledge


def test_knowledge_persists_and_restores(tmp_path: Path):
    path = tmp_path / "knowledge.json"
    store = KnowledgeStore(str(path), window=128)
    store.register("RELIANCE", sector="ENERGY")
    store.observe("RELIANCE", 1, 100.0, volume=10_000, spread_bps=3.0, liquidity=0.9)
    store.observe("RELIANCE", 2, 101.0, volume=10_100, spread_bps=3.0, liquidity=0.9)
    store.record_outcome("RELIANCE", 100.0, won=True)
    store.add_event(FutureEvent("RELIANCE", 10_000, "RESULT", severity=0.7, source="test"))
    store.checkpoint()

    restored = KnowledgeStore(str(path), window=128)
    snap = restored.snapshot("RELIANCE", now_ns=3)
    assert snap.observed_ticks == 2
    assert snap.current_price == 101.0
    assert snap.sector == "ENERGY"
    assert snap.event_risk == 0.7
    assert restored.historical_win_rate("RELIANCE") > 0.5


def test_pretrade_blocks_non_positive_edge():
    store = KnowledgeStore(window=128)
    store.register("TCS", sector="IT")
    for i in range(60):
        store.observe("TCS", i + 1, 100 + i * 0.01, spread_bps=2.0, liquidity=0.9)
    gate = PreTradeKnowledge(store)
    verdict = gate.evaluate(
        "TCS", 61, direction=1, confidence=0.8,
        expected_edge_bps=0.0, current_spread_bps=2.0,
        liquidity=0.9, data_fresh=True)
    assert not verdict.allowed
    assert "non_positive_edge" in verdict.reason


def test_pretrade_blocks_material_future_event():
    store = KnowledgeStore(window=128)
    store.register("INFY", sector="IT")
    for i in range(60):
        store.observe("INFY", i + 1, 100.0, spread_bps=2.0, liquidity=0.9)
    store.add_event(FutureEvent("INFY", 10_000, "RESULT", severity=0.95))
    gate = PreTradeKnowledge(store, max_event_risk=0.90)
    verdict = gate.evaluate(
        "INFY", 61, direction=1, confidence=0.8,
        expected_edge_bps=12.0, current_spread_bps=2.0,
        liquidity=0.9, data_fresh=True)
    assert not verdict.allowed
    assert "event_risk_limit" in verdict.reason


def test_coverage_report_without_mcp_reports_disabled():
    store = KnowledgeStore(window=32)
    store.register("RELIANCE")
    store.register("TCS")
    report = store.coverage_report(now_ns=10_000)
    assert report["mcp_enabled"] is False
    assert report["symbols_tracked"] == 2
    assert report["symbols_synced"] == 0
    assert report["fraction_synced"] == 0.0
    for sym in ("RELIANCE", "TCS"):
        assert report["symbols"][sym]["mcp_synced"] is False
        assert report["symbols"][sym]["mcp_age_s"] is None
        assert report["symbols"][sym]["mcp_stale"] is True


def test_coverage_report_marks_fresh_sync_as_not_stale():
    store = KnowledgeStore(window=32)
    store.register("INFY")
    now_ns = 100 * 1_000_000_000
    tr = store._stocks["INFY"]
    tr.mcp_attempts = 3
    tr.mcp_successes = 2
    tr.mcp_last_success_ns = now_ns - 5 * 1_000_000_000  # 5s ago

    report = store.coverage_report(now_ns)
    sym = report["symbols"]["INFY"]
    assert sym["mcp_synced"] is True
    assert sym["mcp_stale"] is False
    assert sym["mcp_age_s"] == 5.0
    assert report["symbols_synced"] == 1
    assert report["fraction_synced"] == 1.0


def test_coverage_report_flags_stale_sync_and_surfaces_last_error():
    store = KnowledgeStore(window=32)
    store.register("HDFCBANK")
    now_ns = 10_000 * 1_000_000_000
    tr = store._stocks["HDFCBANK"]
    stale_after_s = 3.0 * store._mcp_interval_s
    tr.mcp_successes = 1
    tr.mcp_last_success_ns = now_ns - int((stale_after_s + 60) * 1_000_000_000)
    tr.mcp_last_error = "ConnectionError: timed out"

    report = store.coverage_report(now_ns)
    sym = report["symbols"]["HDFCBANK"]
    assert sym["mcp_synced"] is True       # it did sync at some point...
    assert sym["mcp_stale"] is True        # ...but that data is now too old to trust
    assert sym["mcp_last_error"] == "ConnectionError: timed out"
    assert report["symbols_stale"] == 1


def test_pretrade_approves_clean_setup():
    store = KnowledgeStore(window=128)
    store.register("SBIN", sector="BANK")
    for i in range(120):
        store.observe("SBIN", i + 1, 100 + i * 0.01, spread_bps=1.5, liquidity=0.95)
    for _ in range(8):
        store.record_outcome("SBIN", 10.0, won=True)
    gate = PreTradeKnowledge(store)
    verdict = gate.evaluate(
        "SBIN", 200, direction=1, confidence=0.9,
        expected_edge_bps=15.0, current_spread_bps=1.5,
        liquidity=0.95, data_fresh=True)
    assert verdict.allowed
    assert verdict.score >= gate.min_score
