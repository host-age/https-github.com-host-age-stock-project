import importlib

from gmq.core.capabilities import Capability, all_statuses, enabled, status


def _reload_capabilities():
    import gmq.core.capabilities as cap
    return importlib.reload(cap)


def test_optional_capabilities_are_off_without_credentials(monkeypatch):
    for key in (
        "NSE_MCP_ENABLED",
        "KITE_API_KEY",
        "KITE_ACCESS_TOKEN",
        "GMQ_LIVE_TRADING",
        "PERPLEXITY_API_KEY",
        "PERPLEXITY_ENABLED",
        "OPENROUTER_API_KEY",
        "LING_FIN_ENABLED",
        "LING_FIN_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    cap = _reload_capabilities()
    assert enabled(Capability.KITE_EXECUTION) is False
    assert enabled(Capability.PERPLEXITY_NEWS) is False
    assert enabled(Capability.LING_FIN) is False
    assert isinstance(all_statuses(), dict)


def test_kite_execution_requires_explicit_live_gate(monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "key")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "token")
    monkeypatch.delenv("GMQ_LIVE_TRADING", raising=False)
    cap = _reload_capabilities()
    st = status(Capability.KITE_EXECUTION)
    assert st.configured is True
    assert st.enabled is False

    monkeypatch.setenv("GMQ_LIVE_TRADING", "1")
    cap = _reload_capabilities()
    assert cap.enabled(Capability.KITE_EXECUTION) is True


def test_local_ling_runtime_needs_no_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LING_FIN_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LING_FIN_ENABLED", "1")
    cap = _reload_capabilities()
    st = cap.status(Capability.LING_FIN)
    assert st.configured is True
    assert st.enabled is True
