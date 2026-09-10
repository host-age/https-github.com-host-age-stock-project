import json

from gmq.ai.ling_fin import LingFinAdvisor


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append((url, headers, json, timeout))
        return FakeResponse({
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "action": "WAIT",
                        "confidence": 0.82,
                        "risk_flag": True,
                        "reason": "Historical/news uncertainty is material.",
                    })
                }
            }]
        })


def test_ling_fin_disabled_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LING_FIN_ENABLED", raising=False)
    advisor = LingFinAdvisor(api_key="")
    assert advisor.enabled is False
    assert advisor.cached("RELIANCE") is None


def test_ling_fin_parses_advice(monkeypatch):
    monkeypatch.setenv("LING_FIN_ENABLED", "1")
    advisor = LingFinAdvisor(api_key="test", refresh_s=30)
    advisor._session = FakeSession()
    advice = advisor._ask("RELIANCE", {
        "price": 1000,
        "p_up": 0.61,
        "news": {"count": 3},
    })
    assert advice is not None
    assert advice.action == "WAIT"
    assert advice.risk_flag is True
    assert 0.0 <= advice.confidence <= 1.0


def test_ling_fin_submission_is_async_and_rate_limited(monkeypatch):
    monkeypatch.setenv("LING_FIN_ENABLED", "1")
    advisor = LingFinAdvisor(api_key="test", refresh_s=60)
    calls = []

    def fake_ask(symbol, context):
        calls.append((symbol, context))
        return None

    advisor._ask = fake_ask
    advisor.submit("TCS", {"price": 100})
    advisor.submit("TCS", {"price": 101})

    # Allow the daemon worker a moment to complete without making the test
    # depend on a fixed sleep. Poll briefly for the first worker.
    import time
    deadline = time.time() + 1.0
    while time.time() < deadline and not calls:
        time.sleep(0.01)
    assert len(calls) == 1
