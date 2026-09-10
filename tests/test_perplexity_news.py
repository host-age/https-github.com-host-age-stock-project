from datetime import date

from gmq.news.perplexity import (
    NewsItem,
    PerplexityClient,
    PerplexityNewsMemory,
    NewsStore,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append((url, json))
        if url.endswith("/search"):
            return FakeResponse(
                {
                    "results": [
                        {
                            "title": "Reliance announces results",
                            "url": "https://example.com/ril-results",
                            "snippet": "Quarterly result summary",
                            "date": "2026-09-10T10:00:00Z",
                            "source": "example.com",
                        }
                    ]
                }
            )
        return FakeResponse({"output": []})


def test_search_news_preserves_dates_and_sources():
    client = PerplexityClient(api_key="test-key", min_request_interval_s=0)
    client.session = FakeSession()
    items = client.search_news(
        "RELIANCE",
        after="09/01/2026",
        before="09/11/2026",
        max_results=20,
    )
    assert len(items) == 1
    assert items[0].symbol == "RELIANCE"
    assert items[0].published_at.startswith("2026-09-10")
    assert items[0].source == "example.com"
    assert client.session.calls[0][1]["search_after_date_filter"] == "09/01/2026"


def test_news_memory_deduplicates_persistently(tmp_path):
    store = NewsStore(str(tmp_path / "news.sqlite3"))
    item = NewsItem(
        symbol="TCS",
        title="Test",
        url="https://example.com/a",
        published_at="2026-09-10T10:00:00Z",
        retrieved_at=1.0,
    )
    assert store.upsert([item, item]) == 2
    assert len(store.recent("TCS")) == 1


def test_historical_backfill_uses_explicit_date_window(tmp_path):
    client = PerplexityClient(api_key="test-key", min_request_interval_s=0)
    client.session = FakeSession()
    memory = PerplexityNewsMemory(
        ["RELIANCE"],
        store_path=str(tmp_path / "news.sqlite3"),
        client=client,
    )
    assert memory.historical_backfill(
        "RELIANCE", date(2026, 9, 1), date(2026, 9, 10)
    ) == 1
    assert len(memory.recent("RELIANCE")) == 1


def test_disabled_client_makes_no_external_request():
    client = PerplexityClient(api_key="", min_request_interval_s=0)
    assert client.enabled is False
    assert client.search_news("TCS") == []
    assert client.finance_search("TCS") == {}
