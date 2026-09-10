"""Perplexity Finance/Search integration for historical and current stock news.

Design rules:
* Never call Perplexity from the market-data tick path.
* Persist normalized results locally so repeated decisions use cached memory.
* Respect provider rate limits with pacing, retries and 429/5xx backoff.
* Preserve publication timestamps and source URLs so backtests can enforce
  information-availability time and avoid look-ahead bias.
* Finance Search is used for structured financial context; Search is used for
  timestamped historical/current news discovery.

Environment:
  PERPLEXITY_API_KEY
  PERPLEXITY_ENABLED=1|0
  PERPLEXITY_NEWS_DB=runs/perplexity_news.sqlite3
  PERPLEXITY_TIMEOUT_S=12
  PERPLEXITY_MAX_RETRIES=3
  PERPLEXITY_MIN_REQUEST_INTERVAL_S=0.75
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

import requests


@dataclass(frozen=True)
class NewsItem:
    symbol: str
    title: str
    url: str
    snippet: str = ""
    published_at: str = ""
    source: str = "perplexity"
    retrieved_at: float = 0.0
    query: str = ""

    @property
    def key(self) -> str:
        raw = f"{self.symbol}|{self.url}|{self.published_at}|{self.title}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RateLimiter:
    def __init__(self, min_interval_s: float = 0.75) -> None:
        self.min_interval_s = max(float(min_interval_s), 0.0)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self.min_interval_s - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class NewsStore:
    """Persistent local news archive used by GMQ's knowledge layer."""

    def __init__(self, path: str = "runs/perplexity_news.sqlite3") -> None:
        self.path = path
        self._lock = threading.RLock()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS news (
                    key TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    retrieved_at REAL NOT NULL,
                    query TEXT NOT NULL
                )"""
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_news_symbol_date "
                "ON news(symbol, published_at)"
            )
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def upsert_many(self, items: Iterable[NewsItem]) -> int:
        rows = list(items)
        if not rows:
            return 0
        with self._lock, self._connect() as db:
            db.executemany(
                """INSERT OR REPLACE INTO news
                   (key,symbol,title,url,snippet,published_at,source,retrieved_at,query)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        item.key,
                        item.symbol.upper(),
                        item.title,
                        item.url,
                        item.snippet,
                        item.published_at,
                        item.source,
                        item.retrieved_at,
                        item.query,
                    )
                    for item in rows
                ],
            )
            db.commit()
        return len(rows)

    def recent(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                """SELECT * FROM news WHERE symbol=?
                   ORDER BY published_at DESC, retrieved_at DESC LIMIT ?""",
                (symbol.upper(), max(1, int(limit))),
            ).fetchall()
        return [NewsItem(**dict(row)) for row in rows]

    def between(
        self, symbol: str, after: str, before: str, limit: int = 100
    ) -> list[NewsItem]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                """SELECT * FROM news
                   WHERE symbol=? AND published_at>=? AND published_at<=?
                   ORDER BY published_at DESC LIMIT ?""",
                (symbol.upper(), after, before, max(1, int(limit))),
            ).fetchall()
        return [NewsItem(**dict(row)) for row in rows]


class PerplexityClient:
    """Thin HTTP client for Perplexity Search and Agent Finance Search."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        timeout_s: float = 12.0,
        max_retries: int = 3,
        min_request_interval_s: float = 0.75,
        base_url: str = "https://api.perplexity.ai",
    ) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY", "").strip()
        self.timeout_s = max(float(timeout_s), 1.0)
        self.max_retries = max(int(max_retries), 0)
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.limiter = RateLimiter(min_request_interval_s)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and os.getenv(
            "PERPLEXITY_ENABLED", "1"
        ).lower() in {"1", "true", "yes", "on"}

    def _post(self, path: str, payload: dict) -> dict:
        if not self.api_key:
            raise RuntimeError("PERPLEXITY_API_KEY is not configured")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                response = self.session.post(
                    f"{self.base_url}{path}",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(
                        f"retryable HTTP {response.status_code}"
                    )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError(f"Perplexity request failed: {last}") from last

    def search_news(
        self,
        symbol: str,
        *,
        after: Optional[str] = None,
        before: Optional[str] = None,
        recency: Optional[str] = None,
        max_results: int = 20,
    ) -> list[NewsItem]:
        if not self.enabled:
            return []
        symbol = symbol.upper()
        payload: dict[str, Any] = {
            "query": f"{symbol} NSE stock company financial market news",
            "country": "IN",
            "max_results": max(1, min(int(max_results), 20)),
            "search_language_filter": ["en"],
        }
        if after:
            payload["search_after_date_filter"] = after
        if before:
            payload["search_before_date_filter"] = before
        if recency:
            payload["search_recency_filter"] = recency

        data = self._post("/search", payload)
        now = time.time()
        items: list[NewsItem] = []
        for row in data.get("results", []):
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            title = str(row.get("title") or "").strip()
            if not url or not title:
                continue
            items.append(
                NewsItem(
                    symbol=symbol,
                    title=title,
                    url=url,
                    snippet=str(row.get("snippet") or ""),
                    published_at=str(row.get("date") or ""),
                    source=str(row.get("source") or "perplexity"),
                    retrieved_at=now,
                    query=str(payload["query"]),
                )
            )
        return items

    def finance_search(
        self, symbol: str, question: Optional[str] = None
    ) -> dict:
        """Use Agent API + finance_search for structured financial context."""
        if not self.enabled:
            return {}
        prompt = question or (
            f"Research {symbol} on NSE. Return current and historical financial "
            "context, earnings, valuation, material company developments and "
            "risks. Distinguish dates and do not invent missing data."
        )
        payload = {
            "model": "perplexity/sonar",
            "input": prompt,
            "tools": [{"type": "finance_search"}],
            "max_output_tokens": 1800,
        }
        return self._post("/v1/agent", payload)


class PerplexityNewsMemory:
    """Persistent historical/current news memory with background refresh."""

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        *,
        store_path: Optional[str] = None,
        client: Optional[PerplexityClient] = None,
        recent_refresh_s: float = 900.0,
    ) -> None:
        self.symbols = [s.upper() for s in (symbols or []) if s]
        self.store = NewsStore(
            store_path
            or os.getenv("PERPLEXITY_NEWS_DB", "runs/perplexity_news.sqlite3")
        )
        self.client = client or PerplexityClient(
            timeout_s=float(os.getenv("PERPLEXITY_TIMEOUT_S", "12")),
            max_retries=int(os.getenv("PERPLEXITY_MAX_RETRIES", "3")),
            min_request_interval_s=float(
                os.getenv("PERPLEXITY_MIN_REQUEST_INTERVAL_S", "0.75")
            ),
        )
        self.recent_refresh_s = max(float(recent_refresh_s), 60.0)
        self._last_refresh: dict[str, float] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.client.enabled or (
            self._thread and self._thread.is_alive()
        ):
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="perplexity-news", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def refresh_symbol(self, symbol: str) -> int:
        symbol = symbol.upper()
        now = time.time()
        with self._lock:
            last = self._last_refresh.get(symbol, 0.0)
            if now - last < self.recent_refresh_s:
                return 0
            self._last_refresh[symbol] = now
        items = self.client.search_news(symbol, recency="week", max_results=20)
        return self.store.upsert_many(items)

    def historical_backfill(
        self, symbol: str, after: date, before: date, max_results: int = 20
    ) -> int:
        """Backfill a bounded historical interval; dates become part of the query."""
        items = self.client.search_news(
            symbol,
            after=after.strftime("%m/%d/%Y"),
            before=before.strftime("%m/%d/%Y"),
            max_results=max_results,
        )
        return self.store.upsert_many(items)

    def recent(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        return self.store.recent(symbol, limit)

    def context(self, symbol: str, limit: int = 10) -> dict:
        items = self.recent(symbol, limit)
        return {
            "count": len(items),
            "items": [asdict(item) for item in items],
            "latest_published_at": items[0].published_at if items else "",
            "sources": sorted({x.source for x in items}),
        }

    def features(self, symbol: str) -> dict[str, float]:
        items = self.recent(symbol, 20)
        if not items:
            return {
                "news_count": 0.0,
                "news_recency": 0.0,
                "news_source_count": 0.0,
            }
        now = datetime.now(timezone.utc).timestamp()
        ages_h: list[float] = []
        for item in items:
            try:
                published = item.published_at.replace("Z", "+00:00")
                dt = datetime.fromisoformat(published)
                ages_h.append(max(0.0, (now - dt.timestamp()) / 3600.0))
            except (TypeError, ValueError):
                continue
        recency = max((1.0 / (1.0 + h / 24.0) for h in ages_h), default=0.0)
        return {
            "news_count": float(len(items)),
            "news_recency": float(recency),
            "news_source_count": float(len({x.source for x in items})),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            for symbol in self.symbols:
                if self._stop.is_set():
                    break
                try:
                    self.refresh_symbol(symbol)
                except Exception:
                    # External news is enrichment, never a reason to kill GMQ.
                    pass
            self._stop.wait(self.recent_refresh_s)
