"""Perplexity Search + Finance Search integration for GMQ.

External calls are asynchronous/background enrichment only. The live trading
hot path reads local SQLite memory and never waits on Perplexity.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional

import requests

from ..core.capabilities import Capability, enabled as capability_enabled


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


class NewsStore:
    """Durable local archive so news can be reused without repeated API calls."""

    def __init__(self, path: str = "runs/perplexity_news.sqlite3") -> None:
        self.path = path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS news (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                snippet TEXT NOT NULL,
                published_at TEXT NOT NULL,
                source TEXT NOT NULL,
                retrieved_at REAL NOT NULL,
                query TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol_date ON news(symbol, published_at)")
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _from_row(row: sqlite3.Row) -> NewsItem:
        return NewsItem(
            symbol=row["symbol"], title=row["title"], url=row["url"],
            snippet=row["snippet"], published_at=row["published_at"],
            source=row["source"], retrieved_at=row["retrieved_at"],
            query=row["query"],
        )

    def upsert(self, items: list[NewsItem]) -> int:
        if not items:
            return 0
        with self._lock, self._connect() as db:
            db.executemany(
                """INSERT OR REPLACE INTO news
                (key,symbol,title,url,snippet,published_at,source,retrieved_at,query)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                [(x.key, x.symbol.upper(), x.title, x.url, x.snippet,
                  x.published_at, x.source, x.retrieved_at, x.query) for x in items],
            )
            db.commit()
        return len(items)

    def recent(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT * FROM news WHERE symbol=? ORDER BY published_at DESC, retrieved_at DESC LIMIT ?",
                (symbol.upper(), max(1, int(limit))),
            ).fetchall()
        return [self._from_row(r) for r in rows]

    def between(self, symbol: str, after: str, before: str, limit: int = 100) -> list[NewsItem]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                """SELECT * FROM news WHERE symbol=? AND published_at>=? AND published_at<=?
                   ORDER BY published_at DESC LIMIT ?""",
                (symbol.upper(), after, before, max(1, int(limit))),
            ).fetchall()
        return [self._from_row(r) for r in rows]


class PerplexityClient:
    """HTTP client for Perplexity Search API and Agent API finance_search."""

    def __init__(self, api_key: Optional[str] = None, *, timeout_s: float = 12.0,
                 max_retries: int = 3, min_request_interval_s: float = 0.75) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY", "").strip()
        self.timeout_s = max(float(timeout_s), 1.0)
        self.max_retries = max(int(max_retries), 0)
        self.session = requests.Session()
        self._pace = threading.Lock()
        self._last_request = 0.0

    @property
    def enabled(self) -> bool:
        return capability_enabled(Capability.PERPLEXITY_NEWS)

    def _wait(self, min_interval_s: float) -> None:
        with self._pace:
            delay = min_interval_s - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()

    def _post(self, path: str, payload: dict) -> dict:
        if not self.enabled:
            raise RuntimeError("Perplexity capability is not enabled")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self._wait(float(os.getenv("PERPLEXITY_MIN_REQUEST_INTERVAL_S", "0.75")))
            try:
                r = self.session.post(
                    f"https://api.perplexity.ai{path}", headers=headers,
                    json=payload, timeout=self.timeout_s,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"retryable HTTP {r.status_code}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2.0 ** attempt, 8.0))
        raise RuntimeError(f"Perplexity request failed: {last}") from last

    def search_news(self, symbol: str, *, after: Optional[str] = None,
                    before: Optional[str] = None, recency: Optional[str] = None,
                    max_results: int = 20) -> list[NewsItem]:
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
        out: list[NewsItem] = []
        for row in data.get("results", []):
            if not isinstance(row, dict):
                continue
            title, url = str(row.get("title") or "").strip(), str(row.get("url") or "").strip()
            if not title or not url:
                continue
            out.append(NewsItem(
                symbol=symbol, title=title, url=url,
                snippet=str(row.get("snippet") or ""),
                published_at=str(row.get("date") or ""),
                source=str(row.get("source") or "perplexity"),
                retrieved_at=now, query=payload["query"],
            ))
        return out

    def finance_search(self, symbol: str, question: Optional[str] = None) -> dict:
        if not self.enabled:
            return {}
        prompt = question or (
            f"Research {symbol} on NSE. Return current and historical financial context, "
            "earnings, valuation, material developments and risks. Distinguish dates "
            "and do not invent missing data."
        )
        return self._post("/v1/agent", {
            "model": "perplexity/sonar",
            "input": prompt,
            "tools": [{"type": "finance_search"}],
            "max_output_tokens": 1800,
        })


class PerplexityNewsMemory:
    """Background-refreshing historical/current news memory for GMQ."""

    def __init__(self, symbols: Optional[list[str]] = None, *,
                 store_path: Optional[str] = None, client: Optional[PerplexityClient] = None,
                 refresh_s: float = 900.0) -> None:
        self.symbols = [s.upper() for s in (symbols or []) if s]
        self.store = NewsStore(store_path or os.getenv(
            "PERPLEXITY_NEWS_DB", "runs/perplexity_news.sqlite3"))
        self.client = client or PerplexityClient()
        self.refresh_s = max(float(refresh_s), 60.0)
        self._last: dict[str, float] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.client.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="perplexity-news", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def refresh_symbol(self, symbol: str) -> int:
        symbol = symbol.upper()
        with self._lock:
            now = time.time()
            if now - self._last.get(symbol, 0.0) < self.refresh_s:
                return 0
            self._last[symbol] = now
        return self.store.upsert(self.client.search_news(symbol, recency="week", max_results=20))

    def historical_backfill(self, symbol: str, after: date, before: date, max_results: int = 20) -> int:
        items = self.client.search_news(
            symbol, after=after.strftime("%m/%d/%Y"), before=before.strftime("%m/%d/%Y"),
            max_results=max_results,
        )
        return self.store.upsert(items)

    def recent(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        return self.store.recent(symbol, limit)

    def context(self, symbol: str, limit: int = 10) -> dict:
        items = self.recent(symbol, limit)
        return {
            "symbol": symbol.upper(), "count": len(items),
            "latest_published_at": items[0].published_at if items else "",
            "sources": sorted({x.source for x in items}),
            "items": [asdict(x) for x in items],
        }

    def features(self, symbol: str) -> dict[str, float]:
        items = self.recent(symbol, 20)
        if not items:
            return {"news_count": 0.0, "news_recency": 0.0, "news_source_count": 0.0}
        now = datetime.now(timezone.utc).timestamp()
        ages = []
        for item in items:
            try:
                dt = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
                ages.append(max(0.0, (now - dt.timestamp()) / 3600.0))
            except (TypeError, ValueError):
                continue
        return {
            "news_count": float(len(items)),
            "news_recency": float(max((1.0 / (1.0 + h / 24.0) for h in ages), default=0.0)),
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
                    pass
            self._stop.wait(self.refresh_s)
