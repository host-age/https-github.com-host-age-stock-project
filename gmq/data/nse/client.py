"""Configurable NSE historical-data download contract.

Endpoint templates remain configuration-driven because exchange URLs and report
routes can change. The client writes raw responses to a local immutable cache.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional


class NSEDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadedFile:
    source: str
    path: str
    sha256: str
    bytes: int


class NSEDataClient:
    """Small cache-aware downloader with an injectable transport.

    `fetcher` receives `(source, headers, timeout_s)` and returns raw bytes.
    Keeping transport injectable makes ingestion testable and avoids coupling
    the archive to one HTTP library.
    """

    def __init__(
        self,
        *,
        cache_dir: str = "data/raw/nse",
        fetcher: Optional[Callable[[str, Mapping[str, str], float], bytes]] = None,
        user_agent: str = "GMQ-NSE-DataClient/1.0",
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.fetcher = fetcher
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self.backoff_s = max(0.0, backoff_s)

    def save_bytes(self, source: str, filename: str, data: bytes) -> DownloadedFile:
        if not data:
            raise NSEDownloadError("empty response")
        target = self.cache_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        return DownloadedFile(source, str(target), hashlib.sha256(data).hexdigest(), len(data))

    def download(self, source: str, filename: str, *, headers: Optional[Mapping[str, str]] = None) -> DownloadedFile:
        target = self.cache_dir / filename
        if target.exists() and target.stat().st_size:
            data = target.read_bytes()
            return DownloadedFile(source, str(target), hashlib.sha256(data).hexdigest(), len(data))
        if self.fetcher is None:
            raise NSEDownloadError("no transport configured; supply fetcher or use save_bytes")
        hdrs = {"User-Agent": self.user_agent, "Accept": "*/*"}
        if headers:
            hdrs.update(headers)
        last: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.save_bytes(source, filename, self.fetcher(source, hdrs, self.timeout_s))
            except Exception as exc:
                last = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_s * (2 ** attempt))
        raise NSEDownloadError(f"failed to download {source}: {last}") from last

    def expand(self, template: str, *, trading_date: str, **values: str) -> str:
        return template.format(date=trading_date, **values)
