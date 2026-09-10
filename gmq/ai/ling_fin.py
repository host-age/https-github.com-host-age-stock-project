"""Asynchronous finance-focused second opinion using OpenRouter.

The advisor is enrichment only: no price feed, sizing, risk override, or order
placement. Calls are cached/background so network latency can never sit on the
market tick or execution path.

Environment:
  LING_FIN_ENABLED=1
  OPENROUTER_API_KEY=...
  LING_FIN_MODEL=inclusionai/ling-3.0-flash-fin:free
  LING_FIN_BASE_URL=https://openrouter.ai/api/v1
  LING_FIN_REFRESH_S=60
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


@dataclass(frozen=True)
class LingAdvice:
    symbol: str
    generated_at: float
    action: str = "UNKNOWN"
    confidence: float = 0.0
    risk_flag: bool = False
    reason: str = ""
    raw: str = ""


class LingFinAdvisor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = 10.0,
        refresh_s: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = model or os.getenv(
            "LING_FIN_MODEL", "inclusionai/ling-3.0-flash-fin:free"
        )
        self.base_url = (base_url or os.getenv(
            "LING_FIN_BASE_URL", "https://openrouter.ai/api/v1"
        )).rstrip("/")
        self.timeout_s = max(float(timeout_s), 2.0)
        self.refresh_s = max(float(refresh_s), 30.0)
        self.enabled = bool(self.api_key) and os.getenv(
            "LING_FIN_ENABLED", "1"
        ).lower() in {"1", "true", "yes", "on"}
        self._cache: dict[str, LingAdvice] = {}
        self._last_request: dict[str, float] = {}
        self._active: set[str] = set()
        self._lock = threading.RLock()
        self._session = requests.Session()

    def cached(self, symbol: str, now: Optional[float] = None,
               max_age_s: Optional[float] = None) -> Optional[LingAdvice]:
        now = now or time.time()
        age = self.refresh_s if max_age_s is None else max(float(max_age_s), 0.0)
        with self._lock:
            advice = self._cache.get(symbol.upper())
            if advice is None or now - advice.generated_at > age:
                return None
            return advice

    def submit(self, symbol: str, context: dict[str, Any]) -> None:
        """Queue a challenge request; returns immediately."""
        symbol = symbol.upper()
        if not self.enabled:
            return
        with self._lock:
            now = time.time()
            if symbol in self._active:
                return
            if now - self._last_request.get(symbol, 0.0) < self.refresh_s:
                return
            self._last_request[symbol] = now
            self._active.add(symbol)
        t = threading.Thread(
            target=self._worker, args=(symbol, context),
            name=f"ling-fin-{symbol}", daemon=True,
        )
        t.start()

    def _worker(self, symbol: str, context: dict[str, Any]) -> None:
        try:
            advice = self._ask(symbol, context)
            if advice is not None:
                with self._lock:
                    self._cache[symbol] = advice
        except Exception:
            pass
        finally:
            with self._lock:
                self._active.discard(symbol)

    def _ask(self, symbol: str, context: dict[str, Any]) -> Optional[LingAdvice]:
        system = (
            "You are a finance research challenger inside a quantitative trading system. "
            "Use only the supplied facts. Do not invent prices, news, dates or financials. "
            "Challenge the proposed trade and identify material risks. You are advisory only. "
            "Return ONLY one compact JSON object with keys: action, confidence, risk_flag, reason. "
            "action must be BUY, SELL, WAIT, or REVIEW. confidence is 0..1. "
            "risk_flag is true or false."
        )
        user = f"Symbol: {symbol}\nContext:\n{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 500,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/host-age/https-github-com-host-age-stock-project",
            "X-Title": "GMQ Finance Advisor",
        }
        r = self._session.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=self.timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = self._parse_json(content)
        if not parsed:
            return LingAdvice(symbol=symbol, generated_at=time.time(), raw=content)
        return LingAdvice(
            symbol=symbol,
            generated_at=time.time(),
            action=str(parsed.get("action", "REVIEW")).upper(),
            confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.0)))),
            risk_flag=bool(parsed.get("risk_flag", False)),
            reason=str(parsed.get("reason", ""))[:1000],
            raw=content[:4000],
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        text = str(content).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].lstrip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except (TypeError, ValueError):
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(text[start:end + 1])
                    return obj if isinstance(obj, dict) else {}
                except (TypeError, ValueError):
                    return {}
            return {}
