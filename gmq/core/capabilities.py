"""Central capability/access gates for GMQ external integrations.

Credentials and runtimes unlock capabilities; absence of access never requires
code changes and never changes the deterministic risk/execution contracts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Capability(str, Enum):
    NSE_BHAVCOPY = "nse_bhavcopy"
    NSE_MARKET = "nse_market"
    KITE_MARKET_DATA = "kite_market_data"
    KITE_EXECUTION = "kite_execution"
    PERPLEXITY_NEWS = "perplexity_news"
    LING_FIN = "ling_fin"


@dataclass(frozen=True)
class CapabilityStatus:
    name: Capability
    enabled: bool
    configured: bool
    reason: str


_ENV_KEYS: Mapping[Capability, tuple[str, ...]] = {
    Capability.NSE_BHAVCOPY: ("NSE_MCP_ENABLED",),
    Capability.NSE_MARKET: ("NSE_MCP_ENABLED",),
    Capability.KITE_MARKET_DATA: ("KITE_API_KEY", "KITE_ACCESS_TOKEN"),
    Capability.KITE_EXECUTION: ("KITE_API_KEY", "KITE_ACCESS_TOKEN"),
    Capability.PERPLEXITY_NEWS: ("PERPLEXITY_API_KEY", "PERPLEXITY_ENABLED"),
    Capability.LING_FIN: ("OPENROUTER_API_KEY", "LING_FIN_ENABLED"),
}


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def status(capability: Capability) -> CapabilityStatus:
    keys = _ENV_KEYS[capability]
    values = {k: os.getenv(k, "").strip() for k in keys}

    if capability in {Capability.NSE_BHAVCOPY, Capability.NSE_MARKET}:
        enabled = _truthy(values.get("NSE_MCP_ENABLED"))
        return CapabilityStatus(capability, enabled, enabled,
                                "enabled" if enabled else "NSE_MCP_ENABLED is off")

    if capability is Capability.KITE_EXECUTION:
        configured = bool(values.get("KITE_API_KEY")) and bool(values.get("KITE_ACCESS_TOKEN"))
        live = _truthy(os.getenv("GMQ_LIVE_TRADING"))
        enabled = configured and live
        if not configured:
            reason = "Kite credentials are not configured"
        elif not live:
            reason = "live trading gate is off"
        else:
            reason = "enabled"
        return CapabilityStatus(capability, enabled, configured, reason)

    if capability is Capability.KITE_MARKET_DATA:
        configured = bool(values.get("KITE_API_KEY")) and bool(values.get("KITE_ACCESS_TOKEN"))
        return CapabilityStatus(capability, configured, configured,
                                "enabled" if configured else "Kite credentials are not configured")

    if capability is Capability.PERPLEXITY_NEWS:
        configured = bool(values.get("PERPLEXITY_API_KEY"))
        enabled = configured and _truthy(values.get("PERPLEXITY_ENABLED"), default=True)
        return CapabilityStatus(capability, enabled, configured,
                                "enabled" if enabled else "Perplexity disabled or key missing")

    if capability is Capability.LING_FIN:
        configured = bool(values.get("OPENROUTER_API_KEY")) or bool(os.getenv("LING_FIN_BASE_URL", "").strip())
        enabled = configured and _truthy(values.get("LING_FIN_ENABLED"), default=True)
        return CapabilityStatus(capability, enabled, configured,
                                "enabled" if enabled else "Ling runtime/key is not configured")

    return CapabilityStatus(capability, False, False, "unsupported capability")


def enabled(capability: Capability) -> bool:
    return status(capability).enabled


def all_statuses() -> dict[str, dict[str, object]]:
    return {
        cap.value: {
            "enabled": st.enabled,
            "configured": st.configured,
            "reason": st.reason,
        }
        for cap in Capability
        for st in [status(cap)]
    }
