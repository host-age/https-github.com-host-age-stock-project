"""Minimal Streamable-HTTP MCP client for NSE market knowledge.

The trading hot path must never wait on MCP/network I/O. This client is a
small synchronous transport intended to run from a background worker. It
supports MCP initialization, tool discovery and heuristic tool selection so
we do not hard-code an NSE tool name that may change independently of GMQ.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


@dataclass
class McpTool:
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)


class StreamableHttpMcpClient:
    def __init__(self, url: str, timeout_s: float = 8.0):
        self.url = url
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        })
        self._rpc_id = 0
        self._session_id: Optional[str] = None
        self._initialized = False
        self._lock = threading.Lock()
        self._tools: Optional[List[McpTool]] = None

    def _rpc(self, method: str, params: Optional[dict] = None) -> Any:
        self._rpc_id += 1
        payload = {"jsonrpc": "2.0", "id": self._rpc_id,
                   "method": method, "params": params or {}}
        headers = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        r = self.session.post(self.url, json=payload, headers=headers,
                              timeout=self.timeout_s)
        r.raise_for_status()
        sid = r.headers.get("Mcp-Session-Id")
        if sid:
            self._session_id = sid
        return self._decode_response(r)

    @staticmethod
    def _decode_response(resp: requests.Response) -> Any:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        text = resp.text.strip()
        if "text/event-stream" not in ctype:
            return resp.json()
        data_lines: List[str] = []
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        for raw in reversed(data_lines):
            if not raw or raw == "[DONE]":
                continue
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                continue
        raise RuntimeError("MCP response contained no JSON data event")

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._rpc("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "gmq-nse-client", "version": "1.0"},
            })
            # MCP requires an initialized notification after the initialize
            # response before normal operations continue.
            self._rpc_id += 1
            payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            headers = {}
            if self._session_id:
                headers["Mcp-Session-Id"] = self._session_id
            r = self.session.post(self.url, json=payload, headers=headers,
                                  timeout=self.timeout_s)
            r.raise_for_status()
            self._initialized = True

    def list_tools(self, force: bool = False) -> List[McpTool]:
        self.initialize()
        if self._tools is not None and not force:
            return self._tools
        result = self._rpc("tools/list")
        raw = result.get("result", result).get("tools", [])
        self._tools = [McpTool(
            name=str(x.get("name", "")),
            description=str(x.get("description", "")),
            input_schema=x.get("inputSchema", {}) or {},
        ) for x in raw if x.get("name")]
        return self._tools

    def choose_tool(self, *terms: str) -> Optional[McpTool]:
        terms_l = [x.lower() for x in terms]
        tools = self.list_tools()
        best = None
        best_score = 0
        for t in tools:
            hay = f"{t.name} {t.description}".lower()
            score = sum(1 for term in terms_l if term in hay)
            if score > best_score:
                best_score = score
                best = t
        return best

    def call_tool(self, tool_name: str, arguments: Optional[dict] = None) -> Any:
        self.initialize()
        result = self._rpc("tools/call", {
            "name": tool_name,
            "arguments": arguments or {},
        })
        return result.get("result", result)

    def close(self) -> None:
        try:
            self.session.close()
        finally:
            self._initialized = False
            self._session_id = None


def default_clients(timeout_s: float = 8.0) -> Dict[str, StreamableHttpMcpClient]:
    return {
        "bhavcopy": StreamableHttpMcpClient(
            os.getenv("NSE_BHAVCOPY_MCP_URL",
                      "https://mcp.nseindia.in/bhavcopy/cm/mcp"), timeout_s),
        "cm_market": StreamableHttpMcpClient(
            os.getenv("NSE_CM_MARKET_MCP_URL",
                      "https://mcp.nseindia.in/cmmkt/mcp"), timeout_s),
    }
