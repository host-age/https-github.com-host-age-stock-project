from __future__ import annotations

from unittest.mock import Mock

from gmq.data.nse_mcp import StreamableHttpMcpClient, default_clients


def test_default_clients_use_supplied_nse_endpoints():
    clients = default_clients()
    assert clients["bhavcopy"].url == "https://mcp.nseindia.in/bhavcopy/cm/mcp"
    assert clients["cm_market"].url == "https://mcp.nseindia.in/cmmkt/mcp"


def test_tool_selection_prefers_matching_name_or_description():
    c = StreamableHttpMcpClient("https://example.invalid/mcp")
    c._initialized = True
    c._tools = [
        type("T", (), {"name": "get_quote", "description": "current equity quote", "input_schema": {"properties": {"symbol": {}}}})(),
        type("T", (), {"name": "search_news", "description": "news", "input_schema": {"properties": {"query": {}}}})(),
    ]
    picked = c.choose_tool("quote", "equity", "symbol")
    assert picked is not None
    assert picked.name == "get_quote"


def test_symbol_args_are_limited_to_advertised_fields():
    c = StreamableHttpMcpClient("https://example.invalid/mcp")
    schema = {"properties": {
        "tradingsymbol": {}, "start_date": {}, "symbols": {}, "unsafe": {}
    }}
    args = c._symbol_args(schema, "RELIANCE")
    assert args["tradingsymbol"] == "RELIANCE"
    assert args["symbols"] == ["RELIANCE"]
    assert "unsafe" not in args
    assert "start_date" not in args
