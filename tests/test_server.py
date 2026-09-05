"""Smoke tests for the Render live-trading server.

Prove the box boots and behaves without credentials (the state Render starts
in), serves the dashboard, and guards the token endpoint. The live trading path
itself needs real Kite credentials and market hours, so it is not exercised
here -- these assert the deployable shell is sound.
"""
import os
import sys

sys.path.insert(0, ".")
import pytest


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    from gmq.app import server
    server._status.update(state="starting", detail="", started=False)
    server._engine = None
    return server, server.app.test_client()


def test_health_ok(client):
    _, c = client
    r = c.get("/health")
    assert r.status_code == 200 and r.json["ok"] is True


def test_dashboard_served(client):
    _, c = client
    r = c.get("/")
    assert r.status_code == 200 and b"Paper Trading" in r.data


def test_no_credentials_waits_cleanly(client):
    server, _ = client
    server._start_trader()          # must not raise without creds
    assert server._status["state"] == "waiting_for_credentials"


def test_status_json_without_engine(client):
    _, c = client
    r = c.get("/api/status")
    assert r.status_code == 200 and "state" in r.json


def test_set_token_requires_secret(client, monkeypatch):
    server, c = client
    monkeypatch.setenv("ADMIN_SECRET", "s3cret")
    bad = c.post("/set-token", json={"secret": "wrong", "access_token": "x"})
    assert bad.status_code == 401


def test_symbols_from_env(client, monkeypatch):
    server, _ = client
    monkeypatch.setenv("SYMBOLS", "reliance, tcs ,infy")
    assert server._symbols() == ["RELIANCE", "TCS", "INFY"]


def test_summary_is_secret_free_and_shaped(client):
    _, c = client
    r = c.get("/api/summary")
    assert r.status_code == 200
    body = r.get_data(as_text=True).lower()
    # the compact desk summary must never leak the admin secret or a token
    assert "admin_secret" not in body and "access_token" not in body
    assert "state" in r.json and "live" in r.json
