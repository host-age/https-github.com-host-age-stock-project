"""Zerodha Kite daily login + instrument-map helper.

Kite access tokens expire every morning (~06:00 IST), so a live paper-trading
setup needs a once-a-day login. This script does the two things you actually
need and nothing else:

  1. print the login URL, take the request_token you get back, and exchange it
     for today's access_token (saved to kite_token.json);
  2. build the {instrument_token: tradingsymbol} map for your universe from
     Kite's instrument dump (saved to kite_instruments.json), which KiteFeed
     needs to route ticks.

Nothing here places an order or can. It only authenticates and reads reference
data. Requires:  pip install kiteconnect

Usage:
  # one-time / daily: get today's access token
  python3 tools/kite_login.py login --api-key YOUR_KEY --api-secret YOUR_SECRET

  # same, and push it straight to a running server (skips the manual curl step)
  python3 tools/kite_login.py login --api-key YOUR_KEY --api-secret YOUR_SECRET \\
      --server-url https://YOUR-SERVICE.onrender.com --admin-secret YOUR_ADMIN_SECRET

  # once (or when your universe changes): build the token map
  python3 tools/kite_login.py instruments --symbols RELIANCE,TCS,INFY

The daily login itself (opening the URL, authenticating with Zerodha, 2FA)
cannot be automated away without storing your trading account password and/or
TOTP secret in this codebase, which is a materially worse security trade than
one manual login a day -- so that part stays manual, on purpose. --server-url
only automates the second half: getting the resulting token onto the running
server, which today means a separate manual `curl -X POST .../set-token`.
"""
from __future__ import annotations

import json
import sys


TOKEN_FILE = "kite_token.json"
INSTR_FILE = "kite_instruments.json"


def _kite(api_key: str):
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        sys.exit("kiteconnect not installed. Run:  pip install kiteconnect")
    return KiteConnect(api_key=api_key)


def login(api_key: str, api_secret: str, server_url: str = "",
          admin_secret: str = "") -> None:
    kite = _kite(api_key)
    print("\n1. Open this URL, log in, and approve:\n")
    print("   " + kite.login_url() + "\n")
    print("2. You'll be redirected to a URL containing  request_token=XXXX")
    req = input("3. Paste the request_token here: ").strip()
    data = kite.generate_session(req, api_secret=api_secret)
    access_token = data["access_token"]
    tok = {"api_key": api_key, "access_token": access_token,
           "user_id": data.get("user_id", "")}
    with open(TOKEN_FILE, "w") as fh:
        json.dump(tok, fh, indent=2)
    print(f"\nSaved today's access token to {TOKEN_FILE}. "
          "It is valid until tomorrow ~06:00 IST.")
    if server_url:
        push_token(server_url, admin_secret, access_token)


def push_token(server_url: str, admin_secret: str, access_token: str) -> None:
    """POST today's token to a running server's /set-token, in place of the
    manual curl step DEPLOY.md otherwise walks you through every morning."""
    if not admin_secret:
        print("\nWARNING: --server-url given without --admin-secret; "
              "not pushing the token (the endpoint requires it).",
              file=sys.stderr)
        return
    import requests
    url = server_url.rstrip("/") + "/set-token"
    try:
        r = requests.post(url, json={"secret": admin_secret,
                                     "access_token": access_token},
                          timeout=10)
    except requests.RequestException as e:
        print(f"\nFailed to reach {url}: {e}", file=sys.stderr)
        return
    if r.ok:
        print(f"\nPushed today's token to {url} -- trader (re)starting.")
    else:
        print(f"\n{url} returned {r.status_code}: {r.text}", file=sys.stderr)


def instruments(symbols: list, exchange: str = "NSE") -> None:
    with open(TOKEN_FILE) as fh:
        tok = json.load(fh)
    kite = _kite(tok["api_key"])
    kite.set_access_token(tok["access_token"])
    dump = kite.instruments(exchange)
    want = {s.upper() for s in symbols}
    mp = {int(r["instrument_token"]): r["tradingsymbol"]
          for r in dump if r["tradingsymbol"] in want}
    missing = want - set(mp.values())
    with open(INSTR_FILE, "w") as fh:
        json.dump({str(k): v for k, v in mp.items()}, fh, indent=2)
    print(f"Mapped {len(mp)} symbols -> {INSTR_FILE}")
    if missing:
        print(f"WARNING: not found on {exchange}: {sorted(missing)}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    args = dict(a.split("=", 1) if "=" in a else (a, "")
                for a in sys.argv[2:] if a.startswith("--"))
    if cmd == "login":
        key = _flag("--api-key"); sec = _flag("--api-secret")
        if not (key and sec):
            sys.exit("need --api-key and --api-secret")
        login(key, sec, server_url=_flag("--server-url"),
              admin_secret=_flag("--admin-secret"))
    elif cmd == "instruments":
        syms = _flag("--symbols")
        if not syms:
            sys.exit("need --symbols RELIANCE,TCS,...")
        instruments([s for s in syms.split(",") if s])
    else:
        sys.exit(__doc__)


def _flag(name: str) -> str:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return ""


if __name__ == "__main__":
    main()
