"""Always-on live paper-trading server for Render (or any host with net access).

One process that:

  * on start, if Kite credentials are present, builds the live engine (real
    ticks in, paper fills, no real orders) and runs it in a background thread
    through market hours;
  * serves a dashboard at ``/`` you can open from any browser or phone -- the
    laptop that watches it needs no compute of its own;
  * exposes ``/api/status`` (JSON snapshot) and ``/health`` (for Render's
    health check);
  * accepts the daily Kite access token via ``POST /set-token`` (protected by
    a shared secret), because Kite tokens expire every morning and this avoids
    a redeploy to refresh them.

Everything heavy (training, backtesting) stays off this box; it only trades and
displays. That is what keeps it comfortable on a small instance.

Environment variables (set in the Render dashboard):
  KITE_API_KEY        your Kite Connect api key
  KITE_ACCESS_TOKEN   today's access token (or POST it to /set-token later)
  KITE_USER_ID        your Kite user id (optional)
  SYMBOLS             comma-separated NSE symbols (default: the config's 10)
  ADMIN_SECRET        shared secret required to POST a new token
  START_CAPITAL       paper capital in rupees (default 1,000,000)
  PORT                set by Render automatically
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

from flask import Flask, jsonify, request, Response

from ..core.config import Config
from .engine import TradingEngine

app = Flask(__name__)

# ---- shared state -------------------------------------------------------
_engine: Optional[TradingEngine] = None
_thread: Optional[threading.Thread] = None
_status = {"state": "starting", "detail": "", "started": False}
_lock = threading.Lock()


def _symbols() -> list:
    env = os.environ.get("SYMBOLS", "").strip()
    if env:
        return [s.strip().upper() for s in env.split(",") if s.strip()]
    return list(Config().data.symbols)


def _build_instrument_map(api_key: str, access_token: str, symbols: list) -> dict:
    """Ask Kite for the token->symbol map for our universe."""
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    want = {s.upper() for s in symbols}
    mp = {}
    for r in kite.instruments("NSE"):
        if r["tradingsymbol"] in want and r.get("segment") == "NSE":
            mp[int(r["instrument_token"])] = r["tradingsymbol"]
    return mp


def _start_trader() -> None:
    """Build the live engine and run it. Called in a background thread."""
    global _engine
    api_key = os.environ.get("KITE_API_KEY", "").strip()
    access = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    symbols = _symbols()

    if not (api_key and access):
        with _lock:
            _status.update(state="waiting_for_credentials",
                           detail="Set KITE_API_KEY and KITE_ACCESS_TOKEN "
                                  "(or POST today's token to /set-token).")
        return

    try:
        from kiteconnect import KiteTicker
        token_map = _build_instrument_map(api_key, access, symbols)
        if not token_map:
            with _lock:
                _status.update(state="error",
                               detail=f"No NSE instruments matched {symbols}.")
            return

        cfg = Config()
        cfg.data.symbols = list(token_map.values())
        cfg.initial_capital = float(os.environ.get("START_CAPITAL",
                                                    cfg.initial_capital))
        cfg.execution.mode = "paper"

        eng = TradingEngine(cfg, live=True, journal=True,
                            run_dir=os.environ.get("RUN_DIR", "runs/live"))

        from ..data.kite_feed import KiteFeed
        ticker = KiteTicker(api_key, access)
        feed = KiteFeed(eng.bus, cfg.data.symbols, token_map,
                        clock=eng.clock, ticker=ticker)
        eng.attach_live_feed(feed)

        with _lock:
            _engine = eng
            _status.update(state="running", started=True,
                           detail=f"Live on {len(token_map)} symbols.")
        # run until the session closes; run_live handles feed start/stop
        eng.run_live(poll_s=0.2)
        with _lock:
            _status.update(state="session_closed",
                           detail="Market closed; trader idle until next start.")
    except ImportError:
        with _lock:
            _status.update(state="error",
                           detail="kiteconnect not installed. Add it to "
                                  "requirements.txt.")
    except Exception as e:  # pragma: no cover - live only
        with _lock:
            _status.update(state="error", detail=f"{type(e).__name__}: {e}")


def _ensure_started() -> None:
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
    _thread = threading.Thread(target=_start_trader, daemon=True)
    _thread.start()


# ---- routes -------------------------------------------------------------
@app.route("/health")
def health() -> Response:
    return jsonify(ok=True, state=_status["state"])


@app.route("/api/status")
def api_status() -> Response:
    out = dict(_status)
    with _lock:
        eng = _engine
    if eng is not None:
        try:
            out["snapshot"] = eng.snapshot()
        except Exception as e:
            out["snapshot_error"] = str(e)
    return jsonify(out)


@app.route("/api/summary")
def api_summary() -> Response:
    """A compact, secret-free snapshot for the operations desk to read.

    The scheduled-task 'desk' (market-open check, end-of-day report) fetches
    this. It is deliberately a small, flat, fixed shape -- only what a human
    report needs -- and never contains the admin secret, tokens, or anything
    that could place a trade. Safe to leave public.
    """
    with _lock:
        eng = _engine
        state = _status["state"]
    out = {"state": state, "live": eng is not None}
    if eng is not None:
        try:
            s = eng.snapshot()
            p = s.get("portfolio", {})
            out.update({
                "time_ist": s.get("time"),
                "session": s.get("session"),
                "warmed_up": s.get("warm"),
                "equity": round(float(p.get("equity", 0)), 2),
                "day_pnl": round(float(p.get("day_pnl", 0)), 2),
                "return_pct": round(float(p.get("return_pct", 0)), 3),
                "open_positions": p.get("open_positions", 0),
                "decisions": s.get("decisions", 0),
                "actions": s.get("actions", 0),
                "stop_outs": s.get("stop_outs", 0),
                "risk_halted": bool(s.get("risk", {}).get("halted", False)),
                "halt_reason": s.get("risk", {}).get("halt_reason", "") or "",
                "positions": [
                    {"symbol": x["symbol"], "qty": x["qty"],
                     "pnl": x["pnl"], "r": x["r"]}
                    for x in s.get("positions", [])
                ],
            })
        except Exception as e:
            out["summary_error"] = str(e)
    return jsonify(out)


@app.route("/set-token", methods=["POST"])
def set_token() -> Response:
    """Refresh the daily Kite access token without a redeploy."""
    secret = os.environ.get("ADMIN_SECRET", "")
    given = request.headers.get("X-Admin-Secret") or \
        (request.json or {}).get("secret", "") if request.is_json else \
        request.form.get("secret", "")
    if not secret or given != secret:
        return jsonify(ok=False, error="unauthorized"), 401
    token = (request.json or {}).get("access_token") if request.is_json \
        else request.form.get("access_token")
    if not token:
        return jsonify(ok=False, error="missing access_token"), 400
    os.environ["KITE_ACCESS_TOKEN"] = token.strip()
    # (re)start the trader with the fresh token
    global _thread
    _thread = None
    _ensure_started()
    return jsonify(ok=True, detail="token updated; trader (re)starting")


@app.route("/")
def dashboard() -> Response:
    return Response(_DASHBOARD_HTML, mimetype="text/html")


# minimal, dependency-free dashboard; polls /api/status
_DASHBOARD_HTML = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>NSE Paper Trader</title>
<style>
 :root{color-scheme:light dark}
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0e1116;color:#e6edf3}
 header{padding:14px 18px;background:#161b22;border-bottom:1px solid #30363d;
   display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
 h1{font-size:16px;margin:0;font-weight:600}
 .pill{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
 .ok{background:#1a7f37;color:#fff}.warn{background:#9e6a03;color:#fff}
 .bad{background:#b62324;color:#fff}.idle{background:#30363d;color:#adbac7}
 main{padding:18px;max-width:1100px;margin:0 auto}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px}
 .card .k{font-size:12px;color:#8b949e}.card .v{font-size:22px;font-weight:700;margin-top:2px}
 .pos-p{color:#3fb950}.neg{color:#f85149}
 table{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}
 th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #21262d;font-variant-numeric:tabular-nums}
 th:first-child,td:first-child{text-align:left}
 th{background:#1c2128;color:#8b949e;font-size:12px}
 .muted{color:#8b949e;font-size:13px}
 .setup{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;line-height:1.7}
 code{background:#21262d;padding:1px 6px;border-radius:5px}
</style></head><body>
<header>
 <h1>&#9819; NSE Grandmaster &mdash; Paper Trading (live prices, no real orders)</h1>
 <span id=state class="pill idle">connecting&hellip;</span>
</header>
<main>
 <div id=body><p class=muted>Loading&hellip;</p></div>
</main>
<script>
const money=n=>'₹'+(n||0).toLocaleString('en-IN',{maximumFractionDigits:0});
const cls=n=>n>0?'pos-p':n<0?'neg':'';
async function tick(){
 let r; try{ r=await (await fetch('/api/status')).json(); }catch(e){ return; }
 const st=r.state, el=document.getElementById('state');
 const map={running:'ok',session_closed:'idle',waiting_for_credentials:'warn',
   starting:'warn',error:'bad'};
 el.className='pill '+(map[st]||'idle'); el.textContent=st.replace(/_/g,' ');
 const b=document.getElementById('body');
 if(st==='waiting_for_credentials'||st==='error'){
   b.innerHTML='<div class=setup><b>'+(st==='error'?'Problem':'Setup needed')+
     '</b><br>'+(r.detail||'')+'<br><br>Set <code>KITE_API_KEY</code> and '+
     '<code>KITE_ACCESS_TOKEN</code> in the Render dashboard, or POST today\\'s '+
     'token to <code>/set-token</code>. The trader starts automatically once '+
     'credentials are present and the market is open (09:15&ndash;15:30 IST).</div>';
   return;
 }
 const s=r.snapshot; if(!s){ b.innerHTML='<p class=muted>'+(r.detail||'')+'</p>'; return; }
 const p=s.portfolio||{};
 const cards=[
  ['Equity',money(p.equity)],
  ['Day P&L',money(p.day_pnl),cls(p.day_pnl)],
  ['Return',(p.return_pct>=0?'+':'')+(p.return_pct||0).toFixed(2)+'%',cls(p.return_pct)],
  ['Open positions',p.open_positions||0],
  ['Decisions',s.decisions||0],
  ['Session',s.session||''],
  ['Warmed up',s.warm?'yes':'no'],
  ['Time (IST)',s.time||''],
 ].map(c=>'<div class=card><div class=k>'+c[0]+'</div><div class="v '+(c[2]||'')+
   '">'+c[1]+'</div></div>').join('');
 let rows=(s.positions||[]).map(x=>'<tr><td>'+x.symbol+'</td><td>'+x.qty+
   '</td><td>'+x.avg+'</td><td>'+x.ltp+'</td><td>'+x.stop+'</td><td class="'+
   cls(x.pnl)+'">'+money(x.pnl)+'</td><td class="'+cls(x.r)+'">'+
   (x.r||0).toFixed(2)+'R</td><td class=muted>'+(x.regime||'')+'</td></tr>').join('');
 if(!rows) rows='<tr><td colspan=8 class=muted>No open positions.</td></tr>';
 b.innerHTML='<div class=grid>'+cards+'</div><table><thead><tr><th>Symbol</th>'+
   '<th>Qty</th><th>Avg</th><th>LTP</th><th>Stop</th><th>P&L</th><th>R</th>'+
   '<th>Regime</th></tr></thead><tbody>'+rows+'</tbody></table>'+
   '<p class=muted style=margin-top:12px>Search '+(s.mean_search_ms||0)+
   'ms avg &middot; '+(s.labels||0)+' labels &middot; paper money, no real orders.</p>';
}
tick(); setInterval(tick,2000);
</script></body></html>"""


def main() -> None:
    _ensure_started()
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
