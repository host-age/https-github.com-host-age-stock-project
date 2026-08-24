# NSE AI Trading Agent — Vercel Dashboard

This repository is prepared for Vercel as a **paper-trading dashboard** driven by live NSE market data. Live broker execution is disabled — every fill is simulated against a ₹10,000 paper book.

## Market data
Prices come from the public Yahoo Finance chart API (no API key needed). NSE tickers are looked up as `SYMBOL.NS` (e.g. `RELIANCE` → `RELIANCE.NS`); a few indices have friendly aliases (`NIFTY`, `BANKNIFTY`, `SENSEX`). If the live feed is unreachable or rate-limited, the app falls back to a synthetic price path and reports `data_source: "synthetic_fallback"` in the API response and a badge on the dashboard, so it never fails silently.

## Vercel
Import this repository into Vercel. `vercel.json` routes `/api/*` to the FastAPI serverless function and serves the dashboard from `/public`.

## Endpoints
- `/` dashboard
- `/health` health check
- `/api/quote?symbol=RELIANCE` latest live price
- `/api/status`
- `/api/simulate` — `{"symbol": "RELIANCE"}`
- `/api/backtest` — `{"symbol": "RELIANCE"}` (runs over ~2y of daily closes)
- `/api/reset`

## Limitations
- Portfolio state (`cash`/`positions`/`trades`) lives in the function's memory, so it resets on a cold start — fine for a demo, not for durable state. Wiring in Vercel KV/Upstash would fix that if needed.
- Yahoo's chart API is unofficial and undocumented; treat quotes as near-real-time, not exchange-certified.

## Architecture boundary
Vercel hosts the dashboard/API. The future always-on market-data, ML, risk and broker-execution engine should run separately; broker credentials must remain server-side secrets.
