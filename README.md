# NSE AI Trading Agent — Vercel Dashboard

This repository is prepared for Vercel as a **paper-trading dashboard** driven by live NSE market data. Live broker execution is disabled — every fill is simulated against a ₹10,000 paper book.

## Market data
Prices come from the public Yahoo Finance chart API (no API key needed). NSE tickers are looked up as `SYMBOL.NS` (e.g. `RELIANCE` → `RELIANCE.NS`); a few indices have friendly aliases (`NIFTY`, `BANKNIFTY`, `SENSEX`). If the live feed is unreachable or rate-limited, the app falls back to a synthetic price path and reports `data_source: "synthetic_fallback"` in the API response and a badge on the dashboard, so it never fails silently.

## NIFTY 100 scanner
`/api/scan?horizon=intraday|swing|invest` scans a static snapshot of the NIFTY 100 constituents (defined in `api/index.py`; NSE's official list drifts on periodic reconstitution, so treat this as approximate) and ranks them by signal score. Each horizon uses its own bar interval and fast/slow window — intraday (5m bars, 5/20-bar crossover), swing (daily bars over 6mo, 10/50-bar), invest (daily bars over 2y, 50/200-bar golden-cross style). Symbols are fetched in parallel and the scan result is cached per-horizon.

## Autonomous agent
`GET /api/tick` runs one full decision cycle — scan, exit anything that's turned bearish, buy top BUY-rated candidates up to 8 open positions — without any user interaction. It's wired to **Vercel Cron** in `vercel.json` (`*/5 3-10 * * 1-5`, every 5 minutes during NSE market hours in UTC, refined further inside the handler using precise IST market-hours logic). Two things worth knowing:
- **Cron only fires against your Production deployment**, never PR previews — the agent won't tick on a preview URL until this merges to your default branch.
- **Vercel's cron frequency limits vary by plan.** If you're on the Hobby tier, check your dashboard's Cron Jobs settings — Vercel may cap or reject sub-daily schedules depending on current plan limits.

### Persistent state (Vercel KV)
Without persistent storage, the agent's cash/positions/trade log live only in the serverless function's memory and vanish between invocations — including between cron ticks, which defeats the point of running autonomously. To fix this:
1. In your Vercel project dashboard: **Storage → Create Database → KV**, then connect it to this project (Production + Preview environments).
2. Vercel automatically injects a `KV_URL` env var into the function's runtime — no code changes needed, `api/index.py` picks it up via the standard `redis` client.
3. `/api/status` reports `persisted: true` once it's reading/writing through KV; if it's still `false` after connecting, check the function logs for a connection error.

Without KV connected, the agent still runs (`autonomous: true`) but is amnesiac — every tick and every manual action operates on a fresh in-memory default rather than a coherent portfolio.

## Endpoints
- `/` dashboard
- `/health` health check
- `/api/quote?symbol=RELIANCE` latest live price
- `/api/scan?horizon=intraday|swing|invest` ranked NIFTY 100 scan
- `/api/tick` one autonomous decision cycle (called by Vercel Cron; safe to call manually too)
- `/api/status`
- `/api/simulate` — `{"symbol": "RELIANCE"}` manual single-symbol decision + paper fill
- `/api/backtest` — `{"symbol": "RELIANCE"}` (runs over ~2y of daily closes)
- `/api/reset`

## Limitations
- Position sizing in `/api/tick` is a flat 15% of cash per new position, capped at 8 open positions — a placeholder, not real risk management (no stop-losses, no volatility-scaled sizing, no portfolio-level exposure caps yet).
- The NIFTY 100 list is fundamentals-free — no P/E, earnings, debt ratios, F&O open interest, or news sentiment. The `invest` horizon is a price-trend read (50/200-bar crossover), not a genuine fundamentals-based investing signal.
- Yahoo's chart API is unofficial and undocumented; treat quotes as near-real-time, not exchange-certified.

## Architecture boundary
Vercel hosts the dashboard/API. The future always-on market-data, ML, risk and broker-execution engine should run separately; broker credentials must remain server-side secrets.
