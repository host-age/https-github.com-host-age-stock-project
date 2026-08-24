# NSE AI Trading Agent — Vercel Dashboard

This repository is prepared for Vercel as a **paper-trading dashboard** driven by live NSE market data. Live broker execution is disabled — every fill is simulated against a ₹10,000 paper book.

## Market data
Prices come from the public Yahoo Finance chart API (no API key needed). NSE tickers are looked up as `SYMBOL.NS` (e.g. `RELIANCE` → `RELIANCE.NS`); a few indices have friendly aliases (`NIFTY`, `BANKNIFTY`, `SENSEX`). If the live feed is unreachable or rate-limited, the app falls back to a synthetic price path and reports `data_source: "synthetic_fallback"` in the API response and a badge on the dashboard, so it never fails silently.

## NIFTY 100 scanner
`/api/scan?horizon=intraday|swing|invest` scans a static snapshot of the NIFTY 100 constituents (defined in `api/index.py`; NSE's official list drifts on periodic reconstitution, so treat this as approximate) and ranks them by signal score. Each horizon uses its own bar interval and fast/slow window — intraday (5m bars, 5/20-bar crossover), swing (daily bars over 6mo, 10/50-bar), invest (daily bars over 2y, 50/200-bar golden-cross style). Symbols are fetched in parallel and the scan result is cached per-horizon.

## Autonomous agent
`GET /api/tick` runs one full decision cycle — scan, exit anything that's turned bearish, buy top BUY-rated candidates up to 8 open positions — without any user interaction. Two schedulers drive it, because Vercel's Hobby plan caps Cron Jobs to once per day:

- **Vercel Cron** (`vercel.json`, `50 3 * * 1-5` — once daily at market open, weekdays) is the reliable baseline. Free on any plan, but only ticks once a day on Hobby.
- **GitHub Actions** (`.github/workflows/agent-tick.yml`, every 15 minutes during market hours) is what actually gives the agent a live, autonomous feel without needing Vercel Pro. One-time setup: add a repository variable `TICK_URL` = your Vercel production URL (Settings → Secrets and variables → Actions → Variables; find the exact domain under your Vercel project's Domains tab). Until it's set, the workflow no-ops with a message instead of failing. GitHub disables scheduled workflows after 60 days with no commits to the repo, and schedule timing isn't second-precise under GitHub's own load.
- If you're on **Vercel Pro**, you can drop the GitHub Actions workflow and just tighten `vercel.json`'s cron schedule instead (e.g. `*/5 3-10 * * 1-5`) — Pro doesn't have the daily cap.

Both routes hit the same `/api/tick` endpoint, so it's safe to have both active at once.

**Cron/Actions only fire against your deployed Production URL, never PR previews** — the agent won't tick on this PR's preview link until it merges to your default branch.

### Persistent state (Vercel KV)
Without persistent storage, the agent's cash/positions/trade log live only in the serverless function's memory and vanish between invocations — including between cron ticks, which defeats the point of running autonomously. To fix this:
1. In your Vercel project dashboard: **Storage → Create Database → KV**, then connect it to this project (Production + Preview environments).
2. Vercel automatically injects a `KV_URL` env var into the function's runtime — no code changes needed, `api/index.py` picks it up via the standard `redis` client.
3. `/api/status` reports `persisted: true` once it's reading/writing through KV; if it's still `false` after connecting, check the function logs for a connection error.

Without KV connected, the agent still runs (`autonomous: true`) but is amnesiac — every tick and every manual action operates on a fresh in-memory default rather than a coherent portfolio.

## Does the strategy actually make money?
`/api/backtest_portfolio?horizon=swing|invest&cost_bps=0` answers this empirically instead of assuming it: walk-forward over ~2 years of daily closes for every NIFTY 100 symbol (`decision_for` at bar *i* only ever sees `prices[:i+1]` — no lookahead), each compared against its own buy-and-hold return over the same window, plus the NIFTY 50 index (`^NSEI`) as the market-level bar to clear. The dashboard's "Strategy vs Buy & Hold" panel runs this and shows average/median strategy return, average buy-and-hold return, win rate against buy-and-hold, and the index return, plus a per-symbol breakdown.

**`cost_bps` defaults to 0 — frictionless.** That's a first check for whether the rule has any edge at all before transaction costs are even considered; it is *not* what real trading would return. Pass `cost_bps=10` (or check "include costs" on the dashboard) for a rough ~0.10%-per-leg NSE cost estimate (brokerage + STT + slippage, ~0.20% round trip) — a frictionless-positive, cost-adjusted-negative result means the edge, if any, doesn't survive real trading costs. Results are cached 15 minutes per horizon/cost combination.

## Endpoints
- `/` dashboard
- `/health` health check
- `/api/quote?symbol=RELIANCE` latest live price
- `/api/scan?horizon=intraday|swing|invest` ranked NIFTY 100 scan
- `/api/backtest_portfolio?horizon=swing|invest&cost_bps=0` walk-forward backtest across the NIFTY 100 vs buy-and-hold and the NIFTY 50 index
- `/api/tick` one autonomous decision cycle (called by Vercel Cron; safe to call manually too)
- `/api/status`
- `/api/simulate` — `{"symbol": "RELIANCE"}` manual single-symbol decision + paper fill
- `/api/backtest` — `{"symbol": "RELIANCE"}` (single-symbol backtest over ~2y of daily closes)
- `/api/reset`

## Limitations
- Position sizing in `/api/tick` is a flat 15% of cash per new position, capped at 8 open positions — a placeholder, not real risk management (no stop-losses, no volatility-scaled sizing, no portfolio-level exposure caps yet).
- The NIFTY 100 list is fundamentals-free — no P/E, earnings, debt ratios, F&O open interest, or news sentiment. The `invest` horizon is a price-trend read (50/200-bar crossover), not a genuine fundamentals-based investing signal.
- `/api/backtest_portfolio`'s cost model is a flat bps-per-leg approximation, not real order-book slippage, and the strategy's own thresholds (0.62/0.38 score cutoffs, window sizes) were picked by hand, not walk-forward-optimized or validated out-of-sample — a strong in-sample result here is a reason to test further, not a guarantee of live performance.
- Yahoo's chart API is unofficial and undocumented; treat quotes as near-real-time, not exchange-certified.

## Architecture boundary
Vercel hosts the dashboard/API. The future always-on market-data, ML, risk and broker-execution engine should run separately; broker credentials must remain server-side secrets.
