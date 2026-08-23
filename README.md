# NSE AI Trading Agent — Vercel Dashboard

This repository is prepared for Vercel as a **paper-trading dashboard**. Live broker execution is disabled.

## Vercel
Import this repository into Vercel. `vercel.json` routes `/api/*` to the FastAPI serverless function and serves the dashboard from `/public`.

## Endpoints
- `/` dashboard
- `/health` health check
- `/api/status`
- `/api/simulate`
- `/api/backtest`
- `/api/reset`

## Architecture boundary
Vercel hosts the dashboard/API. The future always-on market-data, ML, risk and broker-execution engine should run separately; broker credentials must remain server-side secrets.
