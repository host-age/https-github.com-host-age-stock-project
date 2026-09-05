# Deploying the NSE Grandmaster paper trader to Render

Real NSE prices in, **paper money out** — no real order is ever placed. This
runs the trader always-on in the cloud so your laptop only ever opens a browser
to watch it.

## What you need first

1. **A Zerodha Kite Connect subscription** (~₹500/month) — this is the only
   source of live NSE data. Sign up at <https://developers.kite.trade>, create
   an app, and note your **API key** and **API secret**.
2. **A GitHub account** (free) — Render deploys from a Git repo.
3. **A Render account** (free) — <https://render.com>.

## One-time setup

### 1. Put this code on GitHub
Create a new repo and push this project to it. (I can do this step for you if
you connect GitHub.)

### 2. Create the service on Render
- In Render, click **New +** → **Blueprint**.
- Point it at your GitHub repo. Render reads `render.yaml` and creates the
  `nse-paper-trader` web service automatically.
- When it asks, set these **environment variables** (they are marked secret in
  the blueprint, so Render prompts you):
  - `KITE_API_KEY` — your Kite api key
  - `KITE_ACCESS_TOKEN` — leave blank for now (set daily, step 4)
  - `KITE_USER_ID` — your Kite user id (optional)
  - `ADMIN_SECRET` — Render generates this automatically; copy it, you'll need
    it to refresh the token.
- Click **Apply**. Render builds and starts the service. It will show
  **"waiting for credentials"** until you add today's token.

### 3. Open the dashboard
Render gives the service a URL like `https://nse-paper-trader.onrender.com`.
Open it in any browser or on your phone. It shows equity, day P&L, open
positions, and live decisions, refreshing every 2 seconds.

## Every trading day (the one daily step)

Kite access tokens **expire every morning (~06:00 IST)**, so once a day you
generate a fresh one and hand it to the service — no redeploy needed.

**Easiest:** run the login helper on any machine with Python + `kiteconnect`:

```bash
python3 tools/kite_login.py login --api-key YOUR_KEY --api-secret YOUR_SECRET
# open the printed URL, log in, paste the request_token; it prints access_token
```

Then send that token to the running service:

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/set-token \
  -H "Content-Type: application/json" \
  -d '{"secret":"YOUR_ADMIN_SECRET","access_token":"TODAYS_TOKEN"}'
```

The trader picks up the token and starts automatically when the market opens
(09:15 IST). Outside 09:15–15:30 IST it sits idle — there are no ticks to trade.

## What runs where

| Job | Where | Why |
|---|---|---|
| Live paper trading + dashboard | **Render** (this service) | light, always-on, open network |
| Watching the dashboard | your laptop/phone browser | zero compute |
| Model training, backtesting, tuning | not here | heavy; run in a bigger box or the dev sandbox |

## Safety

- The order path to a real Kite account (`gmq/execution/kite.py`) is
  intentionally **locked and unimplemented**. This service fills against live
  quotes with the paper broker only. It cannot place a real trade.
- The hard risk limits (max drawdown, daily loss, position size) are enforced
  by the independent risk engine and the self-evolution controller cannot relax
  them.

## Cost

- Render `starter` plan: a few dollars/month (or free tier, which sleeps when
  idle — fine to start, but it may miss the open; upgrade once you're serious).
- Kite Connect: ~₹500/month. Historical data (for retraining on real history)
  is a separate ~₹2,000/month add-on and is **not** needed just to paper-trade
  live.
