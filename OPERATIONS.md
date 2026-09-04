# The Operations Desk (scheduled tasks around the trader)

This is the second half of the system. The **trader** (the Render app) does the
trading. The **desk** is a small set of scheduled tasks that watch over it and
keep you informed — so you don't have to babysit the dashboard.

**The one rule:** the desk is *oversight and convenience only*. Everything
safety-critical and real-time — decisions, stop-losses, risk limits, halts —
lives **inside the trader** and stays there. The desk runs a few times a day; it
can *tell* you something is wrong, it can never be the thing that protects the
money in real time.

The desk reads the trader through **`/api/summary`** — a compact, secret-free
endpoint (no token, no admin secret, nothing that can place a trade). Safe to
fetch from an automated session.

---

## The tasks

| # | Task | When (IST) | Cron (UTC) | What it does |
|---|---|---|---|---|
| 1 | **Kite token reminder** | 08:45 Mon–Fri | `15 3 * * 1-5` | ✅ **Already active.** Reminds you to refresh the daily token. |
| 2 | **Market-open health check** | 09:20 Mon–Fri | `50 3 * * 1-5` | Fetches `/api/summary`; confirms the agent is live and warmed up. Alerts if it's down or the token wasn't refreshed. |
| 3 | **End-of-day report** | 15:35 Mon–Fri | `5 10 * * 1-5` | Fetches `/api/summary`; sends a clean P&L + positions + risk summary for the day. |
| 4 | **Weekly review** | 10:00 Sat | `30 4 * * 6` | Deeper read: the week's results, what worked, what to watch. |

Cron is UTC; IST is UTC+5:30. (e.g. 15:35 IST − 5:30 = 10:05 UTC → `5 10`.)

---

## Ready-to-activate prompts

These are the exact task prompts. They activate the moment your Render URL
exists — replace `YOUR-SERVICE` with your real subdomain in each.

### 2 — Market-open health check
> It's just after the NSE open. Fetch `https://YOUR-SERVICE.onrender.com/api/summary`.
> If `state` is `running` and `live` is true, send TARUN a one-line "✅ agent is
> live and trading" with the current equity. If `state` is `waiting_for_credentials`,
> tell him the token wasn't refreshed — the agent won't trade until he sends today's
> token. If the fetch fails, tell him the service looks down and to check Render.
> Keep it to 1–3 lines.

### 3 — End-of-day report
> The NSE trading day has closed. Fetch `https://YOUR-SERVICE.onrender.com/api/summary`
> and send TARUN a short end-of-day report: day P&L, return %, number of decisions
> and trades, any stop-outs, whether the risk engine halted (and why), and the open
> positions with their P&L and R. Lead with the P&L. Keep it tight and readable — a
> few lines plus a small positions list. If the fetch fails, say the dashboard was
> unreachable at close.

### 4 — Weekly review
> It's the weekend. Fetch `https://YOUR-SERVICE.onrender.com/api/summary` for the
> latest state, and give TARUN a short weekly review of his NSE paper-trading agent:
> how the week went at a high level, anything notable (halts, big winners/losers),
> and one concrete thing to watch or consider next week. Encouraging and honest —
> if it's losing money, say so plainly. A short paragraph.

---

## Activating them

Once the trader is deployed and you give me the Render URL, I create tasks 2–4
with the prompts above (real URL filled in), and update task 1 to include your
real API key / service / admin secret so the reminder is copy-paste ready.

Until then, only task 1 runs (with placeholders).

## Cost & noise

Four tasks — three weekday, one weekly — is a handful of short sessions a day.
Cheap, and few enough that each notification still means something. That's the
sweet spot; resist adding more unless a real need shows up.
