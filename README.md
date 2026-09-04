# Grandmaster Engine

A real-time, data-driven trading system for NSE equities, built around a
chess-engine-style decision architecture: it generates candidate moves,
simulates the market's replies, evaluates the resulting positions, and plays
the move with the best risk-adjusted expected value — then keeps re-evaluating
while the position is open.

It ships with a **simulated NSE** to run against, because the alternative was
shipping an untested live-order path.

---

## Read this first

This system trades a **simulated market**. That simulation is good — prices
emerge from a real price-time-priority matching engine driven by market
makers, informed traders and noise traders, and it reproduces realistic
spreads (~1–2bp), daily volatility (1.5–2.2%) and cross-sectional correlation
(~0.48 at one-minute sampling). But it is still a model.

**A simulator validates the code, not the edge.** Everything measured here
demonstrates that the machinery works and is internally honest. None of it is
evidence that the strategy would make money in a real market, and the live
broker path is deliberately left unimplemented behind a three-part lock
(`gmq/execution/kite.py`).

Sections 12 (latency) and 11 (execution) of the specification are implemented
against the simulator's timings. Those constants would all need re-measuring
against a real venue before they mean anything.

---

## Architecture

The eleven blocks from the specification, each a real module:

```
Market Data ─┬─ SimExchange        real matching engine, emergent prices
             ├─ MarketDataEngine   L2 book, 7 timeframes, breadth, sectors
             └─ ReplayFeed         historical OHLCV with Brownian-bridge paths
        ↓
Features ────┬─ technical.py       ~30 normalised indicators × 7 timeframes
             ├─ microstructure.py  OFI, VPIN, Kyle's λ, queue, sweep cost
             ├─ crosssectional.py  live correlation matrix, beta, breadth
             └─ derivatives.py     Black-Scholes, IV surface, PCR, max pain, GEX
        ↓
Regime ─────── detector.py         filtered HMM over 8 regimes, interpretable
        ↓
Models ──────┬─ base.py            triple-barrier labelling (leak-safe)
             ├─ online.py          AdaGrad logistic / ridge / quantile
             ├─ gbdt.py            periodically-refit trees, purged + embargoed
             └─ ensemble.py        isotonic calibration, skill-weighted blend
        ↓
Strategy ────┬─ moves.py           the 9 legal actions, parameterised
             ├─ scenarios.py       block-bootstrapped, antithetic paths
             ├─ evaluator.py       Indian cost model + objective function
             └─ search.py          expectimax, beam, iterative deepening
        ↓
Risk ────────┬─ engine.py          THE HARD LAYER — no override exists
             ├─ sizing.py          risk budget / vol target / Kelly / liquidity
             ├─ stops.py           dynamic placement + monotonic ratchet
             └─ portfolio.py       correlation clusters, VaR/ES, exposures
        ↓
Execution ───┬─ simbroker.py       latency, rejects, partials, stop triggers
             ├─ router.py          slicing, retries, reconciliation
             └─ kite.py            live adapter, locked
        ↓
Monitor ─────┬─ trade_monitor.py   7-way adverse-move diagnosis
             └─ journal.py         append-only JSONL + queryable SQLite
        ↓
Analytics ─── metrics.py           Sharpe/Sortino/Calmar, MAE-MFE, per-regime
Backtest ──── backtest/engine.py   walk-forward, Monte Carlo, stress, overfit
```

---

## What was measured

All figures from the simulated market, reported honestly.

### The market itself

| Property | Measured | Real NSE large-caps |
|---|---|---|
| Bid-ask spread | 0.9 – 2.2 bp | ~1 – 3 bp |
| Daily volatility | 1.5 – 2.2% | ~1.2 – 2% |
| 1-min cross-correlation | 0.48 | ~0.4 – 0.6 |
| Tick-level noise | 1 – 4.5 bp | comparable |

### Model forecast skill

Measured **out-of-sample on non-overlapping windows** — each sample scored
before the model trains on it, and samples spaced a full horizon apart so
label windows never overlap:

| Horizon | n | Brier | Brier skill | Accuracy |
|---|---|---|---|---|
| 60s | 6496 | 0.226 | **+0.096** | 62.3% |
| 300s | 1277 | 0.193 | **+0.226** | 71.0% |
| 900s | 401 | 0.201 | **+0.191** | 73.8% |
| 3600s | 100 | 0.212 | **+0.129** | 72.0% |

Positive Brier skill means the model beats predicting the base rate. The
overlapping-window figure is roughly twice as flattering and is reported
separately, labelled as inflated, because it is.

### Regime detection

35 – 46% accurate against ground truth versus 12.5% for chance, with 79–84%
precision on its strongest class. Honestly: this is the weakest component.
Regime is used as a feature and a gate, and its confidence is exposed so
downstream logic can discount it.

### Latency

Search: 7.7ms mean / 22.8ms p99 in the full engine against a 25ms budget.

### The strategy itself

A clean 3-session run, 8 symbols, ₹10,00,000, no halts:

| | |
|---|---|
| Trades | 185 (41.6% won) |
| Return | **−0.045%** after all costs |
| Expectancy | −₹2.44 / trade, −0.001R |
| Profit factor | 0.80 |
| Max drawdown | 0.10% |
| Position mismatches | 1 |
| Stops placed / triggered | 763 / 8 |

**It loses a small amount after costs.** That is the honest read: the machinery
works end to end and is internally consistent — all seven leakage checks pass,
every R-multiple is measurable, reconciliation is clean — but the search is not
finding an edge that survives Indian transaction costs at this trade frequency.

Two things stand out as the places to look, and neither is a bug:

- **Median holding period is 15 seconds.** The engine is scalping, not running
  the multi-timeframe theses the architecture is built for. The decision
  cadence and the exit side of the objective are letting it churn.
- **61% of losing trades were meaningfully in profit first** (`excursions.
  losers_that_were_winners`). The exit policy gives back winners. That is a
  target/trail calibration problem, and the MAE/MFE block exists precisely to
  make it visible.

The walk-forward, Monte Carlo and stress tooling is there to test whether any
change to those actually helps or is just another curve fit.

---

## Bugs this build found and fixed

Kept here because each one is a failure mode that is *silent* — the system
keeps running and looks fine.

1. **The objective never traded.** `EV − 1.25·σ` demands a per-trade Sharpe
   above 1.25 before any trade clears. Variance needs the Arrow-Pratt form
   (`γ·Var/2·equity`), not a raw standard deviation.
2. **Costs charged twice** — once into the simulated position's fees, again in
   the objective — halving every entry's score.
3. **Confidence multiplied in the edge magnitude**, so a well-evidenced small
   edge scored as "unconfident" and every entry gate failed.
4. **Equity double-counted the cash** paid for a position, understating equity
   by the whole book and firing every percentage risk limit far too early.
5. **A search timeout discarded better shallow results**, so the answer quietly
   got worse under load — exactly when it matters most.
6. **The regime filter self-trapped**: without a prior floor, one state reached
   0.999 and no contrary evidence could ever climb back.
7. **Volatility measured over a 65-minute window** while regimes last ~25,
   averaging across two or three different regimes.
8. **The order-rate circuit breaker was a flat 30/min** regardless of universe
   size, counted child slices and stop-management traffic rather than
   decisions, and used a one-minute window that cannot tell a correlated
   stop-out burst from a runaway loop. It halted the engine for working.
9. **Stops were computed but never placed.** Every position was sized against
   its stop distance and the risk engine was told about it — and no order
   existed. A stop that lives only in a variable is not a stop. There is now a
   resting SL-M order at the broker *and* a local per-tick check, because a
   broker-held stop can be rejected or lost on a reconnect.
10. **`initial_risk` was recomputed on every fill.** After two partial exits it
    reflected the residual share count while realised P&L still reflected the
    whole trade, so R-multiples exploded — one run reported **+₹1.67
    expectancy and −0.72R simultaneously**. 1R is now fixed at entry.
11. **The multi-day clock never advanced to the next session.** It ran past
    15:30 into the night, so the move generator correctly refused to open new
    intraday risk "minutes before the close" — forever. A three-day run was
    silently a one-day run.
12. **The consecutive-loss halt measured nothing.** At a 50% win rate a run of
    five appears somewhere in sixty trades ~83% of the time, so it fired daily
    on chance alone. The threshold now scales with the observed win rate and
    the day's trade count.
13. **Cancelled orders were decremented from the expected position twice**,
    once in `cancel()` and again in the terminal-order callback. With a stop
    resting on every position — cancelled and replaced on every ratchet — that
    produced 211 phantom position mismatches in one run. A reconciliation
    alarm nobody believes is not an alarm.
14. **Fill rate counted untriggered stops as misses**, dragging the metric to
    ~52% and hiding whether there was a genuine execution problem underneath.
    A protective stop that never fires is a success.

15. **`Order` is a dataclass with a generated `__eq__`**, so partitioning
    order history with `o not in working` compared by *value*: twenty
    identical stop orders tested equal and collapsed to one. Combined with a
    triggered stop being rewritten to `MARKET` by the broker, the report said
    "0 stops triggered" for a run in which every stop worked.

Several of these only became visible *because* an earlier fix exposed them —
the R-multiple explosion was hidden behind a sizing bug, and the reconciliation
double-count could not appear until stops were actually being placed.

---

## The rules that have no override

```python
# gmq/risk/engine.py — holds no reference to the model or strategy layer,
# and exposes no method that relaxes a limit.
```

* An entry **without a stop** is refused outright — unbounded loss has no
  acceptable size.
* A stop **only ever moves toward price.** There is no code path that widens
  one; `StopPolicy.update` asserts monotonicity before returning.
* A halt **never blocks an exit** — a control that traps the book in a position
  it cannot leave has become the risk.
* Correlated same-direction positions count as **one** exposure.
* The risk engine halts on its own for daily loss, drawdown, consecutive
  losses, loss velocity, model degradation, order-rate anomaly, reject rate,
  slippage and VaR breach — without needing anything upstream to agree.

`tests/test_risk.py::test_risk_engine_has_no_override_api` fails if anyone
adds `force=`, `override()` or a settable-limit path the strategy can reach.

---

## Running it

```bash
pip install numpy pandas scikit-learn pytest

python -m gmq run --days 3 --dashboard     # a session + HTML report
python -m gmq backtest --days 2 --seeds 5  # multi-seed + deflated Sharpe
python -m gmq walkforward --folds 4        # rolling out-of-sample
python -m gmq stress                       # crash, illiquidity, gappy news
python -m gmq verify                       # 58 tests
```

Useful knobs live in `gmq/core/config.py`. Set `search.node_budget > 0` for
bit-reproducible research runs — a wall-clock budget makes two folds
incomparable because a faster machine explores more of the tree.

---

## Going live

Don't, yet. If you do:

1. `GMQ_ALLOW_LIVE_TRADING=1`, `GMQ_LIVE_CONFIRM=I-UNDERSTAND-THIS-TRADES-REAL-MONEY`,
   and `acknowledged=True` at the call site. All three, none of them defaults.
2. Implement the three marked methods in `gmq/execution/kite.py`.
3. **Re-measure everything.** Latency, fill probability, impact and slippage
   are calibrated to this simulator and none of those numbers transfer.
4. Run in observation mode against live data first, comparing predicted to
   realised fills, until the cost model is grounded in reality.
5. Check SEBI and exchange rules on automated order flow from retail accounts
   with your broker. That question is not answered by this code compiling.

Start with capital you are fully prepared to lose, keep the hard limits tight,
and watch it. A system's first live session is an experiment, not a deployment.

---

*This is engineering work, not financial advice. Autonomous trading systems
lose money quickly and comprehensively when their assumptions break.*
