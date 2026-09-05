# Self-Evolution Charter (the ABULIDN loop)

This document is the operating charter for how the trading agent is allowed to
improve itself. It is not aspirational: every rule below is enforced in code
under `gmq/evolve/`, and the mapping from rule to implementation is given so
the two cannot drift apart silently.

The one-line statement of the whole thing:

> **A change to the system is applied only when paired, out-of-sample evidence
> says it beat what it replaced — and the hard risk limits are outside the
> controller's reach entirely.**

The loop is the same discipline the rest of the codebase applies to a *trade*,
turned on the *system itself*: measure honestly, distrust anything that looks
good on one draw, and keep a record you cannot quietly rewrite.

---

## The loop

```
OBSERVE → DIAGNOSE → PROPOSE → EXPERIMENT → VERIFY → PROMOTE → MONITOR → ROLLBACK → REPEAT
```

| Stage | What it means here | Where it lives |
|---|---|---|
| OBSERVE / DIAGNOSE | run reports, excursion stats, loss-cause tags | `analytics/metrics.py`, `analytics/excursion.py` |
| PROPOSE | a change is written down as a `Proposal`, never applied directly | `evolve/proposal.py` |
| boundary check | the proposal is refused if it would loosen a risk control | `evolve/boundaries.py` |
| EXPERIMENT | paired-seed A/B: same markets, config is the only difference | `evolve/controller.py`, `evolve/trader.py` |
| VERIFY | statistical gate — significance, effect size, minimum sample | `evolve/stats.py` |
| PROMOTE / REJECT | only a VALIDATED change goes live; its rollback is recorded | `evolve/controller.py` |
| MONITOR / ROLLBACK | re-measure on fresh seeds; auto-revert on regression | `evolve/controller.py` |
| journal | every transition, appended, never edited | `evolve/proposal.py` (`ChangeJournal`) |

---

## The rules, and where each is enforced

**§4 — Structured proposals.** Every candidate change is a `Proposal` with a
problem, a root cause, the concrete knob changes, the expected benefit, the
risk, the metrics, and the rollback condition. → `proposal.Proposal`.

**§5 — Never modify production directly.** A proposal moves through
`PROPOSED → EXPERIMENT → VALIDATED → PROMOTED` (or `REJECTED` /
`INCONCLUSIVE`), and the states cannot be skipped — illegal transitions raise.
→ `proposal._ALLOWED`, `controller._move`.

**§6–7 — Experiment, with statistical discipline.** The baseline arm and the
candidate arm see the **identical** simulated markets (paired on seed); only
the config differs. Fewer than the minimum number of paired samples returns
`INCONCLUSIVE`, not a verdict — thin evidence is never promoted. → `stats.significance_gate`, `controller.experiment`.

**§8 — Shadow mode.** The paired A/B *is* shadow evaluation: the candidate is
scored on the same draws as the incumbent without affecting anything live.
→ `controller.experiment`. *(Graduated live rollout — §9 canary — is the one
stage deferred until a live broker is connected; there is nothing live to
canary against in the simulator.)*

**§10 — Rollback.** Every promoted change records its inverse patch. `MONITOR`
re-measures it on a **fresh** set of seeds and reverts automatically if it
regresses — a curve fit that survived the fit seeds is caught on new ones.
→ `controller.check_rollback`, `controller.rollback`.

**§11 — Self-modification boundaries (the immutable line).** This is the rule
that makes autonomy safe rather than reckless. The controller **may** tune the
strategy — search depth, objective weights, exit rules, model routing. It
**may not** loosen a hard risk limit: max drawdown, max daily loss, max
position size, leverage, the consecutive-loss halt, min model skill. A
proposal to relax any of these is refused *in code, before it is ever
measured* — because a looser risk cap almost always looks better in a
backtest, right up until the draw it was protecting against arrives. The
controller may only ever make these **stricter**. → `boundaries.KnobRegistry`
(guarded knobs, `stricter_sign`).

This mirrors the existing design rule that the model layer has no handle on the
risk engine. The *evolution* layer does not either.

**§12 — Code self-improvement.** Changes ship behind tests: the controller has
its own suite (`tests/test_evolve.py`) covering the boundary guard, the state
machine, the statistical gate, and full promote/rollback cycles.

**§17 — Change journal.** Every transition is appended to a journal that is
never rewritten. A journal you can edit is one that can be made to agree with
whatever the system currently believes — the exact failure the loop exists to
prevent. → `proposal.ChangeJournal` (append-only; failure history is kept even
when inconvenient, §14).

**§20 — The final rule.** The system may not claim it improved itself. It may
only report evidence: *"E-101 raised the objective from X to Y across N paired
seeds at p=Z; on fresh seeds it regressed and was rolled back."* → the journal
and `SigResult` produce exactly that sentence.

---

## What is tunable vs guarded

Defined in `evolve/boundaries.py`. Summary:

**Tunable** (the controller may set any value in range, and test it):
search depth / scenarios / beam width, discount, risk-aversion, drawdown and
CVaR penalties, min edge, min confidence, exit hysteresis, re-entry cooldown,
give-back activation and keep-fraction, objective weights, model learning rate
and ensemble half-life.

**Guarded** (the controller may only *tighten*, never loosen):
`max_risk_per_trade_pct`, `max_position_pct`, `max_daily_loss_pct`,
`max_daily_trades`, `max_drawdown_pct`, `max_gross_exposure_pct`,
`max_leverage`, `max_open_positions`, `consecutive_loss_halt`,
`min_model_brier_score`.

Changing a guarded limit at all requires an operator editing config. The
controller cannot do it, cannot A/B it, and cannot promote it.

---

## Running it

```
python3 tools/evolve_demo.py [days] [n_seeds] [n_symbols]
```

Runs the full cycle on the live engine: refuses a boundary-crossing proposal,
then takes a real strategy proposal through experiment → verify → promote →
monitor → (rollback if it regresses). The journal is written to
`runs/evolve/journal.jsonl`.

### Demonstrated run

On the first live run (`exit_hysteresis 1.0 → 1.5`, 8 paired seeds):

- the loosening proposal (`max_drawdown_pct 8 → 12`) was **refused before any
  compute** — the immutable boundary held;
- the strategy change **validated in-sample at p=0.000**, was promoted…
- …and was then **rolled back automatically** when it regressed on 8 fresh
  seeds.

That rollback is the charter working as intended: a change that looked
significant on its fit seeds did not survive new ones, and the monitor caught
and reverted it. It is also a real result — raising the hysteresis bar is a
curve fit, so the shipped default (`exit_hysteresis = 1.0`) stands.
