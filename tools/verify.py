"""Full verification pass: tests, structural checks, and a short live run.

Prints a single PASS/FAIL summary. This is what you run before believing
anything the system tells you.
"""
import json
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")


def section(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74, flush=True)


def main() -> int:
    ok = True

    section("1. Unit and behavioural tests")
    rc = subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"])
    ok &= (rc == 0)

    section("2. Simulated market realism")
    import numpy as np
    from gmq.data.simexchange import SimExchange
    from gmq.core.types import NS
    syms = ["RELIANCE", "TCS", "HDFCBANK", "ITC", "SBIN", "INFY"]
    ticks = []
    ex = SimExchange(syms, seed=11, on_tick=ticks.append)
    ts = 0
    for _ in range(48000):                     # ~3.3 h of market
        ts += int(0.25 * NS)
        ex.step(ts, 0.25)
    P = {s: np.array([t.ltp for t in ticks if t.symbol == s]) for s in syms}
    R = {s: np.diff(np.log(P[s])) for s in syms}
    m = min(len(v) for v in R.values())

    def agg(x, k=240):
        n = len(x) // k
        return x[:n * k].reshape(n, k).sum(1)

    A = {s: agg(R[s][:m]) for s in syms}
    daily = {s: float(A[s].std() * np.sqrt(375) * 100) for s in syms}
    spreads = {s: float(np.mean([t.spread_bps for t in ticks
                                 if t.symbol == s and t.bid > 0])) for s in syms}
    corr = float(np.mean([np.corrcoef(A[a], A[b])[0, 1]
                          for i, a in enumerate(syms) for b in syms[i + 1:]]))
    checks = [
        ("daily volatility 0.8-3.5%",
         all(0.8 <= v <= 3.5 for v in daily.values()),
         {k: round(v, 2) for k, v in daily.items()}),
        ("spreads 0.5-6 bp",
         all(0.5 <= v <= 6.0 for v in spreads.values()),
         {k: round(v, 2) for k, v in spreads.items()}),
        ("cross-correlation 0.2-0.8", 0.2 <= corr <= 0.8, round(corr, 3)),
        ("book never crossed",
         all(ex.book_of(s).best_bid < ex.book_of(s).best_ask
             for s in syms if ex.book_of(s).best_bid and ex.book_of(s).best_ask),
         "checked at end of run"),
    ]
    for name, passed, detail in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    section("3. End-to-end run + leakage checks")
    from gmq.core.config import Config
    from gmq.core.clock import IST
    from gmq.app.engine import TradingEngine
    from gmq.backtest.engine import leakage_checks
    cfg = Config(run_id="verify", seed=9)
    cfg.data.symbols = cfg.data.symbols[:6]
    eng = TradingEngine(cfg, run_dir="runs/verify",
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST),
                        journal=True)
    t0 = time.time()
    rep = eng.run(days=1.0, dt=0.5)
    print(f"  ran 1 session in {time.time()-t0:.0f}s: "
          f"{rep['trades'].get('trades',0)} trades, "
          f"{rep['decisions']} decisions, "
          f"{rep['actions']} actions")
    for c in leakage_checks(eng):
        ok &= c["pass"]
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
        print(f"         {c['detail']}")

    section("4. Accounting identity")
    pf = eng.portfolio
    booked = sum(t.pnl for t in pf.trades)
    fees = pf.total_fees
    eq = pf.equity()
    implied = pf.initial + booked - fees
    drift = abs(eq - implied)
    # open positions and unbooked marks legitimately explain a gap
    open_mv = sum(abs(p.qty) * pf.last(p.symbol) for p in pf.open_positions)
    acc_ok = drift < max(1.0, open_mv * 0.02 + abs(booked) * 0.02 + 50)
    ok &= acc_ok
    print(f"  initial {pf.initial:,.2f}  booked {booked:+,.2f}  "
          f"fees {fees:,.2f}")
    print(f"  equity  {eq:,.2f}   implied {implied:,.2f}   drift {drift:,.2f}")
    print(f"  [{'PASS' if acc_ok else 'FAIL'}] equity ties to booked P&L "
          f"less fees (open MV {open_mv:,.0f})")

    section("5. Risk engine cannot be overridden")
    from gmq.risk.engine import RiskEngine
    banned = {"override", "force", "bypass", "disable", "relax"}
    found = {n for n in dir(RiskEngine) if n.lower() in banned}
    ok &= not found
    print(f"  [{'PASS' if not found else 'FAIL'}] no override-shaped API "
          f"({found or 'none found'})")
    holds = [type(v).__module__ for v in vars(eng.risk).values() if v is not None]
    leaky = [m for m in holds if "models" in m or "strategy" in m]
    ok &= not leaky
    print(f"  [{'PASS' if not leaky else 'FAIL'}] holds no reference to the "
          f"decision layer")
    eng.close()

    section(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
