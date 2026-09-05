"""Command line interface.

    python -m gmq run        --days 3 --symbols RELIANCE,TCS,...
    python -m gmq backtest   --days 3 --seeds 5
    python -m gmq walkforward --folds 4
    python -m gmq montecarlo --run runs/session
    python -m gmq stress
    python -m gmq dashboard  --run runs/session
    python -m gmq verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from .core.config import Config
from .core.clock import IST


def _cfg(args) -> Config:
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()
    if getattr(args, "symbols", None):
        cfg.data.symbols = [s.strip().upper() for s in args.symbols.split(",")]
    if getattr(args, "capital", None):
        cfg.initial_capital = float(args.capital)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "run_id", None):
        cfg.run_id = args.run_id
    return cfg


def cmd_run(args) -> int:
    from .app.engine import TradingEngine
    cfg = _cfg(args)
    run_dir = os.path.join("runs", cfg.run_id)
    eng = TradingEngine(cfg, run_dir=run_dir,
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST))

    def prog(d, snap):
        p = snap["portfolio"]
        print(f"  day {d:5.2f}  equity {p['equity']:>12,.0f}  "
              f"ret {p['return_pct']:+6.2f}%  dd {p['drawdown_pct']:5.2f}%  "
              f"pos {p['open_positions']}  trades {p['trades']:>3d}  "
              f"halt {snap['risk']['halt_reason'] or '-'}", flush=True)

    rep = eng.run(days=args.days, dt=args.dt, progress=prog)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    _print_report(rep)
    if args.dashboard:
        from .app.dashboard import build_dashboard
        path = build_dashboard(rep, os.path.join(run_dir, "dashboard.html"),
                               journal=eng.journal)
        print(f"\ndashboard -> {path}")
    eng.close()
    return 0


def cmd_backtest(args) -> int:
    from .backtest.engine import run_backtest, overfitting_report, monte_carlo
    cfg = _cfg(args)
    results = []
    for i in range(args.seeds):
        seed = cfg.seed + i * 13
        print(f"[{i+1}/{args.seeds}] seed={seed} ...", flush=True)
        r = run_backtest(cfg, days=args.days, seed=seed, dt=args.dt,
                         label=f"seed{seed}", run_dir=f"runs/bt/s{seed}")
        results.append(r)
        print("   ", json.dumps(r.summary()), flush=True)
    print("\n--- across seeds ---")
    for r in results:
        print(json.dumps(r.summary()))
    print("\n--- overfitting check ---")
    print(json.dumps(overfitting_report(results), indent=2))
    all_trades = [t for r in results for t in r.trades]
    if len(all_trades) >= 10:
        print("\n--- monte carlo (pooled trades) ---")
        print(json.dumps(monte_carlo(all_trades,
                                     initial_capital=cfg.initial_capital),
                         indent=2))
    return 0


def cmd_walkforward(args) -> int:
    from .backtest.engine import walk_forward
    cfg = _cfg(args)
    out = walk_forward(cfg, n_folds=args.folds, train_days=args.train_days,
                       test_days=args.test_days, dt=args.dt)
    print(json.dumps(out, indent=2))
    return 0


def cmd_stress(args) -> int:
    from .backtest.engine import stress_test
    cfg = _cfg(args)
    print(json.dumps(stress_test(cfg, days=args.days, dt=args.dt), indent=2))
    return 0


def cmd_montecarlo(args) -> int:
    from .backtest.engine import monte_carlo
    import pickle
    path = os.path.join(args.run, "trades.pkl")
    if not os.path.exists(path):
        print(f"no trades at {path}; run `gmq run` first", file=sys.stderr)
        return 1
    with open(path, "rb") as fh:
        trades = pickle.load(fh)
    print(json.dumps(monte_carlo(trades), indent=2))
    return 0


def cmd_dashboard(args) -> int:
    from .app.dashboard import build_dashboard
    rep_path = os.path.join(args.run, "report.json")
    if not os.path.exists(rep_path):
        print(f"no report at {rep_path}", file=sys.stderr)
        return 1
    with open(rep_path) as fh:
        rep = json.load(fh)
    out = build_dashboard(rep, os.path.join(args.run, "dashboard.html"))
    print(out)
    return 0


def cmd_verify(args) -> int:
    import subprocess
    return subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"])


def _print_report(rep: dict) -> None:
    t, e = rep.get("trades", {}), rep.get("equity", {})
    print("\n" + "=" * 70)
    print(f"  trades {t.get('trades', 0)}   win rate {t.get('win_rate', 0):.1%}   "
          f"expectancy {t.get('expectancy_r', 0):+.3f}R   "
          f"profit factor {t.get('profit_factor', 0)}")
    print(f"  return {e.get('total_return_pct', 0):+.3f}%   "
          f"sharpe {e.get('sharpe', 0):.2f}   sortino {e.get('sortino', 0):.2f}   "
          f"max dd {e.get('max_drawdown_pct', 0):.2f}%")
    lat = rep.get("latency", {})
    print(f"  search {lat.get('mean_search_ms', 0):.1f}ms mean / "
          f"{lat.get('p99_search_ms', 0):.1f}ms p99   "
          f"decisions {rep.get('decisions', 0)}  actions {rep.get('actions', 0)}")
    ex = rep.get("execution", {})
    print(f"  fills {ex.get('filled', 0)}  rejects {ex.get('rejected', 0)}  "
          f"slippage {ex.get('mean_slippage_bps', 0):+.2f}bps  "
          f"discrepancies {ex.get('discrepancies', 0)}")
    if rep.get("halts"):
        print(f"  halts: {[h['reason'] for h in rep['halts']]}")
    print("=" * 70)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gmq", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--symbols")
    p.add_argument("--capital", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--run-id", dest="run_id", default="session")
    p.add_argument("--dt", type=float, default=0.5,
                   help="simulation step in seconds")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a live-style session")
    r.add_argument("--days", type=float, default=1.0)
    r.add_argument("--dashboard", action="store_true")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("backtest", help="multi-seed backtest + overfit check")
    b.add_argument("--days", type=float, default=2.0)
    b.add_argument("--seeds", type=int, default=3)
    b.set_defaults(fn=cmd_backtest)

    w = sub.add_parser("walkforward", help="rolling out-of-sample evaluation")
    w.add_argument("--folds", type=int, default=3)
    w.add_argument("--train-days", dest="train_days", type=float, default=2.0)
    w.add_argument("--test-days", dest="test_days", type=float, default=1.0)
    w.set_defaults(fn=cmd_walkforward)

    s = sub.add_parser("stress", help="hostile-regime stress tests")
    s.add_argument("--days", type=float, default=1.0)
    s.set_defaults(fn=cmd_stress)

    m = sub.add_parser("montecarlo", help="resample a completed run's trades")
    m.add_argument("--run", default="runs/session")
    m.set_defaults(fn=cmd_montecarlo)

    d = sub.add_parser("dashboard", help="build the HTML dashboard")
    d.add_argument("--run", default="runs/session")
    d.set_defaults(fn=cmd_dashboard)

    v = sub.add_parser("verify", help="run the test suite")
    v.set_defaults(fn=cmd_verify)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
