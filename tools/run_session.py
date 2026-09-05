"""Run a full multi-day session, save everything, and build the dashboard."""
import json
import os
import pickle
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
from gmq.core.config import Config
from gmq.core.clock import IST
from gmq.app.engine import TradingEngine
from gmq.app.dashboard import build_dashboard
from gmq.backtest.engine import leakage_checks


def main(days=3.0, seed=5, run_id="session", dt=0.5, nsym=8):
    cfg = Config(run_id=run_id, seed=seed)
    cfg.data.symbols = cfg.data.symbols[:nsym]
    run_dir = f"runs/{run_id}"
    eng = TradingEngine(cfg, run_dir=run_dir,
                        start=datetime(2026, 8, 17, 9, 15, tzinfo=IST))

    def prog(d, snap):
        p = snap["portfolio"]
        print(f"  d{d:5.2f} eq={p['equity']:>11,.0f} ret={p['return_pct']:+6.2f}% "
              f"dd={p['drawdown_pct']:5.2f}% pos={p['open_positions']} "
              f"trades={p['trades']:3d} act={snap['actions']:4d} "
              f"lbl={snap['labels']:6d} halt={snap['risk']['halt_reason'] or '-'}",
              flush=True)

    t0 = time.time()
    rep = eng.run(days=days, dt=dt, progress=prog)
    rep["wall_s"] = round(time.time() - t0, 1)

    # everything the dashboard and later analysis need
    equity = list(eng.portfolio.equity_curve)
    trades = list(eng.portfolio.trades)
    checks = leakage_checks(eng)
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/report.json", "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    with open(f"{run_dir}/trades.pkl", "wb") as fh:
        pickle.dump(trades, fh)
    # downsample the equity curve for the chart; keep the full one on disk
    step = max(1, len(equity) // 900)
    series = [v for _ts, v in equity[::step]]
    with open(f"{run_dir}/equity.json", "w") as fh:
        json.dump({"series": series, "full_n": len(equity)}, fh)

    dash = build_dashboard(
        rep, f"{run_dir}/dashboard.html",
        extras={
            "equity_series": series,
            "checks": checks,
            "fees": eng.portfolio.total_fees,
            "journal_path": f"{run_dir}/{run_id}_decisions.jsonl",
            "title": "Grandmaster Engine Report",
        })

    print("\n" + "=" * 78)
    print(f"RUN {run_id}  |  {days} days  |  {rep['wall_s']}s wall")
    print("=" * 78)
    print("\nTRADES");  print(json.dumps(rep["trades"], indent=2))
    print("\nEQUITY");  print(json.dumps(rep["equity"], indent=2))
    print("\nEXCURSIONS"); print(json.dumps(rep["excursions"], indent=2))
    print(f"\ndecisions={rep['decisions']} actions={rep['actions']} "
          f"vetoed={rep['vetoed']} action_rate={rep['action_rate']}")
    print("execution:", json.dumps(rep["execution"]))
    print("latency:  ", json.dumps(rep["latency"]))
    print("halts:    ", json.dumps(rep["halts"], default=str))
    print("loss causes:", json.dumps(rep["loss_causes"]))
    print("\nby regime:"); print(json.dumps(rep["by_regime"], indent=1)[:1200])
    print("\nhonest model skill:")
    print(json.dumps(rep["models"].get("honest_skill", {}), indent=1))
    print("\ndecision calibration:")
    print(json.dumps(rep.get("decision_calibration", []), indent=1))
    print("\nleakage checks:")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
    print(f"\ndashboard -> {dash}")
    eng.close()
    return rep


if __name__ == "__main__":
    main(days=float(sys.argv[1]) if len(sys.argv) > 1 else 3.0,
         run_id=sys.argv[2] if len(sys.argv) > 2 else "session")
