"""Does the model layer actually learn anything?

Runs the full data -> feature -> label -> model pipeline against the simulated
exchange and reports out-of-sample forecast quality. The numbers that matter:

  Brier skill  > 0   the model beats predicting the base rate. This is the
                     only headline that cannot be gamed by a lopsided market.
  calibration        does "0.6" mean 60%? If not, every expected value the
                     search engine computes downstream is wrong.

Skill is measured on each sample BEFORE the model trains on it, so every
number here is genuinely out-of-sample.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, ".")
from gmq.core.bus import EventBus, Topic
from gmq.core.clock import SimClock, IST
from gmq.core.config import Config
from gmq.data.feed import SimFeed, MarketDataEngine
from gmq.features.registry import FeatureEngine
from gmq.regime.detector import RegimeDetector
from gmq.models.bank import ModelBank


def run(days: float = 3.0, dt: float = 0.5, seed: int = 5, symbols=None,
        use_gbdt: bool = True, verbose: bool = True):
    cfg = Config()
    syms = symbols or cfg.data.symbols[:8]
    bus = EventBus(swallow_errors=False)
    clock = SimClock(datetime(2026, 8, 17, 9, 15, tzinfo=IST))
    mde = MarketDataEngine(bus, clock, syms)
    fe = FeatureEngine(syms)
    rd = RegimeDetector()
    bank = ModelBank(cfg.models, dim=200, use_gbdt=use_gbdt)
    bus.subscribe(Topic.TICK, fe.on_tick, priority=15)
    bus.subscribe(Topic.DEPTH, fe.on_depth, priority=15)
    feed = SimFeed(bus, clock, syms, seed=seed)

    # resolve pending labels on every tick -- this is what trains the models
    def on_tick(t):
        bank.on_price(t.symbol, t.ts, t.ltp)
    bus.subscribe(Topic.TICK, on_tick, priority=35, name="bank.price")

    steps_per_day = int(22500 / dt)
    total = int(steps_per_day * days)
    per_min = max(1, int(60 / dt))
    t0 = time.time()
    observed = 0
    for i in range(total):
        feed.step(dt)
        if i % per_min == per_min - 1:
            prices = {s: mde.ltp(s) for s in syms}
            fe.on_minute(prices)
            for s in syms:
                fv = fe.build(s, clock.now_ns(), mde)
                if not fv.ready:
                    continue
                if not bank.feature_names:
                    bank.feature_names = fe.feature_names()
                st = rd.update(s, fv, mde)
                bank.observe(s, clock.now_ns(), fv,
                             st.current.value if st.initialised else "")
                observed += 1
        if verbose and i % (steps_per_day // 2) == 0 and i:
            el = time.time() - t0
            print(f"  ... {i/steps_per_day:.1f}d  obs={observed} "
                  f"labels={bank.samples_seen}  {el:.0f}s")

    if verbose:
        print(f"\nwall {time.time()-t0:.0f}s | observations {observed} | "
              f"resolved labels {bank.samples_seen}")
        print(f"features {len(bank.feature_names)}")
        print("\nDIRECTION MODELS (out-of-sample, scored before training)")
        print(f"{'horizon':>8s} {'n':>7s} {'brier':>7s} {'skill':>7s} "
              f"{'acc':>7s} {'logloss':>8s}   members")
        for h in bank.horizons:
            m = bank.models[h].direction
            t = m.skill
            mem = " ".join(f"{k.split('_')[0]}:{v:.2f}"
                           for k, v in sorted(m.last_weights.items()))
            print(f"{int(h):>8d} {t.n_total:>7d} {t.brier():>7.4f} "
                  f"{t.skill_score():>7.4f} {t.accuracy():>7.4f} "
                  f"{t.log_loss():>8.4f}   {mem}")
        print("\nHONEST (NON-OVERLAPPING) OUT-OF-SAMPLE SKILL")
        print("  one sample per horizon-length window, so label windows")
        print("  never overlap -- this is the number that is not inflated")
        print(f"  {'horizon':>8s} {'n':>6s} {'brier':>7s} {'skill':>8s} {'acc':>7s}")
        for h in bank.horizons:
            t_ = bank.honest[h]
            print(f"  {int(h):>8d} {t_.n_total:>6d} {t_.brier():>7.4f} "
                  f"{t_.skill_score():>+8.4f} {t_.accuracy():>7.4f}")

        print("\nPER-MEMBER SKILL (overlapping windows -- inflated)")
        for h in bank.horizons:
            ens = bank.models[h].direction
            for name, tr in ens.trackers.items():
                if tr.ready:
                    print(f"  h={int(h):<5d} {name:<18s} n={tr.n_total:<6d} "
                          f"brier={tr.brier():.4f} skill={tr.skill_score():+.4f} "
                          f"acc={tr.accuracy():.3f}")
        print("\nBARRIER MODELS")
        for h in bank.horizons:
            hm = bank.models[h]
            print(f"  h={int(h):<5d} p_target n={hm.p_target.n:<6d} "
                  f"skill={hm.p_target.skill.skill_score():+.4f} | "
                  f"p_stop n={hm.p_stop.n:<6d} "
                  f"skill={hm.p_stop.skill.skill_score():+.4f}")
        print("\nCALIBRATION of the 900s ensemble (pred vs observed)")
        cal = bank.models[bank.horizons[-1]].direction.cal
        for row in (cal.reliability() if cal else []):
            bar = "#" * int(row["obs"] * 40)
            print(f"  pred={row['pred']:.2f} obs={row['obs']:.2f} "
                  f"n={row['n']:<5d} {bar}")
        print("\nTOP FEATURES (900s logistic, |weight|)")
        h = bank.horizons[-1]
        lg = bank.models[h].direction.members[0]
        for nm, w in lg.top_features(bank.feature_names, 15):
            print(f"  {nm:<28s} {w:+.4f}")
    return bank


if __name__ == "__main__":
    d = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    run(days=d)
