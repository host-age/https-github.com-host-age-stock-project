"""Score the regime detector against the simulator's ground truth."""
import sys, time
from collections import Counter
from datetime import datetime
import numpy as np

sys.path.insert(0, ".")
from gmq.core.bus import EventBus, Topic
from gmq.core.clock import SimClock, IST
from gmq.data.feed import SimFeed, MarketDataEngine
from gmq.features.registry import FeatureEngine
from gmq.regime.detector import RegimeDetector, REGIMES


def run(steps=60000, warm=45, seed=5, syms=None, verbose=True):
    syms = syms or ["RELIANCE", "TCS", "HDFCBANK", "ITC", "SBIN", "INFY"]
    bus = EventBus(swallow_errors=False)
    clock = SimClock(datetime(2026, 8, 20, 9, 15, tzinfo=IST))
    mde = MarketDataEngine(bus, clock, syms)
    fe = FeatureEngine(syms)
    rd = RegimeDetector()
    bus.subscribe(Topic.TICK, fe.on_tick, priority=15)
    bus.subscribe(Topic.DEPTH, fe.on_depth, priority=15)
    feed = SimFeed(bus, clock, syms, seed=seed)

    n = len(REGIMES)
    cm = np.zeros((n, n), dtype=int)
    idx = {r: i for i, r in enumerate(REGIMES)}
    conf = []
    t0 = time.time()
    for i in range(steps):
        feed.step(0.25)
        if i % 240 == 239:
            fe.on_minute({s: mde.ltp(s) for s in syms})
            if i > 240 * warm:
                for s in syms:
                    fv = fe.build(s, clock.now_ns(), mde)
                    if not fv.ready:
                        continue
                    st = rd.update(s, fv, mde)
                    if not st.initialised:
                        continue
                    tr = feed.exchange.true_regime(s)
                    cm[idx[tr], idx[st.current]] += 1
                    conf.append((st.confidence, st.current == tr))
    tot = cm.sum()
    acc = np.trace(cm) / max(tot, 1)
    if verbose:
        print(f"wall {time.time()-t0:.0f}s   obs={tot}   accuracy={acc*100:.1f}%  (chance 12.5%)")
        print("\nconfusion (rows = TRUE, cols = PREDICTED)")
        hdr = "".join(f"{r.value[:6]:>8s}" for r in REGIMES)
        print(f"{'':16s}{hdr}   recall")
        for i, r in enumerate(REGIMES):
            row = cm[i]
            rec = row[i] / row.sum() * 100 if row.sum() else 0.0
            print(f"{r.value:16s}" + "".join(f"{v:8d}" for v in row) +
                  f"   {rec:5.1f}%")
        print(f"{'precision':16s}" + "".join(
            f"{(cm[i,i]/cm[:,i].sum()*100 if cm[:,i].sum() else 0):7.1f}%"
            for i in range(n)))
        c = np.array(conf)
        print("\ncalibration:")
        for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .85), (.85, 1.01)]:
            m = (c[:, 0] >= lo) & (c[:, 0] < hi)
            if m.sum() > 5:
                print(f"  conf [{lo:.2f},{hi:.2f}): n={int(m.sum()):5d} "
                      f"acc={100*c[m,1].mean():5.1f}%")
    return acc, cm


if __name__ == "__main__":
    run(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 60000)
