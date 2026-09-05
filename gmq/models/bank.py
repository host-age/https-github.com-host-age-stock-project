"""The prediction block (spec §3, §16).

`ModelBank` owns every model for every symbol and turns a feature vector into
the full probabilistic picture the decision engine needs:

    P(up)              directional edge
    P(target|stop)     triple-barrier hit probabilities
    E[return]          expected move over the horizon
    E[vol]             expected dispersion of that move
    E[hold]            expected time in the trade
    P(continuation)    trend persists vs reverses
    confidence         how much of the above to believe

Models are **shared across symbols, not per-symbol**. One RELIANCE will never
generate enough resolved labels in a session to train anything; ten names
sharing a model see ten times the data, and because every feature is
normalised (ATR multiples, z-scores, bounded ratios) the pooled model is
learning cross-sectionally valid relationships rather than one stock's quirks.
A per-symbol bias term keeps whatever idiosyncrasy is real.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Prediction, NS
from ..core.config import ModelConfig
from ..core.mathx import clamp, sigmoid
from .base import Predictor, TripleBarrierLabeler, OnlineScaler, Sample, SkillTracker
from .online import OnlineLogistic, OnlineRidge, OnlineQuantile
from .gbdt import GBDTClassifier, GBDTRegressor
from .ensemble import SkillWeightedEnsemble, IsotonicCalibrator


class HorizonModels:
    """The full model set for one prediction horizon."""

    def __init__(self, dim: int, horizon_s: float, cfg: ModelConfig,
                 use_gbdt: bool = True):
        self.horizon_s = horizon_s
        tag = f"h{int(horizon_s)}"
        members: List[Predictor] = [
            OnlineLogistic(dim, name=f"logit_{tag}", lr=cfg.online_lr,
                           l2=cfg.l2, horizon_s=horizon_s),
            OnlineLogistic(dim, name=f"logit_fast_{tag}", lr=cfg.online_lr * 3,
                           l2=cfg.l2 * 4, forget=0.9995, horizon_s=horizon_s),
        ]
        if use_gbdt:
            members.append(GBDTClassifier(name=f"gbdt_{tag}",
                                          horizon_s=horizon_s,
                                          retrain_every=cfg.retrain_every))
        self.direction = SkillWeightedEnsemble(members, name=f"dir_{tag}",
                                               horizon_s=horizon_s,
                                               min_weight=cfg.min_weight)
        # barrier models: separate heads, because P(target) and P(stop) are
        # not complements -- the timeout outcome is neither
        self.p_target = OnlineLogistic(dim, name=f"ptgt_{tag}", lr=cfg.online_lr,
                                       l2=cfg.l2, horizon_s=horizon_s)
        self.p_stop = OnlineLogistic(dim, name=f"pstp_{tag}", lr=cfg.online_lr,
                                     l2=cfg.l2, horizon_s=horizon_s)
        self.cal_target = IsotonicCalibrator()
        self.cal_stop = IsotonicCalibrator()
        self.exp_ret = OnlineRidge(dim, name=f"eret_{tag}", lr=cfg.online_lr,
                                   l2=cfg.l2, horizon_s=horizon_s)
        self.exp_hold = OnlineRidge(dim, name=f"ehold_{tag}", lr=cfg.online_lr,
                                    l2=cfg.l2, horizon_s=horizon_s)
        # 80th percentile adverse excursion -- the number a stop must survive
        self.mae_q = OnlineQuantile(dim, q=0.80, name=f"mae_{tag}")
        self.mfe_q = OnlineQuantile(dim, q=0.60, name=f"mfe_{tag}")
        self.continuation = OnlineLogistic(dim, name=f"cont_{tag}",
                                           lr=cfg.online_lr, l2=cfg.l2)
        self.n = 0

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> Prediction:
        d = self.direction.predict(x)
        pt = self.p_target.prob(x)
        ps = self.p_stop.prob(x)
        if self.cal_target.ready:
            pt = self.cal_target.transform(pt)
        if self.cal_stop.ready:
            ps = self.cal_stop.transform(ps)
        # Skill-gate the barrier heads. Measured out-of-sample, P(target) and
        # P(stop) are a rare-event calibration problem and can post negative
        # skill -- worse than the base rate. A head that does not beat its base
        # rate must not drive expected value at full strength, or a good
        # directional call gets sized and exited on a bad probability. So each
        # head is shrunk toward its own base rate in proportion to how little
        # skill it has: full skill -> use the model, zero-or-negative skill ->
        # fall back to the base rate, which is what "no information" should
        # mean. The direction model (which is genuinely skilful) is untouched;
        # this only quarantines the weak heads until they earn their influence.
        pt = self._gate(self.p_target, pt)
        ps = self._gate(self.p_stop, ps)
        # P(target) + P(stop) <= 1; the remainder is the timeout outcome.
        tot = pt + ps
        if tot > 1.0:
            pt, ps = pt / tot, ps / tot
        er = self.exp_ret.value(x)
        eh = max(1.0, self.exp_hold.value(x))
        ev = math.sqrt(max(self.exp_ret.resid_var, 1e-12))
        return Prediction(
            p_up=d.p_up,
            exp_return=er,
            exp_vol=ev,
            p_target=float(clamp(pt, 0.0, 1.0)),
            p_stop=float(clamp(ps, 0.0, 1.0)),
            exp_hold_s=float(clamp(eh, 1.0, self.horizon_s * 3)),
            p_continuation=self.continuation.prob(x),
            confidence=d.confidence,
            horizon_s=self.horizon_s,
            source=f"bank_h{int(self.horizon_s)}",
            features_used=d.features_used,
        )

    @staticmethod
    def _gate(head, p: float) -> float:
        """Shrink a barrier probability toward its base rate by (1 - skill).

        skill<=0  -> return the base rate (the head is worse than useless);
        skill=1   -> return the head's own estimate untouched;
        in between -> a convex blend. Below the warmup sample count the head
        has no established skill and is fully shrunk, which is the safe default.
        """
        sk = head.skill
        if not sk.ready:
            base = sk.base_rate() if hasattr(sk, "base_rate") else 0.5
            return float(base)
        w = max(0.0, min(1.0, sk.skill_score()))
        base = sk.base_rate() if hasattr(sk, "base_rate") else 0.5
        return float(w * p + (1.0 - w) * base)

    def learn(self, s: Sample) -> None:
        x = s.x
        # Score before training, so every number is out-of-sample for this
        # observation. (Still overlapping-window; ModelBank keeps a separate
        # purged tracker for the honest headline figure.)
        self.p_target.skill.add(self.p_target.prob(x), min(s.y_target, 1.0))
        self.p_stop.skill.add(self.p_stop.prob(x), min(s.y_stop, 1.0))
        if float(s.meta.get("continuation", -1)) >= 0:
            self.continuation.skill.add(self.continuation.prob(x),
                                        float(s.meta["continuation"]))
        # Ambiguous samples (both barriers touched between observations) carry
        # genuinely less information, so they are down-weighted rather than
        # dropped -- dropping them would bias the sample toward calm periods.
        w = 0.5 if (s.y_target == 0.5) else 1.0
        self.direction.partial_fit(x, s.y_dir, w)
        self.cal_target.add(self.p_target.prob(x), min(s.y_target, 1.0))
        self.cal_stop.add(self.p_stop.prob(x), min(s.y_stop, 1.0))
        self.p_target.partial_fit(x, min(s.y_target, 1.0), w)
        self.p_stop.partial_fit(x, min(s.y_stop, 1.0), w)
        self.exp_ret.partial_fit(x, s.ret, w)
        self.exp_hold.partial_fit(x, s.hold_s, w)
        if s.entry_px > 0:
            self.mae_q.partial_fit(x, abs(s.mae) / s.entry_px, w)
            self.mfe_q.partial_fit(x, abs(s.mfe) / s.entry_px, w)
        cont = float(s.meta.get("continuation", -1))
        if cont >= 0:
            self.continuation.partial_fit(x, cont, w)
        self.n += 1

    def state(self) -> dict:
        return {
            "n": self.n,
            "direction": self.direction.state(),
            "p_target": self.p_target.state(),
            "p_stop": self.p_stop.state(),
            "exp_ret": self.exp_ret.state(),
        }


class ModelBank:
    """All horizons, plus the labelling pipeline and per-symbol bias."""

    def __init__(self, cfg: ModelConfig, dim: int = 160,
                 use_gbdt: bool = True):
        self.cfg = cfg
        self.dim = dim
        self.scaler = OnlineScaler(dim)
        self.horizons = [float(h) for h in cfg.horizons_s]
        self.models: Dict[float, HorizonModels] = {
            h: HorizonModels(dim, h, cfg, use_gbdt=use_gbdt and h >= 300)
            for h in self.horizons
        }
        # Barrier width MUST scale with the horizon, or the label is
        # degenerate. A +-1.6/1.0 ATR barrier is calibrated for the ~900s
        # horizon; applied unchanged to a 60s horizon it is almost never
        # touched, because price does not travel a full ATR in a minute. The
        # label then collapses to "timeout" on essentially every sample, the
        # base rate goes to ~0, and the model scores -1.0 skill no matter what
        # it predicts -- which is exactly what the diagnostic showed. Diffusion
        # scales with sqrt(time), so the barrier does too: at 60s the width is
        # sqrt(60/900)=0.26x, at 3600s it is 2.0x. Every horizon now gets a
        # barrier it can actually reach, and therefore a non-degenerate label.
        ref_h = 900.0
        self.labelers: Dict[float, TripleBarrierLabeler] = {}
        for h in self.horizons:
            scale = math.sqrt(max(h, 1.0) / ref_h)
            self.labelers[h] = TripleBarrierLabeler(
                cfg.barrier_target_atr * scale,
                cfg.barrier_stop_atr * scale, h)
        # per-symbol residual bias, learned from the same resolved labels
        self.sym_bias: Dict[str, float] = defaultdict(float)
        self.sym_n: Dict[str, int] = defaultdict(int)
        self.samples_seen = 0
        # Honest, non-overlapping out-of-sample skill.
        #
        # The per-member trackers above score each sample before training on
        # it, which handles direct leakage -- but consecutive samples share
        # most of their label window. Sample t resolves and trains the model;
        # sample t+1min, whose outcome was largely determined by the same price
        # path, is then scored by a model that has just been shown the answer.
        # That inflates measured skill without any bug being visible anywhere.
        # These trackers only accept a sample when a full horizon has elapsed
        # since the last one scored for that symbol, so the sampled windows do
        # not overlap at all.
        self.honest: Dict[float, SkillTracker] = {
            h: SkillTracker(window=4000) for h in self.horizons
        }
        self._last_scored: Dict[Tuple[str, float], int] = {}
        self.last_review: Dict[str, str] = {}
        self.review_every = 500
        self._since_review = 0
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    def observe(self, symbol: str, ts: int, fv, regime: str = "") -> np.ndarray:
        """Register the current state for later labelling; returns scaled x."""
        if not self.feature_names:
            return np.zeros(0)
        raw = fv.vec(self.feature_names)
        x = self.scaler.fit_transform(raw)
        if fv.atr > 0 and fv.price > 0:
            for h, lab in self.labelers.items():
                lab.observe(x, ts, symbol, fv.price, fv.atr, regime,
                            meta={"align": fv.alignment})
        return x

    def transform(self, fv) -> np.ndarray:
        if not self.feature_names:
            return np.zeros(0)
        return self.scaler.transform(fv.vec(self.feature_names))

    # ------------------------------------------------------------------
    def on_price(self, symbol: str, ts: int, px: float) -> int:
        """Resolve pending labels and train on whatever became real."""
        n = 0
        for h, lab in self.labelers.items():
            for s in lab.update(symbol, ts, px):
                key = (symbol, h)
                last = self._last_scored.get(key, -(1 << 62))
                if s.ts - last >= int(h * NS):
                    self._last_scored[key] = s.ts
                    self.honest[h].add(self.models[h].direction.predict(s.x).p_up,
                                       s.y_dir)
                self.models[h].learn(s)
                self._learn_bias(s)
                n += 1
                self.samples_seen += 1
        self._since_review += n
        if self._since_review >= self.review_every:
            self._since_review = 0
            for h, m in self.models.items():
                acts = m.direction.review()
                for k, v in acts.items():
                    self.last_review[k] = v
        return n

    def _learn_bias(self, s: Sample) -> None:
        self.sym_n[s.symbol] += 1
        a = 1.0 / min(self.sym_n[s.symbol], 400)
        self.sym_bias[s.symbol] += a * ((s.y_dir - 0.5) * 2.0 -
                                        self.sym_bias[s.symbol])

    # ------------------------------------------------------------------
    def predict(self, symbol: str, x: np.ndarray,
                horizon_s: Optional[float] = None) -> Prediction:
        h = horizon_s or self._nearest_horizon(horizon_s or 300.0)
        p = self.models[h].predict(x)
        # per-symbol bias, heavily shrunk -- it is a correction, not a signal
        b = self.sym_bias.get(symbol, 0.0) * 0.15
        if b:
            p.p_up = float(clamp(p.p_up + b * 0.1, 0.01, 0.99))
        return p

    def predict_all(self, symbol: str, x: np.ndarray) -> Dict[float, Prediction]:
        return {h: self.predict(symbol, x, h) for h in self.horizons}

    def _nearest_horizon(self, h: float) -> float:
        return min(self.horizons, key=lambda z: abs(z - h))

    def consensus(self, symbol: str, x: np.ndarray) -> Prediction:
        """Blend horizons into one view, weighted by each horizon's skill.

        A short-horizon edge and a long-horizon edge pointing opposite ways is
        a real and common state; collapsing it to a single number loses that,
        so `p_continuation` carries the agreement level forward for the search
        engine to use rather than hiding it.
        """
        preds = self.predict_all(symbol, x)
        num = den = 0.0
        for h, p in preds.items():
            w = max(0.05, p.confidence)
            num += w * (2 * p.p_up - 1)
            den += w
        edge = num / den if den else 0.0
        best = max(preds.values(), key=lambda p: p.confidence)
        sides = [1 if p.p_up > 0.5 else -1 for p in preds.values()]
        agree = abs(sum(sides)) / max(len(sides), 1)
        out = Prediction(
            p_up=float(clamp(0.5 + 0.5 * edge, 0.01, 0.99)),
            exp_return=best.exp_return,
            exp_vol=best.exp_vol,
            p_target=best.p_target,
            p_stop=best.p_stop,
            exp_hold_s=best.exp_hold_s,
            p_continuation=float(agree),
            confidence=float(clamp(best.confidence * (0.75 + 0.25 * agree), 0, 1)),
            horizon_s=best.horizon_s,
            source="consensus",
        )
        return out

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "samples": self.samples_seen,
            "honest_skill": {str(int(h)): t.stats()
                             for h, t in self.honest.items()},
            "pending": {str(int(h)): l.pending_count()
                        for h, l in self.labelers.items()},
            "scaler_n": self.scaler.n,
            "horizons": {str(int(h)): m.state() for h, m in self.models.items()},
            "reviews": dict(self.last_review),
        }

    def health(self, min_samples: int = 300) -> Tuple[bool, str]:
        """Is the model layer sane enough to trade? Used by the risk engine.

        Three things this deliberately does NOT do:

        * It does not judge on raw Brier. Brier depends on the base rate, so
          the same score means "excellent" in a balanced market and "useless"
          in a lopsided one. Brier *skill* against the base rate is the
          comparison that means the same thing everywhere.
        * It does not take the worst of every horizon. The long horizons
          resolve few labels, so their scores are dominated by sampling noise,
          and letting the noisiest estimate halt the whole system means the
          system halts on noise. Only horizons with a real sample count vote.
        * It does not fail on a thin sample. An undertrained model is not a
          degraded one -- that is what the warm-up gate is for.

        Model health is a *current condition*, not an event, so this can and
        does return True again once skill recovers. That is the difference
        between it and a daily-loss halt, which is a decision about a day.
        """
        votes: List[Tuple[float, float]] = []
        for h, m in self.models.items():
            t = self.honest.get(h)
            if t is not None and t.n_total >= min_samples and t.ready:
                votes.append((h, t.skill_score()))
        if not votes:
            return True, ""          # not enough evidence to condemn it
        # The shortest horizon with a real sample is the one the engine leans
        # on most and the one with the most data behind it.
        votes.sort(key=lambda kv: kv[0])
        worst = min(s for _h, s in votes)
        mean_skill = sum(s for _h, s in votes) / len(votes)
        if mean_skill < -0.02 and worst < -0.05:
            return False, (f"model_skill_mean_{mean_skill:+.3f}"
                           f"_worst_{worst:+.3f}")
        return True, ""
