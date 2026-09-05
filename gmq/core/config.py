r"""Central configuration.

Every tunable in the system lives here so a run is fully described by one
serialisable object -- which is what makes walk-forward and Monte Carlo runs
reproducible and what gets stamped into every decision record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class RiskLimits:
    """Spec §14 -- the hard layer. These are enforced by an independent engine
    that the model layer has no handle on and cannot mutate."""
    # per trade
    max_risk_per_trade_pct: float = 0.50       # % of equity at 1R
    max_position_pct: float = 12.0             # % of equity in one symbol
    min_qty: int = 1
    # daily
    max_daily_loss_pct: float = 2.0
    max_daily_trades: int = 60
    # portfolio
    max_drawdown_pct: float = 8.0              # from equity high-water mark
    max_gross_exposure_pct: float = 200.0
    max_net_exposure_pct: float = 100.0
    max_leverage: float = 3.0
    max_open_positions: int = 8
    max_sector_exposure_pct: float = 35.0
    max_correlated_exposure_pct: float = 45.0  # |rho| > corr_threshold cluster
    corr_threshold: float = 0.65
    # tail risk
    max_var_pct: float = 3.0                   # 1-day 99% VaR as % of equity
    max_es_pct: float = 4.5                    # expected shortfall
    # circuit breakers
    consecutive_loss_halt: int = 5
    loss_velocity_halt_pct: float = 1.2        # loss within loss_velocity_win
    loss_velocity_window_s: float = 900.0
    # model sanity -- if the AI misbehaves, risk stops trading on its own
    # Order-rate anomaly detection. Expressed PER SYMBOL, because the whole
    # point is to catch a model stuck in a feedback loop -- and a flat cap
    # across the book means the same "runaway" threshold fires during normal
    # operation with ten names that would never fire with two. It counts parent
    # decisions, not child slices: one decision sliced into five orders is one
    # decision, and counting slices makes a large order look like a malfunction.
    max_orders_per_min_per_symbol: float = 4.0
    max_orders_per_min: int = 30       # absolute floor, for tiny universes
    max_reject_rate: float = 0.25
    min_model_brier_score: float = 0.30        # worse than this -> suspend
    # execution
    max_slippage_bps_halt: float = 45.0
    # square-off
    square_off_intraday: bool = True


@dataclass
class SearchConfig:
    """Spec §2/§5 -- the grandmaster engine."""
    max_depth: int = 3                 # plies of look-ahead
    scenarios_per_node: int = 24       # sampled market replies per move
    beam_width: int = 4                # keep top-K moves per ply
    horizon_s: float = 900.0           # per-ply lookahead horizon
    discount: float = 0.97             # per-ply discount on future EV
    prune_ev_floor_r: float = -0.35    # abandon branches below this EV in R
    time_budget_ms: float = 25.0       # hard wall-clock budget per decision
    # Reproducibility. A wall-clock budget makes the search non-deterministic:
    # the same input can explore a different number of paths depending on how
    # busy the machine is, so a backtest cannot be replayed exactly and two
    # walk-forward folds are not strictly comparable. Setting node_budget > 0
    # replaces the clock with a node count, which is deterministic. Live
    # trading wants the clock (latency is the real constraint); backtests and
    # research want the node count.
    node_budget: int = 0               # 0 -> use the wall clock
    # Objective weights. See evaluator.objective for why these are the shapes
    # and magnitudes they are -- variance is an Arrow-Pratt term scaled by
    # equity, drawdown and tail loss are linear in rupees.
    risk_aversion: float = 2.0         # gamma on variance/equity
    dd_penalty: float = 0.08           # rupees of EV per rupee of E[drawdown]
    cvar_penalty: float = 0.06         # rupees of EV per rupee of tail loss
    min_edge_bps: float = 6.0          # do not act below this net edge
    min_confidence: float = 0.30
    use_antithetic: bool = True        # variance reduction in path sampling
    # Score every candidate move against the same sampled futures, and re-use
    # those futures while the situation is unchanged. Without it, two searches
    # over an identical state can disagree, and a position exits because the
    # dice moved rather than because the market did.
    common_random_numbers: bool = True
    # How many round trips of edge a change to an open position must beat
    # before it is worth making. 1.0 = the change must pay for itself.
    exit_hysteresis: float = 1.0
    # Value an open position at liquidation, not at the mid. Without this the
    # exit leg of a round trip is only charged on paths where a barrier fires,
    # so the search sees a round trip as costing about half what it does.
    liquidation_in_eval: bool = True
    # Seconds before the same symbol may be entered again after being closed.
    reentry_cooldown_s: float = 120.0
    # Give-back exit rule (analytics/excursion.py). The defaults are the
    # starting point; the fitted policy replaces them once there is enough
    # evidence to clear the overfitting guards.
    giveback_enabled: bool = True
    giveback_activation_r: float = 1.4
    giveback_keep_fraction: float = 0.55
    giveback_min_samples: int = 120
    giveback_refit_every: int = 40     # closed trades between refits


@dataclass
class ExecConfig:
    """Spec §11/§12."""
    mode: str = "paper"                # paper | sim | live  (live is gated)
    broker: str = "sim"                # sim | kite | upstox
    # latency model, microseconds
    lat_send_us: float = 900.0
    lat_ack_us: float = 1500.0
    lat_jitter_us: float = 600.0
    lat_feed_us: float = 400.0
    # microstructure costs
    taker_impact_coef: float = 0.55    # sqrt-impact coefficient
    reject_prob: float = 0.004
    partial_prob: float = 0.15
    max_participation: float = 0.12    # of bar volume
    slice_threshold_notional: float = 1_500_000.0
    verify_fills: bool = True
    reconcile_every_s: float = 5.0
    max_retries: int = 3


@dataclass
class ModelConfig:
    """Spec §3/§17."""
    horizons_s: List[float] = field(default_factory=lambda: [60, 300, 900, 3600])
    barrier_target_atr: float = 1.6
    barrier_stop_atr: float = 1.0
    online_lr: float = 0.012
    l2: float = 1e-4
    warmup_samples: int = 250
    ensemble_halflife: int = 400       # samples, for skill-decay weighting
    min_weight: float = 0.02
    calibration_bins: int = 12
    retrain_every: int = 2000
    degrade_window: int = 200
    degrade_brier_delta: float = 0.06  # vs. in-sample -> flag for review


@dataclass
class DataConfig:
    symbols: List[str] = field(default_factory=lambda: [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "LT", "AXISBANK",
    ])
    index_symbol: str = "NIFTY"
    timeframes: List[str] = field(default_factory=lambda:
                                  ["1m", "5m", "15m", "1h", "4h", "1d", "1w"])
    depth_levels: int = 5
    tick_hz: float = 4.0               # ticks/sec/symbol in the simulator
    warmup_bars: int = 240
    history_days: int = 60


@dataclass
class Config:
    run_id: str = "run"
    seed: int = 7
    initial_capital: float = 1_000_000.0
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecConfig = field(default_factory=ExecConfig)
    journal_dir: str = "runs"
    log_level: str = "INFO"
    # objective weights (spec §15)
    obj_return_w: float = 1.0
    obj_dd_w: float = 1.6
    obj_vol_w: float = 0.8
    obj_ploss_w: float = 0.5
    obj_cost_w: float = 1.0
    obj_util_w: float = 0.15

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        c = cls()
        for k, v in d.items():
            if not hasattr(c, k):
                continue
            cur = getattr(c, k)
            if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
                for kk, vv in v.items():
                    if hasattr(cur, kk):
                        setattr(cur, kk, vv)
            else:
                setattr(c, k, v)
        return c

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# NSE sector map for the default universe (spec §10 concentration limits)
SECTORS: Dict[str, str] = {
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK",
    "AXISBANK": "BANK", "KOTAKBANK": "BANK", "INDUSINDBK": "BANK",
    "BHARTIARTL": "TELECOM",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG",
    "LT": "INFRA", "ULTRACEMCO": "CEMENT",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO",
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA",
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "HINDALCO": "METAL",
    "NIFTY": "INDEX", "BANKNIFTY": "INDEX",
}


def sector_of(symbol: str) -> str:
    return SECTORS.get(symbol.upper(), "UNKNOWN")
