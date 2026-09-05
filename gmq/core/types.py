"""Core domain types shared by every layer of the system.

Everything here is deliberately dependency-light (stdlib + numpy only) so the
hot path -- data ingest -> feature -> model -> risk -> execution -- never pays
for heavyweight object construction.
"""
from __future__ import annotations

import time
import itertools
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from typing import Optional, Dict, Any, List, Tuple

# --------------------------------------------------------------------------
# time helpers
# --------------------------------------------------------------------------

NS = 1_000_000_000


def now_ns() -> int:
    """Monotonic-ish wall clock in nanoseconds (epoch based)."""
    return time.time_ns()


# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------


class Side(IntEnum):
    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        return int(self)


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"            # stop-loss limit (NSE style)
    SL_M = "SL-M"        # stop-loss market
    IOC = "IOC"          # immediate-or-cancel limit


class TimeInForce(str, Enum):
    DAY = "DAY"
    IOC = "IOC"


class Product(str, Enum):
    MIS = "MIS"          # intraday, leveraged, auto square-off
    CNC = "CNC"          # delivery
    NRML = "NRML"        # carry-forward F&O


class OrderStatus(str, Enum):
    PENDING = "PENDING"          # created locally, not yet acknowledged
    OPEN = "OPEN"                # live at exchange
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TRIGGER_PENDING = "TRIGGER_PENDING"

    @property
    def terminal(self) -> bool:
        return self in (
            OrderStatus.COMPLETE,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        )


class InstrumentType(str, Enum):
    EQ = "EQ"
    FUT = "FUT"
    CE = "CE"
    PE = "PE"
    INDEX = "INDEX"


class Regime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    MEAN_REVERTING = "MEAN_REVERTING"
    BREAKOUT = "BREAKOUT"
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    ILLIQUID = "ILLIQUID"


class MoveType(str, Enum):
    """The legal 'moves' the grandmaster engine may consider each turn."""
    WAIT = "WAIT"
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    TAKE_PARTIAL = "TAKE_PARTIAL"
    MOVE_STOP = "MOVE_STOP"
    EXIT = "EXIT"
    REVERSE = "REVERSE"


class LossCause(str, Enum):
    NORMAL_VOLATILITY = "NORMAL_VOLATILITY"
    TEMPORARY_PULLBACK = "TEMPORARY_PULLBACK"
    THESIS_WEAKENING = "THESIS_WEAKENING"
    THESIS_INVALIDATED = "THESIS_INVALIDATED"
    REGIME_CHANGE = "REGIME_CHANGE"
    EVENT_SHOCK = "EVENT_SHOCK"
    LIQUIDITY_EXECUTION = "LIQUIDITY_EXECUTION"


class Timeframe(str, Enum):
    TICK = "tick"
    S5 = "5s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def seconds(self) -> int:
        return {
            "tick": 0, "5s": 5, "1m": 60, "5m": 300, "15m": 900,
            "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800,
        }[self.value]


# ordered coarse -> fine, used by the multi-timeframe alignment logic
TF_LADDER: Tuple[Timeframe, ...] = (
    Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1,
    Timeframe.M15, Timeframe.M5, Timeframe.M1, Timeframe.S5,
)


# --------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Instrument:
    symbol: str
    token: int
    itype: InstrumentType = InstrumentType.EQ
    sector: str = "UNKNOWN"
    tick_size: float = 0.05
    lot_size: int = 1
    exchange: str = "NSE"
    # circuit limits as fraction of previous close
    circuit_pct: float = 0.20
    # per-side brokerage + statutory charges as a fraction of turnover
    cost_bps: float = 3.5
    underlying: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[str] = None

    def round_price(self, px: float) -> float:
        return round(round(px / self.tick_size) * self.tick_size, 4)

    @property
    def is_derivative(self) -> bool:
        return self.itype in (InstrumentType.FUT, InstrumentType.CE, InstrumentType.PE)


# --------------------------------------------------------------------------
# market data
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Tick:
    ts: int                  # nanoseconds
    symbol: str
    ltp: float
    ltq: int = 0
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    volume: int = 0          # cumulative day volume
    oi: int = 0              # open interest (derivatives)
    aggressor: int = 0       # +1 buy-initiated, -1 sell-initiated, 0 unknown

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.ltp

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.spread / m * 1e4) if m > 0 else 0.0

    @property
    def microprice(self) -> float:
        """Size-weighted fair value -- leans toward the thinner side."""
        tot = self.bid_qty + self.ask_qty
        if tot <= 0 or self.bid <= 0 or self.ask <= 0:
            return self.mid
        return (self.bid * self.ask_qty + self.ask * self.bid_qty) / tot


@dataclass(slots=True)
class Bar:
    ts: int                  # bar OPEN time, nanoseconds
    symbol: str
    tf: Timeframe
    o: float
    h: float
    l: float
    c: float
    v: int = 0
    n_trades: int = 0
    vwap_num: float = 0.0    # sum(px*qty), for incremental vwap
    buy_v: int = 0
    sell_v: int = 0
    closed: bool = False

    @property
    def vwap(self) -> float:
        return self.vwap_num / self.v if self.v else self.c

    @property
    def range(self) -> float:
        return self.h - self.l

    @property
    def body(self) -> float:
        return self.c - self.o

    @property
    def delta(self) -> int:
        return self.buy_v - self.sell_v

    def as_tuple(self):
        return (self.ts, self.o, self.h, self.l, self.c, self.v)


@dataclass(slots=True)
class BookLevel:
    price: float
    qty: int
    orders: int = 1


@dataclass(slots=True)
class DepthSnapshot:
    """NSE publishes 5 levels per side on the standard feed."""
    ts: int
    symbol: str
    bids: List[BookLevel] = field(default_factory=list)
    asks: List[BookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid(self) -> float:
        if self.bids and self.asks:
            return 0.5 * (self.bids[0].price + self.asks[0].price)
        return self.best_bid or self.best_ask

    def imbalance(self, levels: int = 5) -> float:
        """(bidQ - askQ) / (bidQ + askQ) over the top `levels`. In [-1, 1]."""
        b = sum(l.qty for l in self.bids[:levels])
        a = sum(l.qty for l in self.asks[:levels])
        tot = a + b
        return (b - a) / tot if tot else 0.0

    def depth_value(self, levels: int = 5) -> float:
        return sum(l.price * l.qty for l in self.bids[:levels]) + \
               sum(l.price * l.qty for l in self.asks[:levels])

    def sweep_cost(self, side: Side, qty: int) -> Tuple[float, int]:
        """Walk the book: returns (avg fill price, qty actually fillable)."""
        levels = self.asks if side is Side.BUY else self.bids
        remaining, notional, filled = qty, 0.0, 0
        for lv in levels:
            take = min(remaining, lv.qty)
            notional += take * lv.price
            filled += take
            remaining -= take
            if remaining <= 0:
                break
        if filled == 0:
            return (0.0, 0)
        return (notional / filled, filled)


# --------------------------------------------------------------------------
# orders / fills / positions
# --------------------------------------------------------------------------

_order_seq = itertools.count(1)


@dataclass(slots=True)
class Order:
    symbol: str
    side: Side
    qty: int
    otype: OrderType = OrderType.LIMIT
    price: float = 0.0            # limit price
    trigger: float = 0.0          # stop trigger
    product: Product = Product.MIS
    tif: TimeInForce = TimeInForce.DAY
    tag: str = ""                 # links order -> decision record
    parent_id: Optional[str] = None
    # runtime state
    order_id: str = ""
    broker_order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_price: float = 0.0
    created_ns: int = field(default_factory=now_ns)
    sent_ns: int = 0
    ack_ns: int = 0
    done_ns: int = 0
    reject_reason: str = ""
    intent_price: float = 0.0     # decision-time reference for slippage attribution

    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"O{next(_order_seq):08d}"

    @property
    def pending_qty(self) -> int:
        return max(0, self.qty - self.filled_qty)

    @property
    def is_live(self) -> bool:
        return not self.status.terminal

    @property
    def latency_us(self) -> float:
        if self.ack_ns and self.sent_ns:
            return (self.ack_ns - self.sent_ns) / 1000.0
        return 0.0

    @property
    def slippage_bps(self) -> float:
        ref = self.intent_price or self.price
        if not ref or not self.avg_price:
            return 0.0
        # positive == worse than intended
        return self.side.sign * (self.avg_price - ref) / ref * 1e4


@dataclass(slots=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: int
    price: float
    ts: int
    fee: float = 0.0
    liquidity: str = "TAKER"     # TAKER | MAKER


@dataclass(slots=True)
class Position:
    symbol: str
    qty: int = 0                 # signed: +long, -short
    avg_price: float = 0.0
    realised: float = 0.0
    fees: float = 0.0
    opened_ns: int = 0
    last_ns: int = 0
    # risk plumbing
    stop: float = 0.0
    target: float = 0.0
    initial_stop: float = 0.0
    initial_risk: float = 0.0    # rupees at risk at entry (1R)
    # excursions
    mfe: float = 0.0             # max favourable excursion, rupees
    mae: float = 0.0             # max adverse excursion, rupees
    peak_px: float = 0.0
    trough_px: float = 0.0
    # provenance
    decision_id: str = ""
    thesis: str = ""
    entry_regime: str = ""
    entry_confidence: float = 0.0
    scale_ins: int = 0
    partials: int = 0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @property
    def side(self) -> Optional[Side]:
        if self.qty > 0:
            return Side.BUY
        if self.qty < 0:
            return Side.SELL
        return None

    @property
    def direction(self) -> int:
        return (self.qty > 0) - (self.qty < 0)

    def unrealised(self, px: float) -> float:
        return (px - self.avg_price) * self.qty

    def pnl(self, px: float) -> float:
        return self.realised + self.unrealised(px) - self.fees

    def r_multiple(self, px: float) -> float:
        """Unrealised P&L in units of the risk taken at entry.

        Returns 0.0 when entry risk was never established. That is a
        deliberate neutral, used only for live decision context where a
        missing value would break comparisons; completed trades report an
        honest None instead (see TradeRecord.r_multiple).
        """
        if self.initial_risk <= 0:
            return 0.0
        return self.unrealised(px) / self.initial_risk

    def notional(self, px: float) -> float:
        return abs(self.qty) * px

    def update_excursions(self, px: float) -> None:
        if self.qty == 0:
            return
        u = self.unrealised(px)
        self.mfe = max(self.mfe, u)
        self.mae = min(self.mae, u)
        self.peak_px = max(self.peak_px, px) if self.peak_px else px
        self.trough_px = min(self.trough_px, px) if self.trough_px else px

    def apply_fill(self, f: Fill) -> float:
        """Apply a fill; returns realised pnl booked by this fill."""
        signed = f.qty * f.side.sign
        booked = 0.0
        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            # opening or adding
            new_qty = self.qty + signed
            if new_qty != 0:
                self.avg_price = (
                    self.avg_price * self.qty + f.price * signed
                ) / new_qty
            self.qty = new_qty
            if self.opened_ns == 0:
                self.opened_ns = f.ts
        else:
            # reducing / closing / flipping
            closing = min(abs(signed), abs(self.qty))
            booked = (f.price - self.avg_price) * closing * self.direction
            self.realised += booked
            remaining = self.qty + signed
            if (remaining > 0) != (self.qty > 0) and remaining != 0:
                # Flipped through zero: this is a new trade in the opposite
                # direction, so everything that describes the old one has to
                # be cleared. Leaving initial_risk behind means the new leg is
                # measured in the old leg's risk units -- silently, and only
                # visible as a nonsense R-multiple much later.
                self.avg_price = f.price
                self.opened_ns = f.ts
                self.mfe = self.mae = 0.0
                self.peak_px = self.trough_px = f.price
                self.initial_risk = 0.0
                self.initial_stop = 0.0
                self.scale_ins = 0
                self.partials = 0
            self.qty = remaining
            if self.qty == 0:
                self.avg_price = 0.0
        self.fees += f.fee
        self.last_ns = f.ts
        return booked


# --------------------------------------------------------------------------
# model / decision payloads
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Prediction:
    """What every predictive model must return, in comparable units."""
    p_up: float = 0.5            # P(price higher at horizon)
    exp_return: float = 0.0      # expected return over horizon, fraction
    exp_vol: float = 0.0         # expected stdev of return over horizon
    p_target: float = 0.0        # P(hit target before stop) -- triple barrier
    p_stop: float = 0.0
    exp_hold_s: float = 0.0      # expected holding time, seconds
    p_continuation: float = 0.5  # P(trend continues | trending)
    confidence: float = 0.0      # [0,1] model self-assessed reliability
    horizon_s: float = 300.0
    source: str = ""
    features_used: int = 0

    def edge(self) -> float:
        """Directional edge in [-1, 1]."""
        return 2.0 * self.p_up - 1.0


@dataclass(slots=True)
class Scenario:
    name: str
    prob: float
    ret: float                   # terminal return of the path, fraction
    max_dd: float                # worst adverse excursion along the path
    max_run: float               # best favourable excursion along the path
    hit_stop: bool = False
    hit_target: bool = False
    hold_s: float = 0.0


@dataclass(slots=True)
class MoveEval:
    move: MoveType
    qty: int = 0
    price: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    ev: float = 0.0              # expected value in rupees, net of costs
    ev_r: float = 0.0            # expected value in R multiples
    score: float = 0.0           # risk-adjusted objective (what we maximise)
    exp_dd: float = 0.0
    p_win: float = 0.0
    p_loss: float = 0.0
    variance: float = 0.0
    cvar: float = 0.0            # expected shortfall of the move, rupees
    cost: float = 0.0            # modelled transaction cost + slippage
    depth: int = 0
    scenarios: List[Scenario] = field(default_factory=list)
    children: List["MoveEval"] = field(default_factory=list)
    rationale: str = ""
    rejected_by: str = ""        # populated when the risk engine vetoes

    @property
    def allowed(self) -> bool:
        return not self.rejected_by


@dataclass(slots=True)
class RiskVerdict:
    approved: bool
    reason: str = ""
    max_qty: int = 0
    breached: List[str] = field(default_factory=list)
    scaled_from: int = 0


@dataclass
class DecisionRecord:
    """The full auditable Market State -> ... -> Outcome record (spec §13)."""
    decision_id: str
    ts: int
    symbol: str
    # 1. market state
    market_state: Dict[str, Any] = field(default_factory=dict)
    # 2. data signals
    signals: Dict[str, float] = field(default_factory=dict)
    # 3. model predictions
    predictions: Dict[str, Any] = field(default_factory=dict)
    regime: str = ""
    regime_conf: float = 0.0
    # 4. alternative scenarios / candidate moves
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    # 5. risk assessment
    risk: Dict[str, Any] = field(default_factory=dict)
    # 6. decision
    chosen: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    # 7. execution
    execution: Dict[str, Any] = field(default_factory=dict)
    # 8. outcome (filled in later by the monitor)
    outcome: Dict[str, Any] = field(default_factory=dict)
    search_ms: float = 0.0
    nodes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
