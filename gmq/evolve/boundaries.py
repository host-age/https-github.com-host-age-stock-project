"""What the controller may touch, and what it may never touch (spec §11).

This is the single most important file in the subpackage, because it is the
one that makes autonomous self-modification safe instead of reckless. Every
tunable in the system is registered here with three facts: where it lives, the
range it is allowed to move in, and -- for the ones that matter -- which
*direction* counts as loosening a safety control.

Two classes of knob:

  * **TUNABLE** -- search depth, objective weights, exit rules, model routing.
    The controller may propose any value inside the registered range and test
    it. These are strategy; getting them wrong costs money, which the
    experiment loop is designed to catch before it is spent.

  * **GUARDED** -- the hard risk limits: max drawdown, max daily loss, max
    position size, the consecutive-loss halt, leverage. The controller may
    read them and may propose making them *stricter*, but any proposal that
    would loosen one is rejected here, in code, before it can be tested. It
    cannot be A/B'd, canaried, or promoted, because a looser risk limit will
    almost always look better in a backtest -- right up until the draw it was
    protecting against arrives. This mirrors the existing design rule that the
    model layer has no handle on the risk engine; the *evolution* layer must
    not either.

"Immutable" here does not mean the numbers can never change. It means they
change only by an operator editing config, never by the system deciding on its
own that its own safety margin is inconvenient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any


class BoundaryError(Exception):
    """Raised when a proposal would cross an immutable boundary."""


@dataclass(frozen=True)
class KnobSpec:
    """One controllable parameter and the rules that govern changing it."""
    path: str                    # "search.exit_hysteresis"
    lo: float                    # inclusive lower bound the controller may set
    hi: float                    # inclusive upper bound
    guarded: bool = False        # True -> a risk control, direction-checked
    # For a guarded knob, which way is *safer*. +1 means larger is stricter
    # (e.g. min_model_brier is not one of these); -1 means smaller is stricter
    # (e.g. max_drawdown_pct: a smaller cap is a tighter control). The
    # controller may only move a guarded knob in the stricter direction.
    stricter_sign: int = 0
    kind: type = float

    def coerce(self, value: Any) -> Any:
        if self.kind is int:
            return int(round(float(value)))
        if self.kind is bool:
            return bool(value)
        return float(value)


# The registry. Guarded knobs carry stricter_sign; tunables do not.
# stricter_sign = -1  ->  smaller value = tighter control (a lower loss cap)
# stricter_sign = +1  ->  larger value  = tighter control (a higher min skill)
_DEFAULT_SPECS = [
    # ---- TUNABLE: the grandmaster search -------------------------------
    KnobSpec("search.max_depth", 1, 4, kind=int),
    KnobSpec("search.scenarios_per_node", 8, 48, kind=int),
    KnobSpec("search.beam_width", 2, 8, kind=int),
    KnobSpec("search.discount", 0.90, 0.999),
    KnobSpec("search.risk_aversion", 0.5, 6.0),
    KnobSpec("search.dd_penalty", 0.0, 0.40),
    KnobSpec("search.cvar_penalty", 0.0, 0.40),
    KnobSpec("search.min_edge_bps", 0.0, 20.0),
    KnobSpec("search.min_confidence", 0.10, 0.60),
    KnobSpec("search.exit_hysteresis", 0.0, 3.0),
    KnobSpec("search.reentry_cooldown_s", 0.0, 600.0),
    KnobSpec("search.giveback_activation_r", 0.5, 3.0),
    KnobSpec("search.giveback_keep_fraction", 0.20, 0.90),
    # ---- TUNABLE: objective weights (spec §15) -------------------------
    KnobSpec("obj_dd_w", 0.0, 4.0),
    KnobSpec("obj_vol_w", 0.0, 3.0),
    KnobSpec("obj_ploss_w", 0.0, 3.0),
    # ---- TUNABLE: model routing / calibration (spec §15) --------------
    KnobSpec("models.online_lr", 1e-4, 5e-2),
    KnobSpec("models.ensemble_halflife", 100, 2000, kind=int),
    # ---- GUARDED: the hard risk layer (spec §14) ----------------------
    # The controller may tighten these and may never loosen them.
    KnobSpec("risk.max_risk_per_trade_pct", 0.05, 0.50, guarded=True, stricter_sign=-1),
    KnobSpec("risk.max_position_pct", 2.0, 12.0, guarded=True, stricter_sign=-1),
    KnobSpec("risk.max_daily_loss_pct", 0.5, 2.0, guarded=True, stricter_sign=-1),
    KnobSpec("risk.max_daily_trades", 10, 60, guarded=True, stricter_sign=-1, kind=int),
    KnobSpec("risk.max_drawdown_pct", 2.0, 8.0, guarded=True, stricter_sign=-1),
    KnobSpec("risk.max_gross_exposure_pct", 50.0, 200.0, guarded=True, stricter_sign=-1),
    KnobSpec("risk.max_leverage", 1.0, 3.0, guarded=True, stricter_sign=-1),
    KnobSpec("risk.max_open_positions", 2, 8, guarded=True, stricter_sign=-1, kind=int),
    KnobSpec("risk.consecutive_loss_halt", 3, 8, guarded=True, stricter_sign=-1, kind=int),
    KnobSpec("risk.min_model_brier_score", 0.20, 0.35, guarded=True, stricter_sign=-1),
]


class KnobRegistry:
    """Holds the knob specs and enforces the boundary rules."""

    def __init__(self, specs: Optional[list] = None):
        self.specs: Dict[str, KnobSpec] = {
            s.path: s for s in (specs or _DEFAULT_SPECS)}

    def get(self, path: str) -> KnobSpec:
        if path not in self.specs:
            raise BoundaryError(f"unknown knob {path!r}: not in the registry, "
                                "so the controller has no authority to change it")
        return self.specs[path]

    def validate(self, path: str, new_value: Any,
                 current_value: Optional[float] = None) -> Any:
        """Return the coerced value if the change is permitted, else raise.

        A change is permitted iff the knob is registered, the new value is in
        range, and -- if the knob is guarded -- the change does not loosen the
        control. `current_value` is required to check the direction of a
        guarded change.
        """
        spec = self.get(path)
        v = spec.coerce(new_value)
        if not (spec.lo <= float(v) <= spec.hi):
            raise BoundaryError(
                f"{path}={v} is outside the permitted range "
                f"[{spec.lo}, {spec.hi}]")
        if spec.guarded:
            if current_value is None:
                raise BoundaryError(
                    f"{path} is a guarded risk control; a proposal to change "
                    "it must state the current value so the direction can be "
                    "checked")
            delta = float(v) - float(current_value)
            if delta == 0:
                return v
            # stricter_sign is the direction that TIGHTENS. If the change moves
            # the other way, it is a loosening, and it is refused.
            if (delta > 0) != (spec.stricter_sign > 0):
                raise BoundaryError(
                    f"REFUSED: {path} {current_value} -> {v} loosens a hard "
                    "risk limit. The evolution controller may tighten risk "
                    "controls but never relax them; that requires an operator. "
                    "(spec §11: security boundaries are immutable.)")
        return v

    def is_guarded(self, path: str) -> bool:
        return self.get(path).guarded
