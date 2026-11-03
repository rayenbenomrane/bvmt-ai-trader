"""Long-only signal state transitions for next-session-open execution."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def build_strategy_actions(
    probabilities: Sequence[float],
    signal_opens: Sequence[float],
    signal_closes: Sequence[float],
    entry_threshold: float = 0.60,
    exit_threshold: float = 0.40,
    minimum_holding_period: int = 3,
    maximum_holding_period: Optional[int] = None,
    stop_loss_pct: Optional[float] = None,
    expected_returns: Optional[Sequence[float]] = None,
    estimated_round_trip_cost: float = 0.0,
    safety_margin: float = 0.0,
    cost_filter: bool = False,
    hysteresis: bool = True,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """Build BUY/HOLD/SELL decisions using only information known at each signal close."""
    probabilities = np.asarray(probabilities, dtype=float)
    signal_opens = np.asarray(signal_opens, dtype=float)
    signal_closes = np.asarray(signal_closes, dtype=float)
    n = len(probabilities)
    if len(signal_opens) != n or len(signal_closes) != n:
        raise ValueError("probabilities, opens, and closes must have equal lengths.")
    if expected_returns is not None and len(expected_returns) != n:
        raise ValueError("expected_returns must have the same length as probabilities.")
    if not 0 <= exit_threshold <= entry_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= exit_threshold <= entry_threshold <= 1.")
    if hysteresis and exit_threshold == entry_threshold:
        raise ValueError("hysteresis requires exit_threshold below entry_threshold.")
    if minimum_holding_period < 0:
        raise ValueError("minimum_holding_period must be non-negative.")
    if maximum_holding_period is not None and maximum_holding_period < minimum_holding_period:
        raise ValueError("maximum_holding_period cannot be less than minimum_holding_period.")
    if stop_loss_pct is not None and not 0 < stop_loss_pct < 1:
        raise ValueError("stop_loss_pct must be between 0 and 1.")
    if cost_filter and expected_returns is None:
        raise ValueError("expected_returns are required when cost_filter is enabled.")

    expected = None if expected_returns is None else np.asarray(expected_returns, dtype=float)
    actions = np.zeros(n, dtype=np.int8)
    reasons = [""] * n
    rejections: List[Dict[str, Any]] = []
    position_open = False
    pending_entry = False
    pending_exit = False
    entry_price = 0.0
    entry_execution_index = -1
    required_return = estimated_round_trip_cost + safety_margin

    for index, probability in enumerate(probabilities):
        if pending_exit:
            position_open = False
            pending_exit = False
        if pending_entry:
            position_open = True
            entry_price = signal_opens[index]
            entry_execution_index = index
            pending_entry = False

        if position_open:
            held_for_next_execution = index + 1 - entry_execution_index
            if stop_loss_pct is not None and signal_closes[index] <= entry_price * (1.0 - stop_loss_pct):
                actions[index] = -1
                reasons[index] = "STOP_LOSS"
                pending_exit = True
            elif maximum_holding_period is not None and held_for_next_execution >= maximum_holding_period:
                actions[index] = -1
                reasons[index] = "MAX_HOLDING_PERIOD"
                pending_exit = True
            elif (
                (probability <= exit_threshold if hysteresis else probability < exit_threshold)
                and held_for_next_execution >= minimum_holding_period
            ):
                actions[index] = -1
                reasons[index] = "SELL_SIGNAL"
                pending_exit = True
            continue

        if probability < entry_threshold:
            continue
        if cost_filter:
            estimated_return = float(expected[index])
            if not np.isfinite(estimated_return) or estimated_return <= required_return:
                rejections.append({
                    "signal_index": index,
                    "predicted_probability": float(probability),
                    "estimated_expected_return": estimated_return,
                    "estimated_transaction_cost": estimated_round_trip_cost,
                    "minimum_required_return": required_return,
                    "rejection_reason": "EXPECTED_RETURN_NOT_ABOVE_COST_AND_MARGIN",
                })
                continue

        actions[index] = 1
        pending_entry = True

    return actions, reasons, rejections


def score_validation_configuration(
    simulation: Dict[str, Any],
    signal_count: int,
    drawdown_penalty: float = 0.50,
    frequency_penalty: float = 0.10,
) -> Dict[str, float]:
    """Rank validation runs by net return while accounting for drawdown and churn."""
    if signal_count <= 0:
        raise ValueError("signal_count must be positive.")
    if drawdown_penalty < 0 or frequency_penalty < 0:
        raise ValueError("selection penalties must be non-negative.")

    trades_per_100 = float(simulation["closed_trades"]) / signal_count * 100.0
    net_return = float(simulation["total_return_pct"])
    drawdown = abs(float(simulation["max_drawdown_pct"]))
    score = net_return - drawdown_penalty * drawdown - frequency_penalty * trades_per_100
    return {
        "score": score,
        "net_return_pct": net_return,
        "absolute_max_drawdown_pct": drawdown,
        "closed_trades_per_100_signals": trades_per_100,
    }