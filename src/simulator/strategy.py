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
    hysteresis: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Build BUY/HOLD/SELL decisions using only information known at each signal close."""
    probabilities = np.asarray(probabilities, dtype=float)
    signal_opens = np.asarray(signal_opens, dtype=float)
    signal_closes = np.asarray(signal_closes, dtype=float)
    n = len(probabilities)
    if len(signal_opens) != n or len(signal_closes) != n:
        raise ValueError("probabilities, opens, and closes must have equal lengths.")
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
    actions = np.zeros(n, dtype=np.int8)
    reasons = [""] * n
    position_open = False
    pending_entry = False
    pending_exit = False
    entry_price = 0.0
    entry_execution_index = -1

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
        actions[index] = 1
        pending_entry = True

    return actions, reasons


def score_validation_configuration(
    simulation: Dict[str, Any],
    signal_count: int,
    drawdown_penalty: float = 0.50,
    frequency_penalty: float = 0.10,
) -> Dict[str, float]:
    """Rank validation runs by excess return while accounting for drawdown and churn."""
    if signal_count <= 0:
        raise ValueError("signal_count must be positive.")
    if drawdown_penalty < 0 or frequency_penalty < 0:
        raise ValueError("selection penalties must be non-negative.")

    trades_per_100 = float(simulation["closed_trades"]) / signal_count * 100.0
    net_return = float(simulation["total_return_pct"])
    buy_and_hold_return = float(simulation["buy_and_hold_return_pct"])
    excess_return = net_return - buy_and_hold_return
    drawdown = abs(float(simulation["max_drawdown_pct"]))
    score = excess_return - drawdown_penalty * drawdown - frequency_penalty * trades_per_100
    return {
        "score": score,
        "net_return_pct": net_return,
        "buy_and_hold_return_pct": buy_and_hold_return,
        "excess_return_pct": excess_return,
        "absolute_max_drawdown_pct": drawdown,
        "closed_trades_per_100_signals": trades_per_100,
    }


def _validation_candidate_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    simulation = candidate["validation_simulation"]
    strategy = candidate["strategy"]
    has_holding_constraint = (
        strategy.get("minimum_holding_period", 0) > 0
        or strategy.get("maximum_holding_period") is not None
    )
    config_key = candidate.get("config_key") or "|".join((
        str(candidate.get("horizon", "cash")),
        str(candidate.get("target_col", "")),
        str(candidate.get("model_name", "")),
        str(strategy.get("entry_threshold", "")),
        str(strategy.get("exit_threshold", "")),
        str(strategy.get("hysteresis", False)),
        str(strategy.get("minimum_holding_period", 0)),
        str(strategy.get("maximum_holding_period", "")),
    ))
    return (
        -float(candidate["validation_score"]["score"]),
        abs(float(simulation["max_drawdown_pct"])),
        int(simulation["total_orders"]),
        bool(strategy.get("hysteresis", False)),
        has_holding_constraint,
        config_key.lower(),
    )


def select_validation_candidate(
    candidates: Sequence[Dict[str, Any]],
    min_validation_closed_trades: int = 10,
) -> Tuple[Optional[Dict[str, Any]], int, str]:
    """Apply the minimum-trade gate and deterministic validation-only tie-break."""
    if min_validation_closed_trades < 0:
        raise ValueError("min_validation_closed_trades must be non-negative.")
    eligible = [
        candidate for candidate in candidates
        if int(candidate["validation_simulation"]["closed_trades"]) >= min_validation_closed_trades
    ]
    excluded = len(candidates) - len(eligible)
    if not eligible:
        return None, excluded, "no reliable edge on validation: no candidate met minimum closed trades"

    ranked = sorted(eligible, key=_validation_candidate_key)
    selected = ranked[0]
    if len(ranked) == 1 or ranked[0]["validation_score"]["score"] != ranked[1]["validation_score"]["score"]:
        return selected, excluded, "higher validation score"

    selected_key = _validation_candidate_key(selected)
    runner_up_key = _validation_candidate_key(ranked[1])
    reasons = (
        "higher validation score",
        "lower absolute maximum drawdown",
        "fewer total orders",
        "simpler strategy (no hysteresis)",
        "simpler strategy (no holding constraints)",
        "alphabetical config key",
    )
    for index, (selected_value, runner_value) in enumerate(zip(selected_key, runner_up_key)):
        if selected_value != runner_value:
            return selected, excluded, reasons[index]
    return selected, excluded, "alphabetical config key"


def assess_validation_edge(
    candidates: Sequence[Dict[str, Any]],
    buy_and_hold_return_pct: float,
    legacy_return_pct: float,
    min_validation_closed_trades: int = 10,
) -> Dict[str, Any]:
    """Flag validation benchmark underperformance without consulting test results."""
    eligible = [
        candidate for candidate in candidates
        if int(candidate["validation_simulation"]["closed_trades"]) >= min_validation_closed_trades
    ]
    returns = [float(candidate["validation_simulation"]["total_return_pct"]) for candidate in eligible]
    best_return = max(returns) if returns else None
    beats_buy_and_hold = any(value > buy_and_hold_return_pct for value in returns)
    beats_legacy = any(value > legacy_return_pct for value in returns)
    return {
        "eligible_candidate_count": len(eligible),
        "best_validation_return_pct": best_return,
        "validation_buy_and_hold_return_pct": float(buy_and_hold_return_pct),
        "validation_legacy_return_pct": float(legacy_return_pct),
        "any_candidate_beats_buy_and_hold": beats_buy_and_hold,
        "any_candidate_beats_legacy": beats_legacy,
        "no_candidate_beats_buy_and_hold": not beats_buy_and_hold,
        "no_candidate_beats_legacy": not beats_legacy,
    }


def meets_validation_robustness_criteria(
    candidate: Dict[str, Any],
    min_validation_closed_trades: int = 10,
    min_buy_and_hold_win_rate: float = 0.50,
    min_legacy_win_rate: float = 0.50,
) -> bool:
    """Whether a candidate clears trade-count and cross-window consistency gates."""
    if min_validation_closed_trades < 0:
        raise ValueError("min_validation_closed_trades must be non-negative.")
    if not 0 <= min_buy_and_hold_win_rate <= 1 or not 0 <= min_legacy_win_rate <= 1:
        raise ValueError("validation win-rate criteria must be between zero and one.")
    robustness = candidate["validation_robustness"]
    return (
        int(candidate["validation_simulation"]["closed_trades"]) >= min_validation_closed_trades
        and float(robustness["buy_and_hold_win_rate"]) >= min_buy_and_hold_win_rate
        and float(robustness["legacy_win_rate"]) >= min_legacy_win_rate
    )