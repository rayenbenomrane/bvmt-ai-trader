"""
Unit tests for Trading Simulator & Backtest Audit (Part 6.4).
Verifies:
- Transaction costs (fees & slippage) applied accurately without double counting.
- Integer share position sizing and cash conservation.
- Next-session execution timing (signal at t, execution at t+1).
- Drawdown mathematics (including when Max Drawdown == Final Return).
- Buy & Hold equity curve and Max Drawdown calculation.
- Trade-level closed round-trip auditing (gross P&L, fees, net P&L).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from src.simulator.trading_simulator import (
    TradingSimulator,
    SimulationConfig,
    OrderExecutionRecord,
    ClosedTradeRecord,
)
from src.simulator.strategy import (
    assess_validation_edge,
    build_strategy_actions,
    meets_validation_robustness_criteria,
    score_validation_configuration,
    select_validation_candidate,
)


def test_price_only_fills_have_no_commissions_or_slippage():
    """Buy and sell cash movements equal the unadjusted execution amounts."""
    initial_cash = 10000.0
    open_price = 10.0

    sim = TradingSimulator(SimulationConfig(
        initial_capital=initial_cash,
        allow_fractional_shares=False,
    ))

    # Single buy signal on day 0, exit on day 1
    signals = np.array([1, 0])
    signal_dates = pd.Series([date(2026, 1, 1), date(2026, 1, 2)])
    exec_dates = pd.Series([date(2026, 1, 2), date(2026, 1, 5)])
    exec_prices = pd.Series([open_price, 12.0])
    close_prices = pd.Series([open_price, 12.0])

    res = sim.simulate(
        signals=signals,
        dates=exec_dates,
        exec_prices=exec_prices,
        close_prices=close_prices,
        signal_dates=signal_dates,
    )

    orders = res["orders"]
    assert len(orders) == 2
    buy_order = orders[0]
    sell_order = orders[1]

    expected_shares = int(initial_cash // open_price)
    expected_cash_after_buy = initial_cash - expected_shares * open_price

    assert buy_order["shares"] == expected_shares
    assert buy_order["fill_price"] == open_price
    assert buy_order["total_amount"] == expected_shares * open_price
    assert pytest.approx(buy_order["cash_after"], 0.01) == expected_cash_after_buy

    sell_open = 12.0
    expected_sell_fill = sell_open
    expected_sell_gross = expected_shares * expected_sell_fill
    expected_net_received = expected_sell_gross
    expected_final_cash = expected_cash_after_buy + expected_net_received

    assert sell_order["shares"] == expected_shares
    assert sell_order["fill_price"] == expected_sell_fill
    assert sell_order["total_amount"] == expected_sell_gross
    assert pytest.approx(sell_order["cash_after"], 0.01) == expected_final_cash
    assert all(order["cash_after"] >= 0 for order in orders)
    assert "fee" not in buy_order and "slippage_cost" not in buy_order


def test_integer_shares_and_cash_conservation():
    """Confirms no fractional shares and exact conservation of cash balance."""
    sim = TradingSimulator(SimulationConfig(
        initial_capital=500.0,
        allow_fractional_shares=False,
    ))

    # Asset price 13.5 TND: 500 TND can only afford ~36 shares
    signals = np.array([1, 1, 0])
    dates = pd.date_range("2026-01-01", periods=3)
    exec_prices = pd.Series([13.5, 13.5, 14.0])
    close_prices = pd.Series([13.5, 13.5, 14.0])

    res = sim.simulate(signals, dates, exec_prices, close_prices)
    orders = res["orders"]
    buy = orders[0]

    assert isinstance(buy["shares"], int)
    assert buy["shares"] * 13.5 < 500.0
    # Cash after buy must equal initial minus total paid
    assert pytest.approx(500.0 - buy["total_amount"], 0.001) == buy["cash_after"]


def test_execution_timing_alignment():
    """Confirms signal generated at session t executes at session t+1 with session t+1 date."""
    sig_dates = pd.Series([date(2026, 2, 6), date(2026, 2, 9)])
    exec_dates = pd.Series([date(2026, 2, 9), date(2026, 2, 10)])
    exec_prices = pd.Series([10.0, 10.5])
    close_prices = pd.Series([10.2, 10.7])
    signals = np.array([1, 0])

    sim = TradingSimulator(SimulationConfig())
    res = sim.simulate(
        signals=signals,
        dates=exec_dates,
        exec_prices=exec_prices,
        close_prices=close_prices,
        signal_dates=sig_dates,
    )

    orders = res["orders"]
    # Signal date was 2026-02-06 (Friday close), execution date must be 2026-02-09 (Monday open)
    assert orders[0]["signal_date"] == date(2026, 2, 6)
    assert orders[0]["execution_date"] == date(2026, 2, 9)
    assert orders[0]["action"] == "BUY"


def test_drawdown_equals_loss_when_no_new_highs():
    """Investigates and confirms why Max Drawdown equals Final Return when equity never exceeds initial capital."""
    capital = 10000.0
    signals = np.array([1, 1, 1])
    dates = pd.date_range("2026-01-01", periods=3)
    # Monotonically declining prices: initial 10, then 9, then 8
    exec_prices = pd.Series([10.0, 9.0, 8.0])
    close_prices = pd.Series([9.0, 8.0, 7.0])

    sim = TradingSimulator(SimulationConfig(initial_capital=capital, allow_fractional_shares=True))
    res = sim.simulate(signals, dates, exec_prices, close_prices)

    # Initial peak is 10,000 at start. It never rises above 10,000.
    # Lowest point is at the end: 7,000.
    # Therefore, max_drawdown = (7000 - 10000) / 10000 = -30.0% == final_return.
    assert pytest.approx(res["max_drawdown_pct"], 0.01) == res["total_return_pct"]
    assert pytest.approx(res["max_drawdown_pct"], 0.01) == -30.0


def test_buy_and_hold_equity_curve_and_drawdown():
    """Confirms Buy & Hold benchmark tracks daily equity curve and calculates B&H Max Drawdown."""
    capital = 10000.0
    signals = np.zeros(4, dtype=int)  # Strategy holds cash
    dates = pd.date_range("2026-01-01", periods=4)
    # Asset starts at 10, rises to 15, drops to 9, ends at 12
    exec_prices = pd.Series([10.0, 15.0, 9.0, 12.0])
    close_prices = pd.Series([10.0, 15.0, 9.0, 12.0])

    sim = TradingSimulator(SimulationConfig(initial_capital=capital, allow_fractional_shares=True))
    res = sim.simulate(signals, dates, exec_prices, close_prices)

    # Strategy held cash
    assert res["total_return_pct"] == 0.0
    # Buy and hold started at 10, ended at 12 (+20%)
    assert pytest.approx(res["buy_and_hold_return_pct"], 0.01) == 20.0
    # Peak was 15,000; trough was 9,000 -> Drawdown from peak was (9000 - 15000) / 15000 = -40%
    assert pytest.approx(res["buy_and_hold_max_drawdown_pct"], 0.01) == -40.0


def test_closed_trade_pnl_and_portfolio_value_reconcile_without_costs():
    """Gross and net P&L are equal when no commissions or slippage exist."""
    sim = TradingSimulator(SimulationConfig(
        initial_capital=10000.0,
        allow_fractional_shares=False,
    ))

    signals = np.array([1, 0])
    dates = pd.date_range("2026-01-01", periods=2)
    exec_prices = pd.Series([10.0, 12.0])
    close_prices = pd.Series([10.0, 12.0])

    res = sim.simulate(signals, dates, exec_prices, close_prices)
    closed = res["closed_trades_records"]
    assert len(closed) == 1
    t = closed[0]

    assert t["net_pnl"] == t["gross_pnl"]
    assert t["is_net_win"] == (t["net_pnl"] > 0)
    assert t["holding_sessions"] == 1
    assert res["gross_win_rate_basis"] == "price-only, no commissions or slippage"
    assert res["total_net_pnl"] == res["total_gross_pnl"]
    assert res["final_portfolio_value"] == pytest.approx(
        res["initial_capital"] + res["total_net_pnl"], abs=0.01
    )


def test_minimum_holding_and_end_of_window_liquidation_exception():
    actions = np.array([1, 0, -1])
    dates = pd.date_range("2026-03-01", periods=3)
    result = TradingSimulator(SimulationConfig(
        initial_capital=10000.0,
        minimum_holding_period=3,
    )).simulate(
        signals=np.zeros(3, dtype=int),
        actions=actions,
        exit_reasons=["", "", "SELL_SIGNAL"],
        dates=dates,
        exec_prices=pd.Series([10.0, 11.0, 12.0]),
        close_prices=pd.Series([10.0, 11.0, 12.0]),
    )

    trade = result["closed_trades_records"][0]
    assert trade["holding_sessions"] == 2
    assert trade["exit_reason"] == "END_OF_TEST"


def test_maximum_hold_exit_fills_at_next_open_and_counts_open_to_open_sessions():
    dates = pd.date_range("2026-04-01", periods=5)
    result = TradingSimulator(SimulationConfig(
        initial_capital=10000.0,
        minimum_holding_period=3,
    )).simulate(
        signals=np.zeros(5, dtype=int),
        actions=np.array([1, 0, 0, -1, 0]),
        exit_reasons=["", "", "", "MAX_HOLDING_PERIOD", ""],
        dates=dates,
        exec_prices=pd.Series([10.0, 11.0, 12.0, 13.0, 14.0]),
        close_prices=pd.Series([10.0, 11.0, 12.0, 13.0, 14.0]),
    )

    sell = result["orders"][1]
    trade = result["closed_trades_records"][0]
    assert sell["execution_date"] == dates[3]
    assert sell["fill_price"] == 13.0
    assert trade["holding_sessions"] == 3
    assert trade["exit_reason"] == "MAX_HOLDING_PERIOD"


def test_hysteresis_holds_between_thresholds_and_enforces_minimum_period():
    actions, reasons = build_strategy_actions(
        probabilities=[0.70, 0.30, 0.30, 0.30, 0.70],
        signal_opens=[10, 10, 10, 10, 10],
        signal_closes=[10, 10, 10, 10, 10],
        minimum_holding_period=3,
    )
    np.testing.assert_array_equal(actions, [1, 0, 0, -1, 1])
    assert reasons[3] == "SELL_SIGNAL"


def test_stop_loss_uses_signal_close_and_max_hold_exits():
    actions, reasons = build_strategy_actions(
        probabilities=[0.70, 0.70, 0.70, 0.70],
        signal_opens=[10, 10, 10, 10],
        signal_closes=[10, 10, 8.9, 10],
        minimum_holding_period=3,
        stop_loss_pct=0.10,
    )
    assert actions[0] == 1
    assert actions[2] == -1
    assert reasons[2] == "STOP_LOSS"

    actions, reasons = build_strategy_actions(
        probabilities=[0.70] * 5,
        signal_opens=[10] * 5,
        signal_closes=[10] * 5,
        minimum_holding_period=3,
        maximum_holding_period=3,
    )
    assert actions[3] == -1
    assert reasons[3] == "MAX_HOLDING_PERIOD"


def test_validation_score_uses_return_drawdown_and_trade_frequency():
    score = score_validation_configuration({
        "total_return_pct": 5.0,
        "buy_and_hold_return_pct": 2.0,
        "max_drawdown_pct": -4.0,
        "closed_trades": 2,
    }, signal_count=100)
    assert score["score"] == pytest.approx(0.8)
    assert score["excess_return_pct"] == 3.0
    assert score["closed_trades_per_100_signals"] == 2.0


def test_tied_validation_candidates_use_deterministic_simplicity_tiebreak():
    shared = {
        "validation_score": {"score": 6.643},
        "validation_simulation": {
            "closed_trades": 12,
            "total_orders": 24,
            "max_drawdown_pct": -4.0,
        },
        "horizon": 3,
        "target_col": "target_binary",
        "model_name": "Logistic Regression",
    }
    complex_candidate = {
        **shared,
        "config_key": "a-complex",
        "strategy": {
            "hysteresis": True,
            "minimum_holding_period": 3,
            "maximum_holding_period": 3,
        },
    }
    simple_candidate = {
        **shared,
        "config_key": "z-simple",
        "strategy": {
            "hysteresis": False,
            "minimum_holding_period": 0,
            "maximum_holding_period": None,
        },
    }

    selected, excluded, tie_break = select_validation_candidate(
        [complex_candidate, simple_candidate]
    )
    assert selected is simple_candidate
    assert excluded == 0
    assert tie_break == "simpler strategy (no hysteresis)"


def test_candidate_selection_ignores_test_metrics():
    candidates = [
        {
            "validation_score": {"score": 2.0},
            "validation_simulation": {
                "closed_trades": 12,
                "total_orders": 24,
                "max_drawdown_pct": -4.0,
            },
            "test_return_pct": 1000.0,
            "strategy": {"hysteresis": False, "minimum_holding_period": 0, "maximum_holding_period": None},
            "config_key": "validation-worse-test-better",
        },
        {
            "validation_score": {"score": 3.0},
            "validation_simulation": {
                "closed_trades": 12,
                "total_orders": 24,
                "max_drawdown_pct": -4.0,
            },
            "test_return_pct": -1000.0,
            "strategy": {"hysteresis": False, "minimum_holding_period": 0, "maximum_holding_period": None},
            "config_key": "validation-better-test-worse",
        },
    ]

    selected, _, _ = select_validation_candidate(candidates)
    assert selected is candidates[1]


def test_validation_edge_assessment_flags_benchmark_underperformance():
    candidates = [{
        "validation_simulation": {"closed_trades": 10, "total_return_pct": 4.0},
    }]
    assessment = assess_validation_edge(
        candidates,
        buy_and_hold_return_pct=6.0,
        legacy_return_pct=5.0,
    )
    assert assessment["no_candidate_beats_buy_and_hold"] is True
    assert assessment["no_candidate_beats_legacy"] is True
    assert assessment["best_validation_return_pct"] == 4.0

    assessment = assess_validation_edge([], buy_and_hold_return_pct=0.0, legacy_return_pct=0.0)
    assert assessment["eligible_candidate_count"] == 0
    assert assessment["best_validation_return_pct"] is None
    assert assessment["no_candidate_beats_buy_and_hold"] is True


def test_robustness_gate_can_require_repeatable_benchmark_wins():
    candidate = {
        "validation_simulation": {"closed_trades": 12},
        "validation_robustness": {"buy_and_hold_win_rate": 0.75, "legacy_win_rate": 0.50},
    }
    assert meets_validation_robustness_criteria(candidate)
    candidate["validation_robustness"]["buy_and_hold_win_rate"] = 0.25
    assert not meets_validation_robustness_criteria(candidate)
    candidate["validation_robustness"]["buy_and_hold_win_rate"] = 1.0
    candidate["validation_simulation"]["closed_trades"] = 9
    assert not meets_validation_robustness_criteria(candidate)


def test_minimum_validation_trades_excludes_candidates_and_allows_cash_fallback():
    candidates = [{
        "validation_score": {"score": 100.0},
        "validation_simulation": {
            "closed_trades": 9,
            "total_orders": 18,
            "max_drawdown_pct": -1.0,
        },
        "strategy": {"hysteresis": False, "minimum_holding_period": 0, "maximum_holding_period": None},
    }]
    selected, excluded, reason = select_validation_candidate(candidates)
    assert selected is None
    assert excluded == 1
    assert "no reliable edge on validation" in reason

    candidates[0]["validation_simulation"]["closed_trades"] = 10
    selected, excluded, _ = select_validation_candidate(candidates)
    assert selected is candidates[0]
    assert excluded == 0
