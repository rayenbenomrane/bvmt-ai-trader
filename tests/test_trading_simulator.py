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
from src.simulator.strategy import build_strategy_actions, score_validation_configuration


def test_transaction_cost_and_slippage_accuracy():
    """Confirms fee and slippage apply on entries and exits without double counting."""
    initial_cash = 10000.0
    fee_rate = 0.0040  # 0.40%
    slippage = 0.0010  # 0.10%
    open_price = 10.0

    sim = TradingSimulator(SimulationConfig(
        initial_capital=initial_cash,
        fee_rate=fee_rate,
        slippage_pct=slippage,
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

    # Verify BUY math
    expected_buy_fill = open_price * (1.0 + slippage)  # 10.010
    expected_cost_per_share = expected_buy_fill * (1.0 + fee_rate)  # 10.05004
    expected_shares = int(initial_cash // expected_cost_per_share)  # 995 shares
    expected_buy_gross = expected_shares * expected_buy_fill  # 995 * 10.010 = 9959.95
    expected_buy_fee = expected_buy_gross * fee_rate  # 39.8398
    expected_cash_after_buy = initial_cash - (expected_buy_gross + expected_buy_fee)

    assert buy_order["shares"] == expected_shares
    assert pytest.approx(buy_order["fill_price"], 0.0001) == expected_buy_fill
    assert pytest.approx(buy_order["fee"], 0.01) == expected_buy_fee
    assert pytest.approx(buy_order["cash_after"], 0.01) == expected_cash_after_buy

    # Verify SELL math
    sell_open = 12.0
    expected_sell_fill = sell_open * (1.0 - slippage)  # 11.988
    expected_sell_gross = expected_shares * expected_sell_fill  # 995 * 11.988 = 11928.06
    expected_sell_fee = expected_sell_gross * fee_rate  # 47.71224
    expected_net_received = expected_sell_gross - expected_sell_fee
    expected_final_cash = expected_cash_after_buy + expected_net_received

    assert sell_order["shares"] == expected_shares
    assert pytest.approx(sell_order["fill_price"], 0.0001) == expected_sell_fill
    assert pytest.approx(sell_order["fee"], 0.01) == expected_sell_fee
    assert pytest.approx(sell_order["cash_after"], 0.01) == expected_final_cash


def test_integer_shares_and_cash_conservation():
    """Confirms no fractional shares and exact conservation of cash balance."""
    sim = TradingSimulator(SimulationConfig(
        initial_capital=500.0,
        fee_rate=0.0040,
        slippage_pct=0.0010,
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

    sim = TradingSimulator(SimulationConfig(initial_capital=capital, fee_rate=0.0, slippage_pct=0.0, allow_fractional_shares=True))
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

    sim = TradingSimulator(SimulationConfig(initial_capital=capital, fee_rate=0.0, slippage_pct=0.0, allow_fractional_shares=True))
    res = sim.simulate(signals, dates, exec_prices, close_prices)

    # Strategy held cash
    assert res["total_return_pct"] == 0.0
    # Buy and hold started at 10, ended at 12 (+20%)
    assert pytest.approx(res["buy_and_hold_return_pct"], 0.01) == 20.0
    # Peak was 15,000; trough was 9,000 -> Drawdown from peak was (9000 - 15000) / 15000 = -40%
    assert pytest.approx(res["buy_and_hold_max_drawdown_pct"], 0.01) == -40.0


def test_closed_trade_audit_pnl_math():
    """Confirms closed round-trip trade P&L matches: Net P&L = Gross P&L - Total Fees."""
    sim = TradingSimulator(SimulationConfig(
        initial_capital=10000.0,
        fee_rate=0.0050,
        slippage_pct=0.0010,
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

    assert pytest.approx(t["net_pnl"], 0.01) == (t["gross_pnl"] - t["total_fees"])
    assert t["is_net_win"] == (t["net_pnl"] > 0)
    assert t["holding_sessions"] == 1


def test_hysteresis_holds_between_thresholds_and_enforces_minimum_period():
    actions, reasons, rejections = build_strategy_actions(
        probabilities=[0.70, 0.30, 0.30, 0.30, 0.70],
        signal_opens=[10, 10, 10, 10, 10],
        signal_closes=[10, 10, 10, 10, 10],
        minimum_holding_period=3,
    )
    np.testing.assert_array_equal(actions, [1, 0, 0, -1, 1])
    assert reasons[3] == "SELL_SIGNAL"
    assert rejections == []


def test_stop_loss_uses_signal_close_and_max_hold_exits():
    actions, reasons, _ = build_strategy_actions(
        probabilities=[0.70, 0.70, 0.70, 0.70],
        signal_opens=[10, 10, 10, 10],
        signal_closes=[10, 10, 8.9, 10],
        minimum_holding_period=3,
        stop_loss_pct=0.10,
    )
    assert actions[0] == 1
    assert actions[2] == -1
    assert reasons[2] == "STOP_LOSS"

    actions, reasons, _ = build_strategy_actions(
        probabilities=[0.70] * 5,
        signal_opens=[10] * 5,
        signal_closes=[10] * 5,
        minimum_holding_period=3,
        maximum_holding_period=3,
    )
    assert actions[3] == -1
    assert reasons[3] == "MAX_HOLDING_PERIOD"


def test_cost_filter_rejects_and_records_unprofitable_entry_estimate():
    actions, _, rejections = build_strategy_actions(
        probabilities=[0.80],
        signal_opens=[10],
        signal_closes=[10],
        expected_returns=[0.01],
        estimated_round_trip_cost=0.012,
        safety_margin=0.002,
        cost_filter=True,
    )
    assert actions[0] == 0
    assert rejections[0]["predicted_probability"] == 0.8
    assert rejections[0]["minimum_required_return"] == pytest.approx(0.014)


def test_validation_score_uses_return_drawdown_and_trade_frequency():
    score = score_validation_configuration({
        "total_return_pct": 5.0,
        "max_drawdown_pct": -4.0,
        "closed_trades": 2,
    }, signal_count=100)
    assert score["score"] == pytest.approx(2.8)
    assert score["closed_trades_per_100_signals"] == 2.0
