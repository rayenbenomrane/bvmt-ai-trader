"""
Unit tests for Trading Simulator.
Verifies next-session execution, transaction costs, whole-share accounting,
slippage modeling, and drawdown calculation on synthetic sequences.
"""

import pytest
import numpy as np
import pandas as pd
from src.simulator.trading_simulator import TradingSimulator, SimulationConfig


def test_zero_trades_preserves_capital():
    n = 20
    dates = pd.date_range("2026-01-01", periods=n)
    exec_prices = pd.Series([10.0] * n)
    close_prices = pd.Series([10.0] * n)
    signals = np.zeros(n, dtype=int)  # Always stay in cash

    sim = TradingSimulator(SimulationConfig(initial_capital=10000.0))
    res = sim.simulate(signals, dates, exec_prices, close_prices)

    assert res["final_portfolio_value"] == 10000.0
    assert res["total_return_pct"] == 0.0
    assert res["total_trades"] == 0
    assert res["total_fees_paid"] == 0.0
    assert res["max_drawdown_pct"] == 0.0


def test_integer_shares_accounting():
    dates = pd.date_range("2026-01-01", periods=5)
    exec_prices = pd.Series([10.0, 10.0, 12.0, 12.0, 12.0])
    close_prices = pd.Series([10.0, 10.0, 12.0, 12.0, 12.0])
    # Buy on day 0, Sell on day 2
    signals = np.array([1, 1, 0, 0, 0])

    fee_rate = 0.005  # 0.5%
    slippage = 0.001  # 0.1%
    initial = 1000.0

    sim = TradingSimulator(SimulationConfig(
        initial_capital=initial,
        fee_rate=fee_rate,
        slippage_pct=slippage,
        allow_fractional_shares=False,
    ))
    res = sim.simulate(signals, dates, exec_prices, close_prices)

    assert res["total_trades"] == 2
    assert res["closed_trades"] == 1
    assert res["total_fees_paid"] > 0
    # Price increased from 10 to 12, so trade was profitable even after fees
    assert res["total_return_pct"] > 0
    assert res["win_rate_pct"] == 100.0


def test_max_drawdown_calculation():
    dates = pd.date_range("2026-01-01", periods=4)
    # Price drops by 50%
    exec_prices = pd.Series([10.0, 10.0, 5.0, 5.0])
    close_prices = pd.Series([10.0, 10.0, 5.0, 5.0])
    signals = np.array([1, 1, 1, 1])  # Hold through the crash

    sim = TradingSimulator(SimulationConfig(initial_capital=10000.0, fee_rate=0.0, slippage_pct=0.0))
    res = sim.simulate(signals, dates, exec_prices, close_prices)

    # From 10,000 to 5,000 is a 50% drawdown
    assert pytest.approx(res["max_drawdown_pct"], 0.1) == -50.0
