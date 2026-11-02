"""
BVMT AI Trader - Realistic Trading Simulator & Backtester

Implements rigorous, leak-free trading simulation for the Bourse de Tunis (BVMT):
- Next-session execution (orders execute at session t+1 Open to eliminate look-ahead bias).
- Distinct tracking of signal generation date (session t close) and execution date (session t+1 open).
- Configurable transaction fee rate and execution slippage.
- Integer share accounting (no fictitious fractional shares on BVMT).
- Exact cash, share quantity, and realized/unrealized P&L balance tracking.
- Trade-level closed round-trip auditing (gross P&L, fees, net P&L, duration).
- Buy-and-Hold benchmark with identical cost model, daily equity curve, and Buy & Hold Max Drawdown.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    initial_capital: float = 10000.0
    fee_rate: float = 0.0040  # 0.40% per trade execution (brokerage + exchange fee + VAT)
    slippage_pct: float = 0.0010  # 0.10% slippage on fill
    execution_price: str = "open"  # "open" (next-session open execution) or "close"
    allow_fractional_shares: bool = False
    auto_close_at_end: bool = True  # Liquidate open position on final session close


@dataclass
class OrderExecutionRecord:
    order_id: int
    session_index: int
    signal_date: Any
    execution_date: Any
    action: str  # "BUY" or "SELL"
    shares: int
    fill_price: float
    gross_amount: float
    fee: float
    total_amount: float
    cash_after: float
    reason: str = ""


@dataclass
class ClosedTradeRecord:
    trade_id: int
    shares: int
    entry_signal_date: Any
    entry_date: Any
    exit_signal_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    holding_sessions: int
    gross_amount_invested: float
    gross_proceeds: float
    gross_pnl: float
    gross_return_pct: float
    entry_fee: float
    exit_fee: float
    total_fees: float
    net_pnl: float
    net_return_pct: float
    is_gross_win: bool
    is_net_win: bool


class TradingSimulator:
    """
    Simulates trading strategies chronologically with next-session execution.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()

    def simulate(
        self,
        signals: np.ndarray,
        dates: pd.Series,
        exec_prices: pd.Series,
        close_prices: pd.Series,
        signal_dates: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Runs the simulation.
        
        Args:
            signals: Array of binary signals (1 = LONG, 0 = CASH) generated at session t close.
            dates: Series of session t+1 execution dates (or session t dates if signal_dates provided).
            exec_prices: Execution prices for session t+1 (typically Open price).
            close_prices: Close prices for session t+1 (used for mark-to-market valuation).
            signal_dates: Optional Series of session t signal generation dates.
            
        Returns:
            Dictionary containing performance metrics, equity curves, order logs, and closed trade audit.
        """
        n = len(signals)
        if n == 0 or len(exec_prices) != n or len(close_prices) != n:
            raise ValueError("Mismatched or empty input series for simulation.")

        # Determine signal and execution dates cleanly
        if signal_dates is not None and len(signal_dates) == n:
            sig_dates = signal_dates
            exec_dates = dates
        else:
            sig_dates = dates
            exec_dates = dates

        capital = self.config.initial_capital
        cash = capital
        shares: int = 0
        total_fees = 0.0
        
        orders: List[OrderExecutionRecord] = []
        closed_trades: List[ClosedTradeRecord] = []
        portfolio_values = []
        daily_cash = []
        daily_shares = []

        # Current open position tracking for trade pairing
        open_trade_shares = 0
        open_entry_price = 0.0
        open_entry_fee = 0.0
        open_entry_sig_date = None
        open_entry_exec_date = None
        open_entry_idx = 0

        for i in range(n):
            sig_dt = sig_dates.iloc[i] if hasattr(sig_dates, "iloc") else sig_dates[i]
            exec_dt = exec_dates.iloc[i] if hasattr(exec_dates, "iloc") else exec_dates[i]
            p_exec = float(exec_prices.iloc[i] if hasattr(exec_prices, "iloc") else exec_prices[i])
            p_close = float(close_prices.iloc[i] if hasattr(close_prices, "iloc") else close_prices[i])
            sig = int(signals[i])

            # 1. Entry Execution: Signal generated at i (session t close) fills at session t+1 open
            if sig == 1 and shares == 0 and cash > 0:
                fill_price = p_exec * (1.0 + self.config.slippage_pct)
                effective_cost_per_share = fill_price * (1.0 + self.config.fee_rate)

                if self.config.allow_fractional_shares:
                    buy_shares = cash / effective_cost_per_share
                else:
                    buy_shares = int(cash // effective_cost_per_share)

                if buy_shares > 0:
                    gross_cost = buy_shares * fill_price
                    fee = gross_cost * self.config.fee_rate
                    total_paid = gross_cost + fee
                    cash -= total_paid
                    shares += int(buy_shares) if not self.config.allow_fractional_shares else buy_shares
                    total_fees += fee

                    open_trade_shares = shares
                    open_entry_price = fill_price
                    open_entry_fee = fee
                    open_entry_sig_date = sig_dt
                    open_entry_exec_date = exec_dt
                    open_entry_idx = i

                    orders.append(
                        OrderExecutionRecord(
                            order_id=len(orders) + 1,
                            session_index=i,
                            signal_date=sig_dt,
                            execution_date=exec_dt,
                            action="BUY",
                            shares=int(buy_shares),
                            fill_price=round(fill_price, 4),
                            gross_amount=round(gross_cost, 4),
                            fee=round(fee, 4),
                            total_amount=round(total_paid, 4),
                            cash_after=round(cash, 4),
                            reason="Long Entry (Signal 1)",
                        )
                    )

            # 2. Exit Execution: Signal 0 closes existing long position
            elif sig == 0 and shares > 0:
                fill_price = p_exec * (1.0 - self.config.slippage_pct)
                gross_proceeds = shares * fill_price
                fee = gross_proceeds * self.config.fee_rate
                net_received = gross_proceeds - fee
                cash += net_received
                total_fees += fee

                # Construct ClosedTradeRecord
                gross_pnl = (fill_price - open_entry_price) * shares
                total_trade_fees = open_entry_fee + fee
                net_pnl = gross_pnl - total_trade_fees
                gross_ret_pct = ((fill_price - open_entry_price) / open_entry_price) * 100.0
                capital_at_risk = (open_entry_price * shares) + open_entry_fee
                net_ret_pct = (net_pnl / capital_at_risk) * 100.0

                closed_trades.append(
                    ClosedTradeRecord(
                        trade_id=len(closed_trades) + 1,
                        shares=int(shares),
                        entry_signal_date=open_entry_sig_date,
                        entry_date=open_entry_exec_date,
                        exit_signal_date=sig_dt,
                        exit_date=exec_dt,
                        entry_price=round(open_entry_price, 4),
                        exit_price=round(fill_price, 4),
                        holding_sessions=i - open_entry_idx,
                        gross_amount_invested=round(open_entry_price * shares, 3),
                        gross_proceeds=round(gross_proceeds, 3),
                        gross_pnl=round(gross_pnl, 3),
                        gross_return_pct=round(gross_ret_pct, 2),
                        entry_fee=round(open_entry_fee, 3),
                        exit_fee=round(fee, 3),
                        total_fees=round(total_trade_fees, 3),
                        net_pnl=round(net_pnl, 3),
                        net_return_pct=round(net_ret_pct, 2),
                        is_gross_win=gross_pnl > 0,
                        is_net_win=net_pnl > 0,
                    )
                )

                orders.append(
                    OrderExecutionRecord(
                        order_id=len(orders) + 1,
                        session_index=i,
                        signal_date=sig_dt,
                        execution_date=exec_dt,
                        action="SELL",
                        shares=int(shares),
                        fill_price=round(fill_price, 4),
                        gross_amount=round(gross_proceeds, 4),
                        fee=round(fee, 4),
                        total_amount=round(net_received, 4),
                        cash_after=round(cash, 4),
                        reason="Exit to Cash (Signal 0)",
                    )
                )

                shares = 0
                open_trade_shares = 0

            # 3. Mark-to-market portfolio valuation at session close
            port_val = cash + (shares * p_close)
            portfolio_values.append(port_val)
            daily_cash.append(cash)
            daily_shares.append(shares)

        # 4. Handle any remaining open position at end of simulation if configured
        if shares > 0 and self.config.auto_close_at_end:
            p_final = float(close_prices.iloc[-1] if hasattr(close_prices, "iloc") else close_prices[-1])
            fill_price = p_final * (1.0 - self.config.slippage_pct)
            gross_proceeds = shares * fill_price
            fee = gross_proceeds * self.config.fee_rate
            net_received = gross_proceeds - fee
            cash += net_received
            total_fees += fee

            gross_pnl = (fill_price - open_entry_price) * shares
            total_trade_fees = open_entry_fee + fee
            net_pnl = gross_pnl - total_trade_fees
            gross_ret_pct = ((fill_price - open_entry_price) / open_entry_price) * 100.0
            capital_at_risk = (open_entry_price * shares) + open_entry_fee
            net_ret_pct = (net_pnl / capital_at_risk) * 100.0

            closed_trades.append(
                ClosedTradeRecord(
                    trade_id=len(closed_trades) + 1,
                    shares=int(shares),
                    entry_signal_date=open_entry_sig_date,
                    entry_date=open_entry_exec_date,
                    exit_signal_date=sig_dates.iloc[-1] if hasattr(sig_dates, "iloc") else sig_dates[-1],
                    exit_date=exec_dates.iloc[-1] if hasattr(exec_dates, "iloc") else exec_dates[-1],
                    entry_price=round(open_entry_price, 4),
                    exit_price=round(fill_price, 4),
                    holding_sessions=n - 1 - open_entry_idx,
                    gross_amount_invested=round(open_entry_price * shares, 3),
                    gross_proceeds=round(gross_proceeds, 3),
                    gross_pnl=round(gross_pnl, 3),
                    gross_return_pct=round(gross_ret_pct, 2),
                    entry_fee=round(open_entry_fee, 3),
                    exit_fee=round(fee, 3),
                    total_fees=round(total_trade_fees, 3),
                    net_pnl=round(net_pnl, 3),
                    net_return_pct=round(net_ret_pct, 2),
                    is_gross_win=gross_pnl > 0,
                    is_net_win=net_pnl > 0,
                )
            )
            shares = 0
            portfolio_values[-1] = cash

        # Final portfolio valuation
        final_val = portfolio_values[-1] if portfolio_values else capital
        total_return_pct = ((final_val - capital) / capital) * 100.0

        # Maximum Drawdown calculation for strategy (anchored to starting capital peak)
        equity_with_start = pd.concat([pd.Series([capital]), pd.Series(portfolio_values)], ignore_index=True)
        rolling_peak = equity_with_start.cummax()
        drawdowns = (equity_with_start - rolling_peak) / rolling_peak
        max_drawdown_pct = float(drawdowns.min() * 100.0) if not drawdowns.empty else 0.0

        # Buy-and-Hold Benchmark: identical entry assumptions, cost model, and daily equity curve
        first_exec = float(exec_prices.iloc[0] if hasattr(exec_prices, "iloc") else exec_prices[0])
        bh_fill_price = first_exec * (1.0 + self.config.slippage_pct)
        bh_cost_per_share = bh_fill_price * (1.0 + self.config.fee_rate)

        if self.config.allow_fractional_shares:
            bh_shares = capital / bh_cost_per_share
            bh_cash = 0.0
        else:
            bh_shares = int(capital // bh_cost_per_share)
            bh_cash = capital - (bh_shares * bh_cost_per_share)

        bh_equity_curve = []
        for i in range(n):
            p_c = float(close_prices.iloc[i] if hasattr(close_prices, "iloc") else close_prices[i])
            bh_equity_curve.append(bh_cash + (bh_shares * p_c))

        bh_with_start = pd.concat([pd.Series([capital]), pd.Series(bh_equity_curve)], ignore_index=True)
        bh_peak = bh_with_start.cummax()
        bh_drawdowns = (bh_with_start - bh_peak) / bh_peak
        bh_max_drawdown_pct = float(bh_drawdowns.min() * 100.0) if not bh_drawdowns.empty else 0.0

        bh_final_val = bh_equity_curve[-1] if bh_equity_curve else capital
        bh_return_pct = ((bh_final_val - capital) / capital) * 100.0

        # Closed Trade Performance Aggregates
        n_closed = len(closed_trades)
        gross_wins = sum(1 for t in closed_trades if t.is_gross_win)
        net_wins = sum(1 for t in closed_trades if t.is_net_win)
        gross_win_rate = (gross_wins / n_closed * 100.0) if n_closed > 0 else 0.0
        net_win_rate = (net_wins / n_closed * 100.0) if n_closed > 0 else 0.0
        total_gross_pnl = sum(t.gross_pnl for t in closed_trades)
        total_net_pnl = sum(t.net_pnl for t in closed_trades)
        avg_holding_sessions = (sum(t.holding_sessions for t in closed_trades) / n_closed) if n_closed > 0 else 0.0

        return {
            "initial_capital": capital,
            "final_portfolio_value": round(float(final_val), 3),
            "total_return_pct": round(float(total_return_pct), 2),
            "buy_and_hold_return_pct": round(float(bh_return_pct), 2),
            "max_drawdown_pct": round(float(max_drawdown_pct), 2),
            "buy_and_hold_max_drawdown_pct": round(float(bh_max_drawdown_pct), 2),
            "total_orders": len(orders),
            "closed_trades": n_closed,
            "gross_win_rate_pct": round(float(gross_win_rate), 2),
            "net_win_rate_pct": round(float(net_win_rate), 2),
            "total_gross_pnl": round(float(total_gross_pnl), 3),
            "total_net_pnl": round(float(total_net_pnl), 3),
            "total_fees_paid": round(float(total_fees), 3),
            "average_holding_sessions": round(float(avg_holding_sessions), 2),
            "fee_rate": self.config.fee_rate,
            "slippage_pct": self.config.slippage_pct,
            "equity_curve": portfolio_values,
            "bh_equity_curve": bh_equity_curve,
            "orders": [asdict(o) for o in orders],
            "closed_trades_records": [asdict(t) for t in closed_trades],
        }
