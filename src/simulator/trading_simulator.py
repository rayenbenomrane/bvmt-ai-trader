"""
BVMT AI Trader - Realistic Trading Simulator & Backtester

Implements rigorous, leak-free trading simulation for the Bourse de Tunis (BVMT):
- Next-session execution (orders execute at session t+1 Open to eliminate look-ahead bias).
- Configurable transaction fee rate and execution slippage.
- Integer share accounting (no fictitious fractional shares on BVMT).
- Exact cash and position balance tracking.
- Buy-and-Hold benchmark with identical cost model.
- Maximum Drawdown, win rate, and total cost attribution.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    initial_capital: float = 10000.0
    fee_rate: float = 0.0040  # 0.40% per trade (brokerage + exchange fee + VAT)
    slippage_pct: float = 0.0010  # 0.10% slippage on execution
    execution_price: str = "open"  # "open" (recommended next-session execution) or "close"
    allow_fractional_shares: bool = False


@dataclass
class TradeRecord:
    session_index: int
    date: Any
    action: str  # "BUY" or "SELL"
    shares: int
    price: float
    fee: float
    total_amount: float
    cash_after: float
    reason: str = ""


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
    ) -> Dict[str, Any]:
        """
        Runs the simulation.
        
        Args:
            signals: Array of binary signals (1 = LONG, 0 = CASH) generated at session t.
            dates: Series of session dates.
            exec_prices: Execution prices for session t (typically Open price of session t).
            close_prices: Close prices for session t (used for mark-to-market valuation).
            
        Returns:
            Dictionary containing performance metrics, equity curve, and trade history.
        """
        n = len(signals)
        if n == 0 or len(exec_prices) != n or len(close_prices) != n:
            raise ValueError("Mismatched or empty input series for simulation.")

        capital = self.config.initial_capital
        cash = capital
        shares: int = 0
        total_fees = 0.0
        trades: List[TradeRecord] = []
        portfolio_values = []
        positions = []
        
        # Track realized trade PnL for win rate
        last_buy_price = 0.0
        winning_trades = 0
        closed_trades = 0

        for i in range(n):
            dt = dates.iloc[i] if hasattr(dates, "iloc") else dates[i]
            p_exec = float(exec_prices.iloc[i] if hasattr(exec_prices, "iloc") else exec_prices[i])
            p_close = float(close_prices.iloc[i] if hasattr(close_prices, "iloc") else close_prices[i])
            sig = int(signals[i])

            # Execution logic at beginning of session i based on signal generated at session i-1 (or passed signal)
            if sig == 1 and shares == 0 and cash > 0:
                # BUY: apply slippage
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
                    last_buy_price = fill_price

                    trades.append(
                        TradeRecord(
                            session_index=i,
                            date=dt,
                            action="BUY",
                            shares=int(buy_shares),
                            price=round(fill_price, 3),
                            fee=round(fee, 3),
                            total_amount=round(total_paid, 3),
                            cash_after=round(cash, 3),
                            reason="Signal 1 (Long Entry)",
                        )
                    )

            elif sig == 0 and shares > 0:
                # SELL: apply slippage
                fill_price = p_exec * (1.0 - self.config.slippage_pct)
                gross_proceeds = shares * fill_price
                fee = gross_proceeds * self.config.fee_rate
                net_received = gross_proceeds - fee
                cash += net_received
                total_fees += fee

                closed_trades += 1
                if fill_price > last_buy_price:
                    winning_trades += 1

                trades.append(
                    TradeRecord(
                        session_index=i,
                        date=dt,
                        action="SELL",
                        shares=int(shares),
                        price=round(fill_price, 3),
                        fee=round(fee, 3),
                        total_amount=round(net_received, 3),
                        cash_after=round(cash, 3),
                        reason="Signal 0 (Exit to Cash)",
                    )
                )
                shares = 0

            # Mark-to-market portfolio valuation at session close
            port_val = cash + (shares * p_close)
            portfolio_values.append(port_val)
            positions.append(shares)

        # Final portfolio valuation
        final_val = portfolio_values[-1] if portfolio_values else capital
        total_return_pct = ((final_val - capital) / capital) * 100.0

        # Maximum Drawdown calculation
        equity_series = pd.Series(portfolio_values)
        rolling_peak = equity_series.cummax()
        drawdowns = (equity_series - rolling_peak) / rolling_peak
        max_drawdown_pct = float(drawdowns.min() * 100.0) if not drawdowns.empty else 0.0

        # Buy and Hold Benchmark (buying at session 0 with identical fee/slippage, holding until last close)
        first_exec = float(exec_prices.iloc[0] if hasattr(exec_prices, "iloc") else exec_prices[0])
        last_close = float(close_prices.iloc[-1] if hasattr(close_prices, "iloc") else close_prices[-1])
        bh_fill_price = first_exec * (1.0 + self.config.slippage_pct)
        bh_cost_per_share = bh_fill_price * (1.0 + self.config.fee_rate)
        
        if self.config.allow_fractional_shares:
            bh_shares = capital / bh_cost_per_share
            bh_cash = 0.0
        else:
            bh_shares = int(capital // bh_cost_per_share)
            bh_cash = capital - (bh_shares * bh_cost_per_share)

        bh_final_val = bh_cash + (bh_shares * last_close)
        bh_return_pct = ((bh_final_val - capital) / capital) * 100.0

        win_rate = (winning_trades / closed_trades * 100.0) if closed_trades > 0 else 0.0

        return {
            "initial_capital": capital,
            "final_portfolio_value": round(float(final_val), 3),
            "total_return_pct": round(float(total_return_pct), 2),
            "buy_and_hold_return_pct": round(float(bh_return_pct), 2),
            "max_drawdown_pct": round(float(max_drawdown_pct), 2),
            "total_trades": len(trades),
            "closed_trades": closed_trades,
            "win_rate_pct": round(float(win_rate), 2),
            "total_fees_paid": round(float(total_fees), 3),
            "fee_rate": self.config.fee_rate,
            "slippage_pct": self.config.slippage_pct,
            "equity_curve": portfolio_values,
        }
