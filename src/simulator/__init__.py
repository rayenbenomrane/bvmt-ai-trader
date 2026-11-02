from src.simulator.trading_simulator import (
    TradingSimulator,
    SimulationConfig,
    OrderExecutionRecord,
    ClosedTradeRecord,
)

# Backwards compatibility alias
TradeRecord = OrderExecutionRecord

__all__ = [
    "TradingSimulator",
    "SimulationConfig",
    "OrderExecutionRecord",
    "ClosedTradeRecord",
    "TradeRecord",
]
