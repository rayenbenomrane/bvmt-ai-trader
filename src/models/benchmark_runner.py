"""
BVMT AI Trader - Benchmark Runner & Out-of-Sample Trading Simulation

Orchestrates model training, validation-based selection, and rigorous
out-of-sample trading simulation with realistic next-session execution on the BVMT.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config.database import SessionLocal
from src.database.models import DailyPrice
from src.models.feature_engineering import (
    FEATURE_COLUMNS,
    prepare_labeled_dataset,
)
from src.models.ml_predictor import (
    train_and_select_model,
    get_benchmark_models,
)
from src.simulator.trading_simulator import (
    TradingSimulator,
    SimulationConfig,
)


def run_benchmark_for_symbol(
    symbol: str,
    fee_rate: float = 0.0040,
    slippage_pct: float = 0.0010,
    initial_capital: float = 10000.0,
) -> Dict[str, Any]:
    """
    Executes the full pipeline for a single stock:
    1. Loads historical data for the symbol.
    2. Constructs stationary features without future-data leakage.
    3. Partitions data chronologically (60% Train, 20% Validation, 20% Test).
    4. Selects optimal model and decision threshold based strictly on Validation set.
    5. Evaluates all models out-of-sample on the Test set.
    6. Simulates trading on the Test set with next-session open execution and integer shares.
    7. Saves reports and model artifacts.
    """
    db = SessionLocal()
    try:
        query = (
            db.query(DailyPrice)
            .filter(DailyPrice.symbol == symbol)
            .order_by(DailyPrice.date.asc())
            .statement
        )
        df_raw = pd.read_sql(query, db.bind)
        if df_raw.empty or len(df_raw) < 100:
            print(f"[ERROR] Insufficient data for {symbol} ({len(df_raw)} records).")
            return {"symbol": symbol, "status": "Insufficient data"}

        df_clean = prepare_labeled_dataset(df_raw)

        print(f"\n==========================================")
        print(f" BENCHMARK & SIMULATION: {symbol}")
        print(f"==========================================")
        print(f" Total Sessions Analyzed : {len(df_clean)}")

        # Train and select model on Validation data
        selection_result = train_and_select_model(df_clean, symbol)
        best_model_name = selection_result["best_model_name"]
        best_threshold = selection_result["best_threshold"]
        test_evals = selection_result["test_evaluations"]
        fitted_models = selection_result["fitted_models"]

        print(f"\n--- Validation-Selected Model ---")
        print(f" Winner    : {best_model_name}")
        print(f" Threshold : {best_threshold:.2f}")

        print(f"\n--- Out-of-Sample Test Evaluations ---")
        for name, metrics in test_evals.items():
            print(
                f" [{name:<19}] Acc: {metrics['accuracy']:.4f} | "
                f"Bal-Acc: {metrics['balanced_accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"AUC: {metrics['roc_auc']:.4f}"
            )

        # Simulation setup on the Test Partition
        test_parts = selection_result["test_partitions"]
        X_test = test_parts["X_test"]
        dates_test = test_parts["dates_test"].reset_index(drop=True)
        exec_prices = test_parts["next_open_test"].reset_index(drop=True)
        close_prices = test_parts["next_close_test"].reset_index(drop=True)

        sim_config = SimulationConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_pct=slippage_pct,
            execution_price="open",
            allow_fractional_shares=False,
        )
        simulator = TradingSimulator(config=sim_config)

        # Run simulation for all benchmark models for comparison
        model_sim_results = {}
        for name, model in fitted_models.items():
            th = selection_result["val_evaluations"][name]["selected_threshold"]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = np.full(len(X_test), 0.5)

            signals = (probs >= th).astype(int)
            sim_res = simulator.simulate(
                signals=signals,
                dates=dates_test,
                exec_prices=exec_prices,
                close_prices=close_prices,
            )
            # Remove full equity curve from summary dict for compact JSON report
            eq_curve = sim_res.pop("equity_curve", [])
            model_sim_results[name] = sim_res

        winner_sim = model_sim_results[best_model_name]

        print(f"\n--- Trading Simulation Results ({symbol}) ---")
        print(f" Model Used        : {best_model_name} (Threshold = {best_threshold:.2f})")
        print(f" Initial Capital   : {initial_capital:.3f} TND")
        print(f" Final Portfolio   : {winner_sim['final_portfolio_value']:.3f} TND (Return: {winner_sim['total_return_pct']:+.2f}%)")
        print(f" Buy & Hold Return : {winner_sim['buy_and_hold_return_pct']:+.2f}%")
        print(f" Max Drawdown      : {winner_sim['max_drawdown_pct']:.2f}%")
        print(f" Total Trades      : {winner_sim['total_trades']} (Closed: {winner_sim['closed_trades']}, Win Rate: {winner_sim['win_rate_pct']:.1f}%)")
        print(f" Transaction Costs : {winner_sim['total_fees_paid']:.3f} TND (Fee: {fee_rate*100:.2f}%, Slippage: {slippage_pct*100:.2f}%)")

        # Save benchmark report
        report_dir = "data/processed/reports"
        os.makedirs(report_dir, exist_ok=True)
        report = {
            "symbol": symbol,
            "config": {
                "initial_capital": initial_capital,
                "fee_rate": fee_rate,
                "slippage_pct": slippage_pct,
                "execution": "next_session_open",
                "allow_fractional_shares": False,
            },
            "selected_model": best_model_name,
            "selected_threshold": best_threshold,
            "model_benchmarks": test_evals,
            "simulation": winner_sim,
            "all_model_simulations": model_sim_results,
        }

        report_file = f"{report_dir}/{symbol}_benchmark_simulation.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)

        return report

    finally:
        db.close()


def main():
    for sym in ["BT", "SFBT"]:
        run_benchmark_for_symbol(sym)


if __name__ == "__main__":
    main()
