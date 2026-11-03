"""
BVMT AI Trader - Benchmark Runner & Out-of-Sample Trading Simulation

Orchestrates model training, validation-based selection, and rigorous
out-of-sample trading simulation with realistic next-session execution on the BVMT.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

from src.config.database import SessionLocal
from src.database.models import DailyPrice
from src.models.feature_engineering import (
    prepare_labeled_dataset,
    round_trip_cost_return,
)
from src.models.ml_predictor import (
    train_and_select_model,
    evaluate_predictions,
)
from src.simulator.trading_simulator import (
    TradingSimulator,
    SimulationConfig,
)
from src.simulator.strategy import build_strategy_actions, score_validation_configuration


def load_tunindex_csv(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load checked-in TUNINDEX history without modifying its source CSV."""
    csv_path = Path(path) if path else Path(__file__).resolve().parents[2] / "src" / "data_collection" / "raw" / "Tunindex.csv"
    if not csv_path.exists():
        return None

    raw = pd.read_csv(csv_path, dtype=str)
    date_column = next((column for column in raw.columns if column.strip().lower() == "date"), None)
    close_column = next((column for column in raw.columns if column.strip().lower() in {"dernier", "close", "clôture", "cloture"}), None)
    if date_column is None or close_column is None:
        raise ValueError(f"TUNINDEX CSV must include date and close/Dernier columns: {csv_path}")

    def parse_localized_number(value: str) -> float:
        text = str(value).strip().replace(" ", "")
        if "," in text and "." in text and text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        elif "," in text:
            text = text.replace(",", ".")
        return float(text)

    market = pd.DataFrame({
        "date": pd.to_datetime(raw[date_column], dayfirst=True, errors="coerce"),
        "close": raw[close_column].map(parse_localized_number),
    })
    market = market.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    return market.sort_values("date").reset_index(drop=True)


def _probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            return model.predict_proba(X)[:, classes.index(1)]
    return np.full(len(X), 0.5)


def _simulation_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "final_portfolio_value", "total_return_pct", "buy_and_hold_return_pct",
            "max_drawdown_pct", "total_orders", "closed_trades",
            "average_holding_sessions", "gross_win_rate_pct", "net_win_rate_pct",
            "total_fees_paid", "total_slippage_cost", "total_transaction_costs",
            "total_gross_pnl", "total_net_pnl",
        )
    }


def run_benchmark_for_symbol(
    symbol: str,
    fee_rate: float = 0.0040,
    slippage_pct: float = 0.0010,
    initial_capital: float = 10000.0,
    safety_margin: float = 0.0,
    minimum_holding_period: int = 3,
    stop_loss_pct: Optional[float] = None,
    tunindex_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Select horizon/model/strategy exclusively on validation, then report test results."""
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

        market_df = load_tunindex_csv(tunindex_csv_path)
        market_status = {
            "available": market_df is not None and not market_df.empty,
            "path": str(tunindex_csv_path or (Path(__file__).resolve().parents[2] / "src" / "data_collection" / "raw" / "Tunindex.csv")),
            "rows": len(market_df) if market_df is not None else 0,
            "features": ["tunindex_return_1d", "tunindex_return_5d", "tunindex_volatility_10d", "tunindex_trend_20d"] if market_df is not None else [],
        }
        if not market_status["available"]:
            market_status["missing_data_requirement"] = "CSV columns: date, close (or Dernier); one row per trading session."

        print(f"\n==========================================")
        print(f" VALIDATION-SELECTED BENCHMARK: {symbol}")
        print(f"==========================================")
        print(f" TUNINDEX rows            : {market_status['rows']}")

        fee_cost_return = round_trip_cost_return(fee_rate, slippage_pct)
        candidate_specs = [
            (1, "target_binary"),
            (1, "target_cost_aware"),
            (3, "target_binary"),
            (3, "target_cost_aware"),
            (5, "target_binary"),
            (5, "target_cost_aware"),
        ]
        candidates_by_config: Dict[tuple, Dict[str, Any]] = {}
        legacy_candidate = None

        # Train and search entirely inside train/validation partitions.
        for horizon, target_col in [(1, "target_legacy_1d")] + candidate_specs:
            df_clean = prepare_labeled_dataset(
                df_raw,
                horizon=horizon,
                fee_rate=fee_rate,
                slippage_pct=slippage_pct,
                safety_margin=safety_margin,
                market_df=market_df,
            )
            selection = train_and_select_model(
                df_clean,
                symbol,
                horizon=horizon,
                target_col=target_col,
                evaluate_test=False,
                save_artifacts=False,
            )
            val = selection["val_partitions"]
            probabilities = selection["validation_probabilities"]
            expected_raw = selection["expected_return_model"].predict(val["X_val"])
            calibration_offset = float(np.median(val["forward_return"].to_numpy() - expected_raw))
            expected_val = expected_raw + calibration_offset

            if target_col == "target_legacy_1d":
                model_name = selection["best_model_name"]
                threshold = selection["val_evaluations"][model_name]["selected_threshold"]
                strategy_configs = [{
                    "hysteresis": False,
                    "entry_threshold": threshold,
                    "exit_threshold": threshold,
                    "minimum_holding_period": 0,
                    "maximum_holding_period": None,
                    "cost_filter": False,
                }]
                models_to_evaluate = {model_name: selection["fitted_models"][model_name]}
            else:
                strategy_configs = []
                for entry in (0.55, 0.60, 0.65):
                    for exit_level in (0.35, 0.40, 0.45):
                        if exit_level < entry:
                            for min_hold in (0, minimum_holding_period):
                                for cost_filter in (False, True):
                                    strategy_configs.append({
                                        "hysteresis": True,
                                        "entry_threshold": entry,
                                        "exit_threshold": exit_level,
                                        "minimum_holding_period": min_hold,
                                        "maximum_holding_period": horizon if horizon >= 3 else None,
                                        "cost_filter": cost_filter,
                                    })
                for threshold in (0.50, 0.55, 0.60, 0.65):
                    for min_hold in (0, minimum_holding_period):
                        for cost_filter in (False, True):
                            strategy_configs.append({
                                "hysteresis": False,
                                "entry_threshold": threshold,
                                "exit_threshold": threshold,
                                "minimum_holding_period": min_hold,
                                "maximum_holding_period": horizon if horizon >= 3 else None,
                                "cost_filter": cost_filter,
                            })
                models_to_evaluate = selection["fitted_models"]

            for model_name, model in models_to_evaluate.items():
                val_probs = probabilities[model_name]
                for strategy in strategy_configs:
                    actions, reasons, rejections = build_strategy_actions(
                        val_probs,
                        val["signal_open"],
                        val["signal_close"],
                        entry_threshold=strategy["entry_threshold"],
                        exit_threshold=strategy["exit_threshold"],
                        minimum_holding_period=strategy["minimum_holding_period"],
                        maximum_holding_period=strategy["maximum_holding_period"],
                        stop_loss_pct=stop_loss_pct,
                        expected_returns=expected_val,
                        estimated_round_trip_cost=fee_cost_return,
                        safety_margin=safety_margin,
                        cost_filter=strategy["cost_filter"],
                        hysteresis=strategy["hysteresis"],
                    )
                    sim_config = SimulationConfig(
                        initial_capital=initial_capital,
                        fee_rate=fee_rate,
                        slippage_pct=slippage_pct,
                        execution_price="open",
                        allow_fractional_shares=False,
                        auto_close_at_end=True,
                        minimum_holding_period=strategy["minimum_holding_period"],
                    )
                    val_simulation = TradingSimulator(sim_config).simulate(
                        signals=np.zeros(len(actions), dtype=int),
                        actions=actions,
                        exit_reasons=reasons,
                        dates=val["next_date"],
                        exec_prices=val["next_open"],
                        close_prices=val["next_close"],
                        signal_dates=val["dates"],
                    )
                    validation_score = score_validation_configuration(
                        val_simulation,
                        len(actions),
                    )
                    config_key = (
                        horizon, target_col, strategy["hysteresis"],
                        strategy["entry_threshold"], strategy["exit_threshold"],
                        strategy["minimum_holding_period"], strategy["cost_filter"],
                    )
                    candidate = {
                        "symbol": symbol,
                        "horizon": horizon,
                        "target_col": target_col,
                        "model_name": model_name,
                        "model": model,
                        "expected_return_model": selection["expected_return_model"],
                        "calibration_offset": calibration_offset,
                        "selection": selection,
                        "strategy": strategy,
                        "validation_metrics": selection["val_evaluations"][model_name],
                        "validation_simulation": val_simulation,
                        "validation_score": validation_score,
                        "validation_rejections": rejections,
                        "validation_entry_threshold_crossings": int(np.sum(val_probs >= strategy["entry_threshold"])),
                        "validation_buy_orders": int(np.sum(actions == 1)),
                    }
                    if target_col == "target_legacy_1d":
                        legacy_candidate = candidate
                        continue

                    previous = candidates_by_config.get(config_key)
                    score = validation_score["score"]
                    previous_score = previous["validation_score"]["score"] if previous else None
                    if previous is None or score > previous_score:
                        candidates_by_config[config_key] = candidate

        if not candidates_by_config or legacy_candidate is None:
            return {"symbol": symbol, "status": "No viable validation candidates"}

        candidates = list(candidates_by_config.values())
        selected = max(candidates, key=lambda item: item["validation_score"]["score"])

        # Only now access test labels/results: selection and calibration are already fixed.
        def evaluate_candidate_on_test(candidate: Dict[str, Any]) -> Dict[str, Any]:
            selection = candidate["selection"]
            test = selection["test_partitions"]
            model = candidate["model"]
            test_probs = _probabilities(model, test["X_test"])
            expected_test = candidate["expected_return_model"].predict(test["X_test"]) + candidate["calibration_offset"]
            strategy = candidate["strategy"]
            actions, reasons, rejections = build_strategy_actions(
                test_probs,
                test["signal_open_test"],
                test["signal_close_test"],
                entry_threshold=strategy["entry_threshold"],
                exit_threshold=strategy["exit_threshold"],
                minimum_holding_period=strategy["minimum_holding_period"],
                maximum_holding_period=strategy["maximum_holding_period"],
                stop_loss_pct=stop_loss_pct,
                expected_returns=expected_test,
                estimated_round_trip_cost=fee_cost_return,
                safety_margin=safety_margin,
                cost_filter=strategy["cost_filter"],
                hysteresis=strategy["hysteresis"],
            )
            sim = TradingSimulator(SimulationConfig(
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_pct=slippage_pct,
                execution_price="open",
                allow_fractional_shares=False,
                auto_close_at_end=True,
                minimum_holding_period=strategy["minimum_holding_period"],
            )).simulate(
                signals=np.zeros(len(actions), dtype=int),
                actions=actions,
                exit_reasons=reasons,
                dates=test["next_date_test"],
                exec_prices=test["next_open_test"],
                close_prices=test["next_close_test"],
                signal_dates=test["dates_test"],
            )
            ml_metrics = evaluate_predictions(
                test["y_test"],
                (test_probs >= strategy["entry_threshold"]).astype(int),
                test_probs,
            )
            return {
                "simulation": sim,
                "ml_metrics": ml_metrics,
                "test_probabilities": test_probs,
                "expected_returns": expected_test,
                "rejections": rejections,
            }

        test_results = [evaluate_candidate_on_test(candidate) for candidate in candidates]
        selected_index = next(index for index, candidate in enumerate(candidates) if candidate is selected)
        selected_test = test_results[selected_index]
        legacy_test = evaluate_candidate_on_test(legacy_candidate)

        report_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        selected_slug = f"{symbol}_{selected['horizon']}d_{selected['target_col']}"
        model_path = Path("models/saved") / f"{selected_slug}_{selected['model_name'].lower().replace(' ', '_')}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(selected["model"], model_path)
        joblib.dump(selected["expected_return_model"], model_path.with_name(model_path.stem + "_return.joblib"))

        selected_sim = selected_test["simulation"]
        pd.DataFrame(selected_sim.get("closed_trades_records", [])).to_csv(
            report_dir / f"{symbol}_trade_log.csv", index=False
        )
        pd.DataFrame(selected_test["rejections"]).to_csv(
            report_dir / f"{symbol}_rejected_trades.csv", index=False
        )
        candidate_results = []
        for candidate, result in zip(candidates, test_results):
            candidate_results.append({
                "horizon": candidate["horizon"],
                "target": candidate["target_col"],
                "model": candidate["model_name"],
                "strategy": candidate["strategy"],
                "validation": _simulation_summary(candidate["validation_simulation"]),
                "validation_score": candidate["validation_score"],
                "validation_rejected_trade_count": len(candidate["validation_rejections"]),
                "validation_entry_threshold_crossings": candidate["validation_entry_threshold_crossings"],
                "validation_buy_orders": candidate["validation_buy_orders"],
                "test_ml_metrics": result["ml_metrics"],
                "test_trading_metrics": _simulation_summary(result["simulation"]),
            })

        best_validation_by_horizon = {}
        cost_filter_comparison = {}
        for candidate in candidates:
            key = str(candidate["horizon"])
            current = best_validation_by_horizon.get(key)
            if current is None or candidate["validation_score"]["score"] > current["validation_score"]["score"]:
                best_validation_by_horizon[key] = {
                    "target": candidate["target_col"],
                    "model": candidate["model_name"],
                    "strategy": candidate["strategy"],
                    "validation_score": candidate["validation_score"],
                    "validation_trading_metrics": _simulation_summary(candidate["validation_simulation"]),
                    "validation_rejected_trade_count": len(candidate["validation_rejections"]),
                }
            filter_key = (
                f"{candidate['horizon']}d_{candidate['target_col']}_"
                f"filter_{str(candidate['strategy']['cost_filter']).lower()}"
            )
            filter_best = cost_filter_comparison.get(filter_key)
            if filter_best is None or candidate["validation_score"]["score"] > filter_best["validation_score"]["score"]:
                cost_filter_comparison[filter_key] = {
                    "strategy": candidate["strategy"],
                    "validation_score": candidate["validation_score"],
                    "entry_threshold_crossings": candidate["validation_entry_threshold_crossings"],
                    "buy_orders": candidate["validation_buy_orders"],
                    "rejected_entries": len(candidate["validation_rejections"]),
                }

        report = {
            "symbol": symbol,
            "config": {
                "initial_capital": initial_capital,
                "fee_rate_per_side": fee_rate,
                "slippage_per_side": slippage_pct,
                "estimated_round_trip_cost_return": fee_cost_return,
                "safety_margin_return": safety_margin,
                "execution": "next_session_open",
                "minimum_holding_period_default": minimum_holding_period,
                "stop_loss_pct": stop_loss_pct,
                "test_used_for_selection": False,
            },
            "target_methodology": {
                "forward_return": "open[t + horizon + 1] / open[t + 1] - 1",
                "direction_target": "UP iff forward_return > 0; otherwise DOWN",
                "cost_aware_target": "UP iff forward_return > compounded round-trip cost return + safety margin; otherwise non-profitable",
                "legacy_target": "close[t + 1] / close[t] - 1 > 0",
                "horizon_unit": "trading sessions",
            },
            "tunindex_market_context": market_status,
            "cost_aware_filter": {
                "expected_return_estimator": "RandomForestRegressor fit on training forward returns; median residual offset calibrated on validation only",
                "entry_rule": "calibrated expected gross return > round-trip transaction cost return + safety margin",
                "limitations": "Conditional expected-return forecasts may be noisy; the filter remains optional and is never fitted on test returns.",
                "validation_evaluation": "Each cost_filter on/off configuration is simulated on validation; rejected-entry counts are reported per candidate.",
            },
            "validation_selection_method": {
                "score": "net_return_pct - 0.50 * abs(max_drawdown_pct) - 0.10 * closed_trades_per_100_signals",
                "selection_score": selected["validation_score"],
                "best_by_horizon": best_validation_by_horizon,
                "cost_filter_on_off_comparison": cost_filter_comparison,
                "thresholds_and_holding_periods_selected_per_stock": True,
            },
            "selected_configuration": {
                "horizon": selected["horizon"],
                "target": selected["target_col"],
                "model": selected["model_name"],
                "strategy": selected["strategy"],
                "validation_ml_metrics": selected["validation_metrics"],
                "validation_trading_metrics": _simulation_summary(selected["validation_simulation"]),
                "validation_score": selected["validation_score"],
                "validation_rejected_trade_count": len(selected["validation_rejections"]),
                "test_ml_metrics": selected_test["ml_metrics"],
                "test_trading_metrics": _simulation_summary(selected_sim),
                "rejected_trade_count": len(selected_test["rejections"]),
            },
            "legacy_baseline": {
                "model": legacy_candidate["model_name"],
                "strategy": legacy_candidate["strategy"],
                "test_ml_metrics": legacy_test["ml_metrics"],
                "test_trading_metrics": _simulation_summary(legacy_test["simulation"]),
            },
            "validation_selected_test_evaluations": candidate_results,
            "model_file": str(model_path),
        }
        report_path = report_dir / f"{symbol}_benchmark_simulation.json"
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, default=str)

        print(f" Selected model / target : {selected['model_name']} / {selected['target_col']}")
        print(f" Selected horizon        : {selected['horizon']} trading sessions")
        print(f" Selected strategy       : {selected['strategy']}")
        print(f" Test return              : {selected_sim['total_return_pct']:+.2f}%")
        print(f" Legacy baseline return   : {legacy_test['simulation']['total_return_pct']:+.2f}%")
        print(f" Test max drawdown        : {selected_sim['max_drawdown_pct']:.2f}%")
        print(f" Report                   : {report_path}")
        return report

    finally:
        db.close()


def main():
    results = [run_benchmark_for_symbol(sym) for sym in ["BT", "SFBT"]]
    comparison = {
        "selection_method": "Validation score = net return - 0.50 * absolute drawdown - 0.10 * closed trades per 100 signals",
        "stocks": {},
    }
    for result in results:
        if result.get("status"):
            comparison["stocks"][result["symbol"]] = {"status": result["status"]}
            continue
        selected = result["selected_configuration"]
        comparison["stocks"][result["symbol"]] = {
            "best_validation_configuration": {
                "horizon": selected["horizon"],
                "target": selected["target"],
                "model": selected["model"],
                "strategy": selected["strategy"],
                "validation_score": selected["validation_score"],
                "validation_trading_metrics": selected["validation_trading_metrics"],
            },
            "final_test_return_pct": selected["test_trading_metrics"]["total_return_pct"],
            "legacy_test_return_pct": result["legacy_baseline"]["test_trading_metrics"]["total_return_pct"],
        }
    report_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "reports" / "strategy_selection_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(comparison, report_file, indent=2, default=str)
    print(f"\nValidation comparison report: {report_path}")


if __name__ == "__main__":
    main()
