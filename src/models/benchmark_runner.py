"""
BVMT AI Trader - Benchmark Runner & Out-of-Sample Trading Simulation

Orchestrates model training, validation-based selection, and rigorous
out-of-sample trading simulation with realistic next-session execution on the BVMT.
"""

import os
import sys
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

from src.config.database import SessionLocal
from src.database.models import DailyPrice
from src.models.feature_engineering import (
    get_available_feature_columns,
    prepare_labeled_dataset,
    rolling_walk_forward_plan,
)
from src.models.ml_predictor import (
    get_benchmark_models,
    train_and_select_model,
    evaluate_predictions,
)
from src.simulator.trading_simulator import (
    TradingSimulator,
    SimulationConfig,
)
from src.simulator.strategy import (
    assess_validation_edge,
    build_strategy_actions,
    score_validation_configuration,
    select_validation_candidate,
)


def load_tunindex_csv(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load checked-in TUNINDEX history without modifying its source CSV."""
    csv_path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[2] / "src" / "data_collection" / "raw" / "Tunindex.csv"
    )
    if not csv_path.exists():
        return None

    raw = pd.read_csv(csv_path, dtype=str)
    date_column = next(
        (column for column in raw.columns if column.strip().lower() == "date"), None
    )
    close_column = next(
        (
            column
            for column in raw.columns
            if column.strip().lower() in {"dernier", "close", "clôture", "cloture"}
        ),
        None,
    )
    if date_column is None or close_column is None:
        raise ValueError(
            f"TUNINDEX CSV must include date and close/Dernier columns: {csv_path}"
        )

    def parse_localized_number(value: str) -> float:
        text = str(value).strip().replace(" ", "")
        if "," in text and "." in text and text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        elif "," in text:
            text = text.replace(",", ".")
        return float(text)

    market = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_column], dayfirst=True, errors="coerce"),
            "close": raw[close_column].map(parse_localized_number),
        }
    )
    market = market.dropna(subset=["date", "close"]).drop_duplicates(
        "date", keep="last"
    )
    return market.sort_values("date").reset_index(drop=True)


def _probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            return model.predict_proba(X)[:, classes.index(1)]
    return np.full(len(X), 0.5)


def _simulation_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a stable trading-metrics block, tolerant of missing keys.

    Handles three cases:
      1. Raw TradingSimulator output (has `final_portfolio_value`).
      2. Aggregated window output (has `final_portfolio_value_mean`).
      3. Empty / NO_TRADE output (uses defaults).
    """

    def pick(key: str, default: Any = 0.0) -> Any:
        if not result:
            return default
        if key in result and result[key] is not None:
            return result[key]
        mean_key = f"{key}_mean"
        if mean_key in result and result[mean_key] is not None:
            return result[mean_key]
        return default

    return {
        "final_portfolio_value": pick("final_portfolio_value", 10000.0),
        "total_return_pct": pick("total_return_pct", 0.0),
        "buy_and_hold_return_pct": pick("buy_and_hold_return_pct", 0.0),
        "max_drawdown_pct": pick("max_drawdown_pct", 0.0),
        "total_orders": pick("total_orders", 0),
        "closed_trades": pick("closed_trades", 0),
        "average_holding_sessions": pick("average_holding_sessions", 0.0),
        "gross_win_rate_pct": pick("gross_win_rate_pct", 0.0),
        "net_win_rate_pct": pick("net_win_rate_pct", 0.0),
        "total_gross_pnl": pick("total_gross_pnl", 0.0),
        "total_net_pnl": pick("total_net_pnl", 0.0),
    }


def _validation_metrics_for_report(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metrics = candidate.get("validation_metrics")
    if metrics is None:
        return None
    result = dict(metrics)
    result["classification_threshold"] = result.pop("selected_threshold", None)
    result["entry_threshold"] = candidate["strategy"]["entry_threshold"]
    return result


def _aggregate_window_simulations(window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple independent window simulations into a single block.

    Returns BOTH raw (`final_portfolio_value`) and averaged (`*_mean`) keys so
    downstream reporting code never crashes on missing keys.
    """
    simulations = [item["simulation"] for item in window_results if item.get("simulation") is not None]
    if not simulations:
        return {
            "window_count": 0,
            "aggregation": "no windows simulated",
            "final_portfolio_value": 10000.0,
            "final_portfolio_value_mean": 10000.0,
            "total_return_pct": 0.0,
            "buy_and_hold_return_pct": 0.0,
            "excess_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_pct_mean": 0.0,
            "total_orders": 0,
            "closed_trades": 0,
            "average_holding_sessions": 0.0,
            "gross_win_rate_pct": 0.0,
            "gross_win_rate_pct_mean": 0.0,
            "net_win_rate_pct": 0.0,
            "net_win_rate_pct_mean": 0.0,
            "total_gross_pnl": 0.0,
            "total_net_pnl": 0.0,
        }

    def g(sim: Dict[str, Any], key: str, default: Any = 0.0) -> Any:
        if not sim:
            return default
        return sim.get(key, default)

    trades = sum(int(g(s, "closed_trades", 0)) for s in simulations)
    fv_mean = float(np.mean([g(s, "final_portfolio_value", 10000.0) for s in simulations]))
    ret_mean = float(np.mean([g(s, "total_return_pct", 0.0) for s in simulations]))
    bh_mean = float(np.mean([g(s, "buy_and_hold_return_pct", 0.0) for s in simulations]))
    dd_mean = float(np.mean([g(s, "max_drawdown_pct", 0.0) for s in simulations]))
    gwr_mean = float(np.mean([g(s, "gross_win_rate_pct", 0.0) for s in simulations]))
    nwr_mean = float(np.mean([g(s, "net_win_rate_pct", 0.0) for s in simulations]))
    excess_mean = float(
        np.mean(
            [
                g(s, "total_return_pct", 0.0) - g(s, "buy_and_hold_return_pct", 0.0)
                for s in simulations
            ]
        )
    )

    avg_hold = (
        float(
            sum(
                g(s, "average_holding_sessions", 0.0) * g(s, "closed_trades", 0)
                for s in simulations
            )
            / trades
        )
        if trades
        else 0.0
    )

    return {
        "window_count": len(simulations),
        "aggregation": "mean of independent windows; order and trade counts are summed",
        "final_portfolio_value": round(fv_mean, 3),
        "final_portfolio_value_mean": round(fv_mean, 3),
        "total_return_pct": round(ret_mean, 3),
        "buy_and_hold_return_pct": round(bh_mean, 3),
        "excess_return_pct": round(excess_mean, 3),
        "max_drawdown_pct": round(dd_mean, 3),
        "max_drawdown_pct_mean": round(dd_mean, 3),
        "total_orders": sum(int(g(s, "total_orders", 0)) for s in simulations),
        "closed_trades": trades,
        "average_holding_sessions": round(avg_hold, 3),
        "gross_win_rate_pct": round(gwr_mean, 3),
        "gross_win_rate_pct_mean": round(gwr_mean, 3),
        "net_win_rate_pct": round(nwr_mean, 3),
        "net_win_rate_pct_mean": round(nwr_mean, 3),
        "total_gross_pnl": round(float(sum(g(s, "total_gross_pnl", 0.0) for s in simulations)), 3),
        "total_net_pnl": round(float(sum(g(s, "total_net_pnl", 0.0) for s in simulations)), 3),
    }


def _aggregate_classification_metrics(window_results: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics = [item["ml_metrics"] for item in window_results if item.get("ml_metrics")]
    if not metrics:
        return {}
    keys = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")
    return {
        key: round(float(np.mean([metric[key] for metric in metrics])), 4)
        for key in keys
    }


def _most_common(values: List[Any]) -> Any:
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], str(value)))[0]


def _strategy_configurations(
    horizon: int, minimum_holding_period: int
) -> List[Dict[str, Any]]:
    configurations = []
    maximum_holds = (horizon, None) if horizon >= 3 else (None,)
    for entry in (0.55, 0.60, 0.65):
        for exit_level in (0.35, 0.40, 0.45):
            if exit_level < entry:
                for min_hold in (0, minimum_holding_period):
                    for max_hold in maximum_holds:
                        configurations.append(
                            {
                                "hysteresis": True,
                                "entry_threshold": entry,
                                "exit_threshold": exit_level,
                                "minimum_holding_period": min_hold,
                                "maximum_holding_period": max_hold,
                            }
                        )
    for threshold in (0.50, 0.55, 0.60, 0.65):
        for min_hold in (0, minimum_holding_period):
            for max_hold in maximum_holds:
                configurations.append(
                    {
                        "hysteresis": False,
                        "entry_threshold": threshold,
                        "exit_threshold": threshold,
                        "minimum_holding_period": min_hold,
                        "maximum_holding_period": max_hold,
                    }
                )
    return configurations


def run_benchmark_for_symbol(
    symbol: str,
    initial_capital: float = 10000.0,
    minimum_holding_period: int = 3,
    min_validation_closed_trades: int = 10,
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
            "path": str(
                tunindex_csv_path
                or (
                    Path(__file__).resolve().parents[2]
                    / "src"
                    / "data_collection"
                    / "raw"
                    / "Tunindex.csv"
                )
            ),
            "rows": len(market_df) if market_df is not None else 0,
            "features": [
                "tunindex_return_1d",
                "tunindex_return_5d",
                "tunindex_volatility_10d",
                "tunindex_trend_20d",
            ]
            if market_df is not None
            else [],
        }
        if not market_status["available"]:
            market_status["missing_data_requirement"] = (
                "CSV columns: date, close (or Dernier); one row per trading session."
            )

        print(f"\n==========================================")
        print(f" VALIDATION-SELECTED BENCHMARK: {symbol}")
        print(f"==========================================")
        print(f" TUNINDEX rows            : {market_status['rows']}")

        common_labeled = prepare_labeled_dataset(df_raw, horizon=5, market_df=market_df)
        common_dates = common_labeled["date"].reset_index(drop=True)
        labeled_sets = {}
        for horizon in (1, 3, 5):
            labeled = prepare_labeled_dataset(df_raw, horizon=horizon, market_df=market_df)
            labeled = labeled.loc[labeled["date"].isin(common_dates)].reset_index(drop=True)
            if not labeled["date"].reset_index(drop=True).equals(common_dates):
                raise AssertionError(
                    "Target horizons must share the same signal-date index."
                )
            labeled_sets[horizon] = labeled
        labeled_sets["legacy"] = labeled_sets[1]

        walk_plan = rolling_walk_forward_plan(len(common_dates))
        candidate_records: Dict[tuple, List[Dict[str, Any]]] = {}
        legacy_records: List[Dict[str, Any]] = []
        validation_window_ranges = []

        for fold_index, boundaries in enumerate(
            walk_plan["validation_windows"], start=1
        ):
            validation_start = boundaries["validation_start"]
            validation_stop = boundaries["validation_end"] - 6
            validation_window_ranges.append(
                {
                    "window": fold_index,
                    "signal_date_start": str(common_dates.iloc[validation_start]),
                    "signal_date_end": str(common_dates.iloc[validation_stop - 1]),
                }
            )
            for horizon, target_col in [
                (1, "target_legacy_1d"),
                (1, "target_binary"),
                (3, "target_binary"),
                (5, "target_binary"),
            ]:
                df_clean = labeled_sets[
                    "legacy" if target_col == "target_legacy_1d" else horizon
                ]
                selection = train_and_select_model(
                    df_clean,
                    symbol,
                    horizon=horizon,
                    target_col=target_col,
                    evaluate_test=False,
                    save_artifacts=False,
                    purge_horizon=5,
                    split_boundaries=boundaries,
                )
                val = selection["val_partitions"]
                probabilities = selection["validation_probabilities"]

                if target_col == "target_legacy_1d":
                    model_name = selection["best_model_name"]
                    threshold = selection["val_evaluations"][model_name][
                        "selected_threshold"
                    ]
                    models_to_evaluate = {
                        model_name: selection["fitted_models"][model_name]
                    }
                    strategy_configs = [
                        {
                            "hysteresis": False,
                            "entry_threshold": threshold,
                            "exit_threshold": threshold,
                            "minimum_holding_period": 0,
                            "maximum_holding_period": None,
                        }
                    ]
                else:
                    models_to_evaluate = selection["fitted_models"]
                    strategy_configs = _strategy_configurations(
                        horizon, minimum_holding_period
                    )

                for model_name, model in models_to_evaluate.items():
                    val_probs = probabilities[model_name]
                    for strategy in strategy_configs:
                        actions, reasons = build_strategy_actions(
                            val_probs,
                            val["signal_open"],
                            val["signal_close"],
                            entry_threshold=strategy["entry_threshold"],
                            exit_threshold=strategy["exit_threshold"],
                            minimum_holding_period=strategy["minimum_holding_period"],
                            maximum_holding_period=strategy["maximum_holding_period"],
                            stop_loss_pct=stop_loss_pct,
                            hysteresis=strategy["hysteresis"],
                        )
                        simulation = TradingSimulator(
                            SimulationConfig(
                                initial_capital=initial_capital,
                                execution_price="open",
                                allow_fractional_shares=False,
                                auto_close_at_end=True,
                                minimum_holding_period=strategy[
                                    "minimum_holding_period"
                                ],
                            )
                        ).simulate(
                            signals=np.zeros(len(actions), dtype=int),
                            actions=actions,
                            exit_reasons=reasons,
                            dates=val["next_date"],
                            exec_prices=val["next_open"],
                            close_prices=val["next_close"],
                            signal_dates=val["dates"],
                        )
                        record = {
                            "fold": fold_index,
                            "model_name": model_name,
                            "classification_threshold": selection["val_evaluations"][
                                model_name
                            ]["selected_threshold"],
                            "validation_metrics": selection["val_evaluations"][
                                model_name
                            ],
                            "strategy": strategy,
                            "simulation": simulation,
                            "signal_count": len(actions),
                            "signal_date_start": str(val["dates"].iloc[0]),
                            "signal_date_end": str(val["dates"].iloc[-1]),
                        }
                        if target_col == "target_legacy_1d":
                            legacy_records.append(record)
                            continue

                        config_key = (
                            horizon,
                            model_name,
                            strategy["hysteresis"],
                            strategy["entry_threshold"],
                            strategy["exit_threshold"],
                            strategy["minimum_holding_period"],
                            strategy["maximum_holding_period"],
                        )
                        candidate_records.setdefault(config_key, []).append(record)

        if not candidate_records or len(legacy_records) != len(
            walk_plan["validation_windows"]
        ):
            return {"symbol": symbol, "status": "Insufficient walk-forward validation windows"}

        def summarize_validation_records(
            records: List[Dict[str, Any]],
            horizon: int,
            target_col: str,
            model_name: str,
            strategy: Dict[str, Any],
            config_key: str,
        ) -> Dict[str, Any]:
            simulations = [item["simulation"] for item in records]
            mean_return = float(
                np.mean([item["total_return_pct"] for item in simulations])
            )
            mean_bh = float(
                np.mean([item["buy_and_hold_return_pct"] for item in simulations])
            )
            mean_drawdown = float(
                np.mean([item["max_drawdown_pct"] for item in simulations])
            )
            total_trades = sum(item["closed_trades"] for item in simulations)
            total_signals = sum(item["signal_count"] for item in records)
            aggregate_simulation = {
                "initial_capital": initial_capital,
                "final_portfolio_value": float(
                    np.mean([item["final_portfolio_value"] for item in simulations])
                ),
                "total_return_pct": mean_return,
                "buy_and_hold_return_pct": mean_bh,
                "max_drawdown_pct": mean_drawdown,
                "total_orders": sum(item["total_orders"] for item in simulations),
                "closed_trades": total_trades,
                "average_holding_sessions": float(
                    np.mean([item["average_holding_sessions"] for item in simulations])
                ),
                "gross_win_rate_pct": float(
                    np.mean([item["gross_win_rate_pct"] for item in simulations])
                ),
                "net_win_rate_pct": float(
                    np.mean([item["net_win_rate_pct"] for item in simulations])
                ),
                "total_gross_pnl": float(
                    sum(item["total_gross_pnl"] for item in simulations)
                ),
                "total_net_pnl": float(
                    sum(item["total_net_pnl"] for item in simulations)
                ),
            }
            window_results = []
            for item in records:
                sim = item["simulation"]
                window_results.append(
                    {
                        "window": item["fold"],
                        "signal_date_start": item["signal_date_start"],
                        "signal_date_end": item["signal_date_end"],
                        "strategy_return_pct": sim["total_return_pct"],
                        "buy_and_hold_return_pct": sim["buy_and_hold_return_pct"],
                        "excess_return_pct": sim["total_return_pct"]
                        - sim["buy_and_hold_return_pct"],
                        "max_drawdown_pct": sim["max_drawdown_pct"],
                        "closed_trades": sim["closed_trades"],
                        "total_orders": sim["total_orders"],
                        "gross_win_rate_pct": sim["gross_win_rate_pct"],
                        "net_win_rate_pct": sim["net_win_rate_pct"],
                        "profitable": sim["total_return_pct"] > 0,
                        "beats_buy_and_hold": sim["total_return_pct"]
                        > sim["buy_and_hold_return_pct"],
                        "beats_legacy": None,
                    }
                )
            metric_keys = (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            )
            validation_metrics = {
                metric: round(
                    float(
                        np.mean(
                            [
                                item["validation_metrics"][metric]
                                for item in records
                            ]
                        )
                    ),
                    4,
                )
                for metric in metric_keys
            }
            validation_metrics["selected_threshold"] = _most_common(
                [item["classification_threshold"] for item in records]
            )
            validation_metrics["threshold_sweep"] = []
            return {
                "symbol": symbol,
                "horizon": horizon,
                "target_col": target_col,
                "model_name": model_name,
                "model": None,
                "strategy": strategy,
                "validation_simulation": aggregate_simulation,
                "validation_score": score_validation_configuration(
                    aggregate_simulation, total_signals
                ),
                "validation_metrics": validation_metrics,
                "validation_records": records,
                "validation_windows": window_results,
                "validation_robustness": {},
                "config_key": config_key,
            }

        legacy_model_name = _most_common(
            [item["model_name"] for item in legacy_records]
        )
        legacy_threshold = _most_common(
            [item["strategy"]["entry_threshold"] for item in legacy_records]
        )
        legacy_strategy = {
            "hysteresis": False,
            "entry_threshold": legacy_threshold,
            "exit_threshold": legacy_threshold,
            "minimum_holding_period": 0,
            "maximum_holding_period": None,
        }
        legacy_candidate = summarize_validation_records(
            legacy_records,
            1,
            "target_legacy_1d",
            legacy_model_name,
            legacy_strategy,
            "legacy_baseline",
        )

        candidates = []
        for key, records in candidate_records.items():
            horizon, model_name, hysteresis, entry, exit_level, min_hold, max_hold = key
            strategy = {
                "hysteresis": hysteresis,
                "entry_threshold": entry,
                "exit_threshold": exit_level,
                "minimum_holding_period": min_hold,
                "maximum_holding_period": max_hold,
            }
            candidate = summarize_validation_records(
                records,
                horizon,
                "target_binary",
                model_name,
                strategy,
                "|".join(map(str, key)),
            )
            candidates.append(candidate)

        legacy_by_fold = {
            item["fold"]: item["simulation"]["total_return_pct"]
            for item in legacy_records
        }
        legacy_windows = legacy_candidate["validation_windows"]
        legacy_candidate["validation_robustness"] = {
            "window_count": len(legacy_windows),
            "profitable_windows": sum(w["profitable"] for w in legacy_windows),
            "profitability_consistency": sum(w["profitable"] for w in legacy_windows)
            / len(legacy_windows),
            "windows_beating_buy_and_hold": sum(
                w["beats_buy_and_hold"] for w in legacy_windows
            ),
            "buy_and_hold_win_rate": sum(
                w["beats_buy_and_hold"] for w in legacy_windows
            )
            / len(legacy_windows),
            "windows_beating_legacy": 0,
            "legacy_win_rate": 0.0,
        }
        for candidate in candidates:
            windows = candidate["validation_windows"]
            for record, window in zip(candidate["validation_records"], windows):
                window["beats_legacy"] = (
                    record["simulation"]["total_return_pct"]
                    > legacy_by_fold[record["fold"]]
                )
            count = len(windows)
            candidate["validation_robustness"] = {
                "window_count": count,
                "profitable_windows": sum(w["profitable"] for w in windows),
                "profitability_consistency": sum(w["profitable"] for w in windows)
                / count,
                "windows_beating_buy_and_hold": sum(
                    w["beats_buy_and_hold"] for w in windows
                ),
                "buy_and_hold_win_rate": sum(
                    w["beats_buy_and_hold"] for w in windows
                )
                / count,
                "windows_beating_legacy": sum(w["beats_legacy"] for w in windows),
                "legacy_win_rate": sum(w["beats_legacy"] for w in windows) / count,
            }

        min_validation_bh_win_rate = 0.50
        min_validation_legacy_win_rate = 0.50
        robust_candidates = [
            candidate
            for candidate in candidates
            if candidate["validation_simulation"]["closed_trades"]
            >= min_validation_closed_trades
            and candidate["validation_robustness"]["buy_and_hold_win_rate"]
            >= min_validation_bh_win_rate
            and candidate["validation_robustness"]["legacy_win_rate"]
            >= min_validation_legacy_win_rate
        ]
        selected, _, tie_break_used = select_validation_candidate(
            robust_candidates,
            min_validation_closed_trades=min_validation_closed_trades,
        )
        excluded_candidate_count = len(candidates) - len(robust_candidates)
        is_no_trade = selected is None
        if is_no_trade:
            cash_validation_windows = []
            cash_signal_count = 0
            for fold_index, boundaries in enumerate(
                walk_plan["validation_windows"], start=1
            ):
                start = boundaries["validation_start"]
                stop = boundaries["validation_end"] - 6
                frame = labeled_sets[3].iloc[start:stop].reset_index(drop=True)
                cash_signal_count += len(frame)
                cash_simulation = TradingSimulator(
                    SimulationConfig(
                        initial_capital=initial_capital,
                        auto_close_at_end=False,
                    )
                ).simulate(
                    signals=np.zeros(len(frame), dtype=int),
                    dates=frame["next_date"],
                    exec_prices=frame["next_open"],
                    close_prices=frame["next_close"],
                    signal_dates=frame["date"],
                )
                if not cash_simulation:
                    cash_simulation = {
                        "final_portfolio_value": initial_capital,
                        "total_return_pct": 0.0,
                        "buy_and_hold_return_pct": 0.0,
                        "max_drawdown_pct": 0.0,
                        "total_orders": 0,
                        "closed_trades": 0,
                        "average_holding_sessions": 0.0,
                        "gross_win_rate_pct": 0.0,
                        "net_win_rate_pct": 0.0,
                        "total_gross_pnl": 0.0,
                        "total_net_pnl": 0.0,
                    }
                cash_validation_windows.append(
                    {
                        "window": fold_index,
                        "signal_date_start": str(frame["date"].iloc[0]),
                        "signal_date_end": str(frame["date"].iloc[-1]),
                        "simulation": cash_simulation,
                    }
                )
            cash_aggregate = _aggregate_window_simulations(cash_validation_windows)
            selected = {
                "symbol": symbol,
                "status": "NO_TRADE",
                "horizon": None,
                "target_col": "CASH_BASELINE",
                "model_name": "NO_TRADE",
                "model": None,
                "strategy": None,
                "validation_metrics": None,
                "validation_simulation": cash_aggregate,
                "validation_score": score_validation_configuration(
                    cash_aggregate, cash_signal_count
                ),
                "validation_windows": [
                    {
                        **{key: value for key, value in item.items() if key != "simulation"},
                        "trading_metrics": _simulation_summary(item["simulation"]),
                    }
                    for item in cash_validation_windows
                ],
                "validation_robustness": {
                    "window_count": len(walk_plan["validation_windows"]),
                    "buy_and_hold_win_rate": 0.0,
                    "legacy_win_rate": 0.0,
                },
                "config_key": "NO_TRADE",
            }
            tie_break_used = (
                "no candidate met validation trade and robustness criteria"
            )

        validation_edge = {
            "minimum_validation_closed_trades": min_validation_closed_trades,
            "minimum_validation_windows_beating_buy_and_hold": min_validation_bh_win_rate,
            "minimum_validation_windows_beating_legacy": min_validation_legacy_win_rate,
            "candidates_meeting_robustness_criteria": len(robust_candidates),
            "candidates_excluded_by_robustness_criteria": excluded_candidate_count,
            "no_candidate_beats_buy_and_hold_consistently": not any(
                item["validation_robustness"]["buy_and_hold_win_rate"]
                >= min_validation_bh_win_rate
                for item in candidates
                if item["validation_simulation"]["closed_trades"]
                >= min_validation_closed_trades
            ),
            "no_candidate_beats_legacy_consistently": not any(
                item["validation_robustness"]["legacy_win_rate"]
                >= min_validation_legacy_win_rate
                for item in candidates
                if item["validation_simulation"]["closed_trades"]
                >= min_validation_closed_trades
            ),
        }

        # Fit once on pre-test history only; keep that model fixed over disjoint test windows.
        def evaluate_test_windows(
            data: pd.DataFrame,
            target_col: str,
            model_name: Optional[str],
            strategy: Optional[Dict[str, Any]],
        ) -> tuple:
            feature_columns = get_available_feature_columns(data)
            first_test_start = walk_plan["test_windows"][0]["test_start"]
            train_stop = first_test_start - 6
            test_model = None
            if model_name is not None:
                test_model = get_benchmark_models()[model_name]
                test_model.fit(
                    data[feature_columns].iloc[:train_stop],
                    data[target_col].iloc[:train_stop],
                )

            results = []
            for window_index, window in enumerate(
                walk_plan["test_windows"], start=1
            ):
                test_frame = (
                    data.iloc[window["test_start"] : window["test_end"]]
                    .copy()
                    .reset_index(drop=True)
                )
                if test_model is None:
                    actions = np.zeros(len(test_frame), dtype=int)
                    reasons = [""] * len(test_frame)
                    probabilities = None
                    ml_metrics = None
                else:
                    probabilities = _probabilities(
                        test_model, test_frame[feature_columns]
                    )
                    actions, reasons = build_strategy_actions(
                        probabilities,
                        test_frame["open"],
                        test_frame["close"],
                        entry_threshold=strategy["entry_threshold"],
                        exit_threshold=strategy["exit_threshold"],
                        minimum_holding_period=strategy["minimum_holding_period"],
                        maximum_holding_period=strategy["maximum_holding_period"],
                        stop_loss_pct=stop_loss_pct,
                        hysteresis=strategy["hysteresis"],
                    )
                    ml_metrics = evaluate_predictions(
                        test_frame[target_col],
                        (probabilities >= strategy["entry_threshold"]).astype(int),
                        probabilities,
                    )

                simulation = TradingSimulator(
                    SimulationConfig(
                        initial_capital=initial_capital,
                        execution_price="open",
                        allow_fractional_shares=False,
                        auto_close_at_end=True,
                        minimum_holding_period=strategy[
                            "minimum_holding_period"
                        ]
                        if strategy
                        else 0,
                    )
                ).simulate(
                    signals=np.zeros(len(actions), dtype=int),
                    actions=actions,
                    exit_reasons=reasons,
                    dates=test_frame["next_date"],
                    exec_prices=test_frame["next_open"],
                    close_prices=test_frame["next_close"],
                    signal_dates=test_frame["date"],
                )
                results.append(
                    {
                        "window": window_index,
                        "signal_date_start": str(test_frame["date"].iloc[0]),
                        "signal_date_end": str(test_frame["date"].iloc[-1]),
                        "simulation": simulation,
                        "ml_metrics": ml_metrics,
                    }
                )
            return results, test_model

        if is_no_trade:
            selected_target = "target_binary"
            selected_data = labeled_sets[3]
            selected_test_windows, selected_test_model = evaluate_test_windows(
                selected_data, selected_target, None, None
            )
        else:
            selected_data = labeled_sets[selected["horizon"]]
            selected_test_windows, selected_test_model = evaluate_test_windows(
                selected_data,
                selected["target_col"],
                selected["model_name"],
                selected["strategy"],
            )

        legacy_test_windows, _ = evaluate_test_windows(
            labeled_sets["legacy"],
            "target_legacy_1d",
            legacy_candidate["model_name"],
            legacy_candidate["strategy"],
        )
        for selected_window, legacy_window in zip(
            selected_test_windows, legacy_test_windows
        ):
            if (
                selected_window["signal_date_start"]
                != legacy_window["signal_date_start"]
                or selected_window["signal_date_end"]
                != legacy_window["signal_date_end"]
            ):
                raise AssertionError(
                    "Selected and legacy test windows must have identical signal dates."
                )
            if not np.isclose(
                selected_window["simulation"]["buy_and_hold_return_pct"],
                legacy_window["simulation"]["buy_and_hold_return_pct"],
                atol=0.01,
            ):
                raise AssertionError(
                    "Selected and legacy windows must have identical buy-and-hold assumptions."
                )

        selected_test = {
            "windows": selected_test_windows,
            "simulation": _aggregate_window_simulations(selected_test_windows),
            "ml_metrics": _aggregate_classification_metrics(selected_test_windows),
        }
        legacy_test = {
            "windows": legacy_test_windows,
            "simulation": _aggregate_window_simulations(legacy_test_windows),
            "ml_metrics": _aggregate_classification_metrics(legacy_test_windows),
        }
        selected_sim = selected_test["simulation"]

        report_dir = (
            Path(__file__).resolve().parents[2] / "data" / "processed" / "reports"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        model_path = None
        if selected_test_model is not None:
            selected_slug = f"{symbol}_{selected['horizon']}d_{selected['target_col']}"
            model_path = (
                Path("models/saved")
                / f"{selected_slug}_{selected['model_name'].lower().replace(' ', '_')}.joblib"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(selected_test_model, model_path)

        selected_trade_records = []
        for result in selected_test_windows:
            for record in result["simulation"].get("closed_trades_records", []):
                selected_trade_records.append(
                    {"window": result["window"], **record}
                )
        pd.DataFrame(selected_trade_records).to_csv(
            report_dir / f"{symbol}_trade_log.csv", index=False
        )

        def compact_test_windows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [
                {
                    "window": item["window"],
                    "signal_date_start": item["signal_date_start"],
                    "signal_date_end": item["signal_date_end"],
                    "trading_metrics": _simulation_summary(item["simulation"]),
                    "ml_metrics": item["ml_metrics"],
                }
                for item in results
            ]

        best_validation_by_horizon = {}
        best_validation_by_max_hold = {}
        excluded_by_horizon = {}
        for candidate in candidates:
            if (
                candidate["validation_simulation"]["closed_trades"]
                < min_validation_closed_trades
            ):
                key = str(candidate["horizon"])
                excluded_by_horizon[key] = excluded_by_horizon.get(key, 0) + 1
                continue
            key = str(candidate["horizon"])
            current = best_validation_by_horizon.get(key)
            if (
                current is None
                or candidate["validation_score"]["score"]
                > current["validation_score"]["score"]
            ):
                best_validation_by_horizon[key] = {
                    "target": candidate["target_col"],
                    "model": candidate["model_name"],
                    "strategy": candidate["strategy"],
                    "validation_score": candidate["validation_score"],
                    "validation_trading_metrics": _simulation_summary(
                        candidate["validation_simulation"]
                    ),
                    "validation_robustness": candidate["validation_robustness"],
                }
            max_hold = candidate["strategy"]["maximum_holding_period"]
            max_hold_key = f"{candidate['horizon']}d_max_hold_{max_hold if max_hold is not None else 'unbounded'}"
            max_hold_best = best_validation_by_max_hold.get(max_hold_key)
            if (
                max_hold_best is None
                or candidate["validation_score"]["score"]
                > max_hold_best["validation_score"]["score"]
            ):
                best_validation_by_max_hold[max_hold_key] = {
                    "horizon": candidate["horizon"],
                    "maximum_holding_period": max_hold,
                    "model": candidate["model_name"],
                    "strategy": candidate["strategy"],
                    "validation_score": candidate["validation_score"],
                    "validation_trading_metrics": _simulation_summary(
                        candidate["validation_simulation"]
                    ),
                    "validation_robustness": candidate["validation_robustness"],
                }
        for horizon in (1, 3, 5):
            key = str(horizon)
            if key not in best_validation_by_horizon:
                best_validation_by_horizon[key] = {
                    "status": "no reliable edge on validation: no candidate met minimum closed trades",
                    "excluded_candidates": excluded_by_horizon.get(key, 0),
                }
        report = {
            "symbol": symbol,
            "config": {
                "initial_capital": initial_capital,
                "execution": "next_session_open",
                "execution_cost_model": "none; no commission or slippage deductions",
                "minimum_holding_period_default": minimum_holding_period,
                "min_validation_closed_trades": min_validation_closed_trades,
                "validation_windows": len(walk_plan["validation_windows"]),
                "test_windows": len(walk_plan["test_windows"]),
                "minimum_validation_bh_win_rate": min_validation_bh_win_rate,
                "minimum_validation_legacy_win_rate": min_validation_legacy_win_rate,
                "stop_loss_pct": stop_loss_pct,
                "test_used_for_selection": False,
                "validation_window_signal_dates": validation_window_ranges,
                "test_window_signal_dates": [
                    {
                        "window": index,
                        "signal_date_start": str(
                            common_dates.iloc[window["test_start"]]
                        ),
                        "signal_date_end": str(
                            common_dates.iloc[window["test_end"] - 1]
                        ),
                    }
                    for index, window in enumerate(walk_plan["test_windows"], start=1)
                ],
                "common_signal_dates_across_horizons": True,
                "holding_period_convention": "Holding sessions are open-to-open execution-index differences. A max-hold SELL signal fills at the next available open. Final-window liquidation uses the last close and may be shorter than the minimum hold.",
            },
            "target_methodology": {
                "forward_return": "open[t + horizon + 1] / open[t + 1] - 1",
                "direction_target": "UP iff forward_return > 0; otherwise DOWN",
                "legacy_target": "close[t + 1] / close[t] - 1 > 0",
                "horizon_unit": "trading sessions",
            },
            "tunindex_market_context": market_status,
            "validation_selection_method": {
                "score": "mean window excess_return_pct - 0.50 * mean absolute max_drawdown_pct - 0.10 * aggregate closed_trades_per_100_signals; excess_return_pct = strategy return - same-window buy-and-hold return",
                "tie_break_order": [
                    "higher validation score",
                    "lower absolute maximum drawdown",
                    "fewer total orders",
                    "no hysteresis",
                    "no minimum or maximum holding constraints",
                    "alphabetical config key",
                ],
                "tie_break_used": tie_break_used,
                "selection_score": selected["validation_score"],
                "minimum_validation_closed_trades": min_validation_closed_trades,
                "minimum_validation_bh_win_rate": min_validation_bh_win_rate,
                "minimum_validation_legacy_win_rate": min_validation_legacy_win_rate,
                "candidates_excluded_for_minimum_trades": excluded_candidate_count,
                "candidates_excluded_by_horizon": excluded_by_horizon,
                "selection_status": "NO_TRADE: no candidate met validation robustness criteria"
                if is_no_trade
                else "selected candidate met validation robustness criteria",
                "best_by_horizon": best_validation_by_horizon,
                "best_by_maximum_holding_period": best_validation_by_max_hold,
                "edge_assessment": validation_edge,
                "thresholds_and_holding_periods_selected_per_stock": True,
            },
            "selected_configuration": {
                "status": "NO_TRADE" if is_no_trade else "SELECTED",
                "horizon": selected["horizon"],
                "target": selected["target_col"],
                "model": selected["model_name"],
                "strategy": selected["strategy"],
                "validation_ml_metrics": _validation_metrics_for_report(selected),
                "thresholds": {
                    "classification_threshold": (
                        _validation_metrics_for_report(selected)[
                            "classification_threshold"
                        ]
                        if selected["validation_metrics"] is not None
                        else None
                    ),
                    "trading_entry_threshold": (
                        selected["strategy"]["entry_threshold"]
                        if selected["strategy"]
                        else None
                    ),
                },
                "validation_trading_metrics": _simulation_summary(
                    selected["validation_simulation"]
                ),
                "validation_score": selected["validation_score"],
                "validation_robustness": selected["validation_robustness"],
                "validation_performance_by_window": selected["validation_windows"],
                "test_ml_metrics": selected_test["ml_metrics"],
                "test_trading_metrics": _simulation_summary(selected_sim),
                "test_performance_by_window": compact_test_windows(
                    selected_test_windows
                ),
            },
            "legacy_baseline": {
                "model": legacy_candidate["model_name"],
                "strategy": legacy_candidate["strategy"],
                "validation_score": legacy_candidate["validation_score"],
                "validation_trading_metrics": _simulation_summary(
                    legacy_candidate["validation_simulation"]
                ),
                "validation_robustness": legacy_candidate["validation_robustness"],
                "validation_performance_by_window": legacy_candidate[
                    "validation_windows"
                ],
                "test_ml_metrics": legacy_test["ml_metrics"],
                "test_trading_metrics": _simulation_summary(legacy_test["simulation"]),
                "test_performance_by_window": compact_test_windows(
                    legacy_test_windows
                ),
            },
            "validation_comparison": {
                "selected_return_pct": selected["validation_simulation"][
                    "total_return_pct"
                ],
                "legacy_return_pct": legacy_candidate["validation_simulation"][
                    "total_return_pct"
                ],
                "buy_and_hold_return_pct": selected["validation_simulation"][
                    "buy_and_hold_return_pct"
                ],
                "selected_excess_vs_buy_and_hold_pct": selected["validation_score"][
                    "excess_return_pct"
                ],
                "legacy_excess_vs_buy_and_hold_pct": legacy_candidate[
                    "validation_score"
                ]["excess_return_pct"],
                "selected_excess_vs_legacy_pct": (
                    selected["validation_simulation"]["total_return_pct"]
                    - legacy_candidate["validation_simulation"]["total_return_pct"]
                ),
                "selected_score": selected["validation_score"]["score"],
                "legacy_score": legacy_candidate["validation_score"]["score"],
            },
            "test_comparison": {
                "selected_return_pct": selected_sim["total_return_pct"],
                "legacy_return_pct": legacy_test["simulation"]["total_return_pct"],
                "buy_and_hold_return_pct": selected_sim["buy_and_hold_return_pct"],
                "cash_return_pct": 0.0,
                "selected_excess_vs_buy_and_hold_pct": (
                    selected_sim["total_return_pct"]
                    - selected_sim["buy_and_hold_return_pct"]
                ),
                "legacy_excess_vs_buy_and_hold_pct": (
                    legacy_test["simulation"]["total_return_pct"]
                    - legacy_test["simulation"]["buy_and_hold_return_pct"]
                ),
                "selected_excess_vs_legacy_pct": (
                    selected_sim["total_return_pct"]
                    - legacy_test["simulation"]["total_return_pct"]
                ),
            },
            "benchmarks": {
                "legacy_strategy_test_return_pct": legacy_test["simulation"][
                    "total_return_pct"
                ],
                "buy_and_hold_test_return_pct": selected_sim["buy_and_hold_return_pct"],
                "cash_test": {
                    "final_portfolio_value": initial_capital,
                    "return_pct": 0.0,
                },
            },
            "model_file": str(model_path) if model_path else None,
        }
        report_path = report_dir / f"{symbol}_benchmark_simulation.json"
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, default=str)

        print(f" Selected model / target : {selected['model_name']} / {selected['target_col']}")
        print(f" Selected horizon        : {selected['horizon']} trading sessions")
        print(f" Selected strategy       : {selected['strategy']}")
        print(f" Test return              : {selected_sim['total_return_pct']:+.2f}%")
        print(
            f" Legacy baseline return   : {legacy_test['simulation']['total_return_pct']:+.2f}%"
        )
        print(f" Test max drawdown        : {selected_sim['max_drawdown_pct']:.2f}%")
        if validation_edge["no_candidate_beats_buy_and_hold_consistently"]:
            print(" [VALIDATION WARNING] No eligible candidate beats buy-and-hold.")
        if validation_edge["no_candidate_beats_legacy_consistently"]:
            print(" [VALIDATION WARNING] No eligible candidate beats the legacy baseline.")
        print(f" Report                   : {report_path}")
        return report

    finally:
        db.close()


def main():
    results = [run_benchmark_for_symbol(sym) for sym in ["BT", "SFBT"]]
    comparison = {
        "selection_method": "Validation score = excess return versus buy-and-hold - 0.50 * absolute drawdown - 0.10 * closed trades per 100 signals",
        "stocks": {},
    }
    for result in results:
        if result.get("status"):
            comparison["stocks"][result["symbol"]] = {"status": result["status"]}
            continue
        selected = result["selected_configuration"]
        comparison["stocks"][result["symbol"]] = {
            "best_validation_configuration": {
                "status": selected["status"] if "status" in selected else "SELECTED",
                "horizon": selected["horizon"],
                "target": selected["target"],
                "model": selected["model"],
                "strategy": selected["strategy"],
                "thresholds": selected["thresholds"],
                "validation_score": selected["validation_score"],
                "validation_ml_metrics": selected["validation_ml_metrics"],
                "validation_trading_metrics": selected["validation_trading_metrics"],
            },
            "validation_edge_assessment": result["validation_selection_method"][
                "edge_assessment"
            ],
            "final_test_return_pct": selected["test_trading_metrics"][
                "total_return_pct"
            ],
            "legacy_test_return_pct": result["legacy_baseline"][
                "test_trading_metrics"
            ]["total_return_pct"],
            "validation_comparison": result["validation_comparison"],
            "test_comparison": result["test_comparison"],
            "benchmarks": result["benchmarks"],
        }
    report_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "reports"
        / "strategy_selection_comparison.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(comparison, report_file, indent=2, default=str)
    print(f"\nValidation comparison report: {report_path}")


if __name__ == "__main__":
    main()