"""
Unit tests for Feature Engineering pipeline.
Uses synthetic OHLCV data to verify:
- Stationary features and mathematical correctness.
- No look-ahead bias (strict temporal causality).
- RSI calculation bounds [0, 100].
- Target alignment with session t+1.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from src.models.feature_engineering import (
    FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    build_features,
    prepare_labeled_dataset,
    compute_rsi,
    round_trip_cost_return,
)


def create_synthetic_ohlcv(n_days: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    start = date(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    
    # Random walk prices
    returns = np.random.normal(0.001, 0.01, n_days)
    close = 10.0 * np.cumprod(1.0 + returns)
    high = close * (1.0 + np.random.uniform(0.002, 0.015, n_days))
    low = close * (1.0 - np.random.uniform(0.002, 0.015, n_days))
    open_p = (high + low) / 2.0
    volume = np.random.randint(1000, 20000, n_days)

    return pd.DataFrame({
        "symbol": "TEST",
        "date": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_rsi_bounds_and_continuity():
    s = pd.Series([10.0 + i * 0.5 for i in range(50)])  # Monotonically increasing
    rsi = compute_rsi(s, period=14).dropna()
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()
    # Continuous upward moves should push RSI close to 100
    assert rsi.iloc[-1] > 90.0

    s_down = pd.Series([50.0 - i * 0.5 for i in range(50)])  # Monotonically decreasing
    rsi_down = compute_rsi(s_down, period=14).dropna()
    assert rsi_down.iloc[-1] < 10.0


def test_feature_engineering_columns():
    df = create_synthetic_ohlcv(120)
    df_feat = build_features(df)
    for col in FEATURE_COLUMNS:
        assert col in df_feat.columns, f"Missing feature column: {col}"


def test_no_future_leakage_in_features():
    df = create_synthetic_ohlcv(80)
    feat1 = build_features(df)
    
    # Mutate the last 5 days
    df_mutated = df.copy()
    df_mutated.iloc[-5:, df.columns.get_loc("close")] *= 2.0
    feat2 = build_features(df_mutated)

    # All features prior to index -5 must remain bit-for-bit identical
    for col in FEATURE_COLUMNS:
        np.testing.assert_array_almost_equal(
            feat1[col].iloc[:-5].values,
            feat2[col].iloc[:-5].values,
            err_msg=f"Future leakage detected in feature: {col}",
        )


def test_target_alignment():
    df = create_synthetic_ohlcv(80)
    labeled = prepare_labeled_dataset(df)

    # One held session means entry at open t+1 and exit at open t+2.
    expected_return = labeled["forward_exit_open"] / labeled["next_open"] - 1.0
    np.testing.assert_array_equal(labeled["target_binary"], (expected_return > 0).astype(int))
    legacy_return = (labeled["next_close"] - labeled["close"]) / labeled["close"]
    np.testing.assert_array_equal(labeled["target_legacy_1d"], (legacy_return > 0).astype(int))


def test_horizon_uses_trading_rows_and_drops_unavailable_targets():
    df = create_synthetic_ohlcv(90)
    feature_rows = build_features(df).dropna(subset=FEATURE_COLUMNS)
    ordered = df.sort_values("date").reset_index(drop=True)
    for horizon in (1, 3, 5):
        labeled = prepare_labeled_dataset(df, horizon=horizon)
        assert len(labeled) == len(feature_rows) - horizon - 1
        source_positions = {day: index for index, day in enumerate(ordered["date"])}
        expected = []
        for day in labeled["date"]:
            signal_index = source_positions[day]
            expected.append(
                ordered["open"].iloc[signal_index + horizon + 1]
                / ordered["open"].iloc[signal_index + 1]
                - 1.0
            )
        np.testing.assert_allclose(labeled["forward_return"], expected)


def test_round_trip_cost_includes_both_sides_and_compounding():
    actual = round_trip_cost_return(fee_rate=0.004, slippage_pct=0.001)
    expected = ((1.001 * 1.004) / (0.999 * 0.996)) - 1.0
    assert actual == pytest.approx(expected)


def test_market_features_align_only_to_current_or_prior_market_date():
    stock = create_synthetic_ohlcv(100)
    market = pd.DataFrame({
        "date": [stock["date"].iloc[60], stock["date"].iloc[70], stock["date"].iloc[80]],
        "close": [100.0, 110.0, 99.0],
    })
    features = build_features(stock, market_df=market)

    for column in MARKET_FEATURE_COLUMNS:
        assert column in features.columns
    assert features["tunindex_return_1d"].iloc[70] == pytest.approx(0.1)
    assert pd.isna(features["tunindex_return_1d"].iloc[69])
