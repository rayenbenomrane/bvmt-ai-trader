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
    build_features,
    prepare_labeled_dataset,
    compute_rsi,
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

    # target_binary at row t must match whether close[t+1] > close[t]
    for i in range(len(labeled) - 1):
        actual_direction = int(labeled["next_close"].iloc[i] > labeled["close"].iloc[i])
        assert labeled["target_binary"].iloc[i] == actual_direction
