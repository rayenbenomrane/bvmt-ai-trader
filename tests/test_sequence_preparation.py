"""
Unit tests for chronological sequence preparation (Part 7 architecture).
Uses synthetic data exclusively - never fabricates real BVMT data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from src.models.sequence_preparation import (
    create_sliding_sequences,
    prepare_sequences_for_stock,
)
from src.models.feature_engineering import FEATURE_COLUMNS


def generate_synthetic_stock_history(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    start_date = date(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Generate synthetic geometric random walk
    returns = np.random.normal(0.0005, 0.015, n_days)
    price = 10.0
    prices = []
    for r in returns:
        price = max(1.0, price * (1.0 + r))
        prices.append(price)

    df = pd.DataFrame({
        "symbol": "TESTSTOCK",
        "date": dates,
        "open": [p * (1 - 0.002) for p in prices],
        "high": [p * (1 + 0.01) for p in prices],
        "low": [p * (1 - 0.01) for p in prices],
        "close": prices,
        "volume": np.random.randint(1000, 50000, n_days),
    })
    return df


def test_create_sliding_sequences():
    features = np.arange(20).reshape(10, 2)
    targets = np.arange(10)
    seq_len = 3

    X_seq, y_seq = create_sliding_sequences(features, targets, sequence_length=seq_len)

    # 10 samples - 3 + 1 = 8 windows
    assert X_seq.shape == (8, 3, 2)
    assert len(y_seq) == 8

    # First window should contain rows 0, 1, 2 of features
    np.testing.assert_array_equal(X_seq[0], features[0:3])
    # Target for first window is at index 2 (target for the end of window)
    assert y_seq[0] == targets[2]

    # Last window should contain rows 7, 8, 9
    np.testing.assert_array_equal(X_seq[-1], features[7:10])
    assert y_seq[-1] == targets[9]


def test_prepare_sequences_pipeline():
    df = generate_synthetic_stock_history(n_days=250)
    seq_len = 10
    
    dataset = prepare_sequences_for_stock(
        df_raw=df,
        symbol="TESTSTOCK",
        sequence_length=seq_len,
        train_ratio=0.6,
        val_ratio=0.2,
    )

    # Verify 3D tensor shapes
    assert dataset.X_train.ndim == 3
    assert dataset.X_val.ndim == 3
    assert dataset.X_test.ndim == 3

    assert dataset.X_train.shape[1] == seq_len
    assert dataset.X_train.shape[2] == len(FEATURE_COLUMNS)
    assert dataset.X_val.shape[1] == seq_len
    assert dataset.X_test.shape[1] == seq_len

    # Verify labels match sequence sample counts
    assert len(dataset.X_train) == len(dataset.y_train)
    assert len(dataset.X_val) == len(dataset.y_val)
    assert len(dataset.X_test) == len(dataset.y_test)

    # Check that scaler was fitted (has mean_ attribute)
    assert hasattr(dataset.scaler, "mean_")
    assert dataset.scaler.mean_.shape[0] == len(FEATURE_COLUMNS)

    # Verify summary dictionary output
    summary = dataset.summary()
    assert summary["symbol"] == "TESTSTOCK"
    assert summary["sequence_length"] == seq_len
    assert summary["num_features"] == len(FEATURE_COLUMNS)


def test_insufficient_data_error():
    df_short = generate_synthetic_stock_history(n_days=30)
    with pytest.raises(ValueError, match="Insufficient historical data"):
        prepare_sequences_for_stock(df_short, symbol="SHORT", sequence_length=15)
