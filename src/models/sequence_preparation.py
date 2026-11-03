"""
BVMT AI Trader - Sequence Preparation for Neural Networks (Part 7: LSTM / GRU)

Provides reusable, leak-free interfaces for chronological 3D tensor sequence generation:
- Converts 2D tabular features into 3D sequence arrays: (N, sequence_length, num_features).
- Configurable sequence length (e.g. 10, 20 sessions).
- Feature scaling fitted strictly on training data only (StandardScaler or RobustScaler).
- Target labels perfectly aligned with the subsequent session (t+1).
- Independent processing for BT and SFBT stocks.
- Chronological train/validation/test split without look-ahead bias or future-data leakage.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sqlalchemy.orm import Session

from src.models.feature_engineering import (
    FEATURE_COLUMNS,
    get_available_feature_columns,
    prepare_labeled_dataset,
    purged_chronological_splits,
)
from src.database.models import DailyPrice


@dataclass
class SequenceDataset:
    symbol: str
    sequence_length: int
    feature_names: List[str]
    
    # 3D Tensor arrays: (N, sequence_length, num_features)
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    
    # Associated dates for the prediction target session (t)
    dates_train: pd.Series
    dates_val: pd.Series
    dates_test: pd.Series
    
    # Fitted scaler instance
    scaler: Any
    
    # Execution prices (next-session open) and close prices for simulation alignment
    test_exec_prices: pd.Series
    test_close_prices: pd.Series
    
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Returns structural dimensions and properties of the prepared sequence dataset."""
        return {
            "symbol": self.symbol,
            "sequence_length": self.sequence_length,
            "num_features": len(self.feature_names),
            "train_shape": list(self.X_train.shape),
            "val_shape": list(self.X_val.shape),
            "test_shape": list(self.X_test.shape),
            "train_class_ratio": float(np.mean(self.y_train)),
            "val_class_ratio": float(np.mean(self.y_val)),
            "test_class_ratio": float(np.mean(self.y_test)),
            "date_range_train": f"{self.dates_train.iloc[0]} to {self.dates_train.iloc[-1]}" if len(self.dates_train) > 0 else "N/A",
            "date_range_val": f"{self.dates_val.iloc[0]} to {self.dates_val.iloc[-1]}" if len(self.dates_val) > 0 else "N/A",
            "date_range_test": f"{self.dates_test.iloc[0]} to {self.dates_test.iloc[-1]}" if len(self.dates_test) > 0 else "N/A",
        }


def create_sliding_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs 3D chronological sliding sequences.
    
    For sample index i (from sequence_length - 1 to len(features) - 1):
      sequence_i = features[i - sequence_length + 1 : i + 1]  (shape: sequence_length, D)
      target_i = targets[i]
      
    Args:
        features: 2D array of shape (N, D).
        targets: 1D array of shape (N,).
        sequence_length: Number of time steps per window.
        
    Returns:
        X_seq: 3D array of shape (N - sequence_length + 1, sequence_length, D)
        y_seq: 1D array of shape (N - sequence_length + 1,)
    """
    n_samples, n_features = features.shape
    if n_samples < sequence_length:
        raise ValueError(
            f"Insufficient samples ({n_samples}) for sequence length ({sequence_length})."
        )

    num_windows = n_samples - sequence_length + 1
    X_seq = np.zeros((num_windows, sequence_length, n_features), dtype=np.float32)
    y_seq = np.zeros(num_windows, dtype=targets.dtype)

    for i in range(num_windows):
        X_seq[i] = features[i : i + sequence_length]
        y_seq[i] = targets[i + sequence_length - 1]

    return X_seq, y_seq


def prepare_sequences_for_stock(
    df_raw: pd.DataFrame,
    symbol: str,
    sequence_length: int = 10,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    target_type: str = "binary",
    scaler_type: str = "standard",
    horizon: int = 1,
    market_df: Optional[pd.DataFrame] = None,
) -> SequenceDataset:
    """
    Transforms raw OHLCV stock history into clean 3D sequence tensors ready for LSTM/GRU.
    
    Guarantees:
    - Chronological order is strictly maintained.
    - Scaler is fitted on training split only; validation and test splits are transformed.
    - Zero data leakage between partitions.
    - Target label y_i represents the market direction for session i+1.
    """
    if sequence_length not in {10, 20}:
        raise ValueError("sequence_length must be 10 or 20 trading sessions.")
    if df_raw.empty or len(df_raw) < (sequence_length + 60):
        raise ValueError(f"Insufficient historical data for symbol '{symbol}' (got {len(df_raw)} rows).")

    # 1. Feature Engineering and Labeling
    df_clean = prepare_labeled_dataset(df_raw, horizon=horizon, market_df=market_df)
    target_columns = {
        "binary": "target_binary",
        "3class": "target_3class",
        "legacy_1d": "target_legacy_1d",
    }
    if target_type not in target_columns:
        raise ValueError(f"Unsupported target_type: {target_type}")
    target_col = target_columns[target_type]
    feature_columns = get_available_feature_columns(df_clean)

    total_len = len(df_clean)
    split = purged_chronological_splits(
        total_len,
        horizon,
        train_ratio,
        val_ratio,
        legacy_target=target_col == "target_legacy_1d",
    )
    train_end = split["train_end"]
    val_end = split["test_start"]
    train_stop = split["train_stop"]
    val_stop = split["validation_stop"]
    if train_stop < sequence_length or val_stop <= train_end or total_len <= val_end:
        raise ValueError("Insufficient samples after purging horizon-overlapping sequence labels.")

    df_train = df_clean.iloc[:train_stop].copy().reset_index(drop=True)
    df_val = df_clean.iloc[train_end:val_stop].copy().reset_index(drop=True)
    df_test = df_clean.iloc[val_end:].copy().reset_index(drop=True)

    # 2. Fit Scaler ONLY on Training Data
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    scaler.fit(df_train[feature_columns].values)

    # Apply the training-fitted scaler to the complete chronological matrix.
    all_scaled = scaler.transform(df_clean[feature_columns].values)
    train_scaled = all_scaled[:train_stop]
    val_scaled = all_scaled[train_end:val_stop]
    test_scaled = all_scaled[val_end:]

    # Prepend earlier scaled observations as context without including future rows.
    val_context_start = train_end - sequence_length + 1
    val_context = all_scaled[val_context_start:val_stop]
    val_targets = df_clean[target_col].iloc[val_context_start:val_stop].values
    test_context_start = val_end - sequence_length + 1
    test_context = all_scaled[test_context_start:]
    test_targets = df_clean[target_col].iloc[test_context_start:].values

    X_train, y_train = create_sliding_sequences(train_scaled, df_train[target_col].values, sequence_length)
    X_val, y_val = create_sliding_sequences(val_context, val_targets, sequence_length)
    X_test, y_test = create_sliding_sequences(test_context, test_targets, sequence_length)

    # Slice date series to align with window outputs
    dates_train = df_train["date"].iloc[sequence_length - 1:].reset_index(drop=True)
    dates_val = df_clean["date"].iloc[train_end:val_stop].reset_index(drop=True)
    dates_test = df_test["date"].reset_index(drop=True)

    test_exec_prices = df_test["next_open"].reset_index(drop=True)
    test_close_prices = df_test["next_close"].reset_index(drop=True)

    # Verify partition lengths match
    assert len(X_val) == len(dates_val) == len(df_val), "Validation length mismatch"
    assert len(X_test) == len(dates_test) == len(df_test), "Test length mismatch"

    metadata = {
        "symbol": symbol,
        "target_type": target_type,
        "scaler_type": scaler_type,
        "sequence_length": sequence_length,
        "features": feature_columns,
        "horizon_sessions": horizon,
        "total_sessions": total_len,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
    }

    return SequenceDataset(
        symbol=symbol,
        sequence_length=sequence_length,
        feature_names=feature_columns,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        dates_train=dates_train,
        dates_val=dates_val,
        dates_test=dates_test,
        scaler=scaler,
        test_exec_prices=test_exec_prices,
        test_close_prices=test_close_prices,
        metadata=metadata,
    )


def load_and_prepare_sequences(
    db: Session,
    symbol: str,
    sequence_length: int = 15,
    scaler_type: str = "standard",
) -> SequenceDataset:
    """Loads historical prices from PostgreSQL and generates sequence dataset for the given symbol."""
    query = (
        db.query(DailyPrice)
        .filter(DailyPrice.symbol == symbol)
        .order_by(DailyPrice.date.asc())
        .statement
    )
    df_raw = pd.read_sql(query, db.bind)
    return prepare_sequences_for_stock(
        df_raw=df_raw,
        symbol=symbol,
        sequence_length=sequence_length,
        scaler_type=scaler_type,
    )
