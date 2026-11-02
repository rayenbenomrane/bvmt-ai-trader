"""
BVMT AI Trader - Feature Engineering Pipeline

Generates stationary, scale-invariant technical and financial features for
Tunisian Stock Exchange (BVMT) securities without future data leakage.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


FEATURE_COLUMNS: List[str] = [
    "return_1d",
    "return_2d",
    "return_5d",
    "volatility_5d",
    "volatility_20d",
    "dist_sma20",
    "dist_sma50",
    "volume_rel_20",
    "rsi_14",
    "rsi_14_diff",
    "macd_norm",
    "macd_hist_norm",
    "macd_hist_diff",
    "norm_range",
    "norm_close_pos",
    "atr14_norm",
]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Wilder's Exponential Relative Strength Index (RSI).
    Output range: [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Exponential moving averages for gain and loss (Wilder's smoothing)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stationary technical indicators and normalized financial features.
    
    Guarantees:
    - Chronological ordering is preserved.
    - Zero future-data leakage (all calculations use strictly past and current values).
    - Features are scale-invariant / percentage-based.
    
    Args:
        df: DataFrame containing at least ['date', 'open', 'high', 'low', 'close', 'volume'].
        
    Returns:
        DataFrame with new technical feature columns.
    """
    df = df.sort_values("date").copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Historical Returns over 1, 2, and 5 sessions
    df["return_1d"] = df["close"].pct_change(1)
    df["return_2d"] = df["close"].pct_change(2)
    df["return_5d"] = df["close"].pct_change(5)

    # 2. Rolling Volatility (standard deviation of daily returns)
    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()
    df["volatility_20d"] = df["return_1d"].rolling(window=20).std()

    # 3. Moving Averages & Percentage Distance
    sma20 = df["close"].rolling(window=20).mean()
    sma50 = df["close"].rolling(window=50).mean()
    df["dist_sma20"] = (df["close"] - sma20) / (sma20 + 1e-8)
    df["dist_sma50"] = (df["close"] - sma50) / (sma50 + 1e-8)

    # 4. Relative Volume (ratio to 20-session rolling average)
    vol_sma20 = df["volume"].rolling(window=20).mean()
    df["volume_rel_20"] = df["volume"] / (vol_sma20 + 1e-8)

    # 5. RSI (14-session) and 1-session RSI Change
    df["rsi_14"] = compute_rsi(df["close"], period=14)
    df["rsi_14_diff"] = df["rsi_14"].diff(1)

    # 6. Normalized MACD & Changes (scale-independent relative to price)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    raw_macd = ema12 - ema26
    df["macd_norm"] = raw_macd / (df["close"] + 1e-8)
    macd_signal = df["macd_norm"].ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = df["macd_norm"] - macd_signal
    df["macd_hist_diff"] = df["macd_hist_norm"].diff(1)

    # 7. Normalized Intraday Price Range & Position
    high_low_range = df["high"] - df["low"]
    df["norm_range"] = high_low_range / (df["close"] + 1e-8)
    df["norm_close_pos"] = (df["close"] - df["low"]) / (high_low_range + 1e-8)

    # 8. Normalized Average True Range (14-session ATR / close)
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low_range, high_close, low_close], axis=1).max(axis=1)
    df["atr14_norm"] = true_range.rolling(window=14).mean() / (df["close"] + 1e-8)

    return df


def prepare_labeled_dataset(
    df: pd.DataFrame,
    threshold_pct: float = 0.005,
) -> pd.DataFrame:
    """
    Attaches next-session returns and targets without look-ahead bias:
    - next_close: Close of session t+1
    - next_return: (Close_{t+1} - Close_t) / Close_t
    - next_open: Open of session t+1 (used for realistic next-session execution)
    - target_binary: 1 if next_return > 0 else 0
    - target_3class: 2 if next_return > threshold, 0 if next_return < -threshold, 1 otherwise
    
    Drops any rows with incomplete features or targets.
    """
    df_feat = build_features(df)

    df_feat["next_close"] = df_feat["close"].shift(-1)
    df_feat["next_open"] = df_feat["open"].shift(-1)
    df_feat["next_date"] = df_feat["date"].shift(-1)
    df_feat["next_return"] = (df_feat["next_close"] - df_feat["close"]) / df_feat["close"]

    # Target: binary direction (1 = UP, 0 = FLAT/DOWN)
    df_feat["target_binary"] = (df_feat["next_return"] > 0).astype(int)

    # Target: three-class direction (2 = UP, 0 = DOWN, 1 = NEUTRAL)
    conditions = [
        df_feat["next_return"] > threshold_pct,
        df_feat["next_return"] < -threshold_pct,
    ]
    choices = [2, 0]
    df_feat["target_3class"] = np.select(conditions, choices, default=1)

    # Drop warm-up rows (moving averages, ATR, RSI require up to 50 sessions) and final row (missing next_return)
    clean_cols = FEATURE_COLUMNS + ["target_binary", "target_3class", "next_return", "next_open", "next_close", "next_date"]
    df_clean = df_feat.dropna(subset=clean_cols).copy().reset_index(drop=True)
    return df_clean
