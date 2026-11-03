"""
BVMT AI Trader - Feature Engineering Pipeline

Generates stationary, scale-invariant technical and financial features for
Tunisian Stock Exchange (BVMT) securities without future data leakage.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


FEATURE_COLUMNS: List[str] = [
    "return_1d",
    "return_2d",
    "return_3d",
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
    "ema12_ema26_ratio",
    "norm_range",
    "norm_close_pos",
    "atr14_norm",
]

MARKET_FEATURE_COLUMNS: List[str] = [
    "tunindex_return_1d",
    "tunindex_return_5d",
    "tunindex_volatility_10d",
    "tunindex_trend_20d",
]


def get_available_feature_columns(df: pd.DataFrame) -> List[str]:
    """Use market features only when real aligned market observations exist."""
    columns = list(FEATURE_COLUMNS)
    if all(column in df.columns for column in MARKET_FEATURE_COLUMNS) and df[MARKET_FEATURE_COLUMNS].notna().any().any():
        columns.extend(MARKET_FEATURE_COLUMNS)
    return columns


def purged_chronological_splits(
    total_samples: int,
    horizon: int,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    legacy_target: bool = False,
    purge_horizon: Optional[int] = None,
    split_boundaries: Optional[Dict[str, int]] = None,
) -> dict:
    """Return chronological slices whose labels end before the next partition."""
    if total_samples < 1:
        raise ValueError("total_samples must be positive.")
    if horizon < 1:
        raise ValueError("horizon must be a positive number of trading sessions.")
    if purge_horizon is not None and purge_horizon < horizon:
        raise ValueError("purge_horizon cannot be shorter than the target horizon.")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one.")

    if split_boundaries is None:
        train_end = int(total_samples * train_ratio)
        val_start = train_end
        val_end = int(total_samples * (train_ratio + val_ratio))
        test_start = val_end
        test_end = total_samples
    else:
        train_end = int(split_boundaries["train_end"])
        val_start = int(split_boundaries.get("validation_start", train_end))
        val_end = int(split_boundaries["validation_end"])
        test_start = int(split_boundaries.get("test_start", val_end))
        test_end = int(split_boundaries.get("test_end", total_samples))
        if not (0 < train_end <= val_start < val_end <= test_start <= test_end <= total_samples):
            raise ValueError("split_boundaries must be chronological and within total_samples.")
    purge_periods = (
        purge_horizon + 1
        if purge_horizon is not None
        else (1 if legacy_target else horizon + 1)
    )
    train_stop = max(0, train_end - purge_periods)
    val_stop = max(val_start, val_end - purge_periods)
    return {
        "train": slice(0, train_stop),
        "validation": slice(val_start, val_stop),
        "test": slice(test_start, test_end),
        "train_stop": train_stop,
        "train_end": train_end,
        "validation_start": val_start,
        "validation_stop": val_stop,
        "validation_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "purge_periods": purge_periods,
    }


def rolling_walk_forward_plan(
    total_samples: int,
    validation_window_count: int = 4,
    test_window_count: int = 3,
    initial_train_ratio: float = 0.35,
    validation_window_ratio: float = 0.08,
) -> Dict[str, List[Dict[str, int]]]:
    """Create expanding-train validation windows followed by disjoint final test windows."""
    if total_samples < 1 or validation_window_count < 1 or test_window_count < 1:
        raise ValueError("sample and window counts must be positive.")
    if not 0 < initial_train_ratio < 1 or not 0 < validation_window_ratio < 1:
        raise ValueError("train and validation ratios must be between zero and one.")

    initial_train_end = int(total_samples * initial_train_ratio)
    validation_size = max(1, int(total_samples * validation_window_ratio))
    validation_windows = []
    for window in range(validation_window_count):
        train_end = initial_train_end + window * validation_size
        validation_end = train_end + validation_size
        validation_windows.append({
            "train_end": train_end,
            "validation_start": train_end,
            "validation_end": validation_end,
            "test_start": validation_end,
            "test_end": validation_end,
        })

    final_validation_end = validation_windows[-1]["validation_end"]
    remaining = total_samples - final_validation_end
    if remaining < test_window_count:
        raise ValueError("Insufficient samples for requested rolling validation and test windows.")
    base_test_size = remaining // test_window_count
    test_windows = []
    test_start = final_validation_end
    for window in range(test_window_count):
        test_end = total_samples if window == test_window_count - 1 else test_start + base_test_size
        test_windows.append({"test_start": test_start, "test_end": test_end})
        test_start = test_end

    if validation_windows[0]["train_end"] <= 6:
        raise ValueError("Insufficient initial training samples for the longest target horizon.")
    return {"validation_windows": validation_windows, "test_windows": test_windows}


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


def build_features(df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
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

    # 1. Historical Returns over 1, 2, 3, and 5 sessions
    df["return_1d"] = df["close"].pct_change(1)
    df["return_2d"] = df["close"].pct_change(2)
    df["return_3d"] = df["close"].pct_change(3)
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
    df["ema12_ema26_ratio"] = ema12 / (ema26 + 1e-8) - 1.0
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

    if market_df is not None and not market_df.empty:
        market = market_df[["date", "close"]].copy()

        # --- FIX: Force both date columns to exact nanosecond precision ---
        market["date"] = pd.to_datetime(market["date"], errors="coerce").astype('datetime64[ns]')
        market["close"] = pd.to_numeric(market["close"], errors="coerce")
        market = market.dropna(subset=["date", "close"]).sort_values("date")
        market = market.drop_duplicates("date", keep="last")

        market["tunindex_return_1d"] = market["close"].pct_change(1)
        market["tunindex_return_5d"] = market["close"].pct_change(5)
        market["tunindex_volatility_10d"] = market["tunindex_return_1d"].rolling(10).std()
        market_sma20 = market["close"].rolling(20).mean()
        market["tunindex_trend_20d"] = market["close"] / (market_sma20 + 1e-8) - 1.0

        # --- FIX: Force df date to exact nanosecond precision ---
        df["date"] = pd.to_datetime(df["date"], errors="coerce").astype('datetime64[ns]')

        df = pd.merge_asof(
            df.sort_values("date"),
            market[["date"] + MARKET_FEATURE_COLUMNS].sort_values("date"),
            on="date",
            direction="backward",
        )

    return df

def prepare_labeled_dataset(
    df: pd.DataFrame,
    threshold_pct: float = 0.005,
    horizon: int = 1,
    market_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Labels signal-date features using next-session-open entry and next-session-open exit.

    forward_return[t] = open[t + horizon + 1] / open[t + 1] - 1.
    target_binary is UP when forward_return > 0.
    The historical close-to-close one-session target remains target_legacy_1d.
    """
    if horizon < 1:
        raise ValueError("horizon must be a positive number of trading sessions.")

    df_feat = build_features(df, market_df=market_df)

    df_feat["next_close"] = df_feat["close"].shift(-1)
    df_feat["next_open"] = df_feat["open"].shift(-1)
    df_feat["next_date"] = df_feat["date"].shift(-1)
    df_feat["next_return"] = (df_feat["next_close"] - df_feat["close"]) / df_feat["close"]
    df_feat["forward_exit_open"] = df_feat["open"].shift(-(horizon + 1))
    df_feat["forward_return"] = df_feat["forward_exit_open"] / df_feat["next_open"] - 1.0

    valid_target = df_feat["forward_return"].notna()
    df_feat["target_legacy_1d"] = np.where(
        df_feat["next_return"].notna(), (df_feat["next_return"] > 0).astype(int), np.nan
    )
    df_feat["target_binary"] = np.where(
        valid_target, (df_feat["forward_return"] > 0).astype(int), np.nan
    )
    # Target: three-class direction (2 = UP, 0 = DOWN, 1 = NEUTRAL)
    conditions = [
        df_feat["next_return"] > threshold_pct,
        df_feat["next_return"] < -threshold_pct,
    ]
    choices = [2, 0]
    df_feat["target_3class"] = np.select(conditions, choices, default=1)

    # Drop warm-up rows (moving averages, ATR, RSI require up to 50 sessions) and final row (missing next_return)
    clean_cols = get_available_feature_columns(df_feat) + [
        "target_binary", "target_3class", "forward_return", "forward_exit_open",
        "next_return", "next_open", "next_close", "next_date",
    ]
    df_clean = df_feat.dropna(subset=clean_cols).copy().reset_index(drop=True)
    df_clean["target_binary"] = df_clean["target_binary"].astype(int)
    df_clean["target_legacy_1d"] = df_clean["target_legacy_1d"].astype(int)
    return df_clean
