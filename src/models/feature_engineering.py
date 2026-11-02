import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates stationary technical features and rolling metrics without future leakage."""
    df = df.sort_values('date').copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Returns & Lagged Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_2d'] = df['close'].pct_change(2)
    df['return_5d'] = df['close'].pct_change(5)

    # Moving Averages & Percentage Distance
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['dist_sma20'] = (df['close'] - df['sma20']) / df['sma20']
    df['dist_sma50'] = (df['close'] - df['sma50']) / df['sma50']

    # EMA ratio
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_ratio'] = df['ema12'] / df['ema26']

    # MACD & Changes
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_hist_diff'] = df['macd_hist'].diff()

    # Rolling Volatility (5 and 20 sessions)
    df['volatility_5d'] = df['return_1d'].rolling(window=5).std()
    df['volatility_20d'] = df['return_1d'].rolling(window=20).std()

    # Volume relative to 20-session average
    df['vol_sma20'] = df['volume'].rolling(window=20).mean()
    df['volume_rel_20'] = df['volume'] / (df['vol_sma20'] + 1e-8)

    # ATR 14 normalized by close
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14_norm'] = true_range.rolling(window=14).mean() / df['close']

    return df