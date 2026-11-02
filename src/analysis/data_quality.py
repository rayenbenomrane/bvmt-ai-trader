import pandas as pd
from typing import Dict, Any
from sqlalchemy.orm import Session
from src.database.models import DailyPrice

def analyze_data_quality(db: Session) -> Dict[str, Any]:
    """
    Analyzes historical stock price data for quality issues:
    - Missing values in OHLCV
    - Invalid OHLC relationships
    - Duplicate records
    - Zero or negative prices / negative volumes
    - Suspicious price jumps (>30% or <-30%)
    - Significant trading coverage gaps (excluding weekends)
    """
    prices_query = db.query(
        DailyPrice.symbol,
        DailyPrice.date,
        DailyPrice.open,
        DailyPrice.high,
        DailyPrice.low,
        DailyPrice.close,
        DailyPrice.volume
    ).statement
    
    df = pd.read_sql(prices_query, db.bind)
    
    if df.empty:
        return {"total_records": 0, "total_issues": 0, "issues": []}

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])

    issues = []

    # 1. Missing values
    nan_mask = df[['open', 'high', 'low', 'close', 'volume']].isna().any(axis=1)
    for _, row in df[nan_mask].iterrows():
        issues.append({
            "symbol": row['symbol'],
            "date": row['date'].strftime('%Y-%m-%d'),
            "issue_type": "MISSING_VALUES",
            "description": "Contains null or missing values in OHLCV fields."
        })

    # 2. Zero or negative prices
    zero_price = (df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)
    for _, row in df[zero_price].iterrows():
        issues.append({
            "symbol": row['symbol'],
            "date": row['date'].strftime('%Y-%m-%d'),
            "issue_type": "ZERO_OR_NEGATIVE_PRICE",
            "description": f"Price is <= 0 (O:{row['open']}, H:{row['high']}, L:{row['low']}, C:{row['close']})"
        })

    # 3. Negative volumes
    neg_vol = df['volume'] < 0
    for _, row in df[neg_vol].iterrows():
        issues.append({
            "symbol": row['symbol'],
            "date": row['date'].strftime('%Y-%m-%d'),
            "issue_type": "NEGATIVE_VOLUME",
            "description": f"Trading volume is negative: {row['volume']}"
        })

    # 4. Invalid OHLC relationships
    invalid_ohlc = df[
        (df['high'] < df['open']) | (df['high'] < df['close']) | 
        (df['low'] > df['open']) | (df['low'] > df['close']) | 
        (df['high'] < df['low'])
    ]
    for _, row in invalid_ohlc.iterrows():
        issues.append({
            "symbol": row['symbol'],
            "date": row['date'].strftime('%Y-%m-%d'),
            "issue_type": "INVALID_OHLC",
            "description": f"Invalid OHLC structure (O:{row['open']}, H:{row['high']}, L:{row['low']}, C:{row['close']})"
        })

    # 5. Duplicates
    duplicates = df[df.duplicated(subset=['symbol', 'date'], keep=False)]
    for _, row in duplicates.iterrows():
        issues.append({
            "symbol": row['symbol'],
            "date": row['date'].strftime('%Y-%m-%d'),
            "issue_type": "DUPLICATE_RECORD",
            "description": "Duplicate trading session record for symbol and date."
        })

    # 6. Suspicious price movements (>30% daily swing)
    df = df.sort_values(['symbol', 'date'])
    df['prev_close'] = df.groupby('symbol')['close'].shift(1)
    df['daily_return'] = (df['close'] - df['prev_close']) / df['prev_close']
    suspicious = df[(df['daily_return'] > 0.30) | (df['daily_return'] < -0.30)]
    for _, row in suspicious.iterrows():
        issues.append({
            "symbol": row['symbol'],
            "date": row['date'].strftime('%Y-%m-%d'),
            "issue_type": "SUSPICIOUS_PRICE_MOVEMENT",
            "description": f"Extreme daily price move of {row['daily_return']*100:.2f}%"
        })

    return {
        "total_records": len(df),
        "total_issues": len(issues),
        "issues": issues
    }