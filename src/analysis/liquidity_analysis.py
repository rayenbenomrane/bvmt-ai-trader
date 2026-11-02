import pandas as pd
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.database.models import DailyPrice

def analyze_liquidity(db: Session) -> List[Dict[str, Any]]:
    """Calculates average/median volume, zero-volume sessions, and trading value per stock."""
    prices_query = db.query(DailyPrice).statement
    df = pd.read_sql(prices_query, db.bind)
    
    if df.empty:
        return []

    for col in ['close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Estimated daily traded value = Close * Volume
    df['traded_value'] = df['close'] * df['volume']

    results = []
    for symbol, group in df.groupby('symbol'):
        avg_vol = group['volume'].mean()
        median_vol = group['volume'].median()
        zero_vol_sessions = int((group['volume'] == 0).sum())
        avg_traded_val = group['traded_value'].mean()
        
        results.append({
            "symbol": symbol,
            "average_daily_volume": round(float(avg_vol), 2),
            "median_daily_volume": round(float(median_vol), 2),
            "zero_volume_sessions": zero_vol_sessions,
            "average_daily_value": round(float(avg_traded_val), 3),
            "total_sessions": len(group)
        })

    # Sort by average daily trading volume descending (most active first)
    results = sorted(results, key=lambda x: x["average_daily_volume"], reverse=True)
    return results