import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.database.models import DailyPrice

def analyze_volatility(db: Session, trading_days_per_year: int = 252) -> List[Dict[str, Any]]:
    """
    Calculates returns, annualized volatility, max drawdown, best/worst daily returns.
    Convention: 252 trading days per year for annualized metrics.
    Note: Calculations use unadjusted close prices; returns may be distorted by dividends/splits.
    """
    prices_query = db.query(DailyPrice).statement
    df = pd.read_sql(prices_query, db.bind)
    
    if df.empty:
        return []

    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    results = []
    for symbol, group in df.groupby('symbol'):
        group = group.copy()
        group['daily_return'] = group['close'].pct_change()
        
        returns = group['daily_return'].dropna()
        
        if len(returns) < 2:
            results.append({
                "symbol": symbol,
                "average_daily_return": 0.0,
                "volatility_daily": 0.0,
                "annualized_volatility": 0.0,
                "max_drawdown": 0.0,
                "best_daily_return": 0.0,
                "worst_daily_return": 0.0,
                "status": "Insufficient data"
            })
            continue

        avg_daily_ret = returns.mean()
        std_daily_ret = returns.std()
        ann_volatility = std_daily_ret * np.sqrt(trading_days_per_year)
        best_ret = returns.max()
        worst_ret = returns.min()

        # Maximum Drawdown calculation
        prices = group['close']
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max
        max_dd = drawdown.min()

        results.append({
            "symbol": symbol,
            "average_daily_return": round(float(avg_daily_ret), 6),
            "volatility_daily": round(float(std_daily_ret), 6),
            "annualized_volatility": round(float(ann_volatility), 6),
            "max_drawdown": round(float(max_dd), 6),
            "best_daily_return": round(float(best_ret), 6),
            "worst_daily_return": round(float(worst_ret), 6),
            "status": "Active"
        })

    return results