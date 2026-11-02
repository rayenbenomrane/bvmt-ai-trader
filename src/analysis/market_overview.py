import pandas as pd
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from src.database.models import Instrument, DailyPrice

def get_market_overview(db: Session) -> Dict[str, Any]:
    """Calculates overall market statistics, registered securities, and coverage."""
    instruments = db.query(Instrument).all()
    total_securities = len(instruments)
    
    prices_query = db.query(DailyPrice).statement
    df = pd.read_sql(prices_query, db.bind)
    
    if df.empty:
        return {
            "total_registered_securities": total_securities,
            "securities_with_data": 0,
            "total_historical_records": 0,
            "first_trading_date": None,
            "last_trading_date": None,
            "stock_stats": []
        }

    df['date'] = pd.to_datetime(df['date'])
    securities_with_data = df['symbol'].nunique()
    total_records = len(df)
    first_date = df['date'].min().strftime('%Y-%m-%d')
    last_date = df['date'].max().strftime('%Y-%m-%d')

    stock_stats = []
    for symbol, group in df.groupby('symbol'):
        rec_count = len(group)
        min_d = group['date'].min()
        max_d = group['date'].max()
        # Coverage percentage against calendar business days in range
        b_days = pd.bdate_range(start=min_d, end=max_d)
        coverage_pct = (rec_count / len(b_days)) * 100 if len(b_days) > 0 else 0.0
        
        stock_stats.append({
            "symbol": symbol,
            "records": rec_count,
            "first_date": min_d.strftime('%Y-%m-%d'),
            "last_date": max_d.strftime('%Y-%m-%d'),
            "coverage_percentage": round(min(coverage_pct, 100.0), 2)
        })

    return {
        "total_registered_securities": total_securities,
        "securities_with_data": securities_with_data,
        "total_historical_records": total_records,
        "first_trading_date": first_date,
        "last_trading_date": last_date,
        "stock_stats": stock_stats
    }