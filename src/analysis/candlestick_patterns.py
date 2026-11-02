import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config.database import SessionLocal, Base, engine
from src.database.models import DailyPrice, CandlestickPattern, Instrument

def detect_candlestick_patterns_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects candlestick patterns for a single stock dataframe sorted chronologically by date.
    Required columns in df: ['symbol', 'date', 'open', 'high', 'low', 'close']
    Returns a list of detected pattern dictionaries.
    """
    patterns = []
    if df.empty or len(df) < 3:
        return patterns

    df = df.sort_values('date').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close', 'date'])
    if len(df) < 3:
        return patterns

    symbol = df['symbol'].iloc[0]

    for i in range(2, len(df)):
        c0 = df.iloc[i]     # Current candle
        c1 = df.iloc[i-1]   # Previous candle
        c2 = df.iloc[i-2]   # 2 days ago

        if (c0['high'] < c0['low'] or c1['high'] < c1['low'] or c2['high'] < c2['low'] or
            c0['high'] < max(c0['open'], c0['close']) or c0['low'] > min(c0['open'], c0['close'])):
            continue

        date_val = c0['date']
        if isinstance(date_val, str):
            date_val = datetime.strptime(date_val, '%Y-%m-%d').date()
        elif hasattr(date_val, 'date'):
            date_val = date_val.date()

        o, h, l, c = c0['open'], c0['high'], c0['low'], c0['close']
        body = abs(c - o)
        total_range = h - l
        if total_range == 0:
            continue

        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        o1, h1, l1, c1_val = c1['open'], c1['high'], c1['low'], c1['close']
        body1 = abs(c1_val - o1)

        o2, h2, l2, c2_val = c2['open'], c2['high'], c2['low'], c2['close']
        body2 = abs(c2_val - o2)

        # 1. Doji
        if body <= 0.1 * total_range:
            patterns.append({"symbol": symbol, "date": date_val, "pattern_name": "Doji", "direction": "NEUTRAL"})

        # 2. Hammer
        if lower_shadow >= 2 * body and upper_shadow <= 0.2 * total_range and body > 0:
            patterns.append({"symbol": symbol, "date": date_val, "pattern_name": "Hammer", "direction": "BULLISH"})

        # 3. Inverted Hammer
        if upper_shadow >= 2 * body and lower_shadow <= 0.2 * total_range and body > 0:
            patterns.append({"symbol": symbol, "date": date_val, "pattern_name": "Inverted Hammer", "direction": "BULLISH"})

        # 4. Shooting Star
        if upper_shadow >= 2 * body and lower_shadow <= 0.2 * total_range and body > 0 and c < o:
            patterns.append({"symbol": symbol, "date": date_val, "pattern_name": "Shooting Star", "direction": "BEARISH"})

        # 5. Hanging Man
        if lower_shadow >= 2 * body and upper_shadow <= 0.2 * total_range and body > 0 and c < o:
            patterns.append({"symbol": symbol, "date": date_val, "pattern_name": "Hanging Man", "direction": "BEARISH"})

        # 6. Bullish Engulfing
        if c1_val < o1 and c > o and c >= o1 and o <= c1_val and body > body1:
            patterns.append({"symbol": symbol, "date": date_val, "pattern_name": "Bullish Engulfing", "direction": "BULLISH"})

        # 7. Bearish Engulfing
        if c1_val > o1 and c < o and c <= o1 and o >= c1_val and body > body1:
            patterns.append({"symbol": symbol, "date": date_val, "pattern_name": "Bearish Engulfing", "direction": "BEARISH"})

        # 8. Morning Star
        if c2_val < o2 and body2 > 0.5 * (h2 - l2) and body1 < 0.3 * body2 and c > o and c >= (o2 + c2_val) / 2:
            patterns.append({"symbol": symbol, "date": date_val, "pattern_name": "Morning Star", "direction": "BULLISH"})

        # 9. Evening Star
        if c2_val > o2 and body2 > 0.5 * (h2 - l2) and body1 < 0.3 * body2 and c < o and c <= (o2 + c2_val) / 2:
            patterns.append({"symbol": symbol, "date": date_val, "pattern_name": "Evening Star", "direction": "BEARISH"})

    return patterns

def run_candlestick_analysis(symbols: List[str] = ["BT", "SFBT"]):
    db = SessionLocal()
    Base.metadata.create_all(bind=engine)
    
    total_detected = 0
    summary = {}
    report_dir = "data/processed/reports"
    os.makedirs(report_dir, exist_ok=True)
    all_detected_records = []

    try:
        for symbol in symbols:
            print(f"Analyzing candlestick patterns for {symbol}...")
            prices_query = db.query(DailyPrice).filter(DailyPrice.symbol == symbol).order_by(DailyPrice.date.asc()).statement
            df = pd.read_sql(prices_query, db.bind)

            if df.empty:
                print(f" -> No historical prices found for {symbol}.")
                summary[symbol] = 0
                continue

            patterns = detect_candlestick_patterns_df(df)
            summary[symbol] = len(patterns)
            total_detected += len(patterns)

            if patterns:
                for pat in patterns:
                    stmt = insert(CandlestickPattern).values(
                        symbol=pat['symbol'],
                        date=pat['date'],
                        pattern_name=pat['pattern_name'],
                        direction=pat['direction']
                    )
                    stmt = stmt.on_conflict_do_update(
                        constraint='uq_symbol_date_pattern',
                        set_={'direction': stmt.excluded.direction}
                    )
                    db.execute(stmt)
                db.commit()
                all_detected_records.extend(patterns)

            print(f" -> Detected {len(patterns)} pattern occurrences for {symbol}.")

        if all_detected_records:
            rep_df = pd.DataFrame(all_detected_records)
            rep_path = f"{report_dir}/candlestick_patterns.csv"
            rep_df.to_csv(rep_path, index=False)
            print(f"\n[SUCCESS] Exported candlestick patterns report to {rep_path}")

        print("\n" + "="*45)
        print(" CANDLESTICK PATTERN DETECTION SUMMARY")
        print("="*45)
        for sym, count in summary.items():
            print(f" - {sym}: {count} patterns detected")
        print(f" Total Detected: {total_detected}")
        print("="*45 + "\n")

    except Exception as e:
        db.rollback()
        print(f"[ERROR] Candlestick analysis failed: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    run_candlestick_analysis()