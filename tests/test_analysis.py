import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from src.analysis.data_quality import analyze_data_quality
from src.analysis.liquidity_analysis import analyze_liquidity
from src.analysis.volatility_analysis import analyze_volatility
from src.config.database import SessionLocal
from src.database.models import Instrument, DailyPrice

def seed_test_data(db, symbol, prices_data):
    inst = db.query(Instrument).filter_by(symbol=symbol).first()
    if not inst:
        inst = Instrument(symbol=symbol, company_name=f"Company {symbol}")
        db.add(inst)
        db.commit()
    
    for p in prices_data:
        dp = DailyPrice(
            symbol=symbol,
            date=p['date'],
            open=p['open'],
            high=p['high'],
            low=p['low'],
            close=p['close'],
            volume=p['volume'],
            source="TEST"
        )
        db.add(dp)
    db.commit()

def test_volatility_and_drawdown():
    db = SessionLocal()
    try:
        data = [
            {"date": date(2026, 1, 1), "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.0, "volume": 1000},
            {"date": date(2026, 1, 2), "open": 10.0, "high": 11.0, "low": 9.9, "close": 11.0, "volume": 1500},
            {"date": date(2026, 1, 5), "open": 11.0, "high": 11.2, "low": 10.0, "close": 10.5, "volume": 1200},
            {"date": date(2026, 1, 6), "open": 10.5, "high": 10.8, "low": 9.5, "close": 9.5, "volume": 2000},
        ]
        seed_test_data(db, "TESTSTOCK", data)
        vol_results = analyze_volatility(db)
        
        assert len(vol_results) == 1
        res = vol_results[0]
        assert res["symbol"] == "TESTSTOCK"
        assert res["max_drawdown"] < 0  # Should detect drawdown from 11.0 to 9.5
    finally:
        db.close()

def test_liquidity_ranking():
    db = SessionLocal()
    try:
        data1 = [
            {"date": date(2026, 1, 1), "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 500},
            {"date": date(2026, 1, 2), "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 1500},
        ]
        data2 = [
            {"date": date(2026, 1, 1), "open": 20.0, "high": 20.5, "low": 19.5, "close": 20.0, "volume": 10000},
        ]
        seed_test_data(db, "LOWACT", data1)
        seed_test_data(db, "HIGHACT", data2)

        liq = analyze_liquidity(db)
        assert liq[0]["symbol"] == "HIGHACT"  # Higher average volume
    finally:
        db.close()

def test_invalid_ohlc_detection():
    db = SessionLocal()
    try:
        bad_data = [
            {"date": date(2026, 1, 1), "open": 10.0, "high": 9.0, "low": 11.0, "close": 10.0, "volume": 1000} # High < Low
        ]
        seed_test_data(db, "BAD", bad_data)
        quality = analyze_data_quality(db)
        assert quality["total_issues"] > 0
        assert any(i["issue_type"] == "INVALID_OHLC" for i in quality["issues"])
    finally:
        db.close()

def test_empty_dataset():
    db = SessionLocal()
    try:
        quality = analyze_data_quality(db)
        assert quality["total_records"] == 0
        assert quality["total_issues"] == 0
        
        liq = analyze_liquidity(db)
        assert liq == []
        
        vol = analyze_volatility(db)
        assert vol == []
    finally:
        db.close()
