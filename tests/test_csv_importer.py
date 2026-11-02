import os
import pytest
import csv
from datetime import datetime
from src.data_collection.csv_importer import import_csv_file, parse_date, parse_numeric
from src.config.database import SessionLocal, engine
from src.database.models import Instrument, DailyPrice

def create_synthetic_csv(tmp_path, filename, header, rows):
    file_path = tmp_path / filename
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return str(file_path)

def test_french_csv_columns(tmp_path):
    header = ["Date", "Ouverture", "Plus Haut", "Plus Bas", "Clôture", "Volume"]
    rows = [
        ["2026-06-01", "10,500", "10,800", "10,400", "10,750", "15000"]
    ]
    file_path = create_synthetic_csv(tmp_path, "bt_fr.csv", header, rows)
    
    result = import_csv_file(file_path, symbol="BT", source="ILBOURSA")
    assert result["successful_rows"] == 1
    assert result["rejected_rows"] == 0

def test_english_csv_columns(tmp_path):
    header = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = [
        ["2026-06-02", "20.1", "20.5", "19.9", "20.3", "5000"]
    ]
    file_path = create_synthetic_csv(tmp_path, "sfbt_en.csv", header, rows)
    
    result = import_csv_file(file_path, symbol="SFBT", source="BVMT")
    assert result["successful_rows"] == 1
    assert result["rejected_rows"] == 0

def test_invalid_ohlc_values(tmp_path):
    header = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = [
        ["2026-06-03", "10.0", "9.5", "10.2", "9.8", "1000"] # High < Open, Low > Open
    ]
    file_path = create_synthetic_csv(tmp_path, "invalid.csv", header, rows)
    
    result = import_csv_file(file_path, symbol="TEST", source="ILBOURSA")
    assert result["successful_rows"] == 0
    assert result["rejected_rows"] == 1

def test_duplicate_dates_in_file(tmp_path):
    header = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = [
        ["2026-06-04", "10.0", "10.5", "9.8", "10.2", "1000"],
        ["2026-06-04", "10.2", "10.6", "10.1", "10.4", "1200"] # Duplicate date in same file
    ]
    file_path = create_synthetic_csv(tmp_path, "dup.csv", header, rows)
    
    result = import_csv_file(file_path, symbol="TEST", source="ILBOURSA")
    assert result["successful_rows"] == 1
    assert result["rejected_rows"] == 1

def test_reimport_upsert(tmp_path):
    header = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = [
        ["2026-06-05", "10.0", "10.5", "9.8", "10.2", "1000"]
    ]
    file_path = create_synthetic_csv(tmp_path, "upsert1.csv", header, rows)
    
    # First import
    res1 = import_csv_file(file_path, symbol="BT", source="ILBOURSA")
    assert res1["successful_rows"] == 1

    # Reimport modified price for same date
    rows_updated = [
        ["2026-06-05", "11.0", "11.5", "10.8", "11.2", "2000"]
    ]
    file_path2 = create_synthetic_csv(tmp_path, "upsert2.csv", header, rows_updated)
    res2 = import_csv_file(file_path2, symbol="BT", source="ILBOURSA")
    assert res2["successful_rows"] == 1

    db = SessionLocal()
    try:
        price_rec = db.query(DailyPrice).filter_by(symbol="BT").first()
        assert float(price_rec.close) == 11.2
        assert price_rec.volume == 2000
    finally:
        db.close()

def test_unknown_stock_symbol_auto_registration(tmp_path):
    header = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = [
        ["2026-06-06", "5.0", "5.2", "4.9", "5.1", "3000"]
    ]
    file_path = create_synthetic_csv(tmp_path, "new_stock.csv", header, rows)
    
    result = import_csv_file(file_path, symbol="BIAT", source="ILBOURSA")
    assert result["successful_rows"] == 1

    db = SessionLocal()
    try:
        inst = db.query(Instrument).filter_by(symbol="BIAT").first()
        assert inst is not None
        assert inst.symbol == "BIAT"
    finally:
        db.close()
