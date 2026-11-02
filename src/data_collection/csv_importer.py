import os
import sys
import csv
import argparse
import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path when run as a module or script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy.dialects.postgresql import insert
from src.config.database import SessionLocal
from src.database.models import Instrument, DailyPrice

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Column alias mappings (case-insensitive keys)
COLUMN_ALIASES = {
    "date": ["date", "séance", "seance", "time", "day"],
    "open": ["open", "ouverture", "cours d'ouverture", "cours d ouverture", "ouv."], # Added "ouv."
    "high": ["high", "plus haut", "haut"],
    "low": ["low", "plus bas", "bas"],
    "close": ["close", "clôture", "cloture", "cours de clôture", "cours de cloture", "dernier"], # Added "dernier"
    "volume": ["volume", "quantité", "quantite", "vol", "titres échangés", "vol."] # Added "vol."
}

def detect_delimiter(sample_text: str) -> str:
    """Automatically detect common CSV delimiters (comma, semicolon, tab)."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[',', ';', '\t', '|'])
        return dialect.delimiter
    except Exception:
        # Fallback heuristic if sniffer fails
        if sample_text.count(';') > sample_text.count(','):
            return ';'
        return ','

def parse_date(date_str: str) -> datetime.date:
    """Support multiple common date formats."""
    date_str = date_str.strip()
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", 
        "%Y/%m/%d", "%d.%m.%Y", "%Y%m%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: '{date_str}'")

def parse_numeric(val_str: str) -> Decimal:
    """Handle French/European comma decimals vs US dot decimals, and K/M/B multipliers."""
    if val_str is None or str(val_str).strip() == "":
        raise ValueError("Empty numeric value")
    
    cleaned = str(val_str).strip().upper().replace(" ", "")
    
    # Handle multipliers (K = Thousands, M = Millions, B = Billions)
    multiplier = Decimal(1)
    if cleaned.endswith('K'):
        multiplier = Decimal(1000)
        cleaned = cleaned[:-1]
    elif cleaned.endswith('M'):
        multiplier = Decimal(1000000)
        cleaned = cleaned[:-1]
    elif cleaned.endswith('B'):
        multiplier = Decimal(1000000000)
        cleaned = cleaned[:-1]

    # Handle decimal separators
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    
    return Decimal(cleaned) * multiplier

def normalize_row_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw CSV column headers to standardized keys using COLUMN_ALIASES."""
    normalized = {}
    # Lowercase and strip whitespace from input headers
    cleaned_row = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    
    for std_key, aliases in COLUMN_ALIASES.items():
        found_val = None
        for alias in aliases:
            if alias.lower() in cleaned_row:
                found_val = cleaned_row[alias.lower()]
                break
        normalized[std_key] = found_val
    return normalized

def import_csv_file(file_path: str, symbol: str, source: str = "ILBOURSA") -> Dict[str, Any]:
    """Imports historical stock prices from a CSV file into PostgreSQL with robust validation and upsert logic."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    symbol = symbol.strip().upper()
    source = source.strip().upper()

    total_rows = 0
    successful_rows = 0
    updated_records = 0
    rejected_rows_list = []
    seen_dates = set()
    imported_dates = []

    # Read sample text for delimiter detection
    with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = detect_delimiter(sample)
        reader = csv.DictReader(f, delimiter=delimiter)
        
        raw_rows = list(reader)

    total_rows = len(raw_rows)
    valid_records_to_upsert = []

    for idx, row in enumerate(raw_rows, start=1):
        norm = normalize_row_keys(row)
        reasons = []

        # 1. Check missing fields
        missing = [k for k, v in norm.items() if v is None or str(v).strip() == ""]
        if missing:
            reasons.append(f"Missing required fields: {missing}")
            rejected_rows_list.append({"row_number": idx, "row_data": row, "reason": "; ".join(reasons)})
            continue

        # 2. Parse Date
        try:
            row_date = parse_date(str(norm["date"]))
        except ValueError as e:
            reasons.append(str(e))
            rejected_rows_list.append({"row_number": idx, "row_data": row, "reason": "; ".join(reasons)})
            continue

        # 3. Check duplicate dates within the same file
        if row_date in seen_dates:
            reasons.append(f"Duplicate date within file: {row_date}")
            rejected_rows_list.append({"row_number": idx, "row_data": row, "reason": "; ".join(reasons)})
            continue
        seen_dates.add(row_date)

        # 4. Parse OHLCV Numerics
        try:
            open_val = parse_numeric(norm["open"])
            high_val = parse_numeric(norm["high"])
            low_val = parse_numeric(norm["low"])
            close_val = parse_numeric(norm["close"])
            volume_val = int(parse_numeric(norm["volume"]))
        except (InvalidOperation, ValueError, TypeError) as e:
            reasons.append(f"Invalid numeric format: {e}")
            rejected_rows_list.append({"row_number": idx, "row_data": row, "reason": "; ".join(reasons)})
            continue

        # 5. Validation Rules
        if open_val <= 0 or high_val <= 0 or low_val <= 0 or close_val <= 0:
            reasons.append("Prices must be positive numbers (> 0)")
        if high_val < open_val or high_val < close_val:
            reasons.append(f"High ({high_val}) cannot be less than Open ({open_val}) or Close ({close_val})")
        if low_val > open_val or low_val > close_val:
            reasons.append(f"Low ({low_val}) cannot be greater than Open ({open_val}) or Close ({close_val})")
        if high_val < low_val:
            reasons.append(f"High ({high_val}) cannot be less than Low ({low_val})")
        if volume_val < 0:
            reasons.append(f"Volume cannot be negative ({volume_val})")

        if reasons:
            rejected_rows_list.append({"row_number": idx, "row_data": row, "reason": "; ".join(reasons)})
            continue

        valid_records_to_upsert.append({
            "symbol": symbol,
            "date": row_date,
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": volume_val,
            "source": source
        })
        imported_dates.append(row_date)

    # 6. Database Persistence & Upsert Transaction
    db = SessionLocal()
    try:
        # Ensure instrument exists (automatically register if not present)
        instrument = db.query(Instrument).filter(Instrument.symbol == symbol).first()
        if not instrument:
            logger.info(f"Instrument '{symbol}' not found in database. Registering automatically...")
            new_instrument = Instrument(
                symbol=symbol,
                company_name=f"Company {symbol}",
                is_active=True
            )
            db.add(new_instrument)
            db.commit()

        if valid_records_to_upsert:
            stmt = insert(DailyPrice).values(valid_records_to_upsert)
            # PostgreSQL Upsert: update fields on conflict (symbol, date)
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'date'],
                set_={
                    'open': stmt.excluded.open,
                    'high': stmt.excluded.high,
                    'low': stmt.excluded.low,
                    'close': stmt.excluded.close,
                    'volume': stmt.excluded.volume,
                    'source': stmt.excluded.source
                }
            )
            result = db.execute(stmt)
            db.commit()
            successful_rows = len(valid_records_to_upsert)
            # Rowcount indicates affected rows; in upserts, we track successful imports
            updated_records = result.rowcount or successful_rows

    except Exception as e:
        db.rollback()
        logger.error(f"Database transaction failed during import: {e}")
        raise e
    finally:
        db.close()

    # 7. Save Rejected Rows Report if any
    if rejected_rows_list:
        os.makedirs("data/processed/rejected", exist_ok=True)
        report_filename = f"data/processed/rejected/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(report_filename, 'w', newline='', encoding='utf-8') as rf:
                writer = csv.writer(rf)
                writer.writerow(["Row Number", "Validation Reason", "Raw Data"])
                for rej in rejected_rows_list:
                    writer.writerow([rej["row_number"], rej["reason"], str(rej["row_data"])])
            logger.info(f"Rejected rows report saved to {report_filename}")
        except Exception as ex:
            logger.warning(f"Could not save rejected rows report file: {ex}")

    date_range_str = f"{min(imported_dates)} to {max(imported_dates)}" if imported_dates else "N/A"

    # Display Import Report Summary
    print("\n" + "="*40)
    print(f" IMPORT REPORT: {symbol}")
    print("="*40)
    print(f" Stock Symbol          : {symbol}")
    print(f" Source                : {source}")
    print(f" Total Rows Detected   : {total_rows}")
    print(f" Successfully Imported : {successful_rows}")
    print(f" Rejected Rows         : {len(rejected_rows_list)}")
    print(f" Upserted / Updated    : {updated_records}")
    print(f" Date Range Imported   : {date_range_str}")
    print("="*40 + "\n")

    return {
        "symbol": symbol,
        "source": source,
        "total_rows": total_rows,
        "successful_rows": successful_rows,
        "rejected_rows": len(rejected_rows_list),
        "updated_records": updated_records,
        "date_range": date_range_str
    }

def main():
    parser = argparse.ArgumentParser(description="Import BVMT historical stock prices from CSV.")
    parser.add_argument("--file", required=True, help="Path to the CSV file")
    parser.add_argument("--symbol", required=True, help="Stock symbol (e.g. BT, SFBT)")
    parser.add_argument("--source", default="ILBOURSA", help="Data source provider (default: ILBOURSA)")

    args = parser.parse_args()
    import_csv_file(file_path=args.file, symbol=args.symbol, source=args.source)

if __name__ == "__main__":
    main()