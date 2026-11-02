import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from src.config.database import engine, Base
from src.database.models import Instrument, DailyPrice

def test_connection() -> None:
    print("Testing PostgreSQL connection...")
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.scalar()
            print(f"[SUCCESS] Connected to PostgreSQL successfully!")
            print(f"Database Version: {version}")
        
        print("Initializing database schema...")
        Base.metadata.create_all(bind=engine)
        print("[SUCCESS] Tables 'instruments' and 'daily_prices' created successfully.")
        
    except Exception as e:
        print(f"[ERROR] Database connection or schema creation failed: {e}")

if __name__ == "__main__":
    test_connection()
