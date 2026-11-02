import os
import pytest

# Ensure tests use the test database rather than the production database
os.environ["DB_NAME"] = os.getenv("TEST_DB_NAME", "bvmt_test")

from src.config.database import engine, Base, SessionLocal
from src.database.models import Instrument, DailyPrice, CandlestickPattern

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure schema exists on the test database."""
    Base.metadata.create_all(bind=engine)
    yield

@pytest.fixture(autouse=True)
def clean_test_database():
    """Clean tables in the test database before each test run."""
    db = SessionLocal()
    try:
        db.query(CandlestickPattern).delete()
        db.query(DailyPrice).delete()
        db.query(Instrument).delete()
        db.commit()
    finally:
        db.close()
    yield
