from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Date, Numeric, BigInteger, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from src.config.database import Base

class Instrument(Base):
    __tablename__ = "instruments"

    symbol = Column(String(20), primary_key=True, index=True)
    company_name = Column(String(150), nullable=False)
    isin = Column(String(20), unique=True, nullable=True)
    sector = Column(String(100), nullable=True)
    trading_group = Column(String(20), nullable=True)
    market = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    prices = relationship("DailyPrice", back_populates="instrument", cascade="all, delete-orphan")

class DailyPrice(Base):
    __tablename__ = "daily_prices"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), ForeignKey("instruments.symbol", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Numeric(12, 3), nullable=False)
    high = Column(Numeric(12, 3), nullable=False)
    low = Column(Numeric(12, 3), nullable=False)
    close = Column(Numeric(12, 3), nullable=False)
    volume = Column(BigInteger, default=0, nullable=False)
    source = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_symbol_date'),
        Index('idx_daily_prices_symbol_date', 'symbol', 'date'),
    )

    instrument = relationship("Instrument", back_populates="prices")
