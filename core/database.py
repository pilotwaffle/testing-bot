from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.dialects.sqlite import JSON
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class Trade(Base):
    """FreqTrade-style trade model"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    exchange = Column(String(25), nullable=False)
    pair = Column(String(25), nullable=False, index=True)
    base_currency = Column(String(25), nullable=False)
    quote_currency = Column(String(25), nullable=False)
    is_open = Column(Boolean, nullable=False, default=True)
    fee_open = Column(Float, nullable=False, default=0.0)
    fee_open_cost = Column(Float, nullable=True)
    fee_open_currency = Column(String(25), nullable=True)
    fee_close = Column(Float, nullable=False, default=0.0)
    fee_close_cost = Column(Float, nullable=True)
    fee_close_currency = Column(String(25), nullable=True)
    open_rate = Column(Float, nullable=False)
    open_rate_requested = Column(Float, nullable=True)
    open_trade_value = Column(Float, nullable=False)
    close_rate = Column(Float, nullable=True)
    close_rate_requested = Column(Float, nullable=True)
    close_profit = Column(Float, nullable=True)
    close_profit_abs = Column(Float, nullable=True)
    stake_amount = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    amount_requested = Column(Float, nullable=True)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    close_date = Column(DateTime, nullable=True, index=True)
    open_order_id = Column(String(255), nullable=True)
    stop_loss = Column(Float, nullable=True, default=0.0)
    stop_loss_pct = Column(Float, nullable=True)
    initial_stop_loss = Column(Float, nullable=True, default=0.0)
    initial_stop_loss_pct = Column(Float, nullable=True)
    stoploss_order_id = Column(String(255), nullable=True)
    stoploss_last_update = Column(DateTime, nullable=True)
    max_rate = Column(Float, nullable=True, default=0.0)
    min_rate = Column(Float, nullable=True)
    exit_reason = Column(String(100), nullable=True)
    exit_order_status = Column(String(100), nullable=True)
    strategy = Column(String(100), nullable=False)
    enter_tag = Column(String(100), nullable=True)
    timeframe = Column(Integer, nullable=True)

    trading_mode = Column(String(10), nullable=False, default='spot')
    leverage = Column(Float, nullable=True, default=1.0)
    is_short = Column(Boolean, nullable=False, default=False)

    custom_data = Column(JSON, nullable=True)

    orders = relationship("Order", back_populates="trade", cascade="all, delete-orphan")

class Order(Base):
    """Order model"""
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=False)
    order_id = Column(String(255), nullable=False, index=True)
    order_type = Column(String(50), nullable=False)
    side = Column(String(25), nullable=False)
    amount = Column(Float, nullable=False)
    filled = Column(Float, nullable=False, default=0.0)
    remaining = Column(Float, nullable=False, default=0.0)
    cost = Column(Float, nullable=False, default=0.0)
    average = Column(Float, nullable=True)
    status = Column(String(100), nullable=False)
    order_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    order_filled_date = Column(DateTime, nullable=True)

    trade = relationship("Trade", back_populates="orders")

class PairLock(Base):
    """Pair lock model for risk management"""
    __tablename__ = 'pairlocks'

    id = Column(Integer, primary_key=True)
    pair = Column(String(25), nullable=False, index=True)
    reason = Column(String(255), nullable=True)
    lock_time = Column(DateTime, nullable=False)
    lock_end_time = Column(DateTime, nullable=False, index=True)
    active = Column(Boolean, nullable=False, default=True, index=True)

class GlobalLock(Base):
    """Global lock model"""
    __tablename__ = 'globallocks'

    id = Column(Integer, primary_key=True)
    lock_time = Column(DateTime, nullable=False)
    lock_end_time = Column(DateTime, nullable=False, index=True)
    reason = Column(String(255), nullable=True)
    active = Column(Boolean, nullable=False, default=True, index=True)

class DatabaseManager:
    """Database manager with FreqTrade-style functionality"""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, echo=False, future=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine, future=True)
        self.create_tables()

    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def add_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """Add new trade to database"""
        with self.get_session() as session:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            logger.info(f"Added trade {trade.id} for {trade.pair}")
            return trade

    def update_trade(self, trade_id: int, updates: Dict[str, Any]) -> Optional[Trade]:
        """Update existing trade"""
        with self.get_session() as session:
            trade = session.get(Trade, trade_id)
            if trade:
                for key, value in updates.items():
                    setattr(trade, key, value)
                session.commit()
                session.refresh(trade)
                logger.info(f"Updated trade {trade_id}")
                return trade
            return None

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        with self.get_session() as session:
            return session.query(Trade).filter(Trade.is_open.is_(True)).all()

    def get_trade_history(self, pair: Optional[str] = None, limit: int = 100) -> List[Trade]:
        """Get trade history"""
        with self.get_session() as session:
            query = session.query(Trade).filter(Trade.is_open.is_(False))
            if pair:
                query = query.filter(Trade.pair == pair)
            return query.order_by(Trade.close_date.desc()).limit(limit).all()

    def get_trade_performance(self) -> Dict[str, Any]:
        """Get overall trading performance metrics"""
        with self.get_session() as session:
            closed_trades = session.query(Trade).filter(Trade.is_open.is_(False)).all()
            if not closed_trades:
                return {}

            total_trades = len(closed_trades)
            winning_trades = len([t for t in closed_trades if t.close_profit and t.close_profit > 0])
            total_profit = sum([t.close_profit_abs or 0 for t in closed_trades])

            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_profit': total_profit,
                'avg_profit': total_profit / total_trades if total_trades > 0 else 0
            }