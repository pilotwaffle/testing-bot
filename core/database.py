# === NEW FILE: core/database.py ===
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
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
    is_open = Column(Boolean, nullable=False, default=True, index=True)
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
    open_order_id = Column(String(255), nullable=True, index=True)
    stop_loss = Column(Float, nullable=True, default=0.0)
    stop_loss_pct = Column(Float, nullable=True)
    initial_stop_loss = Column(Float, nullable=True, default=0.0)
    initial_stop_loss_pct = Column(Float, nullable=True)
    stoploss_order_id = Column(String(255), nullable=True, index=True)
    stoploss_last_update = Column(DateTime, nullable=True)
    max_rate = Column(Float, nullable=True, default=0.0)
    min_rate = Column(Float, nullable=True)
    exit_reason = Column(String(100), nullable=True)
    exit_order_status = Column(String(100), nullable=True)
    strategy = Column(String(100), nullable=False)
    enter_tag = Column(String(100), nullable=True)
    timeframe = Column(Integer, nullable=True)
    
    # Trading mode
    trading_mode = Column(String(10), nullable=False, default='spot')
    leverage = Column(Float, nullable=True, default=1.0)
    is_short = Column(Boolean, nullable=False, default=False)
    
    # Custom data
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
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
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
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
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
            return session.query(Trade).filter(Trade.is_open == True).all()
    
    def get_trade_history(self, pair: Optional[str] = None, limit: int = 100) -> List[Trade]:
        """Get trade history"""
        with self.get_session() as session:
            query = session.query(Trade).filter(Trade.is_open == False)
            if pair:
                query = query.filter(Trade.pair == pair)
            return query.order_by(Trade.close_date.desc()).limit(limit).all()
    
    def get_trade_performance(self) -> Dict[str, Any]:
        """Get overall trading performance metrics"""
        with self.get_session() as session:
            closed_trades = session.query(Trade).filter(Trade.is_open == False).all()
            
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

# === NEW FILE: core/risk_manager.py ===
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class PositionSizing:
    """Position sizing calculation result"""
    recommended_amount: float
    risk_amount: float
    risk_percentage: float
    stop_loss_distance: float
    reasoning: str

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    risk_level: RiskLevel
    risk_score: float
    max_position_size: float
    recommended_action: str
    reasons: List[str]

class RiskManager:
    """FreqTrade-style risk management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_open_trades = config.get('max_open_trades', 3)
        self.stake_amount = config.get('stake_amount', 100)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_total_risk = config.get('max_total_risk', 0.10)  # 10%
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.50)  # 50%
        self.max_daily_drawdown = config.get('max_daily_drawdown', 0.05)  # 5%
        
        # Portfolio tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = config.get('max_daily_trades', 10)
        
    def calculate_position_size(self, pair: str, entry_price: float, 
                              stop_loss: Optional[float], account_balance: float,
                              current_positions: Dict[str, Any]) -> PositionSizing:
        """
        Calculate optimal position size based on risk management rules
        Similar to FreqTrade's position sizing
        """
        if stop_loss is None:
            # Default stop loss if not provided
            stop_loss_distance = 0.05  # 5%
            stop_loss = entry_price * (1 - stop_loss_distance)
        else:
            stop_loss_distance = abs(entry_price - stop_loss) / entry_price
        
        # Calculate risk amount based on account balance
        max_risk_amount = account_balance * self.max_risk_per_trade
        
        # Calculate position size based on stop loss distance
        if stop_loss_distance > 0:
            recommended_amount = max_risk_amount / (entry_price * stop_loss_distance)
        else:
            recommended_amount = self.stake_amount / entry_price
        
        # Apply maximum position limits
        max_position_value = account_balance * 0.20  # Max 20% of account per position
        max_amount_by_value = max_position_value / entry_price
        
        if recommended_amount > max_amount_by_value:
            recommended_amount = max_amount_by_value
            reasoning = "Limited by maximum position value (20% of account)"
        else:
            reasoning = f"Based on {self.max_risk_per_trade:.1%} risk with {stop_loss_distance:.1%} stop loss"
        
        # Check correlation limits
        correlation_adjustment = self._check_correlation_limits(pair, current_positions)
        recommended_amount *= correlation_adjustment
        
        if correlation_adjustment < 1.0:
            reasoning += f" (reduced {(1-correlation_adjustment)*100:.0f}% due to correlation)"
        
        risk_percentage = (recommended_amount * entry_price * stop_loss_distance) / account_balance
        
        return PositionSizing(
            recommended_amount=recommended_amount,
            risk_amount=recommended_amount * entry_price * stop_loss_distance,
            risk_percentage=risk_percentage,
            stop_loss_distance=stop_loss_distance,
            reasoning=reasoning
        )
    
    def assess_market_risk(self, market_data: Dict[str, Any], 
                          portfolio: Dict[str, Any]) -> RiskAssessment:
        """Assess overall market risk"""
        risk_factors = []
        risk_score = 0.0
        
        # Volatility risk
        volatility = market_data.get('volatility_24h', 0)
        if volatility > 0.10:  # 10%
            risk_factors.append(f"High volatility: {volatility:.1%}")
            risk_score += 30
        elif volatility > 0.05:  # 5%
            risk_factors.append(f"Moderate volatility: {volatility:.1%}")
            risk_score += 15
        
        # Drawdown risk
        current_drawdown = portfolio.get('current_drawdown', 0)
        if current_drawdown > 0.15:  # 15%
            risk_factors.append(f"High drawdown: {current_drawdown:.1%}")
            risk_score += 40
        elif current_drawdown > 0.10:  # 10%
            risk_factors.append(f"Moderate drawdown: {current_drawdown:.1%}")
            risk_score += 20
        
        # Open positions risk
        open_positions = len(portfolio.get('open_positions', []))
        if open_positions >= self.max_open_trades:
            risk_factors.append(f"Maximum positions reached: {open_positions}")
            risk_score += 20
        
        # Daily trading frequency
        if self.daily_trades >= self.max_daily_trades:
            risk_factors.append(f"Daily trade limit reached: {self.daily_trades}")
            risk_score += 25
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = RiskLevel.EXTREME
            recommended_action = "STOP_TRADING"
            max_position_size = 0.0
        elif risk_score >= 50:
            risk_level = RiskLevel.HIGH
            recommended_action = "REDUCE_EXPOSURE" 
            max_position_size = 0.5
        elif risk_score >= 30:
            risk_level = RiskLevel.MEDIUM
            recommended_action = "NORMAL_CAUTION"
            max_position_size = 0.8
        else:
            risk_level = RiskLevel.LOW
            recommended_action = "NORMAL_TRADING"
            max_position_size = 1.0
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            max_position_size=max_position_size,
            recommended_action=recommended_action,
            reasons=risk_factors
        )
    
    def should_enter_trade(self, pair: str, signal_confidence: float,
                          current_positions: Dict[str, Any], 
                          account_status: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if a trade should be entered based on risk management
        """
        # Check maximum open trades
        open_count = len(current_positions.get('open_trades', []))
        if open_count >= self.max_open_trades:
            return False, f"Maximum open trades reached ({open_count}/{self.max_open_trades})"
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.daily_trades}/{self.max_daily_trades})"
        
        # Check account drawdown
        current_drawdown = account_status.get('drawdown', 0)
        if current_drawdown > self.max_daily_drawdown:
            return False, f"Daily drawdown limit exceeded ({current_drawdown:.1%})"
        
        # Check signal confidence
        min_confidence = 0.6  # 60% minimum confidence
        if signal_confidence < min_confidence:
            return False, f"Signal confidence too low ({signal_confidence:.1%} < {min_confidence:.1%})"
        
        # Check pair-specific risks
        pair_risk = self._assess_pair_risk(pair, current_positions)
        if pair_risk['blocked']:
            return False, pair_risk['reason']
        
        return True, "Risk checks passed"
    
    def _check_correlation_limits(self, pair: str, current_positions: Dict[str, Any]) -> float:
        """Check correlation limits and return position size multiplier"""
        # Simplified correlation check based on base currency
        base_currency = pair.split('/')[0]
        
        # Count positions in same base currency
        same_base_exposure = 0.0
        for pos in current_positions.get('open_trades', []):
            if pos.get('pair', '').startswith(base_currency):
                same_base_exposure += pos.get('stake_amount', 0)
        
        total_exposure = sum([pos.get('stake_amount', 0) for pos in current_positions.get('open_trades', [])])
        
        if total_exposure == 0:
            return 1.0
        
        current_correlation = same_base_exposure / total_exposure
        
        if current_correlation > self.max_correlation_exposure:
            # Reduce position size proportionally
            return max(0.1, 1 - (current_correlation - self.max_correlation_exposure))
        
        return 1.0
    
    def _assess_pair_risk(self, pair: str, current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess pair-specific risks"""
        # Check if pair is already being traded
        for pos in current_positions.get('open_trades', []):
            if pos.get('pair') == pair:
                return {
                    'blocked': True,
                    'reason': f"Already have open position in {pair}"
                }
        
        # Add more pair-specific risk checks here
        # (e.g., recent losses, volatility spikes, etc.)
        
        return {'blocked': False, 'reason': ''}
    
    def update_daily_stats(self, trade_result: Dict[str, Any]):
        """Update daily trading statistics"""
        self.daily_trades += 1
        self.daily_pnl += trade_result.get('profit', 0)
        
        logger.info(f"Daily stats: {self.daily_trades} trades, {self.daily_pnl:.2f} PnL")
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new day)"""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        logger.info("Daily risk management stats reset")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        return {
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'max_daily_trades': self.max_daily_trades,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_total_risk': self.max_total_risk,
            'max_correlation_exposure': self.max_correlation_exposure,
            'max_daily_drawdown': self.max_daily_drawdown,
            'risk_utilization': {
                'trades': f"{self.daily_trades}/{self.max_daily_trades}",
                'daily_pnl': self.daily_pnl
            }
        }