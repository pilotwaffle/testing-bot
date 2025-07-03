# File: E:\Trade Chat Bot\G Trading Bot\exchange_error_handler.py
# Location: E:\Trade Chat Bot\G Trading Bot\exchange_error_handler.py

"""
Comprehensive Error Handling System for Exchange API Failures
Elite Trading Bot V3.0 - Exchange API Error Handler
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import aiohttp
import json
from functools import wraps
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Classification of different error types"""
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    INVALID_REQUEST = "invalid_request"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    MARKET_CLOSED = "market_closed"
    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ExchangeError:
    """Standardized error structure"""
    type: ErrorType
    severity: ErrorSeverity
    message: str
    exchange: str
    endpoint: str
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    raw_error: Optional[str] = None
    http_status: Optional[int] = None

@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    """Circuit breaker for exchange endpoints"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class ExchangeErrorHandler:
    """Comprehensive error handling system for exchange APIs"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ExchangeError] = []
        self.rate_limit_trackers: Dict[str, Dict[str, Any]] = {}
        self.fallback_data_sources: Dict[str, List[str]] = {
            "market_data": ["coingecko", "coinmarketcap", "binance"],
            "trading": ["kraken", "coinbase", "binance"]
        }
        
    def get_circuit_breaker(self, endpoint: str) -> CircuitBreaker:
        """Get or create circuit breaker for endpoint"""
        if endpoint not in self.circuit_breakers:
            self.circuit_breakers[endpoint] = CircuitBreaker()
        return self.circuit_breakers[endpoint]
    
    def classify_error(self, error: Exception, response: Optional[aiohttp.ClientResponse] = None) -> ErrorType:
        """Classify error type from exception and response"""
        error_msg = str(error).lower()
        
        if response:
            status = response.status
            if status == 429:
                return ErrorType.RATE_LIMIT
            elif status in [401, 403]:
                return ErrorType.AUTH_ERROR
            elif status in [400, 422]:
                return ErrorType.INVALID_REQUEST
            elif status >= 500:
                return ErrorType.SERVER_ERROR
        
        # Network-related errors
        if any(keyword in error_msg for keyword in ['timeout', 'read timeout', 'connect timeout']):
            return ErrorType.TIMEOUT
        elif any(keyword in error_msg for keyword in ['connection', 'network', 'dns', 'ssl']):
            return ErrorType.NETWORK_ERROR
        elif 'insufficient' in error_msg and 'funds' in error_msg:
            return ErrorType.INSUFFICIENT_FUNDS
        elif 'market' in error_msg and ('closed' in error_msg or 'inactive' in error_msg):
            return ErrorType.MARKET_CLOSED
        elif any(keyword in error_msg for keyword in ['invalid', 'parse', 'json', 'decode']):
            return ErrorType.INVALID_RESPONSE
        
        return ErrorType.UNKNOWN
    
    def get_error_severity(self, error_type: ErrorType) -> ErrorSeverity:
        """Determine error severity"""
        severity_map = {
            ErrorType.NETWORK_ERROR: ErrorSeverity.MEDIUM,
            ErrorType.RATE_LIMIT: ErrorSeverity.LOW,
            ErrorType.AUTH_ERROR: ErrorSeverity.HIGH,
            ErrorType.INVALID_REQUEST: ErrorSeverity.MEDIUM,
            ErrorType.SERVER_ERROR: ErrorSeverity.HIGH,
            ErrorType.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorType.INVALID_RESPONSE: ErrorSeverity.MEDIUM,
            ErrorType.INSUFFICIENT_FUNDS: ErrorSeverity.CRITICAL,
            ErrorType.MARKET_CLOSED: ErrorSeverity.LOW,
            ErrorType.UNKNOWN: ErrorSeverity.MEDIUM
        }
        return severity_map.get(error_type, ErrorSeverity.MEDIUM)
    
    def should_retry(self, error_type: ErrorType, retry_count: int, max_retries: int) -> bool:
        """Determine if error should be retried"""
        no_retry_errors = {
            ErrorType.AUTH_ERROR,
            ErrorType.INVALID_REQUEST,
            ErrorType.INSUFFICIENT_FUNDS
        }
        
        if error_type in no_retry_errors:
            return False
        
        return retry_count < max_retries
    
    def calculate_retry_delay(self, retry_count: int, config: RetryConfig) -> float:
        """Calculate delay before retry with exponential backoff and jitter"""
        delay = min(config.base_delay * (config.exponential_base ** retry_count), config.max_delay)
        
        if config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)  # Add jitter
        
        return delay
    
    async def handle_rate_limit(self, exchange: str, endpoint: str, retry_after: Optional[int] = None) -> float:
        """Handle rate limiting with intelligent backoff"""
        if retry_after:
            delay = retry_after
        else:
            # Estimate delay based on endpoint type
            endpoint_delays = {
                "trading": 5.0,
                "market_data": 1.0,
                "account": 3.0,
                "default": 2.0
            }
            
            endpoint_type = "default"
            for key in endpoint_delays:
                if key in endpoint.lower():
                    endpoint_type = key
                    break
            
            delay = endpoint_delays[endpoint_type]
        
        logger.warning(f"Rate limited on {exchange} {endpoint}, waiting {delay}s")
        await asyncio.sleep(delay)
        return delay
    
    def log_error(self, error: ExchangeError):
        """Log error with appropriate level"""
        message = f"Exchange Error [{error.exchange}][{error.endpoint}]: {error.message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(message)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(message)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(message)
        else:
            logger.info(message)
        
        # Store in history (keep last 1000 errors)
        self.error_history.append(error)
        if len(self.error_history) > 1000:
            self.error_history.pop(0)
    
    async def get_fallback_data(self, data_type: str, symbol: str) -> Optional[Dict]:
        """Get data from fallback sources"""
        sources = self.fallback_data_sources.get(data_type, [])
        
        for source in sources:
            try:
                # This would integrate with your actual fallback APIs
                logger.info(f"Attempting fallback data from {source} for {symbol}")
                # Implementation would depend on your specific fallback APIs
                return {"source": source, "symbol": symbol, "fallback": True}
            except Exception as e:
                logger.warning(f"Fallback source {source} failed: {e}")
                continue
        
        return None

def with_error_handling(
    exchange: str,
    endpoint: str,
    retry_config: Optional[RetryConfig] = None,
    use_circuit_breaker: bool = True,
    fallback_data_type: Optional[str] = None
):
    """Decorator for adding comprehensive error handling to exchange API calls"""
    
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = ExchangeErrorHandler()
            circuit_breaker = handler.get_circuit_breaker(f"{exchange}_{endpoint}")
            retry_count = 0
            
            while retry_count <= retry_config.max_retries:
                # Check circuit breaker
                if use_circuit_breaker and not circuit_breaker.should_allow_request():
                    logger.warning(f"Circuit breaker open for {exchange} {endpoint}")
                    if fallback_data_type:
                        symbol = kwargs.get('symbol', 'UNKNOWN')
                        fallback_data = await handler.get_fallback_data(fallback_data_type, symbol)
                        if fallback_data:
                            return fallback_data
                    raise Exception("Circuit breaker open - service unavailable")
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record success
                    if use_circuit_breaker:
                        circuit_breaker.record_success()
                    
                    return result
                
                except Exception as e:
                    # Classify error
                    response = getattr(e, 'response', None)
                    error_type = handler.classify_error(e, response)
                    severity = handler.get_error_severity(error_type)
                    
                    # Create error object
                    exchange_error = ExchangeError(
                        type=error_type,
                        severity=severity,
                        message=str(e),
                        exchange=exchange,
                        endpoint=endpoint,
                        retry_count=retry_count,
                        raw_error=str(e),
                        http_status=getattr(response, 'status', None) if response else None
                    )
                    
                    # Log error
                    handler.log_error(exchange_error)
                    
                    # Record failure in circuit breaker
                    if use_circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    # Handle rate limiting
                    if error_type == ErrorType.RATE_LIMIT:
                        retry_after = None
                        if response and 'retry-after' in response.headers:
                            retry_after = int(response.headers['retry-after'])
                        await handler.handle_rate_limit(exchange, endpoint, retry_after)
                        retry_count += 1
                        continue
                    
                    # Check if should retry
                    if handler.should_retry(error_type, retry_count, retry_config.max_retries):
                        delay = handler.calculate_retry_delay(retry_count, retry_config)
                        logger.info(f"Retrying {exchange} {endpoint} in {delay:.2f}s (attempt {retry_count + 1})")
                        await asyncio.sleep(delay)
                        retry_count += 1
                        continue
                    
                    # Try fallback if available
                    if fallback_data_type and severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                        symbol = kwargs.get('symbol', 'UNKNOWN')
                        fallback_data = await handler.get_fallback_data(fallback_data_type, symbol)
                        if fallback_data:
                            logger.info(f"Using fallback data for {exchange} {endpoint}")
                            return fallback_data
                    
                    # Re-raise the original exception if no retry/fallback
                    raise e
            
            # If we get here, all retries failed
            raise Exception(f"All retries failed for {exchange} {endpoint}")
        
        return wrapper
    return decorator

# Health monitoring functions
class ExchangeHealthMonitor:
    """Monitor exchange health and performance"""
    
    def __init__(self, error_handler: ExchangeErrorHandler):
        self.error_handler = error_handler
        self.health_metrics: Dict[str, Dict] = {}
    
    def get_exchange_health(self, exchange: str) -> Dict[str, Any]:
        """Get health metrics for an exchange"""
        recent_errors = [
            err for err in self.error_handler.error_history 
            if err.exchange == exchange and 
            datetime.now() - err.timestamp < timedelta(hours=1)
        ]
        
        total_errors = len(recent_errors)
        critical_errors = len([err for err in recent_errors if err.severity == ErrorSeverity.CRITICAL])
        
        # Get circuit breaker states
        cb_states = {
            endpoint: cb.state.value 
            for endpoint, cb in self.error_handler.circuit_breakers.items()
            if exchange in endpoint
        }
        
        return {
            "exchange": exchange,
            "status": "healthy" if total_errors < 10 and critical_errors == 0 else "degraded" if critical_errors == 0 else "unhealthy",
            "recent_errors": total_errors,
            "critical_errors": critical_errors,
            "circuit_breakers": cb_states,
            "last_error": recent_errors[-1].message if recent_errors else None
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        exchanges = list(set(err.exchange for err in self.error_handler.error_history))
        exchange_health = {ex: self.get_exchange_health(ex) for ex in exchanges}
        
        unhealthy_count = len([h for h in exchange_health.values() if h["status"] == "unhealthy"])
        
        return {
            "overall_status": "healthy" if unhealthy_count == 0 else "degraded" if unhealthy_count < len(exchanges) / 2 else "unhealthy",
            "exchanges": exchange_health,
            "total_errors_1h": len([
                err for err in self.error_handler.error_history 
                if datetime.now() - err.timestamp < timedelta(hours=1)
            ])
        }

# Example usage integration for your existing bot
class EnhancedKrakenAPI:
    """Enhanced Kraken API with comprehensive error handling"""
    
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.error_handler = ExchangeErrorHandler()
        self.health_monitor = ExchangeHealthMonitor(self.error_handler)
    
    @with_error_handling(
        exchange="kraken",
        endpoint="market_data",
        retry_config=RetryConfig(max_retries=3, base_delay=1.0),
        fallback_data_type="market_data"
    )
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker data with error handling"""
        # Your existing ticker implementation
        async with aiohttp.ClientSession() as session:
            url = f"https://api.kraken.com/0/public/Ticker?pair={symbol}"
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
                return await response.json()
    
    @with_error_handling(
        exchange="kraken",
        endpoint="trading",
        retry_config=RetryConfig(max_retries=1, base_delay=2.0),  # Trading should have fewer retries
        use_circuit_breaker=True
    )
    async def place_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """Place order with error handling"""
        # Your existing order placement implementation
        # This is just a placeholder - implement with your actual trading logic
        return {"order_id": "12345", "status": "pending"}
    
    def get_health_status(self) -> Dict:
        """Get current health status"""
        return self.health_monitor.get_exchange_health("kraken")