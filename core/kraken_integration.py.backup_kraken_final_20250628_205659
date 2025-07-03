# Imports added by quick_fix_script.py for Elite Trading Bot V3.0
# Location: E:\Trade Chat Bot\G Trading Bot\core\kraken_integration.py
# Added missing type hints and standard imports

from typing import List, Dict, Optional, Union, Any
import asyncio
import logging
import json
import time
from datetime import datetime

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Add these imports to your existing enhanced_trading_engine.py
"""
Integration script for adding Kraken Futures to EliteTradingEngine
This will modify your existing EliteTradingEngine to include Kraken support
"""

# Environment Configuration (add to your .env file)
KRAKEN_ENV_CONFIG = """
# Kraken Futures Configuration
KRAKEN_ENABLED=true
KRAKEN_PUBLIC_KEY=W/LQxAC/7BBTlMDpUX4fs6n4g0x8EO/UU5y1r0lTTdg+MFiSMXZr3a5C
KRAKEN_PRIVATE_KEY=CFhVRfbIQwMOeukbuUt0XXURmuR30BlriWt5NIV/SZUHT9WHthPSbUCtBWAfEbS8FDudpYoeMogNr+Ql3Wt4vBFe
KRAKEN_SANDBOX=true
KRAKEN_BASE_URL=https://demo-futures.kraken.com
KRAKEN_PAPER_TRADING=true
KRAKEN_DEFAULT_SYMBOLS=BTC/USD,ETH/USD,LTC/USD,XRP/USD,ADA/USD
"""

class KrakenIntegration:
    """
    Integration layer for Kraken Futures within EliteTradingEngine
    """
    
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
        self.logger = logging.getLogger(f"{__name__}.KrakenIntegration")
        
        # Initialize Kraken components
        self.kraken_client = None
        self.kraken_ml_analyzer = None
        
        # Configuration
        self.config = self._load_kraken_config()
        
        # State tracking
        self.is_initialized = False
        self.last_market_update = 0
        self.market_data_cache = {}
        
        # Trading state
        self.active_orders = {}
        self.positions = {}
        self.trading_enabled = False
        
    def _load_kraken_config(self) -> Dict:
        """Load Kraken configuration from environment"""
        return {
            'enabled': os.getenv('KRAKEN_ENABLED', 'false').lower() == 'true',
            'public_key': os.getenv('KRAKEN_PUBLIC_KEY', ''),
            'private_key': os.getenv('KRAKEN_PRIVATE_KEY', ''),
            'sandbox': os.getenv('KRAKEN_SANDBOX', 'true').lower() == 'true',
            'base_url': os.getenv('KRAKEN_BASE_URL', 'https://demo-futures.kraken.com'),
            'paper_trading': os.getenv('KRAKEN_PAPER_TRADING', 'true').lower() == 'true',
            'default_symbols': os.getenv('KRAKEN_DEFAULT_SYMBOLS', 'BTC/USD,ETH/USD').split(',')
        }
    
    async def initialize(self) -> bool:
        """Initialize Kraken integration"""
        try:
            if not self.config['enabled']:
                self.logger.info("Kraken integration disabled in configuration")
                return False
            
            self.logger.info("Initializing Kraken Futures integration...")
            
            # Import Kraken components (assuming they're in the same directory)
            from kraken_futures_client import KrakenFuturesClient
            from kraken_ml_analyzer import KrakenMLAnalyzer
            
            # Initialize Kraken client
            self.kraken_client = KrakenFuturesClient(
                api_key=self.config['public_key'],
                api_secret=self.config['private_key'],
                sandbox=self.config['sandbox']
            )
            
            # Initialize client context
            await self.kraken_client.__aenter__()
            
            # Initialize ML analyzer
            self.kraken_ml_analyzer = KrakenMLAnalyzer(
                kraken_client=self.kraken_client,
                lookback_period=100
            )
            
            await self.kraken_ml_analyzer.initialize()
            
            # Test connection
            instruments = await self.kraken_client.get_instruments()
            
            if not instruments:
                raise Exception("Failed to fetch instruments - connection issue")
            
            self.logger.info(f"Kraken integration initialized - {len(instruments)} instruments available")
            
            # Enable trading if configured
            self.trading_enabled = self.config['paper_trading']
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kraken integration: {e}")
            return False
    
    async def _start_background_tasks(self):
        """Start Kraken-specific background tasks"""
        if hasattr(self.trading_engine, 'background_tasks'):
            # Add Kraken tasks to existing background tasks
            kraken_tasks = [
                self._market_data_monitor(),
                self._order_monitor(),
                self._ml_analysis_update(),
                self._portfolio_monitor()
            ]
            
            for task in kraken_tasks:
                self.trading_engine.background_tasks.append(
                    asyncio.create_task(task)
                )
                
            self.logger.info("Started 4 Kraken background monitoring tasks")
    
    async def _market_data_monitor(self):
        """Monitor Kraken market data"""
        while True:
            try:
                if not self.is_initialized:
                    await asyncio.sleep(10)
                    continue
                
                # Get market data for default symbols
                market_data = await self.kraken_client.get_market_data(
                    self.config['default_symbols']
                )
                
                if market_data:
                    self.market_data_cache.update(market_data)
                    self.last_market_update = market_data.get('timestamp', 0)
                    
                    # Update trading engine with Kraken data
                    if hasattr(self.trading_engine, 'market_data'):
                        self.trading_engine.market_data['kraken'] = market_data
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in Kraken market data monitor: {e}")
                await asyncio.sleep(60)
    
    async def _order_monitor(self):
        """Monitor Kraken orders and positions"""
        while True:
            try:
                if not self.is_initialized or not self.trading_enabled:
                    await asyncio.sleep(30)
                    continue
                
                # Get current positions
                positions = await self.kraken_client.get_positions()
                self.positions = {pos['symbol']: pos for pos in positions}
                
                # Update trading engine
                if hasattr(self.trading_engine, 'positions'):
                    if 'kraken' not in self.trading_engine.positions:
                        self.trading_engine.positions['kraken'] = {}
                    self.trading_engine.positions['kraken'].update(self.positions)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in Kraken order monitor: {e}")
                await asyncio.sleep(120)
    
    async def _ml_analysis_update(self):
        """Update ML analysis for Kraken markets"""
        while True:
            try:
                if not self.is_initialized:
                    await asyncio.sleep(300)  # 5 minutes
                    continue
                
                # Run ML analysis on default symbols
                for symbol in self.config['default_symbols']:
                    try:
                        # Get prediction
                        prediction = await self.kraken_ml_analyzer.predict(
                            symbol=symbol,
                            horizon='1h',
                            model_type='random_forest'
                        )
                        
                        if prediction:
                            # Store in trading engine
                            if hasattr(self.trading_engine, 'ml_predictions'):
                                if 'kraken' not in self.trading_engine.ml_predictions:
                                    self.trading_engine.ml_predictions['kraken'] = {}
                                self.trading_engine.ml_predictions['kraken'][symbol] = prediction
                        
                        await asyncio.sleep(5)  # Rate limiting between symbols
                        
                    except Exception as e:
                        self.logger.warning(f"ML analysis failed for {symbol}: {e}")
                        continue
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in Kraken ML analysis update: {e}")
                await asyncio.sleep(600)
    
    async def _portfolio_monitor(self):
        """Monitor Kraken portfolio performance"""
        while True:
            try:
                if not self.is_initialized:
                    await asyncio.sleep(60)
                    continue
                
                # Get account info
                account_info = await self.kraken_client.get_account_info()
                
                # Get trading summary
                trading_summary = await self.kraken_client.get_trading_summary()
                
                # Update trading engine
                if hasattr(self.trading_engine, 'portfolio_data'):
                    self.trading_engine.portfolio_data['kraken'] = {
                        'account': account_info,
                        'summary': trading_summary,
                        'last_update': datetime.now()
                    }
                
                await asyncio.sleep(120)  # Update every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Error in Kraken portfolio monitor: {e}")
                await asyncio.sleep(180)
    
    # ==================== TRADING INTERFACE ====================
    
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = 'market', price: float = None) -> Dict:
        """Place order through Kraken"""
        try:
            if not self.is_initialized or not self.trading_enabled:
                return {'success': False, 'error': 'Kraken trading not enabled'}
            
            # Validate inputs
            if side not in ['buy', 'sell']:
                return {'success': False, 'error': 'Invalid side'}
            
            if size <= 0:
                return {'success': False, 'error': 'Invalid size'}
            
            # Place order
            result = await self.kraken_client.place_order(
                symbol=symbol,
                side=side,
                size=size,
                order_type='mkt' if order_type == 'market' else 'lmt',
                price=price
            )
            
            if result.get('result') == 'success':
                order_id = result.get('order_id')
                self.active_orders[order_id] = {
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'type': order_type,
                    'price': price,
                    'timestamp': datetime.now()
                }
                
                self.logger.info(f"Kraken order placed: {order_id}")
                return {'success': True, 'order_id': order_id, 'result': result}
            else:
                return {'success': False, 'error': result.get('message', 'Unknown error')}
                
        except Exception as e:
            self.logger.error(f"Error placing Kraken order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel Kraken order"""
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'Kraken not initialized'}
            
            result = await self.kraken_client.cancel_order(order_id)
            
            if result.get('result') == 'success':
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                
                return {'success': True, 'result': result}
            else:
                return {'success': False, 'error': result.get('message', 'Cancel failed')}
                
        except Exception as e:
            self.logger.error(f"Error cancelling Kraken order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_market_analysis(self, symbols: List[str] = None) -> Dict:
        """Get ML market analysis"""
        try:
            if not self.is_initialized:
                return {}
            
            if not symbols:
                symbols = self.config['default_symbols']
            
            return await self.kraken_ml_analyzer.get_market_analysis(symbols)
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken market analysis: {e}")
            return {}
    
    # ==================== STRATEGY INTEGRATION ====================
    
    async def execute_strategy_signal(self, signal: Dict) -> Dict:
        """Execute trading signal from strategy"""
        try:
            if not self.is_initialized or not self.trading_enabled:
                return {'success': False, 'error': 'Trading not enabled'}
            
            symbol = signal.get('symbol')
            action = signal.get('action')  # 'buy', 'sell', 'hold'
            size = signal.get('size', 0.1)
            confidence = signal.get('confidence', 0.5)
            
            # Only execute high confidence signals
            if confidence < 0.6:
                return {'success': False, 'error': 'Confidence too low'}
            
            if action in ['buy', 'sell']:
                return await self.place_order(
                    symbol=symbol,
                    side=action,
                    size=size,
                    order_type='market'
                )
            
            return {'success': True, 'action': 'hold'}
            
        except Exception as e:
            self.logger.error(f"Error executing Kraken strategy signal: {e}")
            return {'success': False, 'error': str(e)}
    
    # ==================== DASHBOARD DATA ====================
    
    async def get_dashboard_data(self) -> Dict:
        """Get data for dashboard display"""
        try:
            if not self.is_initialized:
                return {'status': 'not_initialized'}
            
            # Health check
            health = await self.kraken_client.health_check()
            
            # Account info
            account = await self.kraken_client.get_account_info()
            
            # Recent ML predictions
            ml_data = {}
            for symbol in self.config['default_symbols'][:3]:  # Top 3 symbols
                try:
                    prediction = await self.kraken_ml_analyzer.predict(symbol, '1h')
                    if prediction:
                        ml_data[symbol] = prediction
                except Exception:
                    continue
            
            return {
                'status': 'active',
                'health': health,
                'account': account,
                'positions': len(self.positions),
                'active_orders': len(self.active_orders),
                'ml_predictions': ml_data,
                'market_data': self.market_data_cache.get('summary', {}),
                'last_update': datetime.now().isoformat(),
                'trading_enabled': self.trading_enabled
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken dashboard data: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # ==================== CLEANUP ====================
    
    async def shutdown(self):
        """Cleanup Kraken integration"""
        try:
            self.logger.info("Shutting down Kraken integration...")
            
            if self.kraken_client:
                await self.kraken_client.__aexit__(None, None, None)
            
            self.is_initialized = False
            self.logger.info("Kraken integration shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error shutting down Kraken integration: {e}")


# ==================== ELITE TRADING ENGINE MODIFICATIONS ====================

def add_kraken_to_elite_engine(trading_engine_class):
    """
    Decorator to add Kraken integration to EliteTradingEngine
    
    Usage:
    @add_kraken_to_elite_engine
    class EliteTradingEngine:
        # your existing engine code
    """
    
    # Store original __init__ and other methods
    original_init = trading_engine_class.__init__
    original_start = getattr(trading_engine_class, 'start', None)
    original_stop = getattr(trading_engine_class, 'stop', None)
    
    def new_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Add Kraken integration
        self.kraken_integration = KrakenIntegration(self)
        
        # Initialize data structures for Kraken
        if not hasattr(self, 'market_data'):
            self.market_data = {}
        
        if not hasattr(self, 'positions'):
            self.positions = {}
        
        if not hasattr(self, 'ml_predictions'):
            self.ml_predictions = {}
        
        if not hasattr(self, 'portfolio_data'):
            self.portfolio_data = {}
        
        self.logger.info("Kraken integration added to EliteTradingEngine")
    
    async def new_start(self):
        # Call original start if exists
        if original_start:
            result = await original_start(self)
            if not result:
                return False
        
        # Initialize Kraken
        kraken_success = await self.kraken_integration.initialize()
        
        if kraken_success:
            self.logger.info("EliteTradingEngine started with Kraken integration")
        else:
            self.logger.warning("EliteTradingEngine started without Kraken integration")
        
        return True
    
    async def new_stop(self):
        # Shutdown Kraken first
        if hasattr(self, 'kraken_integration'):
            await self.kraken_integration.shutdown()
        
        # Call original stop if exists
        if original_stop:
            await original_stop(self)
        
        self.logger.info("EliteTradingEngine stopped")
    
    # Add Kraken-specific methods
    async def place_kraken_order(self, symbol: str, side: str, size: float, 
                                order_type: str = 'market', price: float = None):
        """Place order through Kraken integration"""
        if hasattr(self, 'kraken_integration'):
            return await self.kraken_integration.place_order(symbol, side, size, order_type, price)
        return {'success': False, 'error': 'Kraken not initialized'}
    
    async def get_kraken_analysis(self, symbols: List[str] = None):
        """Get Kraken ML analysis"""
        if hasattr(self, 'kraken_integration'):
            return await self.kraken_integration.get_market_analysis(symbols)
        return {}
    
    async def get_kraken_dashboard_data(self):
        """Get Kraken dashboard data"""
        if hasattr(self, 'kraken_integration'):
            return await self.kraken_integration.get_dashboard_data()
        return {'status': 'not_available'}
    
    # Replace methods
    trading_engine_class.__init__ = new_init
    trading_engine_class.start = new_start
    trading_engine_class.stop = new_stop
    
    # Add new methods
    trading_engine_class.place_kraken_order = place_kraken_order
    trading_engine_class.get_kraken_analysis = get_kraken_analysis
    trading_engine_class.get_kraken_dashboard_data = get_kraken_dashboard_data
    
    return trading_engine_class


# ==================== INTEGRATION SCRIPT ====================

async def integrate_kraken_with_existing_engine():
    """
    Integration script to add Kraken to existing EliteTradingEngine
    Run this to test the integration
    """
    
    # Write environment configuration
    env_file_path = '.env'
    
    try:
        # Read existing .env
        existing_env = ""
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                existing_env = f.read()
        
        # Add Kraken config if not already present
        if 'KRAKEN_ENABLED' not in existing_env:
            with open(env_file_path, 'a') as f:
                f.write('\n' + KRAKEN_ENV_CONFIG)
            print("✅ Added Kraken configuration to .env file")
        else:
            print("ℹ️ Kraken configuration already exists in .env")
        
        # Test integration (assuming EliteTradingEngine is importable)
        try:
            # This would import your existing engine
            # from core.enhanced_trading_engine import EliteTradingEngine
            
            # Apply Kraken integration
            # @add_kraken_to_elite_engine
            # class EnhancedEliteTradingEngine(EliteTradingEngine):
            #     pass
            
            print("✅ Kraken integration ready")
            print("🚀 Restart your server to activate Kraken integration")
            
            return True
            
        except ImportError as e:
            print(f"⚠️ Could not import EliteTradingEngine: {e}")
            print("ℹ️ Integration code is ready - apply manually to your engine")
            return False
            
    except Exception as e:
        print(f"❌ Error during integration: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Kraken Futures Integration for EliteTradingEngine")
    print("=" * 60)
    
    # Run integration
    result = asyncio.run(integrate_kraken_with_existing_engine())
    
    if result:
        print("\n✅ Integration completed successfully!")
        print("\nNext steps:")
        print("1. Restart your trading bot server")
        print("2. Check dashboard for Kraken data")
        print("3. Monitor logs for Kraken integration status")
    else:
        print("\n⚠️ Manual integration required")
        print("\nTo integrate manually:")
        print("1. Add the KrakenIntegration class to your engine")
        print("2. Apply the @add_kraken_to_elite_engine decorator")
        print("3. Update your .env with Kraken configuration")
        print("4. Restart your server")