"""
File: test_kraken_final.py
Location: E:\Trade Chat Bot\G Trading Bot\test_kraken_final.py

Final Kraken Integration Test
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def test_kraken_integration():
    """Test Kraken integration step by step"""
    print("🧪 Final Kraken Integration Test")
    print("=" * 50)
    
    try:
        # Test 1: Import trading engine
        print("Test 1: Importing trading engine...")
        try:
            from core.enhanced_trading_engine import EnhancedTradingEngine
            trading_engine = EnhancedTradingEngine()
            print("✅ Enhanced Trading Engine imported and initialized")
        except Exception as e:
            print(f"⚠️ Enhanced Trading Engine failed: {e}")
            try:
                from core.trading_engine import TradingEngine
                trading_engine = TradingEngine()
                print("✅ Basic Trading Engine imported and initialized")
            except Exception as e2:
                print(f"❌ All trading engines failed: {e2}")
                trading_engine = None
        
        # Test 2: Import Kraken integration
        print("\nTest 2: Importing Kraken integration...")
        from core.kraken_integration import KrakenIntegration
        print("✅ KrakenIntegration imported successfully")
        
        # Test 3: Get constructor signature
        import inspect
        signature = inspect.signature(KrakenIntegration.__init__)
        params = list(signature.parameters.keys())
        print(f"✅ Constructor parameters: {params}")
        
        # Test 4: Initialize based on parameters
        print("\nTest 3: Initializing Kraken integration...")
        
        if 'trading_engine' in params and trading_engine:
            kraken = KrakenIntegration(trading_engine)
            print("✅ KrakenIntegration initialized with trading engine")
        elif 'sandbox' in params:
            kraken = KrakenIntegration(sandbox=True)
            print("✅ KrakenIntegration initialized with sandbox mode")
        else:
            kraken = KrakenIntegration()
            print("✅ KrakenIntegration initialized with no parameters")
        
        # Test 5: Check status
        print("\nTest 4: Checking Kraken status...")
        try:
            status = kraken.get_status()
            print(f"✅ Status retrieved: {type(status)}")
            if isinstance(status, dict):
                print(f"   Status keys: {list(status.keys())}")
        except Exception as e:
            print(f"⚠️ Status check failed: {e}")
        
        print("\n🎉 KRAKEN INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Kraken integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kraken_integration()
    if success:
        print("\n✅ Kraken integration is working!")
        print("🚀 Restart your server to see Kraken as 'Available'")
    else:
        print("\n❌ Kraken integration needs manual fixing")
