
# Quick test script - save as test_chat_direct.py
import sys
sys.path.append(".")

try:
    from ai.chat_manager import EnhancedChatManager
    from core.enhanced_trading_engine import EliteTradingEngine
    from core.ml_engine import MLEngine
    from core.data_fetcher import DataFetcher
    
    print("🔧 Testing chat manager initialization...")
    
    # Initialize dependencies
    trading_engine = EliteTradingEngine()
    ml_engine = MLEngine() 
    data_fetcher = DataFetcher()
    
    print("✅ Dependencies initialized")
    
    # Initialize chat manager
    chat_manager = EnhancedChatManager(
        trading_engine=trading_engine,
        ml_engine=ml_engine, 
        data_fetcher=data_fetcher
    )
    
    print("✅ Chat manager initialized")
    
    # Test process_message method
    test_messages = ["help", "status", "hello"]
    
    for message in test_messages:
        try:
            response = chat_manager.process_message(message)
            print(f"✅ '{message}' -> '{response[:100]}...'")
        except Exception as e:
            print(f"❌ '{message}' -> Error: {e}")
            
except Exception as e:
    print(f"❌ Chat manager test failed: {e}")
    import traceback
    traceback.print_exc()
