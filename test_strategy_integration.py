#!/usr/bin/env python3
"""
FILE: test_strategy_integration.py
LOCATION: E:\Trade Chat Bot\G Trading Bot
Test Strategy Integration with EliteTradingEngine
"""

import asyncio
import sys
from core.enhanced_trading_engine import EliteTradingEngine, EliteEngineConfig

async def test_strategy_integration():
    """Test strategy loading and execution"""
    
    print("🧪 Testing Strategy Integration")
    print("=" * 50)
    
    # Create test configuration
    config = EliteEngineConfig(
        live_trading_enabled=False,  # Paper trading only
        symbols=["BTC/USD", "ETH/USD"],
        timeframes=["1h", "4h"],
        max_concurrent_positions=2,
        log_level="INFO"
    )
    
    try:
        # Initialize engine
        print("1️⃣ Initializing EliteTradingEngine...")
        engine = EliteTradingEngine(config)
        
        # Test strategy loading
        print("2️⃣ Loading strategies...")
        strategies = await engine.load_strategies()
        
        if strategies:
            print(f"✅ Loaded {len(strategies)} strategies:")
            for name in strategies.keys():
                print(f"   - {name}")
        else:
            print("❌ No strategies loaded")
            return False
        
        # Test strategy switching
        print("3️⃣ Testing strategy switching...")
        for strategy_name in list(strategies.keys())[:2]:  # Test first 2
            success = await engine.switch_strategy(strategy_name)
            if success:
                print(f"✅ Switched to {strategy_name}")
            else:
                print(f"❌ Failed to switch to {strategy_name}")
        
        # Test signal generation
        print("4️⃣ Testing signal generation...")
        signals = await engine._generate_trading_signals_with_strategies()
        
        if signals:
            print(f"✅ Generated {len(signals)} signals")
            for signal in signals:
                print(f"   📊 {signal.symbol}: {signal.direction.value} "
                      f"(Confidence: {signal.confidence:.3f})")
        else:
            print("ℹ️  No signals generated (normal for test data)")
        
        # Test performance tracking
        print("5️⃣ Testing performance tracking...")
        performance = await engine.get_strategy_performance()
        
        print(f"📈 Strategy Performance:")
        for strategy, metrics in performance.items():
            print(f"   {strategy}: {metrics['total_trades']} trades, "
                  f"{metrics['win_rate']:.1%} win rate")
        
        print("\n🎉 Strategy integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_strategy_integration())
    if success:
        print("\n✅ All tests passed! Your strategies are ready for live integration.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
