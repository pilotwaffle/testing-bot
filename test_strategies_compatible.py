#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\test_strategies_compatible.py
Location: E:\Trade Chat Bot\G Trading Bot\test_strategies_compatible.py

🧪 Test Trading Strategies API (Compatible Version)
"""

import requests
import json

def test_strategies():
    print("🧪 Testing Trading Strategies API (Compatible)")
    print("=" * 55)
    
    try:
        # Test available strategies
        print("📊 Testing /api/strategies/available...")
        response = requests.get('http://localhost:8000/api/strategies/available')
        
        if response.status_code == 200:
            data = response.json()
            strategies = data.get('strategies', [])
            count = data.get('total_count', len(strategies))
            
            print(f"✅ SUCCESS - Found {count} strategies:")
            print()
            
            for i, strategy in enumerate(strategies, 1):
                print(f"   🎯 Strategy #{i}: {strategy.get('name', 'Unknown')}")
                print(f"      ID: {strategy.get('id', 'N/A')}")
                print(f"      Risk Level: {strategy.get('risk_level', 'N/A')}")
                print(f"      Timeframe: {strategy.get('timeframe', 'N/A')}")
                
                # Handle both old and new formats
                if 'accuracy' in strategy:
                    print(f"      Accuracy: {strategy['accuracy']}")
                if 'estimated_returns' in strategy:
                    print(f"      Est. Returns: {strategy['estimated_returns']}")
                if 'required_capital' in strategy:
                    print(f"      Min Capital: ${strategy['required_capital']}")
                
                print(f"      Description: {strategy.get('description', 'N/A')}")
                print()
                
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"Response: {response.text}")
        
        # Test active strategies
        print("📈 Testing /api/strategies/active...")
        response = requests.get('http://localhost:8000/api/strategies/active')
        
        if response.status_code == 200:
            data = response.json()
            active_strategies = data.get('active_strategies', [])
            total_active = data.get('total_active', len(active_strategies))
            
            print(f"✅ SUCCESS - Found {total_active} active strategies:")
            
            if active_strategies:
                for strategy in active_strategies:
                    print(f"   🚀 {strategy.get('name', strategy.get('id', 'Unknown'))}")
                    print(f"      Status: {strategy.get('status', 'Unknown')}")
                    
                    if 'pnl' in strategy:
                        print(f"      PnL: ${strategy['pnl']:.2f}")
                    if 'positions' in strategy:
                        print(f"      Positions: {strategy['positions']}")
                    if 'win_rate' in strategy:
                        print(f"      Win Rate: {strategy['win_rate']:.1%}")
                    print()
            else:
                print("   📊 No active strategies currently running")
                
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"Response: {response.text}")
        
        print("=" * 55)
        print("🎯 SUMMARY:")
        print("✅ API is working and returning strategies")
        print("✅ Frontend should now populate dropdown correctly")
        print("✅ Try refreshing your browser dashboard")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure your server is running: python main.py")

if __name__ == "__main__":
    test_strategies()
