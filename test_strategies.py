#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\test_strategies.py
Location: E:\Trade Chat Bot\G Trading Bot\test_strategies.py

🧪 Test Trading Strategies API
"""

import requests
import json

def test_strategies():
    print("🧪 Testing Trading Strategies API")
    print("=" * 50)
    
    try:
        # Test available strategies
        print("📊 Testing /api/strategies/available...")
        response = requests.get('http://localhost:8000/api/strategies/available')
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS - Found {data.get('count', 0)} strategies:")
            
            for strategy in data.get('strategies', []):
                print(f"   🎯 {strategy['name']} ({strategy['id']})")
                print(f"      Risk: {strategy['risk_level']} | Accuracy: {strategy['accuracy']}")
                print(f"      Timeframe: {strategy['timeframe']}")
                print()
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"Response: {response.text}")
        
        # Test active strategies
        print("📈 Testing /api/strategies/active...")
        response = requests.get('http://localhost:8000/api/strategies/active')
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS - Found {data.get('total_active', 0)} active strategies:")
            
            for strategy in data.get('active_strategies', []):
                print(f"   🚀 {strategy['name']} - {strategy['status']}")
                print(f"      PnL: ${strategy['pnl']:.2f} | Positions: {strategy['positions']}")
                print(f"      Win Rate: {strategy['win_rate']:.1%}")
                print()
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure your server is running: python main.py")

if __name__ == "__main__":
    test_strategies()
