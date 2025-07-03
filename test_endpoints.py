#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\test_endpoints.py
Location: E:\Trade Chat Bot\G Trading Bot\test_endpoints.py

🧪 Quick endpoint test script - Run this AFTER starting your server
"""

import requests
import json
from datetime import datetime

def test_endpoint(url, name):
    """Test a single endpoint"""
    try:
        print(f"🧪 Testing {name}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {name} - SUCCESS")
            
            # Show a sample of the data
            if isinstance(data, dict):
                if 'data' in data and isinstance(data['data'], list):
                    print(f"   📊 Data count: {len(data['data'])} items")
                elif 'strategies' in data:
                    print(f"   📊 Strategies: {len(data['strategies'])} available")
                elif 'active_strategies' in data:
                    print(f"   📊 Active strategies: {len(data['active_strategies'])}")
                elif 'performance' in data:
                    print(f"   📊 Performance data loaded")
                else:
                    print(f"   📊 Response: {list(data.keys())}")
            
            return True
        else:
            print(f"❌ {name} - Status: {response.status_code}")
            print(f"   Error: {response.text[:100]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ {name} - Connection Error (Server not running?)")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ {name} - Timeout (Server slow to respond)")
        return False
    except Exception as e:
        print(f"❌ {name} - Error: {e}")
        return False

def main():
    """Test all endpoints"""
    print("🚀 Elite Trading Bot V3.0 - Endpoint Test")
    print("="*50)
    print(f"🕒 Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    base_url = "http://localhost:8000"
    
    endpoints = [
        ("/health", "Health Check"),
        ("/api/market-data", "Market Data"),
        ("/api/strategies/available", "Available Strategies"),
        ("/api/strategies/active", "Active Strategies"), 
        ("/api/performance", "Performance Metrics"),
        ("/ping", "Ping Test"),
        ("/static/css/style.css", "CSS File")
    ]
    
    results = []
    
    for endpoint, name in endpoints:
        url = f"{base_url}{endpoint}"
        success = test_endpoint(url, name)
        results.append((name, success))
        print()  # Add spacing
    
    # Summary
    print("="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n🎯 Results: {passed}/{total} endpoints working")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your dashboard should work perfectly!")
    elif passed >= total - 1:
        print("✅ Almost perfect! Check the failing endpoint.")
    else:
        print("⚠️ Some endpoints need attention. Check server logs.")
    
    print("\n💡 Next step: Open http://localhost:8000 in your browser")

if __name__ == "__main__":
    main()