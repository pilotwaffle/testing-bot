#!/usr/bin/env python3
"""
Quick API Test Script for Elite Trading Bot V3.0
Run this to quickly test if your API endpoints are working
"""

import requests
import json
import sys
from datetime import datetime

def test_api_endpoint(url, endpoint_name):
    """Test a single API endpoint"""
    print(f"🔍 Testing {endpoint_name}...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
        
        if response.status_code == 200:
            if 'application/json' in response.headers.get('content-type', ''):
                try:
                    data = response.json()
                    print(f"   ✅ {endpoint_name} - JSON response OK")
                    
                    # Show relevant data based on endpoint
                    if 'market-data' in url:
                        print(f"   📊 Success: {data.get('success', 'unknown')}")
                        print(f"   💱 Currency: {data.get('currency', 'unknown')}")
                        symbols = data.get('symbols', {})
                        print(f"   🪙 Symbols: {len(symbols)} found")
                        if symbols:
                            first_symbol = list(symbols.keys())[0]
                            price = symbols[first_symbol].get('price', 'unknown')
                            print(f"   💰 Sample: {first_symbol} = ${price}")
                    elif 'health' in url:
                        print(f"   🏥 Status: {data.get('status', 'unknown')}")
                        components = data.get('components', {})
                        print(f"   🔧 Components: {sum(1 for v in components.values() if v)} active")
                    
                    return True
                except json.JSONDecodeError:
                    print(f"   ❌ {endpoint_name} - Invalid JSON response")
                    print(f"   Raw: {response.text[:100]}...")
                    return False
            else:
                print(f"   ✅ {endpoint_name} - Non-JSON response (might be OK)")
                print(f"   Content: {response.text[:100]}...")
                return True
        elif response.status_code == 404:
            print(f"   ❌ {endpoint_name} - 404 NOT FOUND")
            print("   💡 The endpoint is not registered in main.py")
            return False
        elif response.status_code == 500:
            print(f"   ❌ {endpoint_name} - 500 SERVER ERROR") 
            print(f"   Error: {response.text[:200]}...")
            return False
        else:
            print(f"   ❌ {endpoint_name} - Unexpected status {response.status_code}")
            print(f"   Response: {response.text[:100]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ {endpoint_name} - CONNECTION FAILED")
        print("   💡 Bot is not running or not accessible")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ {endpoint_name} - TIMEOUT (>10 seconds)")
        print("   💡 Bot is running but very slow")
        return False
    except Exception as e:
        print(f"   ❌ {endpoint_name} - ERROR: {e}")
        return False

def main():
    print("🧪 Elite Trading Bot V3.0 - Quick API Test")
    print("=" * 50)
    print(f"⏰ Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    base_url = "http://localhost:8000"
    
    # Test endpoints in order of importance
    endpoints = [
        ("/ping", "Basic Ping"),
        ("/health", "Health Check"),
        ("/api/market-data", "Market Data API"),
        ("/api/trading-pairs", "Trading Pairs API"),
        ("/", "Dashboard"),
    ]
    
    results = []
    
    for endpoint, name in endpoints:
        url = f"{base_url}{endpoint}"
        success = test_api_endpoint(url, name)
        results.append((name, success))
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("=" * 30)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
        if success:
            passed += 1
    
    total = len(results)
    print()
    print(f"📈 Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print()
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your API is working correctly!")
        print()
        print("🌐 Your bot is available at:")
        print(f"   • Dashboard: {base_url}")
        print(f"   • Market Data: {base_url}/api/market-data")
        print(f"   • Health Check: {base_url}/health")
    elif passed == 0:
        print()
        print("❌ ALL TESTS FAILED!")
        print()
        print("💡 Most likely causes:")
        print("   1. Bot is not running - Start with: python main.py")
        print("   2. Wrong port - Check if bot is on different port")
        print("   3. Firewall blocking - Try running as Administrator")
        print("   4. Bot crashed - Check console for error messages")
    else:
        print()
        print("⚠️  PARTIAL SUCCESS")
        print()
        print("💡 Issues found:")
        for name, success in results:
            if not success:
                if "Market Data" in name:
                    print("   • Market Data API not working - Apply enhanced main.py")
                elif "Basic Ping" in name:
                    print("   • Basic connectivity failed - Check if bot is running")
                elif "Health Check" in name:
                    print("   • Health endpoint missing - Update main.py")
    
    print()
    print("🔧 Quick fixes:")
    print("   • If 404 errors: Apply the enhanced main.py from Claude")
    print("   • If connection errors: Make sure bot is running")
    print("   • If all fail: Check Windows Firewall and port 8000")
    print()
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        input("Press Enter to exit...")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test script error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)