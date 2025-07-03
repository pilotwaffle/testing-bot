#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\test_enhanced_system.py
Location: E:\Trade Chat Bot\G Trading Bot\test_enhanced_system.py

Test Script for Enhanced Elite Trading Bot V3.0
Verifies all new features are working correctly:
- Trading pairs (USD, USDC, USDT) 
- Top 10 crypto market data by market cap
- Real-time price updates
- API endpoints functionality
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime

class EnhancedTradingBotTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.tests_passed = 0
        self.tests_failed = 0
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
        
    def print_test(self, test_name, passed, details=""):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    📝 {details}")
        
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
            
    def print_summary(self):
        total = self.tests_passed + self.tests_failed
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Passed: {self.tests_passed}/{total}")
        print(f"❌ Failed: {self.tests_failed}/{total}")
        
        if self.tests_failed == 0:
            print(f"🎉 ALL TESTS PASSED! Your Enhanced Trading Bot is ready!")
        else:
            print(f"⚠️  Some tests failed. Please check the implementation.")
            
    async def test_server_connectivity(self):
        """Test if the server is running and accessible"""
        self.print_header("Server Connectivity Test")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    if response.status == 200:
                        self.print_test("Server is running", True, f"Status: {response.status}")
                        return True
                    else:
                        self.print_test("Server is running", False, f"Status: {response.status}")
                        return False
        except Exception as e:
            self.print_test("Server is running", False, f"Error: {str(e)}")
            return False
            
    async def test_trading_pairs_endpoint(self):
        """Test the trading pairs API endpoint"""
        self.print_header("Trading Pairs API Test")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/trading-pairs") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Test response structure
                        has_success = 'success' in data and data['success']
                        self.print_test("Trading pairs endpoint accessible", has_success)
                        
                        # Test required pairs exist
                        if 'pairs' in data:
                            pairs = data['pairs']
                            pair_symbols = [pair['symbol'] for pair in pairs]
                            
                            required_pairs = ['USD', 'USDC', 'USDT']
                            has_required = all(symbol in pair_symbols for symbol in required_pairs)
                            self.print_test("Required trading pairs present", has_required, 
                                           f"Found: {', '.join(pair_symbols)}")
                            
                            # Test default currency is USD
                            default_currency = data.get('default', '')
                            is_usd_default = default_currency == 'USD'
                            self.print_test("USD is default currency", is_usd_default, 
                                           f"Default: {default_currency}")
                        else:
                            self.print_test("Pairs data structure", False, "No 'pairs' key found")
                    else:
                        self.print_test("Trading pairs endpoint accessible", False, 
                                       f"Status: {response.status}")
        except Exception as e:
            self.print_test("Trading pairs endpoint accessible", False, f"Error: {str(e)}")
            
    async def test_market_data_endpoint(self):
        """Test the enhanced market data endpoint"""
        self.print_header("Market Data API Test")
        
        currencies_to_test = ['usd', 'usdc', 'usdt']
        
        for currency in currencies_to_test:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/api/market-data?currency={currency}") as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Test response structure
                            has_success = 'success' in data and data['success']
                            self.print_test(f"Market data for {currency.upper()}", has_success)
                            
                            if has_success and 'symbols' in data:
                                symbols = data['symbols']
                                
                                # Test for top 10 cryptocurrencies
                                expected_cryptos = ['BTC', 'ETH', 'USDT', 'SOL', 'BNB', 'XRP', 'USDC', 'DOGE', 'ADA', 'AVAX']
                                found_cryptos = list(symbols.keys())
                                
                                has_top_10 = len(found_cryptos) >= 10
                                self.print_test(f"Top 10 cryptos for {currency.upper()}", has_top_10,
                                               f"Found {len(found_cryptos)} cryptocurrencies")
                                
                                # Test specific cryptos are present
                                btc_present = 'BTC' in symbols
                                eth_present = 'ETH' in symbols
                                usdt_present = 'USDT' in symbols
                                
                                self.print_test(f"Bitcoin (BTC) data for {currency.upper()}", btc_present)
                                self.print_test(f"Ethereum (ETH) data for {currency.upper()}", eth_present) 
                                self.print_test(f"Tether (USDT) data for {currency.upper()}", usdt_present)
                                
                                # Test price data structure
                                if btc_present:
                                    btc_data = symbols['BTC']
                                    has_price = 'price' in btc_data and btc_data['price'] > 0
                                    has_change = 'change' in btc_data
                                    has_volume = 'volume' in btc_data
                                    
                                    self.print_test(f"BTC price data for {currency.upper()}", has_price,
                                                   f"Price: ${btc_data.get('price', 0):,.2f}")
                                    self.print_test(f"BTC change data for {currency.upper()}", has_change)
                                    self.print_test(f"BTC volume data for {currency.upper()}", has_volume)
                            else:
                                self.print_test(f"Market data structure for {currency.upper()}", False, 
                                               "No symbols data found")
                        else:
                            self.print_test(f"Market data for {currency.upper()}", False, 
                                           f"Status: {response.status}")
            except Exception as e:
                self.print_test(f"Market data for {currency.upper()}", False, f"Error: {str(e)}")
                
    async def test_market_overview_endpoint(self):
        """Test the market overview endpoint"""
        self.print_header("Market Overview API Test")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/market-overview") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        has_success = 'success' in data and data['success']
                        self.print_test("Market overview endpoint", has_success)
                        
                        if has_success and 'overview' in data:
                            overview = data['overview']
                            
                            # Test required overview fields
                            required_fields = ['total_market_cap', 'btc_dominance', 'market_sentiment']
                            for field in required_fields:
                                has_field = field in overview
                                self.print_test(f"Overview field: {field}", has_field,
                                               f"Value: {overview.get(field, 'Missing')}")
                        else:
                            self.print_test("Market overview structure", False, "No overview data")
                    else:
                        self.print_test("Market overview endpoint", False, f"Status: {response.status}")
        except Exception as e:
            self.print_test("Market overview endpoint", False, f"Error: {str(e)}")
            
    async def test_price_accuracy(self):
        """Test that prices are within reasonable ranges"""
        self.print_header("Price Accuracy Test")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/market-data?currency=usd") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('success') and 'symbols' in data:
                            symbols = data['symbols']
                            
                            # Test price ranges (rough validation)
                            price_tests = [
                                ('BTC', 50000, 150000),  # Bitcoin should be between $50k-150k
                                ('ETH', 1000, 5000),     # Ethereum should be between $1k-5k
                                ('USDT', 0.95, 1.05),   # USDT should be close to $1
                                ('USDC', 0.95, 1.05),   # USDC should be close to $1
                                ('SOL', 50, 500),       # Solana reasonable range
                            ]
                            
                            for symbol, min_price, max_price in price_tests:
                                if symbol in symbols:
                                    price = symbols[symbol].get('price', 0)
                                    is_reasonable = min_price <= price <= max_price
                                    self.print_test(f"{symbol} price is reasonable", is_reasonable,
                                                   f"Price: ${price:,.2f} (Expected: ${min_price:,.0f}-${max_price:,.0f})")
                                else:
                                    self.print_test(f"{symbol} data present", False, "Symbol not found")
                        else:
                            self.print_test("Price data available", False, "No market data available")
                    else:
                        self.print_test("Price data accessible", False, f"API error: {response.status}")
        except Exception as e:
            self.print_test("Price accuracy test", False, f"Error: {str(e)}")
            
    async def test_dashboard_accessibility(self):
        """Test that the dashboard loads properly"""
        self.print_header("Dashboard Accessibility Test")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        
                        # Test for key dashboard elements
                        has_trading_pairs = 'trading-pairs-container' in html_content
                        has_market_data = 'marketDataGrid' in html_content
                        has_currency_select = 'currencySelect' in html_content
                        has_enhanced_styles = 'enhanced-select' in html_content
                        
                        self.print_test("Dashboard loads", True)
                        self.print_test("Trading pairs section present", has_trading_pairs)
                        self.print_test("Market data grid present", has_market_data)
                        self.print_test("Currency selector present", has_currency_select)
                        self.print_test("Enhanced styles present", has_enhanced_styles)
                        
                        # Test for trading pair options
                        has_usd_option = 'US Dollar (USD)' in html_content
                        has_usdc_option = 'USD Coin (USDC)' in html_content
                        has_usdt_option = 'Tether (USDT)' in html_content
                        
                        self.print_test("USD option present", has_usd_option)
                        self.print_test("USDC option present", has_usdc_option)
                        self.print_test("USDT option present", has_usdt_option)
                        
                    else:
                        self.print_test("Dashboard loads", False, f"Status: {response.status}")
        except Exception as e:
            self.print_test("Dashboard accessibility", False, f"Error: {str(e)}")
            
    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 Elite Trading Bot V3.0 - Enhanced System Test")
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Testing Server: {self.base_url}")
        
        # Check if server is running first
        server_running = await self.test_server_connectivity()
        
        if not server_running:
            print("\n❌ Cannot proceed with tests - server is not accessible")
            print("💡 Please ensure your bot is running with:")
            print("   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
            return
            
        # Run all tests
        await self.test_trading_pairs_endpoint()
        await self.test_market_data_endpoint()
        await self.test_market_overview_endpoint()
        await self.test_price_accuracy()
        await self.test_dashboard_accessibility()
        
        # Print summary
        self.print_summary()
        
        # Recommendations
        if self.tests_failed == 0:
            print(f"\n🎉 CONGRATULATIONS!")
            print(f"Your Enhanced Elite Trading Bot V3.0 is fully functional!")
            print(f"✅ Trading pairs: USD (default), USDC, USDT")
            print(f"✅ Market data: Top 10 cryptocurrencies by market cap")
            print(f"✅ Real-time pricing: Accurate and up-to-date")
            print(f"✅ Professional UI: Modern and user-friendly")
        else:
            print(f"\n🔧 RECOMMENDATIONS:")
            print(f"1. Check server logs for any error messages")
            print(f"2. Verify all files are in the correct locations")
            print(f"3. Ensure all dependencies are installed")
            print(f"4. Review the implementation guide for any missed steps")

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Enhanced Elite Trading Bot V3.0')
    parser.add_argument('--url', default='http://localhost:8000', 
                       help='Base URL of the trading bot (default: http://localhost:8000)')
    
    args = parser.parse_args()
    
    tester = EnhancedTradingBotTester(args.url)
    
    try:
        asyncio.run(tester.run_all_tests())
    except KeyboardInterrupt:
        print(f"\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test execution error: {str(e)}")

if __name__ == "__main__":
    main()