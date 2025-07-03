#!/usr/bin/env python3
"""
Complete System Diagnostic Script for Crypto Trading Bot
Tests all components individually to identify hanging issues
"""

import sys
import time
import traceback
import threading
import os
import importlib.util
from datetime import datetime
from pathlib import Path
import concurrent.futures

class TimeoutError(Exception):
    pass

class SystemDiagnostic:
    def __init__(self, timeout_seconds=10):
        self.timeout_seconds = timeout_seconds
        self.results = {}
        self.current_test = None
        
    def run_with_timeout(self, test_name, test_func, *args, **kwargs):
        """Run a test function with timeout protection (Windows compatible)"""
        self.current_test = test_name
        print(f"\n🔍 Testing: {test_name}")
        print(f"⏰ Timeout set to: {self.timeout_seconds} seconds")
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for Windows-compatible timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(test_func, *args, **kwargs)
            
            try:
                result = future.result(timeout=self.timeout_seconds)
                elapsed = time.time() - start_time
                print(f"✅ PASSED in {elapsed:.2f}s: {test_name}")
                self.results[test_name] = {
                    'status': 'PASSED',
                    'time': elapsed,
                    'result': str(result) if result else 'Success'
                }
                return result
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - start_time
                print(f"⏰ TIMEOUT after {elapsed:.2f}s: {test_name}")
                self.results[test_name] = {
                    'status': 'TIMEOUT',
                    'time': elapsed,
                    'error': f"Test '{test_name}' timed out after {self.timeout_seconds} seconds"
                }
                # Cancel the future to prevent resource leaks
                future.cancel()
                return None
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ FAILED after {elapsed:.2f}s: {test_name}")
                print(f"   Error: {str(e)}")
                self.results[test_name] = {
                    'status': 'FAILED',
                    'time': elapsed,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                return None
    
    def test_basic_imports(self):
        """Test basic Python imports"""
        imports_to_test = [
            ('os', 'import os'),
            ('sys', 'import sys'),
            ('time', 'import time'),
            ('json', 'import json'),
            ('asyncio', 'import asyncio'),
            ('threading', 'import threading'),
            ('sqlite3', 'import sqlite3'),
            ('logging', 'import logging'),
        ]
        
        for name, import_stmt in imports_to_test:
            try:
                exec(import_stmt)
                print(f"✅ Basic import OK: {name}")
            except Exception as e:
                print(f"❌ Basic import FAILED: {name} - {e}")
    
    def test_third_party_imports(self):
        """Test third-party library imports"""
        imports_to_test = [
            ('fastapi', 'import fastapi'),
            ('uvicorn', 'import uvicorn'),
            ('pandas', 'import pandas'),
            ('numpy', 'import numpy'),
            ('requests', 'import requests'),
            ('yfinance', 'import yfinance'),
            ('ccxt', 'import ccxt'),
            ('sklearn', 'import sklearn'),
            ('tensorflow', 'import tensorflow'),
            ('websockets', 'import websockets'),
            ('aiohttp', 'import aiohttp'),
        ]
        
        for name, import_stmt in imports_to_test:
            try:
                exec(import_stmt)
                print(f"✅ Third-party import OK: {name}")
            except Exception as e:
                print(f"❌ Third-party import FAILED: {name} - {e}")
    
    def test_file_exists(self, filepath):
        """Test if a file exists and is readable"""
        try:
            path = Path(filepath)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(100)  # Read first 100 chars
                print(f"✅ File exists and readable: {filepath}")
                return True
            else:
                print(f"❌ File not found: {filepath}")
                return False
        except Exception as e:
            print(f"❌ File read error: {filepath} - {e}")
            return False
    
    def test_import_module_by_path(self, module_name, file_path):
        """Test importing a specific module by file path"""
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                raise ImportError(f"Could not create spec for {module_name}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ Module import OK: {module_name} from {file_path}")
            return module
        except Exception as e:
            print(f"❌ Module import FAILED: {module_name} - {e}")
            return None
    
    def test_class_instantiation(self, module, class_name):
        """Test instantiating a class from a module"""
        try:
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                # Try to instantiate with no args first
                try:
                    instance = cls()
                    print(f"✅ Class instantiation OK: {class_name}()")
                    return instance
                except TypeError:
                    # Try with common default args
                    try:
                        instance = cls({})  # Empty config
                        print(f"✅ Class instantiation OK: {class_name}({{}})")
                        return instance
                    except:
                        print(f"⚠️ Class exists but needs specific args: {class_name}")
                        return cls
            else:
                print(f"❌ Class not found: {class_name}")
                return None
        except Exception as e:
            print(f"❌ Class instantiation FAILED: {class_name} - {e}")
            return None
    
    def test_database_connection(self):
        """Test database connectivity"""
        try:
            import sqlite3
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
            conn.close()
            print("✅ SQLite connection OK")
            return True
        except Exception as e:
            print(f"❌ Database connection FAILED: {e}")
            return False
    
    def test_network_connectivity(self):
        """Test basic network connectivity"""
        try:
            import requests
            response = requests.get('https://httpbin.org/get', timeout=5)
            if response.status_code == 200:
                print("✅ Network connectivity OK")
                return True
            else:
                print(f"❌ Network test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Network connectivity FAILED: {e}")
            return False
    
    def test_trading_engine_components(self):
        """Test specific trading engine components"""
        components_to_test = [
            'core/enhanced_trading_engine.py',
            'core/fast_trading_engine.py', 
            'core/enhanced_ml_engine.py',
            'core/fast_ml_engine.py',
            'core/data_fetcher.py',
            'core/risk_manager.py',
            'core/notification_manager.py',
        ]
        
        for component_path in components_to_test:
            if self.test_file_exists(component_path):
                # Extract module name from path
                module_name = Path(component_path).stem
                self.run_with_timeout(
                    f"Import {module_name}",
                    self.test_import_module_by_path,
                    module_name,
                    component_path
                )
    
    def run_comprehensive_diagnostic(self):
        """Run all diagnostic tests"""
        print("🚀 Starting Comprehensive System Diagnostic")
        print("=" * 60)
        
        # Test 1: Basic Python functionality
        print("\n📦 Testing Basic Python Imports...")
        self.run_with_timeout("Basic Imports", self.test_basic_imports)
        
        # Test 2: Third-party libraries
        print("\n📦 Testing Third-Party Library Imports...")
        self.run_with_timeout("Third-Party Imports", self.test_third_party_imports)
        
        # Test 3: File system access
        print("\n📁 Testing File System Access...")
        files_to_check = [
            'main.py',
            'enhanced_trading_engine.py',
            'core/enhanced_trading_engine.py',
            'config.json',
            'requirements.txt'
        ]
        
        for file_path in files_to_check:
            self.run_with_timeout(f"File Check: {file_path}", self.test_file_exists, file_path)
        
        # Test 4: Database connectivity
        print("\n🗄️ Testing Database Connectivity...")
        self.run_with_timeout("Database Connection", self.test_database_connection)
        
        # Test 5: Network connectivity
        print("\n🌐 Testing Network Connectivity...")
        self.run_with_timeout("Network Connection", self.test_network_connectivity)
        
        # Test 6: Trading engine components
        print("\n⚙️ Testing Trading Engine Components...")
        self.test_trading_engine_components()
        
        # Test 7: Main application import
        print("\n🎯 Testing Main Application Import...")
        if self.test_file_exists('main.py'):
            self.run_with_timeout(
                "Main App Import",
                self.test_import_module_by_path,
                "main",
                "main.py"
            )
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC REPORT")
        print("=" * 60)
        
        passed = [k for k, v in self.results.items() if v['status'] == 'PASSED']
        failed = [k for k, v in self.results.items() if v['status'] == 'FAILED']
        timeouts = [k for k, v in self.results.items() if v['status'] == 'TIMEOUT']
        
        print(f"✅ PASSED: {len(passed)}")
        print(f"❌ FAILED: {len(failed)}")
        print(f"⏰ TIMEOUTS: {len(timeouts)}")
        
        if timeouts:
            print(f"\n🚨 HANGING COMPONENTS (LIKELY CAUSE):")
            for test in timeouts:
                result = self.results[test]
                print(f"   - {test}: Hung after {result['time']:.2f}s")
        
        if failed:
            print(f"\n❌ FAILED COMPONENTS:")
            for test in failed:
                result = self.results[test]
                print(f"   - {test}: {result['error']}")
        
        # Identify most likely culprit
        if timeouts:
            print(f"\n🔍 RECOMMENDED ACTION:")
            print(f"   The system is hanging at: {timeouts[0]}")
            print(f"   Check this component for infinite loops, blocking calls, or heavy computations")
        elif failed:
            print(f"\n🔍 RECOMMENDED ACTION:")
            print(f"   Fix the failed imports/components before proceeding")
        else:
            print(f"\n🎉 All tests passed! The hanging issue may be in component initialization")
        
        # Save detailed report
        self.save_detailed_report()
    
    def save_detailed_report(self):
        """Save detailed diagnostic report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"diagnostic_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("CRYPTO TRADING BOT - SYSTEM DIAGNOSTIC REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for test_name, result in self.results.items():
                f.write(f"Test: {test_name}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Time: {result['time']:.2f}s\n")
                
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                
                if 'traceback' in result:
                    f.write(f"Traceback:\n{result['traceback']}\n")
                
                f.write("-" * 30 + "\n")
        
        print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    """Main diagnostic function"""
    print("🔧 Crypto Trading Bot System Diagnostic Tool")
    print("This will test all components to find the hanging issue\n")
    
    # Allow user to set timeout
    try:
        timeout = int(input("Enter timeout per test in seconds (default 10): ") or "10")
    except ValueError:
        timeout = 10
    
    print(f"Using timeout: {timeout} seconds per test\n")
    
    diagnostic = SystemDiagnostic(timeout_seconds=timeout)
    diagnostic.run_comprehensive_diagnostic()
    
    print(f"\n🏁 Diagnostic Complete!")
    print(f"Check the generated report file for detailed analysis.")

if __name__ == "__main__":
    main()