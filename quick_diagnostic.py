#!/usr/bin/env python3
"""
Quick Windows-Compatible Diagnostic for Trading Bot Hanging Issues
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

def safe_test(test_name, test_func, timeout_seconds=10):
    """Run a test with timeout protection"""
    print(f"\n🔍 Testing: {test_name}")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(test_func)
        
        try:
            result = future.result(timeout=timeout_seconds)
            elapsed = time.time() - start_time
            print(f"✅ PASSED ({elapsed:.2f}s): {test_name}")
            return True, result
        except concurrent.futures.TimeoutError:
            elapsed = time.time() - start_time
            print(f"⏰ TIMEOUT ({elapsed:.2f}s): {test_name} - HANGING DETECTED!")
            future.cancel()
            return False, "TIMEOUT"
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ FAILED ({elapsed:.2f}s): {test_name} - {str(e)}")
            return False, str(e)

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic Python imports...")
    import os, sys, time, json
    return "Basic imports OK"

def test_third_party_imports():
    """Test third-party imports one by one"""
    imports_to_test = [
        'fastapi',
        'uvicorn', 
        'pandas',
        'numpy',
        'requests',
        'yfinance',
        'ccxt',
        'sklearn',
        'tensorflow',
        'websockets'
    ]
    
    results = {}
    for imp in imports_to_test:
        try:
            exec(f'import {imp}')
            print(f"  ✅ {imp}")
            results[imp] = "OK"
        except Exception as e:
            print(f"  ❌ {imp}: {e}")
            results[imp] = str(e)
    
    return results

def test_enhanced_engine_import():
    """Test importing enhanced_trading_engine.py"""
    engine_paths = [
        'enhanced_trading_engine.py',
        'core/enhanced_trading_engine.py',
        'engine/enhanced_trading_engine.py'
    ]
    
    for path in engine_paths:
        if Path(path).exists():
            print(f"  Found engine at: {path}")
            
            # Try to import it
            spec = importlib.util.spec_from_file_location("enhanced_engine", path)
            module = importlib.util.module_from_spec(spec)
            
            # This is where it likely hangs
            spec.loader.exec_module(module)
            return f"Successfully imported from {path}"
    
    return "Enhanced engine file not found"

def test_main_py_import():
    """Test importing main.py"""
    if Path('main.py').exists():
        print("  Found main.py, attempting import...")
        spec = importlib.util.spec_from_file_location("main", "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return "Main.py imported successfully"
    else:
        return "main.py not found"

def find_hanging_component():
    """Run tests to identify hanging component"""
    print("🚀 Quick Diagnostic for Trading Bot Hanging Issues")
    print("=" * 60)
    
    # Test 1: Basic functionality
    success, result = safe_test("Basic Python Imports", test_basic_imports, 5)
    if not success:
        print("🚨 Basic Python functionality is broken!")
        return
    
    # Test 2: Third-party libraries  
    success, result = safe_test("Third-Party Library Imports", test_third_party_imports, 15)
    if not success and result == "TIMEOUT":
        print("🚨 HANGING DETECTED: Third-party library imports")
        print("   Likely cause: TensorFlow, sklearn, or other ML library loading")
        return
    
    # Test 3: Enhanced engine import
    success, result = safe_test("Enhanced Trading Engine Import", test_enhanced_engine_import, 20)
    if not success and result == "TIMEOUT":
        print("🚨 HANGING DETECTED: Enhanced Trading Engine Import")
        print("   This is your main issue! The enhanced_trading_engine.py is hanging during import.")
        print("   Common causes:")
        print("   - TensorFlow model loading")
        print("   - Database connection attempts")
        print("   - Network calls during import")
        print("   - Infinite loops in class initialization")
        return
    
    # Test 4: Main application import
    success, result = safe_test("Main Application Import", test_main_py_import, 25)
    if not success and result == "TIMEOUT":
        print("🚨 HANGING DETECTED: Main Application Import")
        print("   The main.py file is hanging during import")
        return
    
    print("\n🎉 No hanging detected in basic tests!")
    print("The issue might be in:")
    print("- Specific function calls during startup")
    print("- Component initialization after import")
    print("- Database/network connections during app startup")

if __name__ == "__main__":
    print("Enter timeout per test (default 15 seconds):")
    try:
        timeout = int(input() or "15")
    except:
        timeout = 15
    
    find_hanging_component()
    
    print(f"\n🏁 Quick diagnostic complete!")
    print(f"If hanging was detected, that's your culprit!")