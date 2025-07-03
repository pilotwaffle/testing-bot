# FILE: debug_kraken_test.py
# LOCATION: E:\Trade Chat Bot\G Trading Bot\debug_kraken_test.py

#!/usr/bin/env python3
"""
🔍 DEBUG KRAKEN IMPLEMENTATION TEST
Simple diagnostic script to identify issues
"""

import sys
import os
import traceback
from pathlib import Path

def test_basic_functionality():
    """Test basic Python functionality"""
    print("=" * 60)
    print("🔍 KRAKEN IMPLEMENTATION DIAGNOSTIC TEST")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Basic Python
        print("✅ Test 1: Basic Python execution - PASSED")
        
        # Test 2: Current directory
        current_dir = Path.cwd()
        print(f"✅ Test 2: Current directory: {current_dir}")
        
        # Test 3: File system access
        files = list(current_dir.iterdir())
        print(f"✅ Test 3: Found {len(files)} files in current directory")
        
        # Test 4: Import test
        print("🔄 Test 4: Testing imports...")
        
        # Test standard library imports
        import asyncio
        print("  ✅ asyncio imported")
        
        import json
        print("  ✅ json imported")
        
        import logging
        print("  ✅ logging imported")
        
        # Test third-party imports
        try:
            import aiohttp
            print("  ✅ aiohttp imported")
        except ImportError as e:
            print(f"  ❌ aiohttp missing: {e}")
        
        try:
            import pandas
            print("  ✅ pandas imported")
        except ImportError as e:
            print(f"  ❌ pandas missing: {e}")
        
        try:
            import numpy
            print("  ✅ numpy imported")
        except ImportError as e:
            print(f"  ❌ numpy missing: {e}")
        
        try:
            import sklearn
            print("  ✅ scikit-learn imported")
        except ImportError as e:
            print(f"  ❌ scikit-learn missing: {e}")
        
        try:
            import fastapi
            print("  ✅ fastapi imported")
        except ImportError as e:
            print(f"  ❌ fastapi missing: {e}")
        
        # Test 5: Check core directory
        core_dir = current_dir / "core"
        if core_dir.exists():
            print(f"✅ Test 5: Core directory exists with {len(list(core_dir.iterdir()))} files")
            
            # Check for Kraken files
            kraken_files = [
                'kraken_futures_client.py',
                'kraken_ml_analyzer.py', 
                'kraken_integration.py',
                'kraken_dashboard_routes.py'
            ]
            
            for file in kraken_files:
                file_path = core_dir / file
                if file_path.exists():
                    print(f"  ✅ {file} exists ({file_path.stat().st_size} bytes)")
                else:
                    print(f"  ❌ {file} missing")
        else:
            print("❌ Test 5: Core directory not found")
        
        # Test 6: Environment check
        print("🔄 Test 6: Environment check...")
        print(f"  Python version: {sys.version}")
        print(f"  Platform: {sys.platform}")
        print(f"  Working directory: {os.getcwd()}")
        
        # Test 7: Try basic async function
        print("🔄 Test 7: Testing async functionality...")
        
        async def test_async():
            await asyncio.sleep(0.1)
            return "async works"
        
        result = asyncio.run(test_async())
        print(f"  ✅ Async test result: {result}")
        
        print()
        print("🎉 DIAGNOSTIC COMPLETE")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR in diagnostic test: {e}")
        print("FULL TRACEBACK:")
        traceback.print_exc()
        return False

def test_kraken_file_creation():
    """Test creating a simple Kraken file"""
    print("\n🔄 Testing Kraken file creation...")
    
    try:
        # Create core directory if needed
        core_dir = Path("core")
        core_dir.mkdir(exist_ok=True)
        
        # Create a simple test file
        test_file = core_dir / "kraken_test.py"
        
        test_content = '''# FILE: core/kraken_test.py
# LOCATION: E:\\Trade Chat Bot\\G Trading Bot\\core\\kraken_test.py

"""
Simple test file for Kraken integration
"""

def test_function():
    return "Kraken test file created successfully!"

if __name__ == "__main__":
    print(test_function())
'''
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        print(f"✅ Test file created: {test_file}")
        
        # Test importing the file
        sys.path.append(str(core_dir.parent))
        from core.kraken_test import test_function
        result = test_function()
        print(f"✅ Test import successful: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating test file: {e}")
        traceback.print_exc()
        return False

def install_missing_packages():
    """Install missing packages"""
    print("\n📦 Checking and installing missing packages...")
    
    packages = [
        'aiohttp',
        'pandas', 
        'numpy',
        'scikit-learn',
        'fastapi',
        'uvicorn'
    ]
    
    missing_packages = []
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package} already installed")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} missing")
    
    if missing_packages:
        print(f"\n🔄 Installing {len(missing_packages)} missing packages...")
        import subprocess
        
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Packages installed successfully!")
                return True
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing packages: {e}")
            return False
    else:
        print("✅ All required packages are already installed!")
        return True

def main():
    """Main diagnostic function"""
    try:
        print("Starting Kraken implementation diagnostic...")
        
        # Run basic tests
        basic_test = test_basic_functionality()
        
        # Install missing packages
        install_test = install_missing_packages()
        
        # Test file creation
        file_test = test_kraken_file_creation()
        
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"Basic functionality: {'✅ PASS' if basic_test else '❌ FAIL'}")
        print(f"Package installation: {'✅ PASS' if install_test else '❌ FAIL'}")
        print(f"File creation test: {'✅ PASS' if file_test else '❌ FAIL'}")
        
        if basic_test and install_test and file_test:
            print("\n🎉 ALL TESTS PASSED!")
            print("💡 Your system is ready for Kraken implementation")
            print("\nNext steps:")
            print("1. Run the full implement_kraken.py script")
            print("2. If it still fails, check the specific error messages")
        else:
            print("\n⚠️ SOME TESTS FAILED")
            print("💡 Fix the failing tests before proceeding")
        
        return basic_test and install_test and file_test
        
    except Exception as e:
        print(f"\n❌ DIAGNOSTIC FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Kraken Implementation Diagnostic...")
    print("This will help identify why implement_kraken.py isn't working")
    print()
    
    try:
        success = main()
        
        if success:
            print("\n✅ Diagnostic completed successfully!")
        else:
            print("\n❌ Diagnostic found issues that need fixing")
            
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n⛔ Diagnostic cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")