"""
File: fix_kraken_final.py
Location: E:\Trade Chat Bot\G Trading Bot\fix_kraken_final.py

Kraken Integration Final Fix Script
Fixes the remaining Kraken integration issue to get it working
"""

import os
import shutil
import importlib.util
from pathlib import Path
from datetime import datetime

def backup_kraken_files():
    """Backup Kraken-related files"""
    files_to_backup = [
        "core/kraken_integration.py",
        "main.py"
    ]
    
    backups = []
    for file_path in files_to_backup:
        if Path(file_path).exists():
            backup_name = f"{file_path}.backup_kraken_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_name)
            backups.append(backup_name)
            print(f"📁 Backup created: {backup_name}")
    
    return backups

def analyze_kraken_integration():
    """Analyze the Kraken integration to find the specific issue"""
    print("🔍 Analyzing Kraken Integration Issue")
    print("=" * 50)
    
    kraken_file = Path("core/kraken_integration.py")
    if not kraken_file.exists():
        print("❌ core/kraken_integration.py not found")
        return False
    
    try:
        with open(kraken_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check the KrakenIntegration class constructor
        if "class KrakenIntegration" in content:
            print("✅ KrakenIntegration class found")
            
            # Find the __init__ method
            init_start = content.find("def __init__(self")
            if init_start != -1:
                init_end = content.find("def ", init_start + 1)
                if init_end == -1:
                    init_end = len(content)
                
                init_method = content[init_start:init_end]
                print(f"🔍 Constructor signature found:")
                
                # Extract just the signature line
                signature_line = init_method.split('\n')[0]
                print(f"   {signature_line}")
                
                # Check what parameters it expects
                if "trading_engine" in signature_line:
                    print("✅ Expects 'trading_engine' parameter")
                    return "needs_trading_engine"
                elif "sandbox" in signature_line:
                    print("✅ Expects 'sandbox' parameter")  
                    return "needs_sandbox"
                else:
                    print("✅ Takes no parameters")
                    return "no_params"
            else:
                print("❌ No __init__ method found")
                return False
        else:
            print("❌ KrakenIntegration class not found")
            return False
            
    except Exception as e:
        print(f"❌ Error analyzing kraken_integration.py: {e}")
        return False

def test_kraken_import():
    """Test importing Kraken integration"""
    print("\n🧪 Testing Kraken Import")
    print("=" * 50)
    
    try:
        # Try to import the KrakenIntegration class
        from core.kraken_integration import KrakenIntegration
        print("✅ KrakenIntegration imported successfully")
        
        # Try to get class info
        import inspect
        signature = inspect.signature(KrakenIntegration.__init__)
        params = list(signature.parameters.keys())
        print(f"🔍 Constructor parameters: {params}")
        
        return params
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error getting class info: {e}")
        return None

def fix_main_py_kraken_init(constructor_params):
    """Fix the Kraken initialization in main.py"""
    print(f"\n🔧 Fixing Kraken Initialization in main.py")
    print("=" * 50)
    
    if not Path("main.py").exists():
        print("❌ main.py not found")
        return False
    
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the Kraken initialization section
        kraken_init_start = content.find("# Initialize Kraken Integration")
        if kraken_init_start == -1:
            kraken_init_start = content.find("from core.kraken_integration import KrakenIntegration")
        
        if kraken_init_start == -1:
            print("❌ Kraken initialization section not found")
            return False
        
        # Determine the correct initialization based on constructor parameters
        if 'trading_engine' in constructor_params:
            new_init = """try:
    from core.kraken_integration import KrakenIntegration
    if trading_engine:
        kraken_integration = KrakenIntegration(trading_engine)
        logger.info("✅ Kraken Integration initialized with trading engine")
    else:
        logger.warning("⚠️ Cannot initialize Kraken without trading engine")
        kraken_integration = None
except Exception as e:
    logger.error(f"❌ Error initializing KrakenIntegration: {e}")
    kraken_integration = None"""
            
        elif 'sandbox' in constructor_params:
            new_init = """try:
    from core.kraken_integration import KrakenIntegration
    kraken_integration = KrakenIntegration(sandbox=True)
    logger.info("✅ Kraken Integration initialized (sandbox mode)")
except Exception as e:
    logger.error(f"❌ Error initializing KrakenIntegration: {e}")
    kraken_integration = None"""
            
        else:
            new_init = """try:
    from core.kraken_integration import KrakenIntegration
    kraken_integration = KrakenIntegration()
    logger.info("✅ Kraken Integration initialized")
except Exception as e:
    logger.error(f"❌ Error initializing KrakenIntegration: {e}")
    kraken_integration = None"""
        
        # Replace the problematic initialization
        # Find the try block for Kraken initialization
        kraken_try_start = content.find("try:", kraken_init_start)
        if kraken_try_start != -1:
            # Find the end of this try block (next 'try:' or 'except:' at same level)
            kraken_try_end = content.find("\nexcept", kraken_try_start)
            if kraken_try_end != -1:
                kraken_try_end = content.find("\n\n", kraken_try_end)
                if kraken_try_end != -1:
                    # Replace the entire try block
                    new_content = content[:kraken_try_start] + new_init + content[kraken_try_end:]
                    
                    with open("main.py", 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    print("✅ Kraken initialization fixed in main.py")
                    return True
        
        print("❌ Could not find exact location to fix")
        return False
        
    except Exception as e:
        print(f"❌ Error fixing main.py: {e}")
        return False

def create_test_script():
    """Create a test script to verify Kraken integration"""
    print("\n🧪 Creating Kraken Test Script")
    print("=" * 50)
    
    test_script = '''"""
File: test_kraken_final.py
Location: E:\\Trade Chat Bot\\G Trading Bot\\test_kraken_final.py

Final Kraken Integration Test
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def test_kraken_integration():
    """Test Kraken integration step by step"""
    print("🧪 Final Kraken Integration Test")
    print("=" * 50)
    
    try:
        # Test 1: Import trading engine
        print("Test 1: Importing trading engine...")
        try:
            from core.enhanced_trading_engine import EnhancedTradingEngine
            trading_engine = EnhancedTradingEngine()
            print("✅ Enhanced Trading Engine imported and initialized")
        except Exception as e:
            print(f"⚠️ Enhanced Trading Engine failed: {e}")
            try:
                from core.trading_engine import TradingEngine
                trading_engine = TradingEngine()
                print("✅ Basic Trading Engine imported and initialized")
            except Exception as e2:
                print(f"❌ All trading engines failed: {e2}")
                trading_engine = None
        
        # Test 2: Import Kraken integration
        print("\\nTest 2: Importing Kraken integration...")
        from core.kraken_integration import KrakenIntegration
        print("✅ KrakenIntegration imported successfully")
        
        # Test 3: Get constructor signature
        import inspect
        signature = inspect.signature(KrakenIntegration.__init__)
        params = list(signature.parameters.keys())
        print(f"✅ Constructor parameters: {params}")
        
        # Test 4: Initialize based on parameters
        print("\\nTest 3: Initializing Kraken integration...")
        
        if 'trading_engine' in params and trading_engine:
            kraken = KrakenIntegration(trading_engine)
            print("✅ KrakenIntegration initialized with trading engine")
        elif 'sandbox' in params:
            kraken = KrakenIntegration(sandbox=True)
            print("✅ KrakenIntegration initialized with sandbox mode")
        else:
            kraken = KrakenIntegration()
            print("✅ KrakenIntegration initialized with no parameters")
        
        # Test 5: Check status
        print("\\nTest 4: Checking Kraken status...")
        try:
            status = kraken.get_status()
            print(f"✅ Status retrieved: {type(status)}")
            if isinstance(status, dict):
                print(f"   Status keys: {list(status.keys())}")
        except Exception as e:
            print(f"⚠️ Status check failed: {e}")
        
        print("\\n🎉 KRAKEN INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Kraken integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kraken_integration()
    if success:
        print("\\n✅ Kraken integration is working!")
        print("🚀 Restart your server to see Kraken as 'Available'")
    else:
        print("\\n❌ Kraken integration needs manual fixing")
'''
    
    try:
        with open("test_kraken_final.py", 'w', encoding='utf-8') as f:
            f.write(test_script)
        print("✅ Created test_kraken_final.py")
        return True
    except Exception as e:
        print(f"❌ Error creating test script: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Kraken Integration Final Fix")
    print("=" * 60)
    
    # Step 1: Backup files
    backup_kraken_files()
    
    # Step 2: Analyze the issue
    constructor_type = analyze_kraken_integration()
    if not constructor_type:
        print("❌ Could not analyze Kraken integration")
        return
    
    # Step 3: Test import and get actual parameters
    actual_params = test_kraken_import()
    if not actual_params:
        print("❌ Could not import Kraken integration")
        return
    
    # Step 4: Fix main.py with correct parameters
    if fix_main_py_kraken_init(actual_params):
        print("✅ main.py Kraken initialization fixed")
    else:
        print("❌ Could not fix main.py")
        return
    
    # Step 5: Create test script
    create_test_script()
    
    print("\n🎉 KRAKEN INTEGRATION FIX COMPLETE!")
    print("=" * 60)
    
    print("🧪 Test the fix:")
    print("   python test_kraken_final.py")
    print()
    print("🚀 Then restart your server:")
    print("   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("📊 Expected health check result:")
    print('   "kraken_integration": true  ✅')
    print()
    print("🌐 Check at: http://localhost:8000/health")

if __name__ == "__main__":
    main()