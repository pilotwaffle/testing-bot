"""
File: kraken_fix.py
Location: E:\Trade Chat Bot\G Trading Bot\kraken_fix.py

Kraken Integration Fix Script
Diagnoses and fixes Kraken integration availability issues
"""

import os
import sys
import json
import importlib.util
from pathlib import Path
import traceback

def check_kraken_files():
    """Check if Kraken integration files exist"""
    print("🔍 Checking Kraken Integration Files")
    print("=" * 50)
    
    kraken_files = [
        "core/kraken_integration.py",
        "core/kraken_futures_client.py", 
        "core/kraken_ml_analyzer.py",
        "core/kraken_dashboard_routes.py"
    ]
    
    found_files = []
    missing_files = []
    
    for file_path in kraken_files:
        if Path(file_path).exists():
            found_files.append(file_path)
            print(f"✅ Found: {file_path}")
            
            # Check file size
            size = Path(file_path).stat().st_size
            print(f"   📏 Size: {size} bytes")
            
            # Check for common imports/classes
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'class' in content:
                    classes = [line.strip() for line in content.split('\n') if line.strip().startswith('class ')]
                    if classes:
                        print(f"   📦 Classes: {len(classes)} found")
                        for cls in classes[:2]:  # Show first 2 classes
                            print(f"      • {cls.split('(')[0].replace('class ', '')}")
                    
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
    
    return found_files, missing_files

def test_kraken_imports():
    """Test importing Kraken modules"""
    print("\n🔍 Testing Kraken Module Imports")
    print("=" * 50)
    
    import_tests = [
        ("core.kraken_integration", "KrakenIntegration"),
        ("core.kraken_futures_client", "KrakenFuturesClient"),
        ("core.kraken_ml_analyzer", "KrakenMLAnalyzer"),
        ("core.kraken_dashboard_routes", None)
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name, class_name in import_tests:
        try:
            print(f"🧪 Testing: {module_name}")
            
            # Try to import the module
            module = importlib.import_module(module_name)
            print(f"   ✅ Module imported successfully")
            
            # Try to access the class if specified
            if class_name:
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    print(f"   ✅ Class {class_name} found")
                    successful_imports.append((module_name, class_name))
                else:
                    print(f"   ❌ Class {class_name} not found in module")
                    failed_imports.append((module_name, f"Class {class_name} missing"))
            else:
                successful_imports.append((module_name, "Module only"))
                
        except ImportError as e:
            print(f"   ❌ Import failed: {e}")
            failed_imports.append((module_name, f"ImportError: {e}"))
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            failed_imports.append((module_name, f"Error: {e}"))
    
    return successful_imports, failed_imports

def check_kraken_credentials():
    """Check Kraken API credentials configuration"""
    print("\n🔍 Checking Kraken Credentials Configuration")
    print("=" * 50)
    
    # Check environment variables
    kraken_env_vars = [
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",
        "KRAKEN_SANDBOX",
        "KRAKEN_FUTURES_API_KEY",
        "KRAKEN_FUTURES_SECRET"
    ]
    
    env_status = {}
    for var in kraken_env_vars:
        value = os.getenv(var)
        if value:
            # Hide the actual value for security
            masked_value = f"{value[:4]}..." if len(value) > 4 else "***"
            print(f"✅ {var}: {masked_value}")
            env_status[var] = True
        else:
            print(f"❌ {var}: Not set")
            env_status[var] = False
    
    # Check config files
    config_files = [
        "config.py",
        "core/config.py", 
        "settings.py",
        ".env"
    ]
    
    print(f"\n📄 Checking configuration files:")
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ Found: {config_file}")
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'kraken' in content.lower():
                        print(f"   📊 Contains Kraken configuration")
            except Exception as e:
                print(f"   ❌ Error reading {config_file}: {e}")
        else:
            print(f"❌ Missing: {config_file}")
    
    return env_status

def check_main_py_kraken_init():
    """Check how Kraken is initialized in main.py"""
    print("\n🔍 Checking Kraken Initialization in main.py")
    print("=" * 50)
    
    if not Path("main.py").exists():
        print("❌ main.py not found")
        return False
    
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for Kraken imports
        kraken_imports = [
            "kraken_integration",
            "KrakenIntegration", 
            "kraken_futures",
            "KrakenFuturesClient"
        ]
        
        found_imports = []
        for imp in kraken_imports:
            if imp in content:
                found_imports.append(imp)
                print(f"✅ Found import/reference: {imp}")
        
        if not found_imports:
            print("❌ No Kraken imports found in main.py")
            return False
        
        # Check for initialization
        init_patterns = [
            "kraken_integration =",
            "KrakenIntegration()",
            "kraken_client =",
            "KrakenFuturesClient("
        ]
        
        found_inits = []
        for pattern in init_patterns:
            if pattern in content:
                found_inits.append(pattern)
                print(f"✅ Found initialization: {pattern}")
        
        if not found_inits:
            print("❌ No Kraken initialization found in main.py")
            return False
        
        print("✅ Kraken appears to be properly referenced in main.py")
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing main.py: {e}")
        return False

def create_kraken_test_script():
    """Create a test script to verify Kraken integration"""
    print("\n🛠️ Creating Kraken Test Script")
    print("=" * 50)
    
    test_script = '''"""
File: test_kraken.py
Location: E:\\Trade Chat Bot\\G Trading Bot\\test_kraken.py

Kraken Integration Test Script
Tests Kraken integration independently
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def test_kraken_integration():
    """Test Kraken integration step by step"""
    print("🧪 Testing Kraken Integration")
    print("=" * 50)
    
    try:
        # Test 1: Import kraken integration
        print("Test 1: Importing Kraken integration...")
        from core.kraken_integration import KrakenIntegration
        print("✅ KrakenIntegration imported successfully")
        
        # Test 2: Initialize Kraken integration (sandbox mode)
        print("\\nTest 2: Initializing Kraken integration...")
        kraken = KrakenIntegration(sandbox=True)
        print("✅ KrakenIntegration initialized successfully")
        
        # Test 3: Check status
        print("\\nTest 3: Checking Kraken status...")
        status = kraken.get_status()
        print(f"✅ Status: {status}")
        
        # Test 4: Test connection
        print("\\nTest 4: Testing connection...")
        is_connected = kraken.test_connection()
        print(f"✅ Connection test: {'PASSED' if is_connected else 'FAILED'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   → Check if core/kraken_integration.py exists")
        print("   → Check for missing dependencies")
        return False
        
    except Exception as e:
        print(f"❌ Kraken integration test failed: {e}")
        print(f"   → Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kraken_integration()
    if success:
        print("\\n🎉 Kraken integration is working!")
    else:
        print("\\n❌ Kraken integration needs fixing")
'''
    
    try:
        with open("test_kraken.py", 'w', encoding='utf-8') as f:
            f.write(test_script)
        print("✅ Created test_kraken.py")
        print("📋 Run with: python test_kraken.py")
        return True
    except Exception as e:
        print(f"❌ Error creating test script: {e}")
        return False

def provide_kraken_fixes(found_files, failed_imports, env_status):
    """Provide specific fixes based on diagnostic results"""
    print("\n💡 Kraken Integration Fixes")
    print("=" * 50)
    
    fixes_needed = []
    
    # Check for missing files
    if len(found_files) < 4:
        fixes_needed.append("missing_files")
        print("🔧 Fix 1: Missing Kraken Files")
        print("   → Some Kraken integration files are missing")
        print("   → Check if files were moved or renamed")
        print("   → Restore from backup if available")
        print()
    
    # Check for import failures
    if failed_imports:
        fixes_needed.append("import_errors")
        print("🔧 Fix 2: Import Errors")
        for module, error in failed_imports:
            print(f"   → {module}: {error}")
        print("   → Check for syntax errors in Kraken files")
        print("   → Verify all dependencies are installed")
        print()
    
    # Check for missing credentials
    if not any(env_status.values()):
        fixes_needed.append("no_credentials")
        print("🔧 Fix 3: Missing Credentials")
        print("   → No Kraken API credentials found")
        print("   → For paper trading, credentials may not be required")
        print("   → Set KRAKEN_SANDBOX=true for testing")
        print()
    
    # Provide general solution
    print("🔧 General Solution Steps:")
    print("1. Run: python test_kraken.py")
    print("2. Check the specific error messages")
    print("3. Fix import errors first")
    print("4. Configure credentials if needed")
    print("5. Restart the server")
    
    return fixes_needed

def main():
    """Main diagnostic function"""
    print("🔍 Kraken Integration Diagnostic")
    print("=" * 50)
    
    # Step 1: Check files
    found_files, missing_files = check_kraken_files()
    
    # Step 2: Test imports
    successful_imports, failed_imports = test_kraken_imports()
    
    # Step 3: Check credentials
    env_status = check_kraken_credentials()
    
    # Step 4: Check main.py initialization
    main_py_ok = check_main_py_kraken_init()
    
    # Step 5: Create test script
    create_kraken_test_script()
    
    # Step 6: Provide fixes
    provide_kraken_fixes(found_files, failed_imports, env_status)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Files found: {len(found_files)}/4")
    print(f"   Successful imports: {len(successful_imports)}")
    print(f"   Failed imports: {len(failed_imports)}")
    print(f"   Credentials configured: {sum(env_status.values())}")
    print(f"   Main.py integration: {'✅' if main_py_ok else '❌'}")
    
    print(f"\n🚀 Next Steps:")
    print("1. Run: python test_kraken.py")
    print("2. Fix any errors shown")
    print("3. Restart your trading bot server")
    print("4. Check dashboard - Kraken should show as 'Available'")

if __name__ == "__main__":
    main()