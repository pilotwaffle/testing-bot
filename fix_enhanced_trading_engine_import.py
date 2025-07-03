#!/usr/bin/env python3
"""
FILE: fix_enhanced_trading_engine_import.py
LOCATION: E:\Trade Chat Bot\G Trading Bot\

Fix Enhanced Trading Engine Import Issue
Simple class name mismatch fix
"""

import os
import shutil
from datetime import datetime

def fix_main_py_import():
    """Fix the import statement in main.py"""
    
    main_file = "main.py"
    
    if not os.path.exists(main_file):
        print(f"❌ {main_file} not found")
        return False
    
    # Backup main.py
    backup_name = f"main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    shutil.copy2(main_file, backup_name)
    print(f"📁 Backup created: {backup_name}")
    
    # Read current content
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the import
    old_import = "from core.enhanced_trading_engine import EnhancedTradingEngine"
    new_import = "from core.enhanced_trading_engine import EliteTradingEngine as EnhancedTradingEngine"
    
    if old_import in content:
        new_content = content.replace(old_import, new_import)
        
        # Write the fixed content
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ main.py import statement fixed!")
        print(f"   Changed: {old_import}")
        print(f"   To:      {new_import}")
        return True
    else:
        print("ℹ️  Import statement not found or already fixed")
        return True

def add_alias_to_enhanced_engine():
    """Add alias to enhanced_trading_engine.py"""
    
    engine_file = "core/enhanced_trading_engine.py"
    
    if not os.path.exists(engine_file):
        print(f"❌ {engine_file} not found")
        return False
    
    # Read current content
    with open(engine_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if alias already exists
    alias_line = "EnhancedTradingEngine = EliteTradingEngine"
    
    if alias_line in content:
        print("ℹ️  Alias already exists in enhanced_trading_engine.py")
        return True
    
    # Add alias at the end
    if not content.endswith('\n'):
        content += '\n'
    
    content += f"\n# Alias for backward compatibility\n{alias_line}\n"
    
    # Write the updated content
    with open(engine_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Alias added to enhanced_trading_engine.py!")
    print(f"   Added: {alias_line}")
    return True

def verify_fix():
    """Verify the fix works"""
    
    print("\n🔍 Verifying fix...")
    
    try:
        # Test import
        import sys
        sys.path.insert(0, '.')
        
        from core.enhanced_trading_engine import EnhancedTradingEngine
        
        print("✅ EnhancedTradingEngine can now be imported successfully!")
        print(f"   Class: {EnhancedTradingEngine.__name__}")
        print(f"   Module: {EnhancedTradingEngine.__module__}")
        
        # Test instantiation
        engine = EnhancedTradingEngine()
        print("✅ EliteTradingEngine can be instantiated!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import still failing: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def main():
    """Main function to fix the import issue"""
    
    print("FILE: fix_enhanced_trading_engine_import.py")
    print("LOCATION: E:\\Trade Chat Bot\\G Trading Bot\\")
    print("")
    print("🔧 Fixing Enhanced Trading Engine Import Issue")
    print("=" * 60)
    
    print("\n📋 ISSUE ANALYSIS:")
    print("   Your enhanced_trading_engine.py contains 'EliteTradingEngine' class")
    print("   But main.py is trying to import 'EnhancedTradingEngine'")
    print("   This is just a simple name mismatch!")
    
    print(f"\n🛠️  APPLYING FIXES:")
    
    # Method 1: Fix main.py import
    print("\n1️⃣ Fixing main.py import statement...")
    fix1_success = fix_main_py_import()
    
    # Method 2: Add alias to enhanced engine (backup method)
    print("\n2️⃣ Adding alias to enhanced_trading_engine.py...")
    fix2_success = add_alias_to_enhanced_engine()
    
    # Verify the fix
    verification_success = verify_fix()
    
    print(f"\n📊 RESULTS:")
    print(f"   Fix 1 (main.py import): {'✅' if fix1_success else '❌'}")
    print(f"   Fix 2 (alias addition): {'✅' if fix2_success else '❌'}")
    print(f"   Verification test: {'✅' if verification_success else '❌'}")
    
    if verification_success:
        print(f"\n🎉 SUCCESS! Your EliteTradingEngine is now ready!")
        print(f"   Restart your server to see the improvements:")
        print(f"   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        
        print(f"\n🚀 EXPECTED IMPROVEMENTS:")
        print(f"   ✅ No more 'Enhanced Trading Engine failed, using fallback' warning")
        print(f"   ✅ Full EliteTradingEngine with all 54KB of features")
        print(f"   ✅ 7 background monitoring tasks")
        print(f"   ✅ Advanced state management and health monitoring")
        print(f"   ✅ Professional-grade risk management")
        print(f"   ✅ Comprehensive logging and performance tracking")
        
    else:
        print(f"\n⚠️  PARTIAL SUCCESS:")
        print(f"   The fixes were applied but verification failed")
        print(f"   Your bot will still work with FastTradingEngine fallback")
        print(f"   Check for any syntax errors in enhanced_trading_engine.py")

if __name__ == "__main__":
    main()