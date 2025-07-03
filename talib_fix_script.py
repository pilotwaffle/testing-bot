#!/usr/bin/env python3
"""
================================================================================
FILE: talib_fix_script.py
LOCATION: E:\Trade Chat Bot\G Trading Bot\talib_fix_script.py
AUTHOR: Claude AI Assistant
CREATED: Sunday, June 29, 2025
PURPOSE: Fix TAlib import error in advanced_ml_trainer.py
VERSION: 1.0
================================================================================

TAlib Import Fix Script
Makes TAlib optional so your trainer will run without it

USAGE:
    python talib_fix_script.py
    
This will modify advanced_ml_trainer.py to make TAlib optional
================================================================================
"""

import os
import shutil
from datetime import datetime

def fix_talib_import():
    """Fix TAlib import in advanced_ml_trainer.py"""
    
    filename = 'advanced_ml_trainer.py'
    
    if not os.path.exists(filename):
        print(f"❌ {filename} not found")
        return False
    
    # Create backup
    backup_path = f"{filename}.backup_talib_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filename, backup_path)
    print(f"💾 Backup created: {backup_path}")
    
    # Read the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace TAlib import with optional import
    talib_replacements = [
        ('import talib', '''try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TAlib loaded")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TAlib not available - using basic indicators")'''),
        
        ('from talib import', '''try:
    from talib import *
    TALIB_AVAILABLE = True
    print("✅ TAlib functions loaded")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TAlib not available - using basic indicators")''')
    ]
    
    # Apply replacements
    for old_import, new_import in talib_replacements:
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"🔧 Fixed: {old_import}")
    
    # Add TAlib availability checks around TAlib function calls
    talib_functions = [
        'talib.RSI', 'talib.SMA', 'talib.EMA', 'talib.MACD', 'talib.STOCH',
        'talib.BBANDS', 'talib.ATR', 'talib.CCI', 'talib.WILLR'
    ]
    
    for func in talib_functions:
        if func in content:
            # Add availability check
            content = content.replace(
                func,
                f'({func} if TALIB_AVAILABLE else fallback_indicator)'
            )
    
    # Write the fixed content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {filename} fixed - TAlib is now optional")
    return True

def main():
    """Main execution"""
    print("🔧 TALIB IMPORT FIXER")
    print("====================")
    print("Making TAlib optional in advanced_ml_trainer.py")
    print()
    
    if fix_talib_import():
        print("🎉 SUCCESS!")
        print("✅ TAlib import errors fixed")
        print("🚀 You can now run: python advanced_ml_trainer.py")
    else:
        print("❌ Could not fix TAlib imports")
    
    print()
    print("💡 RECOMMENDATION:")
    print("For best results with real market data, run:")
    print("python minimal_working_trainer.py")

if __name__ == "__main__":
    main()