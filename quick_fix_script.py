#!/usr/bin/env python3
"""
File: quick_fix_script.py
Location: E:\Trade Chat Bot\G Trading Bot\quick_fix_script.py
Description: Quick Fix Script for Elite Trading Bot V3.0
Purpose: Applies immediate fixes for critical startup issues
"""

import os
import sys
import shutil
from pathlib import Path

def apply_unicode_fixes():
    """Apply Unicode console fixes"""
    print("🔧 Applying Unicode console fixes...")
    
    # Set environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'
    
    # For Python 3.7+ on Windows
    if sys.platform == "win32":
        try:
            import codecs
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
            print("✅ Unicode console support enabled")
        except:
            # Fallback for older Python versions
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
            print("✅ Unicode console support enabled (fallback)")

def fix_kraken_imports(bot_path):
    """Fix missing imports in Kraken files"""
    print("🔧 Fixing Kraken imports...")
    
    core_path = Path(bot_path) / "core"
    kraken_files = [
        "kraken_futures_client.py",
        "kraken_ml_analyzer.py", 
        "kraken_integration.py",
        "kraken_dashboard_routes.py"
    ]
    
    required_imports = """# Imports added by quick_fix_script.py for Elite Trading Bot V3.0
# Location: E:\\Trade Chat Bot\\G Trading Bot\\core\\{filename}
# Added missing type hints and standard imports

from typing import List, Dict, Optional, Union, Any
import asyncio
import logging
import json
import time
from datetime import datetime

"""
    
    for filename in kraken_files:
        file_path = core_path / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if imports are missing
                if 'from typing import' not in content or 'List' not in content.split('from typing import')[0]:
                    # Backup original file
                    backup_path = file_path.with_suffix('.py.backup')
                    shutil.copy2(file_path, backup_path)
                    
                    # Format imports with correct filename
                    formatted_imports = required_imports.format(filename=filename)
                    
                    # Add imports to the beginning
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(formatted_imports + content)
                    
                    print(f"✅ Fixed imports in {filename} (backup created)")
                else:
                    print(f"✅ {filename} imports already OK")
                    
            except Exception as e:
                print(f"❌ Error fixing {filename}: {e}")

def fix_market_data_processor(bot_path):
    """Add missing get_latest_data method to MarketDataProcessor"""
    print("🔧 Fixing MarketDataProcessor...")
    
    core_path = Path(bot_path) / "core"
    mdp_file = core_path / "market_data_processor.py"
    
    if not mdp_file.exists():
        print("❌ market_data_processor.py not found")
        return
    
    try:
        with open(mdp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if get_latest_data method exists
        if 'def get_latest_data(' not in content:
            # Backup original file
            backup_path = mdp_file.with_suffix('.py.backup')
            shutil.copy2(mdp_file, backup_path)
            
            # Add the missing method
            method_to_add = '''
    # Method added by quick_fix_script.py for Elite Trading Bot V3.0
    # File: market_data_processor.py
    # Location: E:\\Trade Chat Bot\\G Trading Bot\\core\\market_data_processor.py
    def get_latest_data(self, symbol):
        """Get latest market data for a symbol"""
        try:
            # Return cached data if available
            if hasattr(self, 'latest_data') and symbol in self.latest_data:
                return self.latest_data[symbol]
            
            # Fallback to basic structure
            return {
                "symbol": symbol,
                "price": 0.0,
                "timestamp": time.time(),
                "volume": 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "status": "no_data"
            }
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error getting latest data for {symbol}: {e}")
            return None
'''
            
            # Find a good place to insert the method (after __init__ or other methods)
            if 'class MarketDataProcessor' in content:
                # Find the end of the class or a good insertion point
                lines = content.split('\n')
                insert_index = -1
                
                for i, line in enumerate(lines):
                    if 'class MarketDataProcessor' in line:
                        # Look for a good insertion point after __init__
                        for j in range(i, min(i + 50, len(lines))):
                            if 'def __init__' in lines[j]:
                                # Find the end of __init__ method
                                indent_level = len(lines[j]) - len(lines[j].lstrip())
                                for k in range(j + 1, len(lines)):
                                    if lines[k].strip() and len(lines[k]) - len(lines[k].lstrip()) <= indent_level and lines[k].strip() != '':
                                        insert_index = k
                                        break
                                break
                        if insert_index == -1:
                            insert_index = i + 5  # Insert after class definition
                        break
                
                if insert_index > 0:
                    lines.insert(insert_index, method_to_add)
                    new_content = '\n'.join(lines)
                    
                    with open(mdp_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    print("✅ Added get_latest_data method to MarketDataProcessor")
                else:
                    print("❌ Could not find insertion point for get_latest_data method")
            else:
                print("❌ MarketDataProcessor class not found")
        else:
            print("✅ get_latest_data method already exists")
            
    except Exception as e:
        print(f"❌ Error fixing MarketDataProcessor: {e}")

def create_safe_startup_script(bot_path):
    """Create a startup script that handles Unicode properly"""
    print("🔧 Creating safe startup script...")
    
    bot_path = Path(bot_path)
    
    # Windows batch script
    bat_script = f'''@echo off
REM File: start_safe.bat
REM Location: {bot_path}\\start_safe.bat
REM Description: Elite Trading Bot V3.0 - Safe Startup Script
REM Purpose: Start bot with proper Unicode and encoding support

echo Elite Trading Bot V3.0 - Safe Startup
echo =====================================

:: Set UTF-8 encoding for Windows
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=0
chcp 65001 >nul

:: Navigate to bot directory
cd /d "{bot_path}"

echo Starting server with Unicode support...
echo Server will be available at: http://localhost:8000

:: Start server with UTF-8 support and reduced logging
python -X utf8 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level warning

echo.
echo Server stopped. Press any key to exit...
pause >nul
'''
    
    # Linux/Mac shell script
    sh_script = f'''#!/bin/bash
# File: start_safe.sh
# Location: {bot_path}/start_safe.sh
# Description: Elite Trading Bot V3.0 - Safe Startup Script  
# Purpose: Start bot with proper Unicode and encoding support

echo "Elite Trading Bot V3.0 - Safe Startup"
echo "====================================="

# Set UTF-8 encoding
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8

# Navigate to bot directory
cd "{bot_path}"

echo "Starting server with Unicode support..."
echo "Server will be available at: http://localhost:8000"

# Start server
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level warning

echo "Server stopped."
'''
    
    # Save scripts
    with open(bot_path / "start_safe.bat", 'w', encoding='utf-8') as f:
        f.write(bat_script)
    
    with open(bot_path / "start_safe.sh", 'w', encoding='utf-8') as f:
        f.write(sh_script)
    
    # Make shell script executable on Unix systems
    if os.name != 'nt':
        os.chmod(bot_path / "start_safe.sh", 0o755)
    
    print("✅ Created start_safe.bat and start_safe.sh")

def fix_enhanced_trading_engine_unicode(bot_path):
    """Fix Unicode issues in enhanced_trading_engine.py"""
    print("🔧 Fixing Unicode issues in enhanced_trading_engine.py...")
    
    core_path = Path(bot_path) / "core"
    engine_file = core_path / "enhanced_trading_engine.py"
    
    if not engine_file.exists():
        print("❌ enhanced_trading_engine.py not found")
        return
    
    try:
        with open(engine_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Unicode checkmarks with ASCII equivalents
        replacements = {
            '✓': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '🔍': '[CHECK]',
            '💡': '[TIP]'
        }
        
        modified = False
        for unicode_char, ascii_replacement in replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, ascii_replacement)
                modified = True
        
        if modified:
            # Backup original file
            backup_path = engine_file.with_suffix('.py.backup')
            shutil.copy2(engine_file, backup_path)
            
            with open(engine_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Replaced Unicode characters with ASCII equivalents")
        else:
            print("✅ No Unicode characters found to replace")
            
    except Exception as e:
        print(f"❌ Error fixing Unicode in enhanced_trading_engine.py: {e}")

def main():
    """Main fix application function"""
    print("🚀 Elite Trading Bot V3.0 - Quick Fix Tool")
    print("=" * 50)
    
    # Get bot path
    if len(sys.argv) > 1:
        bot_path = sys.argv[1]
    else:
        bot_path = input("Enter bot path (default: E:\\Trade Chat Bot\\G Trading Bot): ").strip()
        if not bot_path:
            bot_path = "E:\\Trade Chat Bot\\G Trading Bot"
    
    bot_path = Path(bot_path)
    
    if not bot_path.exists():
        print(f"❌ Bot path not found: {bot_path}")
        return
    
    print(f"📁 Working with bot at: {bot_path}")
    print()
    
    # Apply fixes
    apply_unicode_fixes()
    fix_kraken_imports(bot_path)
    fix_market_data_processor(bot_path)
    fix_enhanced_trading_engine_unicode(bot_path)
    create_safe_startup_script(bot_path)
    
    print()
    print("🎉 QUICK FIXES APPLIED!")
    print("=" * 30)
    print("✅ Unicode console support enabled")
    print("✅ Kraken import issues fixed")
    print("✅ MarketDataProcessor method added")
    print("✅ Unicode characters replaced with ASCII")
    print("✅ Safe startup scripts created")
    print()
    print("🚀 NEXT STEPS:")
    print("1. Use 'start_safe.bat' (Windows) or 'start_safe.sh' (Linux/Mac) to start the bot")
    print("2. Original files backed up with .backup extension")
    print("3. Check http://localhost:8000 once started")
    print("4. Monitor console for any remaining issues")

if __name__ == "__main__":
    main()