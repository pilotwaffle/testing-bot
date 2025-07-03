#!/usr/bin/env python3
"""
fix_imports.py - Fix import statements to use core module
This script will update import statements in your Python files to use the correct core module paths
"""

import os
import re

def fix_import_in_file(filepath, old_import, new_import):
    """Fix a specific import in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace the import statement
        updated_content = content.replace(old_import, new_import)
        
        if content != updated_content:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(updated_content)
            print(f"✅ Fixed imports in {filepath}")
            return True
        else:
            print(f"ℹ️  No changes needed in {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def main():
    """Fix common import issues"""
    print("🔧 Fixing import statements to use core module...")
    print()
    
    # Common import fixes needed
    import_fixes = [
        # Core ML Engine
        {
            'files': ['enhanced_model_trainer.py', 'main_trading_bot.py', 'web_dashboard.py'],
            'old': 'from enhanced_ml_engine import AdaptiveMLEngine',
            'new': 'from core.enhanced_ml_engine import AdaptiveMLEngine'
        },
        {
            'files': ['enhanced_model_trainer.py', 'main_trading_bot.py', 'web_dashboard.py'],
            'old': 'import enhanced_ml_engine',
            'new': 'import core.enhanced_ml_engine as enhanced_ml_engine'
        },
        # Data Fetcher
        {
            'files': ['enhanced_model_trainer.py', 'main_trading_bot.py', 'web_dashboard.py'],
            'old': 'from enhanced_data_fetcher import',
            'new': 'from core.enhanced_data_fetcher import'
        },
        # Trading Strategy
        {
            'files': ['main_trading_bot.py', 'web_dashboard.py'],
            'old': 'from enhanced_trading_strategy import',
            'new': 'from core.enhanced_trading_strategy import'
        },
        # Performance Monitor
        {
            'files': ['main_trading_bot.py', 'web_dashboard.py'],
            'old': 'from performance_monitor import',
            'new': 'from core.performance_monitor import'
        },
        # Config imports
        {
            'files': ['enhanced_model_trainer.py', 'main_trading_bot.py', 'web_dashboard.py'],
            'old': 'from config import',
            'new': 'from core.config import'
        }
    ]
    
    files_updated = 0
    
    for fix in import_fixes:
        for filename in fix['files']:
            if os.path.exists(filename):
                if fix_import_in_file(filename, fix['old'], fix['new']):
                    files_updated += 1
            else:
                print(f"⚠️  File not found: {filename}")
    
    print()
    print(f"🎉 Import fixing completed! Updated {files_updated} files.")
    print()
    
    # Additional check: look for any remaining import errors
    print("🔍 Checking for remaining potential import issues...")
    
    python_files = [f for f in os.listdir('.') if f.endswith('.py')]
    
    for filename in python_files:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Look for problematic imports
                problematic_patterns = [
                    r'from enhanced_\w+ import',
                    r'import enhanced_\w+',
                    r'from performance_monitor import',
                    r'import performance_monitor'
                ]
                
                for pattern in problematic_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        print(f"⚠️  Potential import issue in {filename}: {matches}")
                        
        except Exception as e:
            print(f"❌ Error checking {filename}: {e}")
    
    print()
    print("💡 If you still have import errors, try:")
    print("1. Check that all files exist in the core/ directory")
    print("2. Make sure core/__init__.py exists (create empty file if needed)")
    print("3. Run: python -c \"import core.enhanced_ml_engine\" to test")

if __name__ == "__main__":
    main()