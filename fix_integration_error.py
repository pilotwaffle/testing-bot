#!/usr/bin/env python3
"""
FILE: fix_integration_error.py
LOCATION: E:\Trade Chat Bot\G Trading Bot\

Fix Integration Indentation Error
Quick fix for the syntax error in enhanced_trading_engine.py
"""

import os
import re
from datetime import datetime

def fix_enhanced_trading_engine():
    """Fix the indentation and syntax errors in enhanced_trading_engine.py"""
    
    print("FILE: fix_integration_error.py")
    print("LOCATION: E:\\Trade Chat Bot\\G Trading Bot\\")
    print("")
    print("🔧 Fixing Integration Syntax Error")
    print("=" * 50)
    
    engine_file = "core/enhanced_trading_engine.py"
    
    if not os.path.exists(engine_file):
        print("❌ enhanced_trading_engine.py not found")
        return False
    
    # Read the file
    with open(engine_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔍 Analyzing file for syntax errors...")
    
    # Fix common issues from the integration
    fixes_applied = []
    
    # Fix 1: Method name formatting issues
    if '*enhance*signal_with_multi_strategy' in content:
        content = content.replace('*enhance*signal_with_multi_strategy', '_enhance_signal_with_multi_strategy')
        fixes_applied.append("Fixed method name formatting")
    
    # Fix 2: Indentation issues - ensure all methods are properly indented within the class
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    
    for i, line in enumerate(lines):
        if line.strip().startswith('class EliteTradingEngine'):
            in_class = True
            fixed_lines.append(line)
        elif in_class and line.strip().startswith('class ') and not line.strip().startswith('class EliteTradingEngine'):
            # End of EliteTradingEngine class
            in_class = False
            fixed_lines.append(line)
        elif in_class and line.strip().startswith('def ') and not line.startswith('    '):
            # Method not properly indented - fix it
            fixed_lines.append('    ' + line.strip())
            fixes_applied.append(f"Fixed indentation for method on line {i+1}")
        elif in_class and line.strip().startswith('async def ') and not line.startswith('    '):
            # Async method not properly indented - fix it
            fixed_lines.append('    ' + line.strip())
            fixes_applied.append(f"Fixed indentation for async method on line {i+1}")
        else:
            fixed_lines.append(line)
    
    # Fix 3: Remove any duplicate or malformed method definitions
    content = '\n'.join(fixed_lines)
    
    # Fix 4: Clean up any double spacing or formatting issues
    content = re.sub(r'\n\n\n+', '\n\n', content)  # Remove excessive blank lines
    
    # Fix 5: Ensure proper class structure
    if 'async def _enhance_signal_with_multi_strategy' in content:
        # Make sure this method is properly formatted
        content = re.sub(
            r'async def _enhance_signal_with_multi_strategy\(self, primary_signal: TradingSignal,\s*symbol: str, market_data: Dict\)',
            'async def _enhance_signal_with_multi_strategy(self, primary_signal: TradingSignal, symbol: str, market_data: Dict)',
            content
        )
        fixes_applied.append("Fixed _enhance_signal_with_multi_strategy method signature")
    
    # Write the fixed content
    with open(engine_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if fixes_applied:
        print("✅ Applied fixes:")
        for fix in fixes_applied:
            print(f"   - {fix}")
    else:
        print("ℹ️  No obvious syntax errors found - checking Python syntax...")
    
    # Test if the file can be parsed by Python
    try:
        compile(content, engine_file, 'exec')
        print("✅ Python syntax check passed!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error still exists at line {e.lineno}: {e.msg}")
        print(f"   Text: {e.text}")
        
        # Try a more aggressive fix
        return fix_aggressive_syntax_errors(content, engine_file, e)
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def fix_aggressive_syntax_errors(content, engine_file, syntax_error):
    """Apply more aggressive fixes for syntax errors"""
    
    print("🔧 Applying aggressive syntax fixes...")
    
    lines = content.split('\n')
    error_line = syntax_error.lineno - 1  # Convert to 0-based index
    
    if error_line < len(lines):
        problematic_line = lines[error_line]
        print(f"Problematic line {syntax_error.lineno}: {problematic_line}")
        
        # Try to fix common issues
        fixed_line = problematic_line
        
        # Fix method definition issues
        if 'async def' in fixed_line and not fixed_line.strip().endswith(':'):
            # Method definition might be split across lines
            if error_line + 1 < len(lines):
                next_line = lines[error_line + 1].strip()
                if next_line and not next_line.startswith('"""') and not next_line.startswith('def'):
                    # Combine lines
                    fixed_line = fixed_line.rstrip() + ' ' + next_line
                    lines[error_line + 1] = ''  # Remove the next line
        
        # Fix indentation
        if 'async def' in fixed_line or 'def ' in fixed_line:
            if not fixed_line.startswith('    ') and not fixed_line.startswith('class'):
                fixed_line = '    ' + fixed_line.strip()
        
        lines[error_line] = fixed_line
        
        # Write the aggressively fixed content
        fixed_content = '\n'.join(lines)
        
        with open(engine_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Test again
        try:
            compile(fixed_content, engine_file, 'exec')
            print("✅ Aggressive fix successful!")
            return True
        except SyntaxError as e2:
            print(f"❌ Still has syntax error: {e2}")
            return False
    
    return False

def restore_from_backup_if_needed():
    """Restore from backup if the file is too broken"""
    
    print("\n🔄 Checking if backup restoration is needed...")
    
    # Find the most recent backup
    import glob
    backups = glob.glob("core/enhanced_trading_engine_backup_*.py")
    
    if not backups:
        print("❌ No backups found")
        return False
    
    # Get the most recent backup
    latest_backup = max(backups, key=os.path.getctime)
    
    print(f"📁 Latest backup found: {latest_backup}")
    
    choice = input("Would you like to restore from backup and try a simpler integration? (y/n): ").lower()
    
    if choice == 'y':
        # Restore from backup
        with open(latest_backup, 'r', encoding='utf-8') as f:
            backup_content = f.read()
        
        with open("core/enhanced_trading_engine.py", 'w', encoding='utf-8') as f:
            f.write(backup_content)
        
        print("✅ Restored from backup")
        
        # Add simple strategy integration
        add_simple_strategy_integration()
        
        return True
    
    return False

def add_simple_strategy_integration():
    """Add a simpler, safer strategy integration"""
    
    print("🔧 Adding simple strategy integration...")
    
    engine_file = "core/enhanced_trading_engine.py"
    
    with open(engine_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple addition at the end of the class
    simple_integration = '''
    # Strategy Integration - Simple Version
    async def load_strategies(self):
        """Load available trading strategies"""
        self.strategies = {}
        try:
            from strategies.enhanced_trading_strategy import EnhancedTradingStrategy
            self.strategies['enhanced'] = EnhancedTradingStrategy()
            self.logger.info("Enhanced Trading Strategy loaded")
        except:
            pass
        
        try:
            from strategies.ml_strategy import MLStrategy  
            self.strategies['ml'] = MLStrategy()
            self.logger.info("ML Strategy loaded")
        except:
            pass
        
        self.logger.info(f"Loaded {len(self.strategies)} strategies")
        return self.strategies
'''
    
    # Insert before the placeholder classes
    insertion_point = content.find("# Placeholder component classes")
    if insertion_point == -1:
        insertion_point = len(content) - 100  # Near the end
    
    new_content = content[:insertion_point] + simple_integration + '\n' + content[insertion_point:]
    
    with open(engine_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Simple strategy integration added")

def test_python_syntax():
    """Test if the enhanced_trading_engine.py file has valid Python syntax"""
    
    engine_file = "core/enhanced_trading_engine.py"
    
    try:
        with open(engine_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compile(content, engine_file, 'exec')
        print("✅ File syntax is valid!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    """Main function to fix integration errors"""
    
    print("🔧 FIXING INTEGRATION SYNTAX ERROR")
    print("=" * 50)
    
    # Step 1: Try to fix the current file
    print("1️⃣ Attempting to fix current file...")
    success = fix_enhanced_trading_engine()
    
    if success:
        print("\n✅ FIX SUCCESSFUL!")
        print("🧪 Now test the integration:")
        print("   python test_strategy_integration.py")
        return
    
    # Step 2: If that fails, offer backup restoration
    print("\n2️⃣ Attempting backup restoration...")
    restored = restore_from_backup_if_needed()
    
    if restored:
        if test_python_syntax():
            print("\n✅ BACKUP RESTORATION SUCCESSFUL!")
            print("🧪 Now test the simple integration:")
            print("   python test_strategy_integration.py")
        else:
            print("❌ Even backup has issues")
    else:
        print("❌ Could not fix or restore")
        print("💡 Try restarting the server anyway - it might still work!")

if __name__ == "__main__":
    main()