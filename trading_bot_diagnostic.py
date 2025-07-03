#!/usr/bin/env python3
"""
File: trading_bot_diagnostic.py
Location: E:\Trade Chat Bot\G Trading Bot\trading_bot_diagnostic.py
Description: Elite Trading Bot V3.0 - Diagnostic Test Script
Purpose: Identifies and fixes common startup issues
"""

import os
import sys
import traceback
import importlib.util
import logging
import asyncio
from pathlib import Path
import json
from datetime import datetime
import time

class TradingBotDiagnostic:
    def __init__(self, bot_path="E:\\Trade Chat Bot\\G Trading Bot"):
        self.bot_path = Path(bot_path)
        self.core_path = self.bot_path / "core"
        self.issues = []
        self.fixes_applied = []
        
        # Setup UTF-8 console output for Windows
        if sys.platform == "win32":
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    
    def log_issue(self, severity, component, description, fix_suggestion=None):
        """Log an issue found during diagnosis"""
        issue = {
            "severity": severity,
            "component": component,
            "description": description,
            "fix_suggestion": fix_suggestion,
            "timestamp": str(datetime.now())
        }
        self.issues.append(issue)
        
        severity_symbol = {"CRITICAL": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}
        print(f"{severity_symbol.get(severity, '•')} [{severity}] {component}: {description}")
        if fix_suggestion:
            print(f"   💡 Fix: {fix_suggestion}")
    
    def test_unicode_console_support(self):
        """Test if console supports Unicode characters"""
        print("\n🔍 Testing Unicode console support...")
        try:
            test_chars = ["✓", "❌", "⚠️", "🔍", "💡"]
            for char in test_chars:
                print(f"Testing: {char}")
            print("✅ Unicode console support: OK")
            return True
        except UnicodeEncodeError as e:
            self.log_issue("CRITICAL", "Console", 
                         f"Unicode encoding error: {e}",
                         "Set PYTHONIOENCODING=utf-8 environment variable")
            return False
    
    def check_imports_in_file(self, file_path):
        """Check for missing imports in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            issues_found = []
            
            # Check for common missing imports
            if 'List[' in content and 'from typing import' not in content and 'import typing' not in content:
                issues_found.append("Missing 'from typing import List' for List type hints")
            
            if 'Dict[' in content and 'Dict' not in content.split('from typing import')[1].split('\n')[0] if 'from typing import' in content else True:
                issues_found.append("Missing 'Dict' in typing imports")
            
            if 'Optional[' in content and 'Optional' not in content.split('from typing import')[1].split('\n')[0] if 'from typing import' in content else True:
                issues_found.append("Missing 'Optional' in typing imports")
            
            return issues_found
            
        except Exception as e:
            return [f"Error reading file: {e}"]
    
    def test_kraken_integration(self):
        """Test Kraken integration files for issues"""
        print("\n🔍 Testing Kraken integration...")
        
        kraken_files = [
            "kraken_futures_client.py",
            "kraken_ml_analyzer.py", 
            "kraken_integration.py",
            "kraken_dashboard_routes.py"
        ]
        
        for filename in kraken_files:
            file_path = self.core_path / filename
            if not file_path.exists():
                self.log_issue("CRITICAL", f"Kraken/{filename}", 
                             "File not found",
                             f"Ensure {filename} exists in {self.core_path}")
                continue
            
            # Check for import issues
            import_issues = self.check_imports_in_file(file_path)
            for issue in import_issues:
                self.log_issue("WARNING", f"Kraken/{filename}", issue,
                             "Add missing imports to file header")
    
    def test_market_data_processor(self):
        """Test MarketDataProcessor for missing methods"""
        print("\n🔍 Testing MarketDataProcessor...")
        
        try:
            # Try to import and inspect the MarketDataProcessor
            sys.path.insert(0, str(self.core_path))
            
            # Try to load the module
            mdp_file = self.core_path / "market_data_processor.py"
            if not mdp_file.exists():
                self.log_issue("CRITICAL", "MarketDataProcessor", 
                             "market_data_processor.py not found",
                             "Create market_data_processor.py file")
                return
            
            spec = importlib.util.spec_from_file_location("market_data_processor", mdp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if MarketDataProcessor class exists
            if hasattr(module, 'MarketDataProcessor'):
                mdp_class = getattr(module, 'MarketDataProcessor')
                
                # Check for required methods
                required_methods = ['get_latest_data', 'process_data', 'get_historical_data']
                missing_methods = []
                
                for method in required_methods:
                    if not hasattr(mdp_class, method):
                        missing_methods.append(method)
                
                if missing_methods:
                    self.log_issue("CRITICAL", "MarketDataProcessor",
                                 f"Missing methods: {', '.join(missing_methods)}",
                                 "Add missing methods to MarketDataProcessor class")
                else:
                    print("✅ MarketDataProcessor methods: OK")
            else:
                self.log_issue("CRITICAL", "MarketDataProcessor",
                             "MarketDataProcessor class not found in module",
                             "Define MarketDataProcessor class")
                
        except Exception as e:
            self.log_issue("CRITICAL", "MarketDataProcessor",
                         f"Import error: {e}",
                         "Fix import/syntax errors in market_data_processor.py")
    
    def test_logging_configuration(self):
        """Test logging configuration for Unicode issues"""
        print("\n🔍 Testing logging configuration...")
        
        try:
            # Test basic logging
            logger = logging.getLogger("test_logger")
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
            # Test with Unicode characters
            logger.info("Testing Unicode: ✓ ❌ ⚠️")
            print("✅ Logging Unicode support: OK")
            
        except Exception as e:
            self.log_issue("WARNING", "Logging",
                         f"Unicode logging error: {e}",
                         "Configure logging with UTF-8 encoding")
    
    def test_async_health_checks(self):
        """Test for infinite async loops in health checks"""
        print("\n🔍 Testing async health check patterns...")
        
        # Check main.py for health check loops
        main_file = self.bot_path / "main.py"
        if main_file.exists():
            try:
                with open(main_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for potential infinite loops
                if 'while True:' in content and 'health' in content.lower():
                    self.log_issue("WARNING", "HealthCheck",
                                 "Potential infinite health check loop detected",
                                 "Add proper sleep/interval controls in health check loops")
                
                if 'asyncio.sleep(' not in content and 'await asyncio.sleep(' not in content:
                    if 'async def' in content and 'while' in content:
                        self.log_issue("WARNING", "AsyncLoop",
                                     "Async loops without sleep may cause high CPU usage",
                                     "Add await asyncio.sleep() in async loops")
                        
            except Exception as e:
                self.log_issue("WARNING", "HealthCheck",
                             f"Error analyzing main.py: {e}")
    
    def generate_fixes(self):
        """Generate fix files for common issues"""
        print("\n🔧 Generating fixes...")
        
        # Fix 1: Create missing get_latest_data method
        mdp_fix = '''"""
File: market_data_processor_fix.py
Location: E:\\Trade Chat Bot\\G Trading Bot\\diagnostic_fixes\\market_data_processor_fix.py
Description: Fix for missing get_latest_data method in MarketDataProcessor
"""

# Add this method to your MarketDataProcessor class:

def get_latest_data(self, symbol):
    """Get latest market data for a symbol"""
    try:
        # Placeholder implementation - replace with actual market data fetching
        return {
            "symbol": symbol,
            "price": 0.0,
            "timestamp": time.time(),
            "volume": 0.0,
            "status": "placeholder"
        }
    except Exception as e:
        self.logger.error(f"Error getting latest data for {symbol}: {e}")
        return None
'''
        
        # Fix 2: Unicode logging configuration
        logging_fix = '''"""
File: logging_unicode_fix.py
Location: E:\\Trade Chat Bot\\G Trading Bot\\diagnostic_fixes\\logging_unicode_fix.py
Description: Unicode logging configuration fix for Windows console
"""

# Add this to your main.py or logging configuration:

import sys
import codecs
import logging

# Fix Windows console Unicode support
if sys.platform == "win32":
    # Reconfigure stdout/stderr for UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    # Set environment variable
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging with UTF-8 support
def setup_utf8_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('trading_bot.log', encoding='utf-8')
        ]
    )
'''
        
        # Fix 3: Import fixes for Kraken files
        import_fix = '''"""
File: import_fixes.py
Location: E:\\Trade Chat Bot\\G Trading Bot\\diagnostic_fixes\\import_fixes.py
Description: Missing import statements for Kraken integration files
"""

# Add these imports to the top of your Kraken files:

from typing import List, Dict, Optional, Union, Any
import asyncio
import logging
import json
import time
from datetime import datetime
'''
        
        # Save fixes to files
        fixes_dir = self.bot_path / "diagnostic_fixes"
        fixes_dir.mkdir(exist_ok=True)
        
        with open(fixes_dir / "market_data_processor_fix.py", 'w', encoding='utf-8') as f:
            f.write(mdp_fix)
        
        with open(fixes_dir / "logging_unicode_fix.py", 'w', encoding='utf-8') as f:
            f.write(logging_fix)
        
        with open(fixes_dir / "import_fixes.py", 'w', encoding='utf-8') as f:
            f.write(import_fix)
        
        print(f"✅ Fix files generated in: {fixes_dir}")
        
        # Generate startup script with fixes
        startup_script = f'''@echo off
REM File: start_bot_fixed.bat
REM Location: {self.bot_path}\\start_bot_fixed.bat
REM Description: Elite Trading Bot V3.0 startup script with Unicode fixes
REM Purpose: Start bot with proper UTF-8 encoding support

echo Starting Elite Trading Bot V3.0 with fixes...

:: Set UTF-8 encoding
set PYTHONIOENCODING=utf-8
chcp 65001

:: Navigate to bot directory
cd /d "{self.bot_path}"

:: Start with UTF-8 support
python -X utf8 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
'''
        
        with open(self.bot_path / "start_bot_fixed.bat", 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        print("✅ Fixed startup script created: start_bot_fixed.bat")
    
    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        print("🚀 Elite Trading Bot V3.0 - Diagnostic Test Suite")
        print("=" * 60)
        
        # Test basic functionality
        self.test_unicode_console_support()
        self.test_kraken_integration()
        self.test_market_data_processor()
        self.test_logging_configuration()
        self.test_async_health_checks()
        
        # Generate summary
        print("\n📊 DIAGNOSTIC SUMMARY")
        print("=" * 40)
        
        critical_issues = [i for i in self.issues if i["severity"] == "CRITICAL"]
        warning_issues = [i for i in self.issues if i["severity"] == "WARNING"]
        
        print(f"❌ Critical Issues: {len(critical_issues)}")
        print(f"⚠️  Warning Issues: {len(warning_issues)}")
        print(f"ℹ️  Total Issues: {len(self.issues)}")
        
        if critical_issues:
            print("\n🔥 CRITICAL ISSUES TO FIX:")
            for issue in critical_issues:
                print(f"   • {issue['component']}: {issue['description']}")
        
        # Generate fixes
        if self.issues:
            self.generate_fixes()
        
        # Provide next steps
        print("\n🎯 NEXT STEPS:")
        print("1. Apply the generated fixes from diagnostic_fixes/ folder")
        print("2. Use start_bot_fixed.bat to start with UTF-8 support")
        print("3. Check the generated fix files for specific solutions")
        print("4. Run this diagnostic again after applying fixes")
        
        return len(critical_issues) == 0

def main():
    # Check if running from correct directory
    if len(sys.argv) > 1:
        bot_path = sys.argv[1]
    else:
        bot_path = "E:\\Trade Chat Bot\\G Trading Bot"
    
    diagnostic = TradingBotDiagnostic(bot_path)
    
    try:
        success = diagnostic.run_full_diagnostic()
        
        if success:
            print("\n✅ Diagnostic completed successfully - Bot should start properly!")
        else:
            print("\n❌ Critical issues found - Fix these before starting the bot")
            
    except Exception as e:
        print(f"\n💥 Diagnostic failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()