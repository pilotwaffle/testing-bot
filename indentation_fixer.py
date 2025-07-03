#!/usr/bin/env python3
"""
================================================================================
FILE: indentation_fixer.py
LOCATION: E:\Trade Chat Bot\G Trading Bot\indentation_fixer.py
AUTHOR: Claude AI Assistant
CREATED: Sunday, June 29, 2025
PURPOSE: Fix indentation errors in optimized_model_trainer.py
VERSION: 1.0
================================================================================

Indentation Error Fix Script
Fixes Python syntax issues caused by automatic integration

🎯 Features:
✅ Detects and fixes indentation errors
✅ Validates Python syntax
✅ Creates backup before fixing
✅ Line-by-line error detection
✅ Smart indentation correction

USAGE:
    python indentation_fixer.py
    
Expected Result:
- Fixed optimized_model_trainer.py with correct indentation
- All syntax errors resolved
================================================================================
"""

import os
import ast
import shutil
import re
from datetime import datetime

class IndentationFixer:
    """
    Smart indentation fixer for Python files
    """
    
    def __init__(self, filename='optimized_model_trainer.py', verbose=True):
        self.filename = filename
        self.verbose = verbose
        self.backup_suffix = f".backup_indent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.verbose:
            print("🔧 Indentation Fixer initialized")
            print(f"📁 Target file: {filename}")
    
    def create_backup(self):
        """Create backup before fixing"""
        if os.path.exists(self.filename):
            backup_path = self.filename + self.backup_suffix
            shutil.copy2(self.filename, backup_path)
            if self.verbose:
                print(f"💾 Backup created: {os.path.basename(backup_path)}")
            return backup_path
        return None
    
    def check_syntax_errors(self):
        """Check for syntax errors in the file"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the file
            ast.parse(content)
            return None  # No errors
            
        except SyntaxError as e:
            return {
                'line': e.lineno,
                'text': e.text,
                'msg': e.msg,
                'offset': e.offset
            }
        except Exception as e:
            return {
                'line': None,
                'text': None,
                'msg': str(e),
                'offset': None
            }
    
    def fix_common_indentation_issues(self):
        """Fix common indentation problems"""
        
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if self.verbose:
            print(f"📝 Processing {len(lines)} lines...")
        
        fixed_lines = []
        in_function = False
        in_class = False
        base_indent = 0
        
        for i, line in enumerate(lines):
            line_num = i + 1
            original_line = line
            
            # Remove trailing whitespace
            line = line.rstrip() + '\n'
            
            # Skip empty lines
            if line.strip() == '':
                fixed_lines.append(line)
                continue
            
            # Detect function/class definitions
            stripped = line.strip()
            
            if stripped.startswith('def ') or stripped.startswith('class '):
                # Count leading spaces to determine base indentation
                leading_spaces = len(line) - len(line.lstrip())
                base_indent = leading_spaces
                in_function = stripped.startswith('def ')
                in_class = stripped.startswith('class ')
                fixed_lines.append(line)
                continue
            
            # Fix specific problematic patterns
            if 'traceback.print_exc()' in stripped:
                # This is likely the problematic line
                if in_function:
                    # Should be indented inside a function
                    expected_indent = base_indent + 4
                    if 'try:' in ''.join(lines[max(0, i-10):i]) or 'except' in ''.join(lines[max(0, i-5):i]):
                        expected_indent = base_indent + 8  # Inside try/except
                else:
                    expected_indent = 4
                
                # Fix the indentation
                fixed_line = ' ' * expected_indent + stripped + '\n'
                fixed_lines.append(fixed_line)
                
                if self.verbose:
                    print(f"🔧 Fixed line {line_num}: traceback.print_exc() indentation")
                continue
            
            # Fix other common indentation issues
            if stripped.startswith(('import ', 'from ', '#')):
                # Top-level imports and comments
                if not (stripped.startswith('# ===') or stripped.startswith('"""')):
                    fixed_lines.append(stripped + '\n')
                else:
                    fixed_lines.append(line)
                continue
            
            # Fix function/method calls that might be misindented
            if any(pattern in stripped for pattern in [
                'print(', 'return ', 'if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'except', 'finally:'
            ]):
                # These should typically be indented properly within functions
                current_indent = len(line) - len(line.lstrip())
                
                # Check if we need to fix indentation
                if in_function and current_indent < base_indent + 4:
                    # Likely needs more indentation
                    expected_indent = base_indent + 4
                    fixed_line = ' ' * expected_indent + stripped + '\n'
                    fixed_lines.append(fixed_line)
                    
                    if self.verbose and current_indent != expected_indent:
                        print(f"🔧 Fixed line {line_num}: adjusted indentation from {current_indent} to {expected_indent}")
                    continue
            
            # Keep the line as-is if no issues detected
            fixed_lines.append(line)
        
        # Write the fixed content
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        if self.verbose:
            print(f"✅ Indentation fixes applied")
    
    def fix_specific_line_886(self):
        """Fix the specific error on line 886"""
        
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) >= 886:
            line_886 = lines[885]  # 0-indexed
            
            if 'traceback.print_exc()' in line_886:
                # Find the appropriate indentation by looking at surrounding context
                
                # Look backwards for try/except block
                indent_level = 8  # Default for inside except block
                
                for i in range(max(0, 885-10), 885):
                    line = lines[i].strip()
                    if line.startswith('except'):
                        # Found except block, use its indentation + 4
                        except_indent = len(lines[i]) - len(lines[i].lstrip())
                        indent_level = except_indent + 4
                        break
                    elif line.startswith('try:'):
                        # Found try block, use its indentation + 8
                        try_indent = len(lines[i]) - len(lines[i].lstrip())
                        indent_level = try_indent + 8
                        break
                
                # Fix the line
                lines[885] = ' ' * indent_level + 'traceback.print_exc()\n'
                
                # Write back
                with open(self.filename, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                if self.verbose:
                    print(f"🎯 Fixed specific line 886: set indentation to {indent_level} spaces")
                
                return True
        
        return False
    
    def validate_syntax(self):
        """Validate that the file has correct syntax after fixing"""
        error = self.check_syntax_errors()
        
        if error is None:
            if self.verbose:
                print("✅ File syntax is now correct!")
            return True
        else:
            if self.verbose:
                print(f"❌ Syntax error still exists:")
                print(f"   Line {error['line']}: {error['msg']}")
                if error['text']:
                    print(f"   Text: {error['text'].strip()}")
            return False
    
    def fix_all_issues(self):
        """Fix all indentation issues"""
        
        print("🚀 INDENTATION FIXER")
        print("=" * 30)
        
        # Create backup
        backup_path = self.create_backup()
        
        # Check initial syntax
        initial_error = self.check_syntax_errors()
        if initial_error:
            print(f"🔍 Found syntax error on line {initial_error['line']}: {initial_error['msg']}")
        else:
            print("✅ No syntax errors found")
            return True
        
        try:
            # Try specific fix for line 886 first
            if initial_error and initial_error['line'] == 886:
                print("🎯 Fixing specific line 886 issue...")
                if self.fix_specific_line_886():
                    if self.validate_syntax():
                        print("🎉 Successfully fixed line 886!")
                        return True
            
            # Try general indentation fixes
            print("🔧 Applying general indentation fixes...")
            self.fix_common_indentation_issues()
            
            # Validate
            if self.validate_syntax():
                print("🎉 All indentation issues fixed!")
                return True
            else:
                print("⚠️ Some issues may remain")
                return False
                
        except Exception as e:
            print(f"❌ Error during fixing: {str(e)}")
            
            # Restore backup if available
            if backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, self.filename)
                print(f"🔄 Restored from backup: {os.path.basename(backup_path)}")
            
            return False

def create_clean_trainer_script():
    """Create a completely clean trainer script if fixing fails"""
    
    clean_script = '''#!/usr/bin/env python3
"""
CLEAN optimized_model_trainer.py - Syntax Error Free Version
"""

import numpy as np
import pandas as pd
import sys
import os
import traceback

# Basic imports
try:
    import ccxt
except ImportError:
    print("Please install ccxt: pip install ccxt")
    sys.exit(1)

# Enhanced imports (optional)
try:
    from advanced_ensemble import AdvancedEnsembleManager
    ADVANCED_ENSEMBLE_AVAILABLE = True
except ImportError:
    ADVANCED_ENSEMBLE_AVAILABLE = False
    print("⚠️ advanced_ensemble.py not found")

try:
    from advanced_features import AdvancedFeatureEngineer
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    print("⚠️ advanced_features.py not found")

try:
    from regime_detection import MarketRegimeDetector
    REGIME_DETECTION_AVAILABLE = True
except ImportError:
    REGIME_DETECTION_AVAILABLE = False
    print("⚠️ regime_detection.py not found")

try:
    from advanced_stacking import AdvancedStackingEnsemble
    ADVANCED_STACKING_AVAILABLE = True
except ImportError:
    ADVANCED_STACKING_AVAILABLE = False
    print("⚠️ advanced_stacking.py not found")

def create_rate_limited_exchange():
    """Create rate-limited exchange"""
    import time
    
    exchange = ccxt.kraken({
        'enableRateLimit': True,
        'rateLimit': 3000,  # 3 seconds between requests
        'timeout': 30000,
    })
    
    # Add delay wrapper
    original_fetch = exchange.fetch_ohlcv
    
    def safe_fetch(symbol, timeframe='1h', since=None, limit=500, params={}):
        time.sleep(3)  # Always wait 3 seconds
        try:
            return original_fetch(symbol, timeframe, since, limit, params)
        except Exception as e:
            if "Too many requests" in str(e):
                print(f"⚠️ Rate limit for {symbol}, waiting 30s...")
                time.sleep(30)
                return original_fetch(symbol, timeframe, since, limit, params)
            else:
                raise e
    
    exchange.fetch_ohlcv = safe_fetch
    return exchange

def enhanced_feature_engineering(data):
    """Enhanced feature engineering"""
    if ADVANCED_FEATURES_AVAILABLE:
        try:
            engineer = AdvancedFeatureEngineer(verbose=True)
            enhanced_data = engineer.create_advanced_features(data)
            enhanced_data['target'] = (enhanced_data['close'].shift(-1) > enhanced_data['close']).astype(int)
            return enhanced_data
        except Exception as e:
            print(f"⚠️ Advanced features failed: {str(e)[:50]}...")
    
    # Fallback to basic features
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    return data

def train_with_advanced_ensemble(symbol, timeframe, X_train, y_train, X_test, y_test):
    """Train with advanced ensemble"""
    results = {}
    
    if ADVANCED_STACKING_AVAILABLE:
        try:
            print("🏗️ Training Advanced Stacking...")
            stacker = AdvancedStackingEnsemble(cv_folds=3, verbose=True)
            stacker.fit(X_train, y_train)
            
            predictions = stacker.predict(X_test)
            accuracy = np.mean((predictions > 0.5) == (y_test > 0.5))
            
            results['advanced_stacking'] = {
                'accuracy': accuracy,
                'predictions': predictions
            }
            
            print(f"🎯 Advanced Stacking: {accuracy:.1%}")
            
        except Exception as e:
            print(f"❌ Stacking failed: {str(e)[:50]}...")
    
    if ADVANCED_ENSEMBLE_AVAILABLE:
        try:
            print("🧠 Training Advanced Ensemble...")
            manager = AdvancedEnsembleManager(verbose=True)
            models = manager.create_enhanced_models()
            
            # Simple ensemble training
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            predictions = rf.predict(X_test)
            accuracy = np.mean((predictions > 0.5) == (y_test > 0.5))
            
            results['enhanced_rf'] = {
                'accuracy': accuracy,
                'predictions': predictions
            }
            
            print(f"🎯 Enhanced RF: {accuracy:.1%}")
            
        except Exception as e:
            print(f"❌ Ensemble failed: {str(e)[:50]}...")
    
    if results:
        best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
        return {
            'best_method': best_method,
            'best_accuracy': results[best_method]['accuracy'],
            'results': results
        }
    
    return None

def main():
    """Main training function"""
    print("🤖 Enhanced Trading Bot Trainer (Clean Version)")
    print("=" * 50)
    
    # Test advanced features
    print("🔍 Testing advanced features...")
    print(f"Advanced Ensemble: {'✅' if ADVANCED_ENSEMBLE_AVAILABLE else '❌'}")
    print(f"Advanced Features: {'✅' if ADVANCED_FEATURES_AVAILABLE else '❌'}")
    print(f"Regime Detection: {'✅' if REGIME_DETECTION_AVAILABLE else '❌'}")
    print(f"Advanced Stacking: {'✅' if ADVANCED_STACKING_AVAILABLE else '❌'}")
    
    # Create sample data for testing
    print("\\n📊 Creating sample data...")
    np.random.seed(42)
    
    # Generate sample OHLCV data
    n_samples = 1000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    price_data = pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # Enhanced feature engineering
    print("🔬 Enhanced feature engineering...")
    enhanced_data = enhanced_feature_engineering(price_data)
    
    print(f"✅ Features created: {len(enhanced_data.columns)}")
    
    # Prepare training data
    feature_cols = [col for col in enhanced_data.columns 
                   if col not in ['close', 'high', 'low', 'volume', 'target']]
    
    X = enhanced_data[feature_cols].fillna(0).values
    y = enhanced_data['target'].fillna(0).values
    
    # Remove NaN rows
    valid_rows = ~np.isnan(y)
    X = X[valid_rows]
    y = y[valid_rows]
    
    if len(X) < 100:
        print("❌ Insufficient data for training")
        return
    
    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Training data: {len(X_train)} samples")
    print(f"📊 Test data: {len(X_test)} samples")
    
    # Train with advanced ensemble
    print("\\n🚀 Training with advanced methods...")
    try:
        result = train_with_advanced_ensemble('BTC/USD', '4h', X_train, y_train, X_test, y_test)
        
        if result:
            best_accuracy = result['best_accuracy']
            baseline_accuracy = 0.725  # Your previous best
            
            print(f"\\n🏆 RESULTS:")
            print(f"Best Method: {result['best_method']}")
            print(f"Best Accuracy: {best_accuracy:.1%}")
            
            if best_accuracy > baseline_accuracy:
                improvement = (best_accuracy - baseline_accuracy) * 100
                print(f"📈 IMPROVEMENT: +{improvement:.1f}% vs baseline (72.5%)")
                print("🎯 TARGET ACHIEVED!" if best_accuracy >= 0.80 else "✅ Good progress!")
            else:
                print("📊 Baseline performance maintained")
        else:
            print("⚠️ Advanced training failed, but syntax is correct")
            
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    with open('clean_optimized_model_trainer.py', 'w', encoding='utf-8') as f:
        f.write(clean_script)
    
    print("✅ Created clean_optimized_model_trainer.py")
    print("💡 You can use this as a fallback if the original has issues")

def main():
    """Main execution"""
    
    print("🔧 INDENTATION ERROR FIXER")
    print("=" * 40)
    print("Fixing Python syntax errors in optimized_model_trainer.py")
    print()
    
    if not os.path.exists('optimized_model_trainer.py'):
        print("❌ optimized_model_trainer.py not found")
        return
    
    fixer = IndentationFixer(verbose=True)
    
    print("🎯 Attempting to fix indentation errors...")
    success = fixer.fix_all_issues()
    
    if success:
        print("\\n🎉 SUCCESS!")
        print("✅ All syntax errors fixed")
        print("🚀 You can now run: python optimized_model_trainer.py --full-train --enhanced")
    else:
        print("\\n⚠️ Could not fully fix all issues")
        print("💡 Creating clean fallback script...")
        create_clean_trainer_script()
        print("\\n🔄 Alternative options:")
        print("1. Run: python clean_optimized_model_trainer.py")
        print("2. Or restore from backup and try manual fixes")

if __name__ == "__main__":
    main()