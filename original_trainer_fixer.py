#!/usr/bin/env python3
"""
================================================================================
FILE: original_trainer_fixer.py
LOCATION: E:\Trade Chat Bot\G Trading Bot\original_trainer_fixer.py
AUTHOR: Claude AI Assistant
CREATED: Sunday, June 29, 2025
PURPOSE: Fix syntax in original optimized_model_trainer.py with real data
VERSION: 1.0
================================================================================

Original Trainer Syntax Fixer
Fixes your original trainer while preserving real market data pipeline

🎯 Goal: Fix syntax while keeping your proven 72.5% baseline + add advanced features
✅ Preserves your real CCXT data fetching
✅ Keeps your proven training methods  
✅ Adds advanced features carefully
✅ Maintains your excellent results

USAGE:
    python original_trainer_fixer.py
================================================================================
"""

import os
import shutil
import re
from datetime import datetime

def create_syntax_fixed_trainer():
    """Create a syntax-fixed version of the original trainer with real data"""
    
    print("🔧 FIXING ORIGINAL TRAINER SYNTAX")
    print("==================================")
    
    # Create backup first
    backup_path = f"optimized_model_trainer.py.backup_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists('optimized_model_trainer.py'):
        shutil.copy2('optimized_model_trainer.py', backup_path)
        print(f"💾 Backup created: {backup_path}")
    
    # Read the file and fix syntax issues
    try:
        with open('optimized_model_trainer.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("🔍 Analyzing syntax issues...")
        
        # Split into lines for easier processing
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Fix the specific indentation issues around imports
            if 'try:' in line and i > 0:
                # Check if this is an import block
                if any(keyword in ''.join(lines[max(0, i-5):i+10]) for keyword in ['import', 'from']):
                    # This is likely a try/except import block
                    current_indent = len(line) - len(line.lstrip())
                    
                    # Process the try block
                    fixed_lines.append(line)
                    i += 1
                    
                    # Process lines until we find except or end of block
                    while i < len(lines):
                        current_line = lines[i]
                        
                        if current_line.strip() == '':
                            fixed_lines.append(current_line)
                            i += 1
                            continue
                        
                        if current_line.strip().startswith('except'):
                            # This is the except line, should align with try
                            fixed_except = ' ' * current_indent + current_line.strip()
                            fixed_lines.append(fixed_except)
                            i += 1
                            break
                        else:
                            # This should be indented inside the try block
                            line_indent = len(current_line) - len(current_line.lstrip())
                            if line_indent <= current_indent and current_line.strip():
                                # Needs more indentation
                                fixed_line = ' ' * (current_indent + 4) + current_line.strip()
                                fixed_lines.append(fixed_line)
                            else:
                                fixed_lines.append(current_line)
                            i += 1
                    continue
            
            # Fix traceback.print_exc() indentation
            if 'traceback.print_exc()' in line:
                # Find appropriate indentation by looking backwards
                indent_level = 8  # Default for except block
                
                for j in range(max(0, i-10), i):
                    prev_line = lines[j].strip()
                    if prev_line.startswith('except'):
                        except_indent = len(lines[j]) - len(lines[j].lstrip())
                        indent_level = except_indent + 4
                        break
                    elif prev_line.startswith('try:'):
                        try_indent = len(lines[j]) - len(lines[j].lstrip())
                        indent_level = try_indent + 8
                        break
                
                fixed_line = ' ' * indent_level + 'traceback.print_exc()'
                fixed_lines.append(fixed_line)
                print(f"🔧 Fixed traceback.print_exc() at line {i+1}")
            
            # Fix any lines that were incorrectly unindented to 0
            elif line.strip() and not line.startswith(' ') and not line.startswith('#') and not line.startswith('"""'):
                # Check if this should be indented (inside a function/class)
                in_function = False
                in_class = False
                base_indent = 0
                
                for j in range(max(0, i-20), i):
                    prev_line = lines[j].strip()
                    if prev_line.startswith('def '):
                        in_function = True
                        base_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
                    elif prev_line.startswith('class '):
                        in_class = True
                        base_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
                
                if in_function and not line.strip().startswith(('def ', 'class ', 'if __name__')):
                    # Should be indented inside function
                    fixed_line = ' ' * (base_indent + 4) + line.strip()
                    fixed_lines.append(fixed_line)
                    print(f"🔧 Fixed indentation at line {i+1}")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
            
            i += 1
        
        # Join lines back together
        fixed_content = '\n'.join(fixed_lines)
        
        # Write the fixed content
        with open('optimized_model_trainer.py', 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("✅ Syntax fixes applied")
        
        # Test syntax
        try:
            import ast
            ast.parse(fixed_content)
            print("✅ File syntax is now correct!")
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error still exists at line {e.lineno}: {e.msg}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing file: {str(e)}")
        return False

def create_minimal_working_trainer():
    """Create a minimal but working version using real data"""
    
    minimal_trainer = '''#!/usr/bin/env python3
"""
MINIMAL WORKING TRAINER - Real Market Data + Advanced Features
Based on your proven optimized_model_trainer.py architecture
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import traceback
import ccxt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# Advanced imports (your files)
try:
    from advanced_ensemble import AdvancedEnsembleManager
    ADVANCED_ENSEMBLE_AVAILABLE = True
    print("✅ Advanced Ensemble loaded")
except ImportError:
    ADVANCED_ENSEMBLE_AVAILABLE = False
    print("⚠️ Advanced Ensemble not available")

try:
    from advanced_features import AdvancedFeatureEngineer
    ADVANCED_FEATURES_AVAILABLE = True
    print("✅ Advanced Features loaded")
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    print("⚠️ Advanced Features not available")

try:
    from advanced_stacking import AdvancedStackingEnsemble
    ADVANCED_STACKING_AVAILABLE = True
    print("✅ Advanced Stacking loaded")
except ImportError:
    ADVANCED_STACKING_AVAILABLE = False
    print("⚠️ Advanced Stacking not available")

def create_safe_exchange():
    """Create rate-limited exchange like your original"""
    exchange = ccxt.kraken({
        'enableRateLimit': True,
        'rateLimit': 3000,  # 3 seconds between requests
        'timeout': 30000,
    })
    
    # Add rate limiting wrapper
    original_fetch = exchange.fetch_ohlcv
    
    def safe_fetch_ohlcv(symbol, timeframe='1h', since=None, limit=500, params={}):
        time.sleep(3)  # Always wait 3 seconds
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"📡 Fetching {symbol} {timeframe} (attempt {attempt + 1})")
                return original_fetch(symbol, timeframe, since, limit, params)
            except Exception as e:
                if "Too many requests" in str(e):
                    wait_time = 30 * (attempt + 1)
                    print(f"⚠️ Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(10)
        return None
    
    exchange.fetch_ohlcv = safe_fetch_ohlcv
    return exchange

def fetch_real_market_data(exchange, symbol, timeframe='4h', limit=1000):
    """Fetch real market data like your original trainer"""
    try:
        print(f"📊 Fetching real market data: {symbol} {timeframe}")
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            print(f"❌ No data received for {symbol}")
            return None
        
        # Convert to DataFrame (like your original)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Fetched {len(df)} candles for {symbol}")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {str(e)}")
        return None

def enhanced_feature_engineering(df):
    """Enhanced feature engineering using your real data"""
    if ADVANCED_FEATURES_AVAILABLE:
        try:
            print("🔬 Creating advanced features from real market data...")
            engineer = AdvancedFeatureEngineer(verbose=False)
            enhanced_df = engineer.create_advanced_features(df)
            
            # Create target (direction prediction like your original)
            enhanced_df['target'] = (enhanced_df['close'].shift(-1) > enhanced_df['close']).astype(int)
            
            # Remove rows with NaN target
            enhanced_df = enhanced_df.dropna(subset=['target'])
            
            print(f"✅ Created {len(enhanced_df.columns)} features from real data")
            return enhanced_df
            
        except Exception as e:
            print(f"⚠️ Advanced features failed: {str(e)[:50]}...")
            print("📊 Falling back to basic features...")
    
    # Basic features (your original approach)
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df.dropna()

def calculate_rsi(prices, period=14):
    """Calculate RSI like your original"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_advanced_models(X_train, y_train, X_test, y_test, symbol, timeframe):
    """Train with advanced methods (preserving your data pipeline)"""
    results = {}
    
    print(f"🚀 Training advanced models for {symbol} {timeframe}")
    print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Try Advanced Stacking first (best performance)
    if ADVANCED_STACKING_AVAILABLE and len(X_train) >= 200:
        try:
            print("🏗️ Training Advanced Stacking Ensemble...")
            stacker = AdvancedStackingEnsemble(cv_folds=3, verbose=False)
            stacker.fit(X_train, y_train)
            
            predictions = stacker.predict(X_test)
            accuracy = accuracy_score((y_test > 0.5).astype(int), (predictions > 0.5).astype(int))
            
            results['advanced_stacking'] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'model': stacker
            }
            
            print(f"🎯 Advanced Stacking: {accuracy:.1%}")
            
        except Exception as e:
            print(f"❌ Advanced stacking failed: {str(e)[:50]}...")
    
    # Try Enhanced Random Forest (your proven approach + enhancements)
    try:
        print("🌲 Training Enhanced Random Forest...")
        
        if ADVANCED_ENSEMBLE_AVAILABLE:
            manager = AdvancedEnsembleManager(verbose=False)
            models = manager.create_enhanced_models()
            rf_model = models.get('enhanced_random_forest', RandomForestRegressor(n_estimators=200, random_state=42))
        else:
            rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        
        rf_model.fit(X_train, y_train)
        predictions = rf_model.predict(X_test)
        accuracy = accuracy_score((y_test > 0.5).astype(int), (predictions > 0.5).astype(int))
        
        results['enhanced_rf'] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'model': rf_model
        }
        
        print(f"🎯 Enhanced RF: {accuracy:.1%}")
        
    except Exception as e:
        print(f"❌ Enhanced RF failed: {str(e)[:50]}...")
    
    # Basic Gradient Boosting (fallback)
    try:
        print("📈 Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        gb_model.fit(X_train, y_train)
        predictions = gb_model.predict(X_test)
        accuracy = accuracy_score((y_test > 0.5).astype(int), (predictions > 0.5).astype(int))
        
        results['gradient_boosting'] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'model': gb_model
        }
        
        print(f"🎯 Gradient Boosting: {accuracy:.1%}")
        
    except Exception as e:
        print(f"❌ Gradient Boosting failed: {str(e)[:50]}...")
    
    return results

def main():
    """Main training function using real market data"""
    print("🤖 ENHANCED TRAINER - Real Market Data + Advanced ML")
    print("=" * 60)
    
    # Test symbols (like your original)
    symbols = ['BTC/USD', 'ETH/USD']
    timeframes = ['4h', '1d']  # Your best performing timeframes
    
    # Create exchange
    exchange = create_safe_exchange()
    
    all_results = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\\n🔄 Processing {symbol} {timeframe}...")
            
            try:
                # Fetch real market data
                df = fetch_real_market_data(exchange, symbol, timeframe, limit=1000)
                
                if df is None or len(df) < 200:
                    print(f"❌ Insufficient data for {symbol} {timeframe}")
                    continue
                
                # Enhanced feature engineering
                enhanced_df = enhanced_feature_engineering(df)
                
                if len(enhanced_df) < 100:
                    print(f"❌ Insufficient data after feature engineering")
                    continue
                
                # Prepare training data
                feature_cols = [col for col in enhanced_df.columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
                
                X = enhanced_df[feature_cols].fillna(0).values
                y = enhanced_df['target'].values
                
                # Train/test split (keeping recent data for testing like your original)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Train advanced models
                results = train_advanced_models(X_train, y_train, X_test, y_test, symbol, timeframe)
                
                if results:
                    # Find best result
                    best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
                    best_accuracy = results[best_method]['accuracy']
                    
                    # Performance evaluation (like your original)
                    baseline_accuracy = 0.725  # Your previous best
                    target_achieved = best_accuracy >= 0.65
                    
                    status = "🎯 TARGET!" if target_achieved else "✅"
                    print(f"{status} {symbol} {timeframe}: Best {best_accuracy:.1%} ({best_method})")
                    
                    if best_accuracy > baseline_accuracy:
                        improvement = (best_accuracy - baseline_accuracy) * 100
                        print(f"📈 IMPROVEMENT: +{improvement:.1f}% vs baseline (72.5%)")
                    
                    all_results.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'accuracy': best_accuracy,
                        'method': best_method,
                        'target_achieved': target_achieved
                    })
                
            except Exception as e:
                print(f"❌ Error processing {symbol} {timeframe}: {str(e)[:50]}...")
                traceback.print_exc()
    
    # Final summary (like your original)
    if all_results:
        print(f"\\n{'='*60}")
        print("🎉 ENHANCED TRAINING COMPLETE - Real Market Data Results")
        print(f"{'='*60}")
        
        accuracies = [r['accuracy'] for r in all_results]
        targets_achieved = sum(1 for r in all_results if r['target_achieved'])
        
        avg_accuracy = np.mean(accuracies)
        max_accuracy = np.max(accuracies)
        target_rate = targets_achieved / len(all_results)
        
        print(f"📊 Average Accuracy: {avg_accuracy:.1%}")
        print(f"🏆 Peak Accuracy: {max_accuracy:.1%}")
        print(f"🎯 Target Achievement Rate: {target_rate:.1%} ({targets_achieved}/{len(all_results)})")
        
        # Show best results
        print(f"\\n🏆 TOP PERFORMERS:")
        sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
        for i, result in enumerate(sorted_results[:5], 1):
            status = "🎯" if result['target_achieved'] else "✅"
            print(f"  {i}. {result['symbol']} {result['timeframe']}: {result['accuracy']:.1%} ({result['method']}) {status}")
        
        # Improvement analysis
        baseline = 0.725
        improved_count = sum(1 for r in all_results if r['accuracy'] > baseline)
        if improved_count > 0:
            print(f"\\n📈 IMPROVEMENTS:")
            print(f"  {improved_count}/{len(all_results)} combinations beat baseline (72.5%)")
            print(f"  Best improvement: +{(max_accuracy - baseline) * 100:.1f}%")
        
        if max_accuracy >= 0.80:
            print(f"\\n🎉 🎯 TARGET ACHIEVED! 80%+ accuracy reached!")
        elif max_accuracy >= 0.75:
            print(f"\\n✅ Great progress! Close to 80% target.")
        
    else:
        print("❌ No successful training runs")

if __name__ == "__main__":
    main()
'''
    
    with open('minimal_working_trainer.py', 'w', encoding='utf-8') as f:
        f.write(minimal_trainer)
    
    print("✅ Created minimal_working_trainer.py")
    print("💡 This version uses your real market data pipeline")

def main():
    """Main execution"""
    print("🎯 ORIGINAL TRAINER FIXER")
    print("=========================")
    print("Goal: Fix syntax while preserving your proven 72.5% real data results")
    print()
    
    if not os.path.exists('optimized_model_trainer.py'):
        print("❌ optimized_model_trainer.py not found")
        return
    
    # Try to fix the original
    print("🔧 Attempting to fix original trainer syntax...")
    success = create_syntax_fixed_trainer()
    
    if success:
        print("\\n🎉 SUCCESS! Original trainer syntax fixed")
        print("🚀 Try: python optimized_model_trainer.py --full-train --enhanced")
    else:
        print("\\n⚠️ Original still has syntax issues")
    
    # Always create the minimal working version
    print("\\n💡 Creating minimal working trainer with real data...")
    create_minimal_working_trainer()
    
    print("\\n🎯 READY TO TEST:")
    print("Option 1: python optimized_model_trainer.py --full-train --enhanced")
    print("Option 2: python minimal_working_trainer.py")
    print("\\nOption 2 uses your proven real data pipeline + advanced features!")

if __name__ == "__main__":
    main()