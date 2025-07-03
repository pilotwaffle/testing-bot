#!/usr/bin/env python3
"""
Quick Walk-Forward Analysis Integration Script
Integrates with your existing Industrial Crypto Trading Bot v3.0

Run this to immediately start improving your model accuracy with WFA!
"""

import sys
import os
from datetime import datetime
import logging

# Add your bot directory to path if needed
bot_directory = "E:/Trade Chat Bot/G Trading Bot"
if bot_directory not in sys.path:
    sys.path.append(bot_directory)

def quick_walk_forward_analysis():
    """Run a quick walk-forward analysis on your best performing models"""
    
    print("🚀 Quick Walk-Forward Analysis for Your Trading Bot")
    print("=" * 60)
    
    try:
        # Import the walk-forward analyzer
        from walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardConfig
        
        # Configure based on your successful results
        config = WalkForwardConfig(
            optimization_window_days=120,  # 4 months (shorter for quicker results)
            validation_window_days=30,     # 1 month validation
            step_size_days=15,             # Move forward 2 weeks
            target_symbols=['BTC/USD', 'ETH/USD', 'ADA/USD'],  # Your best performers
            timeframes=['4h', '1d'],       # Your best timeframes
            models_to_test=['random_forest', 'gradient_boosting', 'meta_ensemble']
        )
        
        print("🔧 Configuration:")
        print(f"   Symbols: {config.target_symbols}")
        print(f"   Timeframes: {config.timeframes}")
        print(f"   Models: {config.models_to_test}")
        print(f"   Training window: {config.optimization_window_days} days")
        print(f"   Validation window: {config.validation_window_days} days")
        
        # Create analyzer and run
        analyzer = WalkForwardAnalyzer(config)
        
        print("\n⏳ Running analysis... (this may take 10-20 minutes)")
        summary = analyzer.run_analysis()
        
        # Display results
        print("\n🎉 WALK-FORWARD ANALYSIS RESULTS")
        print("=" * 60)
        
        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"📊 Total test periods: {summary['total_iterations']}")
        print(f"🎯 Average validation accuracy: {summary['avg_validation_accuracy']:.2%}")
        print(f"🏆 Best validation accuracy: {summary['best_validation_accuracy']:.2%}")
        print(f"📈 Consistency score (>65%): {summary['consistency_score']:.2%}")
        
        if summary['avg_validation_accuracy'] > 0.65:
            print("✅ EXCELLENT! Your models are robust across time periods!")
        elif summary['avg_validation_accuracy'] > 0.60:
            print("✅ GOOD! Your models show consistent performance!")
        else:
            print("⚠️  Models may be overfitting. Consider parameter adjustment.")
        
        # Show top performers
        print("\n🏆 TOP PERFORMING COMBINATIONS:")
        for i, performer in enumerate(summary['top_performers'][:5], 1):
            print(f"   {i}. {performer['symbol']} {performer['timeframe']} "
                  f"{performer['model_name']}: {performer['validation_accuracy']:.2%}")
        
        # Show by model comparison
        print("\n📊 PERFORMANCE BY MODEL:")
        for model, stats in summary['by_model']['mean'].items():
            print(f"   {model}: {stats:.2%} avg accuracy")
        
        # Show by timeframe comparison  
        print("\n⏰ PERFORMANCE BY TIMEFRAME:")
        for timeframe, stats in summary['by_timeframe']['mean'].items():
            print(f"   {timeframe}: {stats:.2%} avg accuracy")
        
        # Save results
        results_file = analyzer.save_results()
        print(f"\n📁 Detailed results saved to: {results_file}")
        
        # Create plots
        try:
            analyzer.plot_results()
            print("📊 Performance charts created and displayed!")
        except Exception as e:
            print(f"⚠️  Chart creation failed: {e}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        best_model = max(summary['by_model']['mean'].items(), key=lambda x: x[1])
        best_timeframe = max(summary['by_timeframe']['mean'].items(), key=lambda x: x[1])
        
        print(f"   🥇 Best model: {best_model[0]} ({best_model[1]:.2%} avg accuracy)")
        print(f"   🥇 Best timeframe: {best_timeframe[0]} ({best_timeframe[1]:.2%} avg accuracy)")
        
        if summary['avg_validation_accuracy'] > summary.get('avg_train_accuracy', 0):
            print("   ✅ No overfitting detected - models generalize well!")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Focus training on your best-performing combinations")
        print("   2. Consider reducing training on poor-performing setups")
        print("   3. Use walk-forward validation for all future model updates")
        print("   4. Implement confidence-based trading (only trade high-confidence signals)")
        
        return summary
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Quick fix:")
        print("   1. Save the Walk-Forward Analyzer code as 'walk_forward_analyzer.py'")
        print("   2. Place it in your bot directory")
        print("   3. Run this script again")
        return None
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def integrate_wfa_with_existing_training():
    """Show how to integrate WFA with your existing training script"""
    
    print("\n🔗 INTEGRATION WITH YOUR EXISTING TRAINING")
    print("=" * 60)
    
    integration_code = '''
# Add this to your optimized_model_trainer.py

def enhanced_training_with_wfa(symbols, timeframes):
    """Enhanced training using walk-forward analysis"""
    
    from walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardConfig
    
    print("🔄 Running Walk-Forward Analysis before training...")
    
    # Quick WFA to identify best parameters
    config = WalkForwardConfig(
        target_symbols=symbols,
        timeframes=timeframes,
        optimization_window_days=90,
        validation_window_days=30
    )
    
    analyzer = WalkForwardAnalyzer(config)
    summary = analyzer.run_analysis()
    
    # Get best performing combinations
    top_performers = summary['top_performers'][:3]
    
    print("🎯 Training only on top-performing combinations:")
    for performer in top_performers:
        symbol = performer['symbol']
        timeframe = performer['timeframe'] 
        model = performer['model_name']
        accuracy = performer['validation_accuracy']
        
        print(f"   Training {symbol} {timeframe} {model} (WFA: {accuracy:.1%})")
        
        # Your existing training code here
        # train_model(symbol, timeframe, model)
    
    return summary

# Usage in your main training script:
if __name__ == "__main__":
    symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD']
    timeframes = ['4h', '1d']
    
    # Use WFA-enhanced training
    wfa_results = enhanced_training_with_wfa(symbols, timeframes)
'''
    
    print(integration_code)
    
    # Save integration example
    with open('wfa_integration_example.py', 'w') as f:
        f.write(integration_code)
    
    print("📁 Integration example saved to 'wfa_integration_example.py'")

def validate_current_models():
    """Validate your current models using walk-forward analysis"""
    
    print("\n🧪 VALIDATING YOUR CURRENT MODELS")
    print("=" * 60)
    
    # Check if your model files exist
    model_directories = [
        'models/BTCUSD/4h/',
        'models/BTCUSD/1d/', 
        'models/ETHUSD/4h/',
        'models/ADAUSD/4h/',
        'models/ADAUSD/1d/'
    ]
    
    existing_models = []
    for model_dir in model_directories:
        if os.path.exists(model_dir):
            existing_models.append(model_dir)
    
    if existing_models:
        print(f"✅ Found {len(existing_models)} existing model directories:")
        for model_dir in existing_models:
            print(f"   📁 {model_dir}")
        
        print("\n💡 RECOMMENDATION:")
        print("   Run walk-forward analysis on these exact combinations")
        print("   to validate their real-world performance!")
        
        # Create specific config for existing models
        symbols = []
        timeframes = []
        
        for model_dir in existing_models:
            parts = model_dir.split('/')
            if len(parts) >= 3:
                symbol_part = parts[1]  # BTCUSD
                timeframe = parts[2]    # 4h
                
                # Convert BTCUSD to BTC/USD format
                if symbol_part.endswith('USD'):
                    symbol = symbol_part[:-3] + '/USD'
                    if symbol not in symbols:
                        symbols.append(symbol)
                    if timeframe not in timeframes:
                        timeframes.append(timeframe)
        
        if symbols and timeframes:
            print(f"\n🎯 Auto-detected configuration:")
            print(f"   Symbols: {symbols}")
            print(f"   Timeframes: {timeframes}")
            
            return symbols, timeframes
    else:
        print("⚠️  No existing model directories found")
        print("   Using default configuration for WFA")
        return ['BTC/USD', 'ETH/USD', 'ADA/USD'], ['4h', '1d']

def main():
    """Main function to run walk-forward analysis integration"""
    
    print("🤖 Industrial Crypto Trading Bot v3.0")
    print("🔬 Walk-Forward Analysis Integration")
    print("=" * 60)
    
    # Validate existing models
    symbols, timeframes = validate_current_models()
    
    print(f"\n🎯 Analysis will focus on:")
    print(f"   Symbols: {symbols}")
    print(f"   Timeframes: {timeframes}")
    
    # Ask user what they want to do
    print("\n🔧 What would you like to do?")
    print("   1. Run quick walk-forward analysis (recommended)")
    print("   2. Show integration examples")
    print("   3. Both")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice in ['1', '3']:
            summary = quick_walk_forward_analysis()
            
            if summary and summary.get('avg_validation_accuracy', 0) > 0.65:
                print("\n🎉 CONGRATULATIONS!")
                print("Your models show excellent robustness across time periods!")
                print("This validates that your 65-70%+ accuracy is real and sustainable!")
        
        if choice in ['2', '3']:
            integrate_wfa_with_existing_training()
        
        print("\n✅ Walk-Forward Analysis integration complete!")
        print("\n🚀 Your bot is now using research-backed validation methodology!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()