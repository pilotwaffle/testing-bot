#!/usr/bin/env python3
"""
FILE: strategy_analyzer.py
LOCATION: E:\Trade Chat Bot\G Trading Bot\

Strategy Analyzer & Integration Tool
Analyzes your strategy collection and shows integration options
"""

import os
import ast
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

class StrategyAnalyzer:
    """Analyze and catalog trading strategies"""
    
    def __init__(self):
        self.strategies_dir = Path("strategies")
        self.strategies = {}
        self.analysis_results = {}
    
    def analyze_all_strategies(self) -> Dict[str, Any]:
        """Analyze all strategy files in the strategies directory"""
        
        print("FILE: strategy_analyzer.py")
        print("LOCATION: E:\\Trade Chat Bot\\G Trading Bot\\")
        print("")
        print("🎯 Trading Strategy Analysis & Integration Tool")
        print("=" * 70)
        
        if not self.strategies_dir.exists():
            print("❌ Strategies directory not found")
            return {}
        
        strategy_files = list(self.strategies_dir.glob("*.py"))
        strategy_files = [f for f in strategy_files if not f.name.startswith("__")]
        
        print(f"\n📁 Found {len(strategy_files)} strategy files:")
        for file in strategy_files:
            size_kb = file.stat().st_size / 1024
            print(f"   📄 {file.name:<35} ({size_kb:.1f} KB)")
        
        print(f"\n🔍 Analyzing Strategy Implementations...")
        print("-" * 50)
        
        for strategy_file in strategy_files:
            try:
                analysis = self._analyze_strategy_file(strategy_file)
                self.analysis_results[strategy_file.name] = analysis
                
                if analysis['success']:
                    print(f"✅ {strategy_file.name}")
                    self._print_strategy_summary(analysis)
                else:
                    print(f"❌ {strategy_file.name}: {analysis['error']}")
                
            except Exception as e:
                print(f"❌ {strategy_file.name}: Analysis failed - {e}")
        
        return self._generate_integration_report()
    
    def _analyze_strategy_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single strategy file"""
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find classes and methods
            tree = ast.parse(content)
            
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        'name': node.name,
                        'methods': methods,
                        'is_strategy': self._is_strategy_class(node.name, methods)
                    })
                elif isinstance(node, ast.FunctionDef) and not hasattr(node, 'parent_class'):
                    functions.append(node.name)
                elif isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    imports.extend([f"{module}.{alias.name}" for alias in node.names])
            
            # Try to import the module to check for runtime errors
            can_import = False
            import_error = None
            try:
                spec = importlib.util.spec_from_file_location("temp_strategy", file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    can_import = True
            except Exception as e:
                import_error = str(e)
            
            # Analyze strategy characteristics
            strategy_features = self._analyze_strategy_features(content)
            
            return {
                'success': True,
                'file_size': file_path.stat().st_size,
                'classes': classes,
                'functions': functions,
                'imports': imports,
                'can_import': can_import,
                'import_error': import_error,
                'strategy_classes': [c for c in classes if c['is_strategy']],
                'features': strategy_features,
                'complexity_score': self._calculate_complexity_score(classes, functions, content)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _is_strategy_class(self, class_name: str, methods: List[str]) -> bool:
        """Determine if a class is likely a trading strategy"""
        
        strategy_indicators = [
            'strategy' in class_name.lower(),
            'trading' in class_name.lower(),
            any(method in ['execute', 'run', 'trade', 'signal', 'generate_signal'] for method in methods),
            any(method.startswith('on_') for method in methods),
            'init' in methods or '__init__' in methods
        ]
        
        return sum(strategy_indicators) >= 2
    
    def _analyze_strategy_features(self, content: str) -> Dict[str, bool]:
        """Analyze what features a strategy implements"""
        
        content_lower = content.lower()
        
        return {
            'uses_ml': any(keyword in content_lower for keyword in ['sklearn', 'tensorflow', 'keras', 'model', 'predict']),
            'uses_indicators': any(keyword in content_lower for keyword in ['rsi', 'macd', 'sma', 'ema', 'bollinger']),
            'risk_management': any(keyword in content_lower for keyword in ['stop_loss', 'take_profit', 'risk', 'position_size']),
            'multi_timeframe': any(keyword in content_lower for keyword in ['timeframe', '1h', '4h', '1d']),
            'portfolio_management': any(keyword in content_lower for keyword in ['portfolio', 'balance', 'allocation']),
            'backtesting': any(keyword in content_lower for keyword in ['backtest', 'historical', 'performance']),
            'async_capable': 'async def' in content_lower,
            'uses_websocket': any(keyword in content_lower for keyword in ['websocket', 'real_time', 'streaming']),
            'uses_database': any(keyword in content_lower for keyword in ['database', 'sqlite', 'sql']),
            'has_logging': any(keyword in content_lower for keyword in ['logging', 'logger', 'log'])
        }
    
    def _calculate_complexity_score(self, classes: List[Dict], functions: List[str], content: str) -> int:
        """Calculate a complexity score for the strategy"""
        
        score = 0
        score += len(classes) * 10
        score += len(functions) * 5
        score += len(content.split('\n')) // 10  # Lines of code
        score += content.count('def ') * 3
        score += content.count('class ') * 15
        score += content.count('import ') * 2
        
        return min(score, 100)  # Cap at 100
    
    def _print_strategy_summary(self, analysis: Dict[str, Any]):
        """Print a summary of strategy analysis"""
        
        strategy_classes = analysis['strategy_classes']
        features = analysis['features']
        
        if strategy_classes:
            class_names = [c['name'] for c in strategy_classes]
            print(f"   🎯 Strategy Classes: {', '.join(class_names)}")
        
        if analysis['can_import']:
            print(f"   ✅ Importable: Yes")
        else:
            print(f"   ❌ Import Error: {analysis['import_error'][:50]}...")
        
        print(f"   📊 Complexity: {analysis['complexity_score']}/100")
        
        # Feature summary
        active_features = [feature for feature, active in features.items() if active]
        if active_features:
            print(f"   🔧 Features: {', '.join(active_features[:3])}{'...' if len(active_features) > 3 else ''}")
        
        print()
    
    def _generate_integration_report(self) -> Dict[str, Any]:
        """Generate integration recommendations"""
        
        print("\n🚀 INTEGRATION ANALYSIS & RECOMMENDATIONS")
        print("=" * 70)
        
        # Categorize strategies
        flagship_strategies = []
        ml_strategies = []
        template_strategies = []
        broken_strategies = []
        
        for filename, analysis in self.analysis_results.items():
            if not analysis['success']:
                broken_strategies.append(filename)
                continue
            
            if analysis['file_size'] > 30000:  # > 30KB
                flagship_strategies.append((filename, analysis))
            elif analysis['features']['uses_ml']:
                ml_strategies.append((filename, analysis))
            elif 'template' in filename.lower() or 'base' in filename.lower():
                template_strategies.append((filename, analysis))
        
        print(f"\n📊 STRATEGY CATEGORIZATION:")
        print(f"   🏆 Flagship Strategies: {len(flagship_strategies)}")
        print(f"   🧠 ML Strategies: {len(ml_strategies)}")
        print(f"   📋 Templates/Base: {len(template_strategies)}")
        print(f"   ❌ Issues Found: {len(broken_strategies)}")
        
        # Recommendations
        print(f"\n💡 INTEGRATION RECOMMENDATIONS:")
        
        if flagship_strategies:
            best_strategy = max(flagship_strategies, key=lambda x: x[1]['complexity_score'])
            print(f"   🎯 PRIMARY STRATEGY: {best_strategy[0]}")
            print(f"      Size: {best_strategy[1]['file_size']/1024:.1f} KB")
            print(f"      Complexity: {best_strategy[1]['complexity_score']}/100")
            print(f"      Classes: {len(best_strategy[1]['strategy_classes'])}")
        
        if ml_strategies:
            print(f"   🧠 ML INTEGRATION: Use {ml_strategies[0][0]} for AI-powered signals")
        
        if broken_strategies:
            print(f"   🔧 FIXES NEEDED:")
            for strategy in broken_strategies:
                error = self.analysis_results[strategy]['error'][:60]
                print(f"      - {strategy}: {error}...")
        
        # Integration code generation
        print(f"\n🔌 INTEGRATION CODE:")
        self._generate_integration_code()
        
        return {
            'flagship_strategies': flagship_strategies,
            'ml_strategies': ml_strategies,
            'broken_strategies': broken_strategies,
            'total_strategies': len(self.analysis_results),
            'total_size_kb': sum(a.get('file_size', 0) for a in self.analysis_results.values() if a['success']) / 1024
        }
    
    def _generate_integration_code(self):
        """Generate code to integrate strategies with EliteTradingEngine"""
        
        print("   Add this to your EliteTradingEngine:")
        
        integration_code = '''
# Add to EliteTradingEngine class:

async def load_strategies(self) -> Dict[str, Any]:
    """Load and initialize trading strategies"""
    strategies = {}
    
    try:
        # Load Enhanced Trading Strategy (Flagship)
        from strategies.enhanced_trading_strategy import *
        strategies['enhanced'] = EnhancedTradingStrategy(self.config)
        
        # Load ML Strategy
        from strategies.ml_strategy import *
        strategies['ml'] = MLStrategy(self.config)
        
        # Load Custom Strategy
        from strategies.my_enhanced_strategy import *
        strategies['custom'] = MyEnhancedStrategy(self.config)
        
        self.logger.info(f"Loaded {len(strategies)} trading strategies")
        return strategies
        
    except Exception as e:
        self.logger.error(f"Error loading strategies: {e}")
        return {}

async def execute_strategy(self, strategy_name: str, market_data: Dict) -> Optional[TradingSignal]:
    """Execute a specific trading strategy"""
    if strategy_name in self.strategies:
        try:
            strategy = self.strategies[strategy_name]
            signal = await strategy.generate_signal(market_data)
            return signal
        except Exception as e:
            self.logger.error(f"Strategy {strategy_name} execution failed: {e}")
    return None
'''
        
        print(integration_code)
        
        print(f"\n📝 STRATEGY ACTIVATION STEPS:")
        print(f"   1. Add the integration code above to your EliteTradingEngine")
        print(f"   2. Call self.strategies = await self.load_strategies() in __init__")
        print(f"   3. Modify _generate_trading_signals() to use strategies")
        print(f"   4. Test with paper trading first")

def main():
    """Main function to analyze strategies"""
    
    analyzer = StrategyAnalyzer()
    results = analyzer.analyze_all_strategies()
    
    if results:
        print(f"\n📈 SUMMARY:")
        print(f"   Total Strategies Analyzed: {results['total_strategies']}")
        print(f"   Total Strategy Code: {results['total_size_kb']:.1f} KB")
        print(f"   Flagship Strategies: {len(results['flagship_strategies'])}")
        print(f"   ML-Powered Strategies: {len(results['ml_strategies'])}")
        
        if results['flagship_strategies']:
            print(f"\n🏆 RECOMMENDED NEXT STEPS:")
            print(f"   1. Restart your server (it's already enhanced with EliteTradingEngine)")
            print(f"   2. Test strategy integration with your largest strategy")
            print(f"   3. Monitor performance through the dashboard")
            print(f"   4. Consider Walk-Forward Analysis on your best strategies")
        
        print(f"\n🎯 Your trading system is ENTERPRISE-GRADE!")
        print(f"   EliteTradingEngine + Strategy Arsenal = Professional Trading Platform")

if __name__ == "__main__":
    main()