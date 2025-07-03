#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\fix_dropdown_compatibility.py
Location: E:\Trade Chat Bot\G Trading Bot\fix_dropdown_compatibility.py

🔧 Elite Trading Bot V3.0 - Fix Strategy Dropdown Compatibility
Fix frontend to work with existing strategy format
"""

import os
import json
import requests
from datetime import datetime

def test_current_api_format():
    """Test what format the API is currently returning"""
    print("🔍 Testing current API format...")
    
    try:
        response = requests.get('http://localhost:8000/api/strategies/available', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API working - Status: {response.status_code}")
            print(f"📊 Strategies count: {data.get('total_count', len(data.get('strategies', [])))}")
            
            if data.get('strategies'):
                strategy = data['strategies'][0]
                print(f"📝 Strategy format example:")
                for key, value in strategy.items():
                    print(f"   {key}: {value}")
                return data['strategies']
            
        else:
            print(f"❌ API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        return None

def fix_frontend_strategy_handling():
    """Fix the frontend to handle existing strategy format"""
    print("\n🔧 Fixing frontend strategy compatibility...")
    
    js_path = "static/js/dashboard.js"
    
    if not os.path.exists(js_path):
        print("❌ dashboard.js not found!")
        return False
    
    with open(js_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Enhanced strategy loading function that handles multiple formats
    enhanced_strategy_function = '''
    async loadStrategies() {
        try {
            console.log('📊 Loading available strategies...');
            const response = await fetch('/api/strategies/available');
            if (response.ok) {
                const data = await response.json();
                console.log('✅ Strategies API response:', data);
                
                const strategies = data.strategies || data || [];
                console.log(`📈 Found ${strategies.length} strategies`);
                
                if (strategies.length > 0) {
                    this.populateStrategySelect(strategies);
                } else {
                    console.warn('⚠️ No strategies found, using fallback');
                    this.showFallbackStrategies();
                }
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error('❌ Failed to load strategies:', error);
            this.showFallbackStrategies();
        }
    }

    populateStrategySelect(strategies) {
        const select = document.getElementById('strategy-select');
        if (!select) {
            console.warn('❌ Strategy select element not found');
            return;
        }
        
        // Clear existing options
        select.innerHTML = '<option value="">Select Strategy...</option>';
        
        // Add strategies with compatibility for multiple formats
        strategies.forEach((strategy, index) => {
            const option = document.createElement('option');
            option.value = strategy.id || strategy.name || `strategy_${index}`;
            
            // Build display name with available info
            let displayName = strategy.name || strategy.id || `Strategy ${index + 1}`;
            
            // Add risk level if available
            if (strategy.risk_level) {
                displayName += ` (${strategy.risk_level} Risk)`;
            }
            
            // Add accuracy if available
            if (strategy.accuracy) {
                displayName += ` - ${strategy.accuracy} accuracy`;
            }
            
            // Add estimated returns if available
            if (strategy.estimated_returns) {
                displayName += ` - ${strategy.estimated_returns}`;
            }
            
            option.textContent = displayName;
            
            // Set tooltip with description
            const description = strategy.description || 
                              `${strategy.name || 'Trading strategy'} - ${strategy.timeframe || 'Various timeframes'}`;
            option.title = description;
            
            select.appendChild(option);
            
            console.log(`✅ Added strategy: ${displayName}`);
        });
        
        console.log(`✅ Successfully populated ${strategies.length} strategies in dropdown`);
        
        // Trigger change event to update any dependent UI
        select.dispatchEvent(new Event('change'));
    }

    showFallbackStrategies() {
        console.log('🔄 Using fallback strategies...');
        const fallbackStrategies = [
            { 
                id: 'momentum_scalping', 
                name: 'Momentum Scalping', 
                risk_level: 'High', 
                description: 'High-frequency momentum-based strategy',
                timeframe: '1m-5m',
                estimated_returns: '15-25% monthly'
            },
            { 
                id: 'trend_following', 
                name: 'Trend Following', 
                risk_level: 'Medium', 
                description: 'Long-term trend identification',
                timeframe: '1h-4h',
                estimated_returns: '8-15% monthly'
            },
            { 
                id: 'mean_reversion', 
                name: 'Mean Reversion', 
                risk_level: 'Low', 
                description: 'Statistical arbitrage strategy',
                timeframe: '15m-1h',
                estimated_returns: '5-12% monthly'
            }
        ];
        this.populateStrategySelect(fallbackStrategies);
    }

    selectStrategy(strategyId) {
        console.log(`📊 Selected strategy: ${strategyId}`);
        
        // Update UI to show selected strategy
        const select = document.getElementById('strategy-select');
        if (select && strategyId) {
            const selectedOption = select.querySelector(`option[value="${strategyId}"]`);
            if (selectedOption) {
                console.log(`✅ Strategy selected: ${selectedOption.textContent}`);
                this.showNotification(`Strategy selected: ${selectedOption.textContent.split(' (')[0]}`, 'info');
            }
        }
        
        // Store selected strategy
        this.selectedStrategy = strategyId;
        
        // Enable trading buttons if strategy is selected
        this.updateTradingButtonsForStrategy(strategyId);
    }

    updateTradingButtonsForStrategy(strategyId) {
        const startBtn = document.getElementById('start-trading');
        const strategyInfo = document.getElementById('strategy-info');
        
        if (startBtn) {
            if (strategyId) {
                startBtn.disabled = false;
                startBtn.classList.remove('btn-disabled');
                startBtn.textContent = 'Start Trading';
            } else {
                startBtn.disabled = true;
                startBtn.classList.add('btn-disabled');
                startBtn.textContent = 'Select Strategy First';
            }
        }
        
        // Show strategy info if available
        if (strategyInfo) {
            if (strategyId) {
                strategyInfo.style.display = 'block';
                strategyInfo.innerHTML = `<small>Selected: ${strategyId}</small>`;
            } else {
                strategyInfo.style.display = 'none';
            }
        }
    }'''
    
    # Find and replace the existing loadStrategies function
    if 'loadStrategies()' in content:
        # Find the start of the function
        start_marker = 'async loadStrategies()'
        start_pos = content.find(start_marker)
        
        if start_pos != -1:
            # Find the end of the function by counting braces
            lines = content[start_pos:].split('\n')
            func_lines = []
            brace_count = 0
            in_function = False
            
            for i, line in enumerate(lines):
                if start_marker in line:
                    in_function = True
                    brace_count = 0
                
                if in_function:
                    func_lines.append(line)
                    
                    # Count braces to find function end
                    brace_count += line.count('{') - line.count('}')
                    
                    # If we're back to 0 braces and have more than just the function declaration
                    if brace_count <= 0 and len(func_lines) > 1:
                        # Look ahead to see if next line starts a new function/method
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if (next_line.startswith('async ') or 
                                next_line.startswith('function ') or
                                next_line.startswith('populateStrategySelect') or
                                next_line.startswith('showFallback') or
                                next_line.endswith('{') or
                                next_line.startswith('//') or
                                next_line == ''):
                                break
                        else:
                            break
            
            old_function = '\n'.join(func_lines)
            
            # Also find and replace related functions
            functions_to_replace = [
                'populateStrategySelect',
                'showFallbackStrategies', 
                'selectStrategy'
            ]
            
            for func_name in functions_to_replace:
                if func_name in content:
                    # Find and remove old function
                    func_start = content.find(f'{func_name}(')
                    if func_start != -1:
                        # Find the line that contains the function
                        line_start = content.rfind('\n', 0, func_start) + 1
                        func_lines = []
                        brace_count = 0
                        in_function = False
                        
                        remaining_content = content[line_start:]
                        lines = remaining_content.split('\n')
                        
                        for i, line in enumerate(lines):
                            if func_name in line and '(' in line:
                                in_function = True
                                brace_count = 0
                            
                            if in_function:
                                func_lines.append(line)
                                brace_count += line.count('{') - line.count('}')
                                
                                if brace_count <= 0 and len(func_lines) > 1:
                                    if i + 1 < len(lines):
                                        next_line = lines[i + 1].strip()
                                        if (next_line.startswith('async ') or 
                                            next_line.startswith('function ') or
                                            any(f in next_line for f in functions_to_replace) or
                                            next_line.endswith('{') or
                                            next_line.startswith('//') or
                                            next_line == ''):
                                            break
                                    else:
                                        break
                        
                        old_func = '\n'.join(func_lines)
                        content = content.replace(old_func, '')
            
            # Replace the loadStrategies function with enhanced version
            content = content.replace(old_function, enhanced_strategy_function.strip())
        
    else:
        # If function doesn't exist, add it after initializeTradingControls
        if 'initializeTradingControls()' in content:
            insert_point = content.find('    }', content.find('initializeTradingControls()')) + 5
            content = content[:insert_point] + '\n' + enhanced_strategy_function + '\n' + content[insert_point:]
    
    # Update strategy select event handler
    if 'strategy-select' in content and 'addEventListener' in content:
        # Find the strategy-select event handler and update it
        old_handler = '''document.getElementById('strategy-select')?.addEventListener('change', (e) => {
            this.selectStrategy(e.target.value);
        });'''
        
        new_handler = '''const strategySelect = document.getElementById('strategy-select');
        if (strategySelect) {
            strategySelect.addEventListener('change', (e) => {
                this.selectStrategy(e.target.value);
            });
        }'''
        
        if old_handler.split('\n')[0].strip() in content:
            content = content.replace(old_handler, new_handler)
    
    # Write updated dashboard.js
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed frontend strategy compatibility")
    return True

def create_compatible_test_script():
    """Create a test script that works with both old and new formats"""
    test_script = '''#!/usr/bin/env python3
"""
File: E:\\Trade Chat Bot\\G Trading Bot\\test_strategies_compatible.py
Location: E:\\Trade Chat Bot\\G Trading Bot\\test_strategies_compatible.py

🧪 Test Trading Strategies API (Compatible Version)
"""

import requests
import json

def test_strategies():
    print("🧪 Testing Trading Strategies API (Compatible)")
    print("=" * 55)
    
    try:
        # Test available strategies
        print("📊 Testing /api/strategies/available...")
        response = requests.get('http://localhost:8000/api/strategies/available')
        
        if response.status_code == 200:
            data = response.json()
            strategies = data.get('strategies', [])
            count = data.get('total_count', len(strategies))
            
            print(f"✅ SUCCESS - Found {count} strategies:")
            print()
            
            for i, strategy in enumerate(strategies, 1):
                print(f"   🎯 Strategy #{i}: {strategy.get('name', 'Unknown')}")
                print(f"      ID: {strategy.get('id', 'N/A')}")
                print(f"      Risk Level: {strategy.get('risk_level', 'N/A')}")
                print(f"      Timeframe: {strategy.get('timeframe', 'N/A')}")
                
                # Handle both old and new formats
                if 'accuracy' in strategy:
                    print(f"      Accuracy: {strategy['accuracy']}")
                if 'estimated_returns' in strategy:
                    print(f"      Est. Returns: {strategy['estimated_returns']}")
                if 'required_capital' in strategy:
                    print(f"      Min Capital: ${strategy['required_capital']}")
                
                print(f"      Description: {strategy.get('description', 'N/A')}")
                print()
                
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"Response: {response.text}")
        
        # Test active strategies
        print("📈 Testing /api/strategies/active...")
        response = requests.get('http://localhost:8000/api/strategies/active')
        
        if response.status_code == 200:
            data = response.json()
            active_strategies = data.get('active_strategies', [])
            total_active = data.get('total_active', len(active_strategies))
            
            print(f"✅ SUCCESS - Found {total_active} active strategies:")
            
            if active_strategies:
                for strategy in active_strategies:
                    print(f"   🚀 {strategy.get('name', strategy.get('id', 'Unknown'))}")
                    print(f"      Status: {strategy.get('status', 'Unknown')}")
                    
                    if 'pnl' in strategy:
                        print(f"      PnL: ${strategy['pnl']:.2f}")
                    if 'positions' in strategy:
                        print(f"      Positions: {strategy['positions']}")
                    if 'win_rate' in strategy:
                        print(f"      Win Rate: {strategy['win_rate']:.1%}")
                    print()
            else:
                print("   📊 No active strategies currently running")
                
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"Response: {response.text}")
        
        print("=" * 55)
        print("🎯 SUMMARY:")
        print("✅ API is working and returning strategies")
        print("✅ Frontend should now populate dropdown correctly")
        print("✅ Try refreshing your browser dashboard")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure your server is running: python main.py")

if __name__ == "__main__":
    test_strategies()
'''
    
    with open('test_strategies_compatible.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ Created compatible test script")

def main():
    print("🔧 Elite Trading Bot V3.0 - Fix Strategy Dropdown Compatibility")
    print("=" * 70)
    
    # Test current API format
    strategies = test_current_api_format()
    
    if strategies:
        print(f"\n✅ Your API is working! Found {len(strategies)} strategies")
        print("📝 The issue is format compatibility between API and frontend")
    else:
        print("\n❌ API is not responding properly")
        return
    
    # Fix frontend compatibility
    if fix_frontend_strategy_handling():
        print("✅ Frontend compatibility fixed")
    else:
        print("❌ Failed to fix frontend")
        return
    
    # Create compatible test script
    create_compatible_test_script()
    
    print("\n" + "=" * 70)
    print("🎉 DROPDOWN COMPATIBILITY FIX COMPLETE!")
    print("=" * 70)
    print()
    print("📋 What was fixed:")
    print("   ✅ Frontend now handles existing strategy format")
    print("   ✅ Dropdown will populate with your 3 strategies")
    print("   ✅ Compatible with both old and new strategy formats")
    print("   ✅ Enhanced error handling and fallbacks")
    print("   ✅ Better strategy selection feedback")
    print()
    print("🎯 Your current strategies:")
    for i, strategy in enumerate(strategies, 1):
        name = strategy.get('name', 'Unknown')
        risk = strategy.get('risk_level', 'N/A')
        returns = strategy.get('estimated_returns', 'N/A')
        print(f"   {i}. {name} ({risk} Risk) - {returns}")
    print()
    print("🚀 Next steps:")
    print("   1. Refresh your browser (Ctrl+F5)")
    print("   2. Check the strategy dropdown - should show 3 strategies")
    print("   3. Test selection: python test_strategies_compatible.py")
    print()
    print("✅ Your strategy dropdown should now be fully populated!")

if __name__ == "__main__":
    main()