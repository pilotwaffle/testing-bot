#!/usr/bin/env python3
"""
class_inspector.py - Inspect available methods in your existing classes
This will help us understand what methods are actually available
"""

import inspect

def inspect_class(class_obj, class_name):
    """Inspect a class and show its methods"""
    print(f"\n🔍 Inspecting {class_name}:")
    print("=" * (15 + len(class_name)))
    
    # Get all methods and attributes
    methods = []
    attributes = []
    
    for name, obj in inspect.getmembers(class_obj):
        if not name.startswith('_'):  # Skip private methods
            if inspect.ismethod(obj) or inspect.isfunction(obj):
                # Get method signature if possible
                try:
                    sig = inspect.signature(obj)
                    methods.append(f"{name}{sig}")
                except:
                    methods.append(name)
            else:
                attributes.append(name)
    
    print("📋 Available Methods:")
    for method in sorted(methods):
        print(f"  • {method}")
    
    if attributes:
        print("\n📝 Attributes:")
        for attr in sorted(attributes):
            print(f"  • {attr}")
    
    print()

def main():
    """Main inspection function"""
    print("🔍 Enhanced Trading Bot - Class Inspector")
    print("=" * 50)
    
    # Inspect EnhancedDataFetcher
    try:
        from core.enhanced_data_fetcher import EnhancedDataFetcher
        
        # Create instance to inspect
        data_fetcher = EnhancedDataFetcher()
        inspect_class(data_fetcher, "EnhancedDataFetcher")
        
        # Look for methods that might fetch data
        print("🎯 Data fetching methods (likely candidates):")
        for name in dir(data_fetcher):
            if not name.startswith('_') and any(keyword in name.lower() for keyword in ['fetch', 'get', 'data', 'ohlcv', 'candle']):
                print(f"  • {name}")
        
    except Exception as e:
        print(f"❌ Error inspecting EnhancedDataFetcher: {e}")
    
    # Inspect AdaptiveMLEngine
    try:
        from core.enhanced_ml_engine import AdaptiveMLEngine
        
        # Create instance to inspect
        ml_engine = AdaptiveMLEngine()
        inspect_class(ml_engine, "AdaptiveMLEngine")
        
        # Look for methods that might train models
        print("🎯 Training methods (likely candidates):")
        for name in dir(ml_engine):
            if not name.startswith('_') and any(keyword in name.lower() for keyword in ['train', 'fit', 'learn', 'model', 'predict']):
                print(f"  • {name}")
        
    except Exception as e:
        print(f"❌ Error inspecting AdaptiveMLEngine: {e}")
    
    # Additional checks
    print("\n🔧 Recommendations:")
    print("Based on the inspection above, we can create a compatible trainer that uses")
    print("the actual methods available in your existing classes.")

if __name__ == "__main__":
    main()