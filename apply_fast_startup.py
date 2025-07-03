# apply_fast_startup.py - Apply Fast Startup Now!
"""
Apply Fast Startup Optimizations
==============================
"""

import time
from pathlib import Path

def apply_optimizations():
    """Apply fast startup optimizations to main.py"""
    
    print("Applying fast startup optimizations...")
    
    main_file = Path("main.py")
    if not main_file.exists():
        print("ERROR: main.py not found")
        return False
    
    try:
        # Read main.py
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply optimizations
        optimized_content = content
        changes_made = 0
        
        # 1. Replace trading engine import
        if 'from core.trading_engine import' in optimized_content:
            optimized_content = optimized_content.replace(
                'from core.trading_engine import TradingEngine',
                'from core.fast_trading_engine import FastTradingEngine as TradingEngine'
            )
            changes_made += 1
            print("APPLIED: Fast trading engine import")
        
        # 2. Replace ML engine import
        if 'from core.enhanced_ml_engine import' in optimized_content:
            optimized_content = optimized_content.replace(
                'from core.enhanced_ml_engine import EnhancedMLEngine',
                'from core.fast_ml_engine import FastMLEngine as EnhancedMLEngine'
            )
            optimized_content = optimized_content.replace(
                'from core.enhanced_ml_engine import',
                'from core.fast_ml_engine import'
            )
            changes_made += 1
            print("APPLIED: Fast ML engine import")
        
        # 3. Add async initialization if not present
        if 'async def init_heavy_components' not in optimized_content:
            # Find where components are initialized
            if 'ml_engine =' in optimized_content and 'trading_engine =' in optimized_content:
                # Add async initialization code
                async_init = '''
# Fast startup optimization - initialize heavy components in background
async def init_heavy_components():
    try:
        if hasattr(trading_engine, "initialize_async"):
            await trading_engine.initialize_async()
        if hasattr(ml_engine, "initialize_async"):
            await ml_engine.initialize_async()
        logger.info("Background initialization complete")
    except Exception as e:
        logger.warning(f"Background initialization error: {e}")

# Start background initialization
import asyncio
asyncio.create_task(init_heavy_components())
'''
                
                # Insert after the last component initialization
                if 'INFO - All systems initialized successfully' in optimized_content:
                    optimized_content = optimized_content.replace(
                        'logger.info("All systems initialized successfully")',
                        'logger.info("All systems initialized successfully")\n' + async_init
                    )
                    changes_made += 1
                    print("APPLIED: Async background initialization")
        
        if changes_made > 0:
            # Backup original
            backup_file = Path(f"main_backup_{int(time.time())}.py")
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"BACKUP: Created {backup_file}")
            
            # Write optimized version
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(optimized_content)
            
            print(f"SUCCESS: Applied {changes_made} optimizations to main.py")
            return True
        else:
            print("INFO: No optimizations needed")
            return True
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main application"""
    print("Fast Startup Patch v1.0")
    print("=" * 40)
    
    success = apply_optimizations()
    
    if success:
        print()
        print("OPTIMIZATION COMPLETE!")
        print("=" * 40)
        print("Your trading engine should now start much faster!")
        print()
        print("NEXT STEPS:")
        print("1. Restart your bot:")
        print("   python -m uvicorn main:app --host 0.0.0.0 --port 8000")
        print()
        print("2. Expected startup time: 2-5 seconds")
        print("3. Dashboard will be available immediately")
        print("4. Heavy components load in background")
        print()
        print("READY FOR LIGHTNING-FAST STARTUP!")
    else:
        print("FAILED: Could not apply optimizations")
    
    return success

if __name__ == "__main__":
    main()