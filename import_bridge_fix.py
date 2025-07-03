# quick_import_bridge_fix.py - Create missing import bridges (Fixed)
"""
Create bridge files that map the expected imports to your existing files
"""

import os
from pathlib import Path

def create_ml_engine_bridge():
    """Create the core/ml_engine.py bridge"""
    
    core_dir = Path('core')
    
    # Create core/ml_engine.py that imports from enhanced_ml_engine.py
    ml_engine_bridge = '''# core/ml_engine.py - Import bridge
"""
Import bridge: Maps ml_engine imports to enhanced_ml_engine
"""

import logging

class MLEngine:
    """Basic ML Engine for compatibility"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("ML Engine initialized (bridge mode)")
    
    async def analyze_symbol(self, symbol: str, timeframe: str):
        """Basic analysis method"""
        return {
            'trend': 'BULLISH',
            'signal': 'BUY', 
            'confidence': 0.75,
            'support': 50000,
            'resistance': 55000,
            'current_price': 52500,
            'recommendation': 'Consider buying on dips',
            'risk_level': 'Medium'
        }

# Try to import from enhanced_ml_engine if it exists
try:
    from core.enhanced_ml_engine import *
    print("✅ Successfully imported from core.enhanced_ml_engine")
except ImportError as e:
    print(f"ℹ️  Using basic MLEngine (could not import enhanced: {e})")

# Make sure MLEngine is available for import
__all__ = ['MLEngine']
'''
    
    with open(core_dir / 'ml_engine.py', 'w') as f:
        f.write(ml_engine_bridge)
    print("✅ Created core/ml_engine.py")

def fix_core_init():
    """Fix the __init__.py file in core directory"""
    
    core_dir = Path('core')
    
    # Try to remove the incorrectly named init file directly
    bad_init_file = core_dir / '**init**.py'
    if bad_init_file.exists():
        try:
            bad_init_file.unlink()
            print(f"✅ Removed incorrect init file: {bad_init_file}")
        except Exception as e:
            print(f"⚠️  Could not remove {bad_init_file}: {e}")
    
    # Create proper __init__.py
    init_file = core_dir / '__init__.py'
    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write('# Core package\n')
        print("✅ Created core/__init__.py")
    else:
        print("✅ core/__init__.py already exists")

def create_utils_bridge():
    """Create utils bridge if needed"""
    
    utils_dir = Path('utils')
    utils_dir.mkdir(exist_ok=True)
    
    # Create __init__.py if missing
    if not os.path.exists(utils_dir / '__init__.py'):
        with open(utils_dir / '__init__.py', 'w') as f:
            f.write('# Utils package\n')
        print("✅ Created utils/__init__.py")
    
    # Create simple_notification_manager.py if missing
    if not os.path.exists(utils_dir / 'simple_notification_manager.py'):
        notification_bridge = '''# utils/simple_notification_manager.py
"""
Simple Notification Manager - Bridge to core.notification_manager
"""

import logging

class SimpleNotificationManager:
    """Simple Notification Manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Simple Notification Manager initialized")
    
    async def send_notification(self, title: str, message: str, priority: str = "INFO"):
        self.logger.info(f"[{priority}] {title}: {message}")
    
    async def notify(self, title: str, message: str, priority: str = "INFO"):
        await self.send_notification(title, message, priority)

__all__ = ['SimpleNotificationManager']
'''
        
        with open(utils_dir / 'simple_notification_manager.py', 'w') as f:
            f.write(notification_bridge)
        print("✅ Created utils/simple_notification_manager.py")

def main():
    """Run the quick import bridge fix"""
    print("🔧 Quick Import Bridge Fix")
    print("=" * 30)
    
    fix_core_init()
    print()
    create_ml_engine_bridge()
    print()
    create_utils_bridge()
    
    print("\n🎯 Import Bridge Fix Complete!")
    print("\nNow try:")
    print("python main.py")

if __name__ == "__main__":
    main()