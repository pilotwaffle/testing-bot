# system_diagnostic.py - Run this to test your system integration
"""
System Diagnostic Tool for Enhanced Trading Bot
Checks all components and identifies integration issues
"""

import os
import sys
import importlib
from pathlib import Path

def test_component(module_name, component_name):
    """Test if a component can be imported and initialized"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, component_name):
            component_class = getattr(module, component_name)
            # Try to initialize with minimal config
            if component_name in ['AdaptiveMLEngine', 'EnhancedDataFetcher']:
                instance = component_class()
            elif component_name == 'EnhancedTradingStrategy':
                instance = component_class()
            else:
                instance = component_class
            print(f"✅ {component_name} - OK")
            return True
        else:
            print(f"❌ {component_name} - Class not found in {module_name}")
            return False
    except ImportError as e:
        print(f"❌ {component_name} - Import Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {component_name} - Init Error: {e}")
        return True  # Import worked, just init failed

def check_environment():
    """Check environment setup"""
    print("🔍 Environment Check:")
    print(f"   Python Version: {sys.version}")
    print(f"   Working Directory: {os.getcwd()}")
    print(f"   Python Path includes current dir: {'.' in sys.path or os.getcwd() in sys.path}")
    
    # Check key files
    key_files = [
        'main.py', 'enhanced_ml_engine.py', 'enhanced_data_fetcher.py',
        'enhanced_trading_strategy.py', 'trading_engine.py', '.env', 'config.json'
    ]
    
    print("📁 Key Files:")
    for file in key_files:
        exists = os.path.exists(file)
        print(f"   {file}: {'✅' if exists else '❌'}")
    
    # Check directories
    key_dirs = ['core', 'ai', 'utils', 'database', 'strategies', 'templates', 'static']
    print("📂 Key Directories:")
    for dir_name in key_dirs:
        exists = os.path.exists(dir_name)
        print(f"   {dir_name}/: {'✅' if exists else '❌'}")

def test_dashboard_systems():
    """Test different dashboard systems"""
    print("\n🖥️  Dashboard Systems Test:")
    
    # Test FastAPI main.py
    try:
        import main
        print("✅ main.py (FastAPI) - Import OK")
    except Exception as e:
        print(f"❌ main.py (FastAPI) - {e}")
    
    # Test Flask dashboard_app.py
    try:
        import dashboard_app
        print("✅ dashboard_app.py (Flask) - Import OK")
    except Exception as e:
        print(f"❌ dashboard_app.py (Flask) - {e}")
    
    # Test web_dashboard.py
    try:
        import web_dashboard
        print("✅ web_dashboard.py - Import OK")
    except Exception as e:
        print(f"❌ web_dashboard.py - {e}")

def test_core_components():
    """Test core trading components"""
    print("\n🧠 Core Components Test:")
    
    components = [
        ('enhanced_ml_engine', 'AdaptiveMLEngine'),
        ('enhanced_data_fetcher', 'EnhancedDataFetcher'), 
        ('enhanced_trading_strategy', 'EnhancedTradingStrategy'),
        ('trading_engine', 'IndustrialTradingEngine'),
    ]
    
    for module_name, component_name in components:
        test_component(module_name, component_name)

def test_config_system():
    """Test configuration system"""
    print("\n⚙️  Configuration Test:")
    
    # Test .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env file - Loaded")
        
        # Check key variables
        key_vars = ['DATABASE_URL', 'APP_NAME', 'KRAKEN_API_KEY']
        for var in key_vars:
            value = os.getenv(var)
            status = "Set" if value else "Not set"
            print(f"   {var}: {status}")
            
    except ImportError:
        print("❌ python-dotenv not installed")
    except Exception as e:
        print(f"❌ .env error: {e}")
    
    # Test config.json
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("✅ config.json - Loaded")
        print(f"   Keys: {list(config.keys())}")
    except Exception as e:
        print(f"❌ config.json error: {e}")

def test_dependencies():
    """Test required dependencies"""
    print("\n📦 Dependencies Test:")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'sklearn', 
        'tensorflow', 'ccxt', 'flask', 'plotly', 'websockets'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")

def main():
    """Run complete system diagnostic"""
    print("🚀 Enhanced Trading Bot - System Diagnostic")
    print("=" * 50)
    
    # Add current directory to Python path
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    
    check_environment()
    test_dependencies()
    test_config_system() 
    test_core_components()
    test_dashboard_systems()
    
    print("\n🎯 Diagnostic Complete!")
    print("\nNext Steps:")
    print("1. Fix any ❌ issues shown above")
    print("2. Install missing dependencies: pip install -r requirements.txt")
    print("3. Try starting your preferred dashboard:")
    print("   - FastAPI: python main.py")
    print("   - Flask: python dashboard_app.py") 
    print("   - Web Dashboard: python web_dashboard.py")

if __name__ == "__main__":
    main()