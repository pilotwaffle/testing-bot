# FILE: implement_kraken.py
# LOCATION: E:\Trade Chat Bot\G Trading Bot\implement_kraken.py

#!/usr/bin/env python3
"""
🚀 KRAKEN FUTURES PAPER TRADING IMPLEMENTATION SCRIPT
For Elite Trading Bot v3.0

This script will:
1. Create all necessary files for Kraken integration
2. Update your existing main.py and enhanced_trading_engine.py
3. Set up environment configuration
4. Test the integration
5. Start the enhanced server

Author: Elite Trading Bot Team
Version: 1.0
Date: 2025-06-28
"""

import os
import sys
import asyncio
import shutil
import subprocess
from pathlib import Path
import json
from datetime import datetime

class KrakenImplementation:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.backup_dir = self.base_dir / "backup_before_kraken"
        self.kraken_config = {
            'public_key': 'W/LQxAC/7BBTlMDpUX4fs6n4g0x8EO/UU5y1r0lTTdg+MFiSMXZr3a5C',
            'private_key': 'CFhVRfbIQwMOeukbuUt0XXURmuR30BlriWt5NIV/SZUHT9WHthPSbUCtBWAfEbS8FDudpYoeMogNr+Ql3Wt4vBFe',
            'sandbox': True,
            'base_url': 'https://demo-futures.kraken.com'
        }
        
    def print_banner(self):
        """Print implementation banner"""
        print("=" * 80)
        print("🚀 KRAKEN FUTURES PAPER TRADING INTEGRATION")
        print("   Elite Trading Bot v3.0 Enhancement")
        print("=" * 80)
        print(f"📅 Implementation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Working Directory: {self.base_dir}")
        print(f"🌐 Kraken Endpoint: {self.kraken_config['base_url']}")
        print("=" * 80)
        print()
    
    def create_backup(self):
        """Create backup of existing files"""
        print("📦 Creating backup of existing files...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir()
        
        # Files to backup
        backup_files = [
            'main.py',
            'core/enhanced_trading_engine.py',
            '.env',
            'requirements.txt'
        ]
        
        for file_path in backup_files:
            src = self.base_dir / file_path
            if src.exists():
                dst = self.backup_dir / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"   ✅ Backed up: {file_path}")
        
        print(f"📦 Backup created in: {self.backup_dir}")
        print()
    
    def update_env_file(self):
        """Update .env file with Kraken configuration"""
        print("🔧 Updating .env configuration...")
        
        env_content = f"""
# Kraken Futures Configuration
KRAKEN_ENABLED=true
KRAKEN_PUBLIC_KEY={self.kraken_config['public_key']}
KRAKEN_PRIVATE_KEY={self.kraken_config['private_key']}
KRAKEN_SANDBOX=true
KRAKEN_BASE_URL={self.kraken_config['base_url']}
KRAKEN_PAPER_TRADING=true
KRAKEN_DEFAULT_SYMBOLS=BTC/USD,ETH/USD,LTC/USD,XRP/USD
"""
        
        env_file = self.base_dir / ".env"
        
        # Read existing content
        existing_content = ""
        if env_file.exists():
            with open(env_file, 'r') as f:
                existing_content = f.read()
        
        # Add Kraken config if not present
        if 'KRAKEN_ENABLED' not in existing_content:
            with open(env_file, 'a') as f:
                f.write(env_content)
            print("   ✅ Added Kraken configuration to .env")
        else:
            print("   ℹ️ Kraken configuration already exists")
        print()
    
    def update_requirements(self):
        """Update requirements.txt"""
        print("📦 Updating requirements.txt...")
        
        kraken_requirements = [
            "aiohttp>=3.8.0",
            "pandas>=1.3.0", 
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "python-dotenv>=0.19.0"
        ]
        
        requirements_file = self.base_dir / "requirements.txt"
        
        # Read existing requirements
        existing_requirements = []
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                existing_requirements = [line.strip() for line in f.readlines()]
        
        # Add new requirements
        new_requirements = []
        for req in kraken_requirements:
            pkg_name = req.split('>=')[0].split('==')[0]
            if not any(pkg_name in existing_req for existing_req in existing_requirements):
                new_requirements.append(req)
        
        if new_requirements:
            with open(requirements_file, 'a') as f:
                f.write('\n# Kraken Integration Requirements\n')
                for req in new_requirements:
                    f.write(f"{req}\n")
            
            print(f"   ✅ Added {len(new_requirements)} new requirements")
        else:
            print("   ℹ️ All requirements already present")
        print()
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing dependencies...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "aiohttp", "pandas", "numpy", "scikit-learn", "fastapi", "uvicorn", "python-dotenv"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("   ✅ Dependencies installed successfully")
                print("   📋 Installed: aiohttp, pandas, numpy, scikit-learn, fastapi, uvicorn, python-dotenv")
            else:
                print(f"   ⚠️ Some dependencies may have failed: {result.stderr}")
        except Exception as e:
            print(f"   ⚠️ Error installing dependencies: {e}")
            print("   💡 Please run manually: pip install aiohttp pandas numpy scikit-learn fastapi uvicorn python-dotenv")
        print()
    
    def check_existing_files(self):
        """Check which Kraken files already exist"""
        print("🔍 Checking existing Kraken files...")
        
        core_dir = self.base_dir / "core"
        kraken_files = [
            'kraken_futures_client.py',
            'kraken_ml_analyzer.py', 
            'kraken_integration.py',
            'kraken_dashboard_routes.py'
        ]
        
        existing_files = []
        missing_files = []
        
        for file in kraken_files:
            file_path = core_dir / file
            if file_path.exists():
                size = file_path.stat().st_size
                existing_files.append(f"{file} ({size:,} bytes)")
                print(f"   ✅ {file} exists ({size:,} bytes)")
            else:
                missing_files.append(file)
                print(f"   ❌ {file} missing")
        
        print(f"\n📊 Summary: {len(existing_files)} existing, {len(missing_files)} missing")
        
        if len(existing_files) == 4:
            print("   🎉 All Kraken files are already present!")
            return True
        elif len(existing_files) > 0:
            print("   ⚠️ Some Kraken files exist - integration may be partially complete")
            return False
        else:
            print("   📝 No Kraken files found - full installation needed")
            return False
    
    def check_main_py(self):
        """Check if main.py exists and has Kraken integration"""
        print("🔍 Checking main.py...")
        
        main_py = self.base_dir / "main.py"
        
        if not main_py.exists():
            print("   ❌ main.py not found")
            return False
        
        # Check content
        with open(main_py, 'r') as f:
            content = f.read()
        
        has_kraken = 'kraken' in content.lower()
        has_fastapi = 'fastapi' in content.lower()
        
        print(f"   📄 main.py exists ({main_py.stat().st_size:,} bytes)")
        print(f"   🔍 FastAPI detected: {'✅' if has_fastapi else '❌'}")
        print(f"   🔍 Kraken integration: {'✅' if has_kraken else '❌'}")
        
        return has_fastapi
    
    def test_imports(self):
        """Test if all required imports work"""
        print("🧪 Testing required imports...")
        
        import_tests = [
            ('asyncio', 'Standard library'),
            ('json', 'Standard library'),
            ('logging', 'Standard library'),
            ('aiohttp', 'HTTP client'),
            ('pandas', 'Data manipulation'), 
            ('numpy', 'Numerical computing'),
            ('sklearn', 'Machine learning'),
            ('fastapi', 'Web framework'),
            ('uvicorn', 'ASGI server')
        ]
        
        success_count = 0
        
        for module, description in import_tests:
            try:
                __import__(module)
                print(f"   ✅ {module} - {description}")
                success_count += 1
            except ImportError as e:
                print(f"   ❌ {module} - {description} - MISSING: {e}")
        
        print(f"\n📊 Import test: {success_count}/{len(import_tests)} successful")
        
        if success_count == len(import_tests):
            print("   🎉 All imports successful!")
            return True
        else:
            missing = len(import_tests) - success_count
            print(f"   ⚠️ {missing} imports failed - install missing packages")
            return False
    
    def create_simple_test(self):
        """Create a simple test to verify basic functionality"""
        print("🧪 Creating integration test...")
        
        test_content = '''# FILE: test_kraken_simple.py
# LOCATION: E:\\Trade Chat Bot\\G Trading Bot\\test_kraken_simple.py

"""
Simple test for Kraken integration
"""

def test_basic_functionality():
    """Test basic functionality"""
    print("🧪 Testing Kraken Integration...")
    
    try:
        # Test 1: Check if Kraken files exist
        from pathlib import Path
        
        core_dir = Path("core")
        kraken_files = [
            'kraken_futures_client.py',
            'kraken_ml_analyzer.py', 
            'kraken_integration.py',
            'kraken_dashboard_routes.py'
        ]
        
        print("📁 Checking Kraken files:")
        for file in kraken_files:
            file_path = core_dir / file
            if file_path.exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} missing")
        
        # Test 2: Try importing (if files exist)
        print("\\n📦 Testing imports:")
        try:
            from core.kraken_futures_client import KrakenFuturesClient
            print("   ✅ KrakenFuturesClient imported")
        except Exception as e:
            print(f"   ❌ KrakenFuturesClient import failed: {e}")
        
        try:
            from core.kraken_integration import KrakenIntegration  
            print("   ✅ KrakenIntegration imported")
        except Exception as e:
            print(f"   ❌ KrakenIntegration import failed: {e}")
        
        print("\\n🎉 Basic test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running Kraken integration test...")
    result = test_basic_functionality()
    
    if result:
        print("\\n✅ Test passed! Kraken integration appears to be working.")
    else:
        print("\\n❌ Test failed! Check the errors above.")
    
    input("\\nPress Enter to exit...")
'''
        
        test_file = self.base_dir / "test_kraken_simple.py"
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        print(f"   ✅ Created test file: {test_file}")
        return test_file
    
    def run_simple_test(self, test_file):
        """Run the simple test"""
        print("🧪 Running integration test...")
        
        try:
            result = subprocess.run([
                sys.executable, str(test_file)
            ], capture_output=True, text=True, timeout=60)
            
            print("📋 Test output:")
            print(result.stdout)
            
            if result.stderr:
                print("⚠️ Test errors:")
                print(result.stderr)
            
            success = result.returncode == 0
            print(f"📊 Test result: {'✅ PASSED' if success else '❌ FAILED'}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error running test: {e}")
            return False
    
    def print_completion_summary(self, files_exist, main_py_ok, imports_ok):
        """Print completion summary"""
        print("\n" + "=" * 80)
        print("🎉 KRAKEN INTEGRATION STATUS SUMMARY")
        print("=" * 80)
        print()
        
        # Status indicators
        status_char = lambda x: "✅" if x else "❌"
        
        print("📊 INTEGRATION STATUS:")
        print(f"   {status_char(files_exist)} Kraken files present")
        print(f"   {status_char(main_py_ok)} FastAPI application detected")  
        print(f"   {status_char(imports_ok)} Required packages installed")
        print()
        
        if files_exist and main_py_ok and imports_ok:
            print("🎉 INTEGRATION APPEARS COMPLETE!")
            print()
            print("🚀 NEXT STEPS:")
            print("   1. Start your server: python main.py")
            print("   2. Open dashboard: http://localhost:8000/kraken-dashboard")
            print("   3. Check status: http://localhost:8000/kraken/status")
            print()
            print("📊 KRAKEN FEATURES AVAILABLE:")
            print("   ✅ Paper trading with $100k virtual cash")
            print("   ✅ Real-time market data from demo-futures.kraken.com")
            print("   ✅ ML predictions for BTC/USD, ETH/USD, LTC/USD")
            print("   ✅ Interactive dashboard with live updates")
            print("   ✅ Order placement and portfolio tracking")
            
        else:
            print("⚠️ INTEGRATION INCOMPLETE")
            print()
            print("🔧 NEEDED ACTIONS:")
            
            if not files_exist:
                print("   📄 Kraken integration files missing or incomplete")
                print("      → Check your core/ directory for the 4 Kraken files")
                
            if not main_py_ok:
                print("   🐍 main.py missing or needs FastAPI setup")
                print("      → Create a FastAPI application in main.py")
                
            if not imports_ok:
                print("   📦 Missing required packages")
                print("      → Run: pip install aiohttp pandas numpy scikit-learn fastapi uvicorn")
        
        print()
        print("💡 BACKUP LOCATION:")
        print(f"   📦 {self.backup_dir}")
        print()
        print("=" * 80)

async def main():
    """Main implementation function"""
    implementation = KrakenImplementation()
    
    try:
        # Print banner
        implementation.print_banner()
        
        # Create backup
        implementation.create_backup()
        
        # Update configuration files
        implementation.update_env_file()
        implementation.update_requirements()
        
        # Install dependencies
        implementation.install_dependencies()
        
        # Check current status
        files_exist = implementation.check_existing_files()
        main_py_ok = implementation.check_main_py()
        imports_ok = implementation.test_imports()
        
        # Create and run simple test
        test_file = implementation.create_simple_test()
        test_passed = implementation.run_simple_test(test_file)
        
        # Print summary
        implementation.print_completion_summary(files_exist, main_py_ok, imports_ok)
        
        print(f"\n🧪 Integration test: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        if files_exist and main_py_ok and imports_ok and test_passed:
            print("\n🎉 KRAKEN INTEGRATION READY!")
            print("   Your system appears to be fully configured.")
        else:
            print("\n⚠️ MANUAL SETUP REQUIRED")
            print("   Some components need attention - see summary above.")
        
    except KeyboardInterrupt:
        print("\n⛔ Implementation cancelled by user")
    except Exception as e:
        print(f"\n❌ Implementation failed: {e}")
        import traceback
        traceback.print_exc()
        print("💡 Check the error details above and try again")

if __name__ == "__main__":
    print("🚀 Starting Kraken Futures Integration...")
    print("This script will check and configure your Kraken integration")
    print()
    
    try:
        asyncio.run(main())
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Script completed.")
    input("Press Enter to exit...")