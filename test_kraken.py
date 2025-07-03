"""
File: test_kraken.py
Location: E:\Trade Chat Bot\G Trading Bot\test_kraken.py

Kraken Integration Test Script
Tests Kraken integration independently
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def test_kraken_integration():
    """Test Kraken integration step by step"""
    print("🧪 Testing Kraken Integration")
    print("=" * 50)
    
    try:
        # Test 1: Import kraken integration
        print("Test 1: Importing Kraken integration...")
        from core.kraken_integration import KrakenIntegration
        print("✅ KrakenIntegration imported successfully")
        
        # Test 2: Initialize Kraken integration (sandbox mode)
        print("\nTest 2: Initializing Kraken integration...")
        kraken = KrakenIntegration(sandbox=True)
        print("✅ KrakenIntegration initialized successfully")
        
        # Test 3: Check status
        print("\nTest 3: Checking Kraken status...")
        status = kraken.get_status()
        print(f"✅ Status: {status}")
        
        # Test 4: Test connection
        print("\nTest 4: Testing connection...")
        is_connected = kraken.test_connection()
        print(f"✅ Connection test: {'PASSED' if is_connected else 'FAILED'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   → Check if core/kraken_integration.py exists")
        print("   → Check for missing dependencies")
        return False
        
    except Exception as e:
        print(f"❌ Kraken integration test failed: {e}")
        print(f"   → Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kraken_integration()
    if success:
        print("\n🎉 Kraken integration is working!")
    else:
        print("\n❌ Kraken integration needs fixing")
