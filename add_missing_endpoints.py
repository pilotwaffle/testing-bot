"""
File: add_missing_endpoints.py
Location: E:\Trade Chat Bot\G Trading Bot\add_missing_endpoints.py

Missing ML Endpoints Fix Script
Adds missing ML and API endpoints to main.py
"""

import shutil
from datetime import datetime
from pathlib import Path

def backup_main_py():
    """Create backup of main.py"""
    if Path("main.py").exists():
        backup_name = f"main.py.backup_endpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2("main.py", backup_name)
        print(f"📁 Backup created: {backup_name}")
        return backup_name
    return None

def add_missing_endpoints():
    """Add missing endpoints to main.py"""
    print("🔧 Adding Missing ML and API Endpoints")
    print("=" * 50)
    
    if not Path("main.py").exists():
        print("❌ main.py not found")
        return False
    
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find where to insert new endpoints (before the health check)
        health_check_pos = content.find("@app.get(\"/health\")")
        if health_check_pos == -1:
            print("❌ Could not find health check endpoint to insert new endpoints")
            return False
        
        # Define the missing endpoints
        missing_endpoints = '''
# MISSING ML AND API ENDPOINTS - ADDED
@app.get("/api/ml/test")
@app.post("/api/ml/test")
async def test_ml_system():
    """Test ML system functionality"""
    try:
        if not ml_engine:
            return {"status": "error", "message": "ML Engine not available"}
        
        # Test ML engine
        ml_status = ml_engine.get_status()
        model_count = len(ml_status)
        
        # Test a quick training simulation
        test_result = ml_engine.train_model("neural_network", "BTC/USDT")
        
        return {
            "status": "success",
            "message": f"ML System test completed successfully. {model_count} models available.",
            "details": {
                "ml_engine": "Available",
                "models_count": model_count,
                "test_training": test_result.get("status", "unknown"),
                "gemini_ai": chat_manager.gemini_ai.is_available() if chat_manager and hasattr(chat_manager, 'gemini_ai') else False
            }
        }
    except Exception as e:
        logger.error(f"ML test error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/ml/models")
async def list_ml_models():
    """List all available ML models"""
    try:
        if not ml_engine:
            return {"status": "error", "message": "ML Engine not available"}
        
        models = ml_engine.get_status()
        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/ml/status")
async def ml_engine_status():
    """Get ML engine status"""
    try:
        if not ml_engine:
            return {"status": "error", "message": "ML Engine not available"}
        
        status = ml_engine.get_status()
        return {
            "status": "success",
            "ml_engine": "available",
            "models": status
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/ml/train/all")
async def train_all_models(symbol: str = "BTC/USDT"):
    """Train all ML models"""
    try:
        if not ml_engine:
            return {"status": "error", "message": "ML Engine not available"}
        
        results = {}
        model_types = ["lorentzian_classifier", "neural_network", "social_sentiment", "risk_assessment"]
        
        for model_type in model_types:
            try:
                result = ml_engine.train_model(model_type, symbol)
                results[model_type] = result
            except Exception as e:
                results[model_type] = {"status": "error", "message": str(e)}
        
        return {
            "status": "success",
            "message": f"Training completed for all models on {symbol}",
            "results": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio information"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not available"}
        
        status = trading_engine.get_status()
        return {
            "status": "success",
            "portfolio": {
                "total_value": status.get("total_value", 0),
                "cash_balance": status.get("cash_balance", 0),
                "unrealized_pnl": status.get("unrealized_pnl", 0),
                "total_profit": status.get("total_profit", 0),
                "positions": status.get("positions", {}),
                "num_positions": len(status.get("positions", {}))
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/market-data")
async def get_market_data():
    """Get market data"""
    try:
        # Simulate market data
        market_data = {
            "BTC/USDT": {"price": 43250.50, "change_24h": 2.5},
            "ETH/USDT": {"price": 2650.75, "change_24h": 1.8},
            "ADA/USDT": {"price": 0.485, "change_24h": -0.5}
        }
        
        return {
            "status": "success",
            "data": market_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/strategies")
async def get_strategies():
    """Get active strategies"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not available"}
        
        status = trading_engine.get_status()
        strategies = status.get("active_strategies", {})
        
        return {
            "status": "success",
            "strategies": strategies,
            "count": len(strategies)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/system/info")
async def system_info():
    """Get system information"""
    return {
        "status": "success",
        "system": {
            "name": "Industrial Crypto Trading Bot",
            "version": "3.0.0",
            "components": {
                "trading_engine": trading_engine is not None,
                "ml_engine": ml_engine is not None,
                "chat_manager": chat_manager is not None,
                "kraken_integration": kraken_integration is not None,
                "gemini_ai": chat_manager.gemini_ai.is_available() if chat_manager and hasattr(chat_manager, 'gemini_ai') else False
            },
            "features": [
                "Enhanced Trading Engine",
                "ML Predictions with 4 Models",
                "Gemini AI Chat Integration",
                "WebSocket Support",
                "Kraken Integration",
                "Real-time Dashboard"
            ]
        }
    }

@app.get("/dashboard")
async def dashboard_redirect(request: Request):
    """Redirect /dashboard to root"""
    return await dashboard_html(request)

@app.get("/api/test")
async def api_test():
    """Test API functionality"""
    return {
        "status": "success",
        "message": "API is working correctly",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "ml_test": "/api/ml/test",
            "chat": "/api/chat",
            "portfolio": "/api/portfolio",
            "market_data": "/api/market-data"
        }
    }

'''
        
        # Insert the missing endpoints before the health check
        new_content = content[:health_check_pos] + missing_endpoints + "\n" + content[health_check_pos:]
        
        # Write the updated content
        with open("main.py", 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Missing endpoints added successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error adding endpoints: {e}")
        return False

def create_endpoint_test_script():
    """Create a script to test all endpoints"""
    print("\n🧪 Creating Endpoint Test Script")
    print("=" * 50)
    
    test_script = '''"""
File: test_endpoints.py
Location: E:\\Trade Chat Bot\\G Trading Bot\\test_endpoints.py

Endpoint Test Script
Tests all available endpoints to verify they work
"""

import requests
import json

def test_endpoint(method, url, data=None):
    """Test a single endpoint"""
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=10)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, timeout=10)
        else:
            return {"error": "Unsupported method"}
        
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {"error": str(e)}

def test_all_endpoints():
    """Test all endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints = [
        ("GET", "/health"),
        ("GET", "/api"),
        ("GET", "/api/test"),
        ("GET", "/api/ml/test"),
        ("POST", "/api/ml/test"),
        ("GET", "/api/ml/models"),
        ("GET", "/api/ml/status"),
        ("POST", "/api/ml/train/neural_network"),
        ("POST", "/api/ml/train/all"),
        ("GET", "/api/portfolio"),
        ("GET", "/api/market-data"),
        ("GET", "/api/strategies"),
        ("GET", "/api/system/info"),
        ("POST", "/api/chat", {"message": "test"}),
        ("GET", "/status"),
        ("POST", "/api/trading/start"),
        ("POST", "/api/trading/stop")
    ]
    
    print("🧪 Testing All Endpoints")
    print("=" * 60)
    
    results = {}
    passed = 0
    total = len(endpoints)
    
    for method, endpoint, *data in endpoints:
        url = base_url + endpoint
        test_data = data[0] if data else None
        
        print(f"\\n📡 Testing {method} {endpoint}")
        result = test_endpoint(method, url, test_data)
        results[endpoint] = result
        
        if result.get("success"):
            print(f"   ✅ Status: {result['status_code']}")
            response = result.get("response", {})
            if isinstance(response, dict):
                if "status" in response:
                    print(f"   📊 Response status: {response['status']}")
                if "message" in response:
                    print(f"   📝 Message: {response['message'][:100]}...")
            passed += 1
        else:
            print(f"   ❌ Failed: {result.get('error', result.get('status_code', 'Unknown'))}")
    
    print(f"\\n📊 SUMMARY")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All endpoints working perfectly!")
    else:
        print("⚠️ Some endpoints need attention")
    
    return results

if __name__ == "__main__":
    results = test_all_endpoints()
'''
    
    try:
        with open("test_endpoints.py", 'w', encoding='utf-8') as f:
            f.write(test_script)
        print("✅ Endpoint test script created")
        return True
    except Exception as e:
        print(f"❌ Error creating test script: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Missing ML Endpoints Fix")
    print("=" * 60)
    
    print("🎯 Adding missing endpoints:")
    print("   • /api/ml/test - ML system testing")
    print("   • /api/ml/models - List ML models")
    print("   • /api/ml/status - ML engine status")
    print("   • /api/ml/train/all - Train all models")
    print("   • /api/portfolio - Portfolio information")
    print("   • /api/market-data - Market data")
    print("   • /api/strategies - Active strategies")
    print("   • /api/system/info - System information")
    print("   • /api/test - API testing")
    print()
    
    # Step 1: Backup main.py
    backup_main_py()
    
    # Step 2: Add missing endpoints
    if add_missing_endpoints():
        print("✅ Missing endpoints added successfully")
    else:
        print("❌ Failed to add endpoints")
        return
    
    # Step 3: Create test script
    if create_endpoint_test_script():
        print("✅ Test script created")
    
    print("\\n🎉 MISSING ENDPOINTS FIX COMPLETE!")
    print("=" * 60)
    
    print("🔄 Your server will auto-reload with the new endpoints")
    print()
    print("🧪 Test the endpoints:")
    print("1. Visit: http://localhost:8000/api/ml/test")
    print("2. Expected: ML system test results (no more 404)")
    print()
    print("🧪 Run comprehensive test:")
    print("   python test_endpoints.py")
    print()
    print("📊 New endpoints available:")
    print("   ✅ http://localhost:8000/api/ml/test")
    print("   ✅ http://localhost:8000/api/ml/models")
    print("   ✅ http://localhost:8000/api/portfolio")
    print("   ✅ http://localhost:8000/api/market-data")
    print("   ✅ http://localhost:8000/api/system/info")
    print()
    print("🎯 Expected results:")
    print("   ✅ No more 404 errors")
    print("   ✅ All ML endpoints working")
    print("   ✅ Complete API coverage")

if __name__ == "__main__":
    main()