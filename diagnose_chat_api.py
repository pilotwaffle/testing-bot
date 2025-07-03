"""
File: diagnose_chat_api.py
Location: E:\Trade Chat Bot\G Trading Bot\diagnose_chat_api.py

Diagnose Chat API Error
Tests the /api/chat endpoint to identify why it's returning server errors
"""

import requests
import json
from datetime import datetime

def test_chat_api_direct():
    """Test the chat API directly to see the exact error"""
    print("🧪 Testing Chat API Direct")
    print("=" * 50)
    
    test_messages = ["help", "status", "hello", "test"]
    
    for message in test_messages:
        print(f"\n💬 Testing message: '{message}'")
        
        try:
            response = requests.post(
                "http://localhost:8000/api/chat",
                json={"message": message},
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📊 Status Code: {response.status_code}")
            print(f"📊 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ Response JSON: {json.dumps(data, indent=2)}")
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON response: {response.text}")
            else:
                print(f"❌ Error Response:")
                print(f"   Status: {response.status_code}")
                print(f"   Text: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        print("-" * 30)

def test_server_health():
    """Test basic server health"""
    print("\n🏥 Testing Server Health")
    print("=" * 50)
    
    endpoints_to_test = [
        ("/health", "Health Check"),
        ("/", "Dashboard"),
        ("/api", "API Info")
    ]
    
    for endpoint, name in endpoints_to_test:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
            print(f"✅ {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {e}")

def test_chat_manager_methods():
    """Test if we can access the chat manager methods via other endpoints"""
    print("\n🤖 Testing Chat Manager Methods")
    print("=" * 50)
    
    # Test if we can create a simple endpoint to test chat manager
    test_script = '''
# Quick test script - save as test_chat_direct.py
import sys
sys.path.append(".")

try:
    from ai.chat_manager import EnhancedChatManager
    from core.enhanced_trading_engine import EliteTradingEngine
    from core.ml_engine import MLEngine
    from core.data_fetcher import DataFetcher
    
    print("🔧 Testing chat manager initialization...")
    
    # Initialize dependencies
    trading_engine = EliteTradingEngine()
    ml_engine = MLEngine() 
    data_fetcher = DataFetcher()
    
    print("✅ Dependencies initialized")
    
    # Initialize chat manager
    chat_manager = EnhancedChatManager(
        trading_engine=trading_engine,
        ml_engine=ml_engine, 
        data_fetcher=data_fetcher
    )
    
    print("✅ Chat manager initialized")
    
    # Test process_message method
    test_messages = ["help", "status", "hello"]
    
    for message in test_messages:
        try:
            response = chat_manager.process_message(message)
            print(f"✅ '{message}' -> '{response[:100]}...'")
        except Exception as e:
            print(f"❌ '{message}' -> Error: {e}")
            
except Exception as e:
    print(f"❌ Chat manager test failed: {e}")
    import traceback
    traceback.print_exc()
'''
    
    try:
        with open("test_chat_direct.py", 'w', encoding='utf-8') as f:
            f.write(test_script)
        print("✅ Created test_chat_direct.py")
        print("🧪 Run this to test chat manager directly:")
        print("   python test_chat_direct.py")
    except Exception as e:
        print(f"❌ Failed to create test script: {e}")

def check_server_logs():
    """Check what we should look for in server logs"""
    print("\n📋 Server Log Analysis Guide")
    print("=" * 50)
    
    print("🔍 When you send a 'help' message, look for these in server logs:")
    print()
    print("✅ Expected successful pattern:")
    print("   INFO: 127.0.0.1:xxxxx - 'POST /api/chat HTTP/1.1' 200 OK")
    print()
    print("❌ Expected error patterns:")
    print("   ERROR: Exception in ASGI application")
    print("   INFO: 127.0.0.1:xxxxx - 'POST /api/chat HTTP/1.1' 500 Internal Server Error")
    print()
    print("🔍 Key things to look for:")
    print("   • Does the POST request appear in logs?")
    print("   • Is there a 500 error with stack trace?")
    print("   • Any 'AttributeError' or 'TypeError' messages?")
    print("   • Gemini AI related errors?")
    print()
    print("💡 If you see errors, paste them here for analysis!")

def create_simple_chat_test():
    """Create a simple chat test endpoint"""
    print("\n🔧 Creating Simple Chat Test")
    print("=" * 50)
    
    test_endpoint = '''
# Add this to your main.py to test chat manager directly

@app.post("/api/chat/test")
async def chat_test_endpoint(request: Request):
    """Simple chat test endpoint for debugging"""
    try:
        data = await request.json()
        message = data.get("message", "test")
        
        # Test 1: Basic response
        basic_response = f"Test received: {message}"
        
        # Test 2: Check if chat_manager exists
        if 'chat_manager' not in globals():
            return {
                "status": "error",
                "error": "chat_manager not in globals",
                "basic_response": basic_response
            }
        
        # Test 3: Check chat_manager type
        chat_manager_type = str(type(chat_manager).__name__)
        
        # Test 4: Check if process_message method exists
        has_process_message = hasattr(chat_manager, 'process_message')
        
        # Test 5: Try to call process_message
        if has_process_message:
            try:
                ai_response = chat_manager.process_message(message)
                return {
                    "status": "success",
                    "basic_response": basic_response,
                    "chat_manager_type": chat_manager_type,
                    "has_process_message": has_process_message,
                    "ai_response": ai_response
                }
            except Exception as e:
                return {
                    "status": "partial_success",
                    "basic_response": basic_response,
                    "chat_manager_type": chat_manager_type,
                    "has_process_message": has_process_message,
                    "process_message_error": str(e)
                }
        else:
            return {
                "status": "missing_method",
                "basic_response": basic_response,
                "chat_manager_type": chat_manager_type,
                "has_process_message": has_process_message
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": str(type(e).__name__)
        }
'''
    
    print("🔧 Add this test endpoint to your main.py:")
    print("Then test with: http://localhost:8000/api/chat/test")
    print("POST body: {\"message\": \"help\"}")
    print()
    print("Or save this code and manually add it to main.py")

def main():
    """Main diagnostic function"""
    print("🔧 Chat API Error Diagnosis")
    print("=" * 60)
    
    print("🎯 Issue: Chat engines initialized successfully, but /api/chat fails")
    print()
    print("✅ What's working:")
    print("   ✅ Enhanced Chat Manager initialized with dependencies")
    print("   ✅ Gemini AI initialized successfully")
    print("   ✅ Frontend chat interface loads")
    print()
    print("❌ What's failing:")
    print("   ❌ /api/chat returns server error")
    print("   ❌ Messages don't get processed")
    print()
    
    # Run diagnostics
    test_server_health()
    test_chat_api_direct()
    test_chat_manager_methods()
    create_simple_chat_test()
    check_server_logs()
    
    print("\n🎯 NEXT STEPS:")
    print("=" * 60)
    print()
    print("1. 🧪 Run the direct chat test:")
    print("   python test_chat_direct.py")
    print()
    print("2. 📋 Check your server terminal for errors when you type 'help'")
    print("   Look for any stack traces or error messages")
    print()
    print("3. 🔧 If test_chat_direct.py works but /api/chat fails:")
    print("   The issue is in the FastAPI endpoint, not the chat manager")
    print()
    print("4. 📊 Paste any error messages here for analysis")
    print()
    print("💡 Most likely causes:")
    print("   • process_message() method signature changed")
    print("   • Missing await for async operations")
    print("   • Gemini AI API call failing") 
    print("   • JSON serialization error in response")

if __name__ == "__main__":
    main()