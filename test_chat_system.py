"""
File: test_chat_system.py
Location: E:\Trade Chat Bot\G Trading Bot\test_chat_system.py

Chat System Diagnostic
Tests exactly what's happening with the chat functionality
"""

import os
import requests
import json
import asyncio
from pathlib import Path

def test_environment_variables():
    """Test if environment variables are loaded correctly"""
    print("🔍 Testing Environment Variables")
    print("=" * 50)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if api_key:
            print(f"✅ GOOGLE_AI_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
        else:
            print("❌ GOOGLE_AI_API_KEY not found")
        
        enabled = os.getenv('GOOGLE_AI_ENABLED')
        print(f"📊 GOOGLE_AI_ENABLED: {enabled}")
        
        return api_key is not None
        
    except Exception as e:
        print(f"❌ Error loading environment: {e}")
        return False

def test_gemini_directly():
    """Test Gemini AI directly"""
    print("\n🧪 Testing Gemini AI Directly")
    print("=" * 50)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from ai.gemini_ai import GeminiAI
        
        gemini = GeminiAI()
        print(f"📊 Gemini Status: {gemini.get_status()}")
        
        if gemini.is_available():
            print("✅ Gemini AI is available")
            
            # Test chat
            response = asyncio.run(gemini.chat("Hello! Test message."))
            if response:
                print(f"✅ Direct chat test successful")
                print(f"📝 Response: {response[:100]}...")
                return True
            else:
                print("❌ Direct chat test failed - no response")
                return False
        else:
            print("❌ Gemini AI not available")
            return False
            
    except Exception as e:
        print(f"❌ Gemini AI direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_manager():
    """Test Enhanced Chat Manager"""
    print("\n🧪 Testing Enhanced Chat Manager")
    print("=" * 50)
    
    try:
        from core.enhanced_trading_engine import EnhancedTradingEngine
        from core.ml_engine import MLEngine
        from ai.chat_manager import EnhancedChatManager
        
        # Initialize components
        trading_engine = EnhancedTradingEngine()
        ml_engine = MLEngine()
        
        chat_manager = EnhancedChatManager(
            trading_engine=trading_engine,
            ml_engine=ml_engine,
            data_fetcher=None,
            notification_manager=None
        )
        
        print("✅ Enhanced Chat Manager initialized")
        
        # Test process_message
        response = asyncio.run(chat_manager.process_message("Hello! Test message."))
        print(f"📊 Response type: {type(response)}")
        print(f"📊 Response: {response}")
        
        if isinstance(response, dict):
            if 'response' in response:
                print(f"✅ Chat manager test successful")
                print(f"📝 Response content: {response['response'][:100]}...")
                return True
            else:
                print(f"❌ Response missing 'response' key")
                print(f"📊 Available keys: {list(response.keys())}")
                return False
        else:
            print(f"❌ Response is not a dict: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Chat manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_api_endpoint():
    """Test the chat API endpoint"""
    print("\n🧪 Testing Chat API Endpoint")
    print("=" * 50)
    
    try:
        test_data = {
            "message": "Hello! This is a test message.",
            "session_id": "test_session"
        }
        
        response = requests.post(
            "http://localhost:8000/api/chat",
            json=test_data,
            timeout=10
        )
        
        print(f"📊 HTTP Status: {response.status_code}")
        print(f"📊 Content-Type: {response.headers.get('content-type')}")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                print(f"📊 Response data: {response_data}")
                
                if 'response' in response_data:
                    print(f"✅ API response successful")
                    print(f"📝 Response: {response_data['response']}")
                    return True
                else:
                    print(f"❌ API response missing 'response' field")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"📄 Raw response: {response.text}")
                return False
        else:
            print(f"❌ API returned error: {response.status_code}")
            print(f"📄 Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_websocket_chat():
    """Test WebSocket chat"""
    print("\n🧪 Testing WebSocket Chat")
    print("=" * 50)
    
    try:
        import websocket
        import threading
        import time
        
        responses = []
        
        def on_message(ws, message):
            print(f"📨 WebSocket message: {message}")
            responses.append(message)
        
        def on_error(ws, error):
            print(f"❌ WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("🔌 WebSocket connection closed")
        
        def on_open(ws):
            print("🔌 WebSocket connection opened")
            
            # Send test message
            test_message = {
                "type": "chat",
                "message": "Hello! WebSocket test.",
                "session_id": "test_ws"
            }
            ws.send(json.dumps(test_message))
        
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            "ws://localhost:8000/ws",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run WebSocket in a thread for 5 seconds
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        time.sleep(5)
        ws.close()
        
        if responses:
            print(f"✅ WebSocket test successful - {len(responses)} responses")
            for response in responses:
                print(f"📝 Response: {response[:100]}...")
            return True
        else:
            print("❌ WebSocket test failed - no responses")
            return False
            
    except ImportError:
        print("⚠️ websocket-client not installed. Install with: pip install websocket-client")
        return False
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def check_frontend_javascript():
    """Check if frontend JavaScript files are correct"""
    print("\n🔍 Checking Frontend JavaScript")
    print("=" * 50)
    
    js_files = [
        "static/js/chat.js",
        "static/js/dashboard.js"
    ]
    
    for js_file in js_files:
        if Path(js_file).exists():
            print(f"✅ {js_file} exists")
            try:
                with open(js_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for key functions
                if 'sendMessage' in content:
                    print(f"   ✅ Contains sendMessage function")
                if 'chat' in content:
                    print(f"   ✅ Contains chat functionality")
                if 'WebSocket' in content:
                    print(f"   ✅ Contains WebSocket code")
                    
            except Exception as e:
                print(f"   ❌ Error reading {js_file}: {e}")
        else:
            print(f"❌ {js_file} missing")

def provide_specific_fix(test_results):
    """Provide specific fix based on test results"""
    print("\n💡 Specific Fix Recommendations")
    print("=" * 50)
    
    if not test_results.get('env_vars', False):
        print("🔧 Environment Variables Issue:")
        print("   → Check if .env file exists and has GOOGLE_AI_API_KEY")
        print("   → Verify dotenv is loading correctly")
        print("   → Try: python -c \"from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('GOOGLE_AI_API_KEY'))\"")
    
    if not test_results.get('gemini_direct', False):
        print("🔧 Gemini AI Issue:")
        print("   → API key not working or incorrect model")
        print("   → Check API key permissions at https://makersuite.google.com/app/apikey")
        print("   → Verify model name is correct (gemini-1.5-flash)")
    
    if not test_results.get('chat_manager', False):
        print("🔧 Chat Manager Issue:")
        print("   → Enhanced Chat Manager not processing messages correctly")
        print("   → Check for missing methods or import errors")
        print("   → Verify all dependencies are initialized")
    
    if not test_results.get('api_endpoint', False):
        print("🔧 API Endpoint Issue:")
        print("   → /api/chat endpoint not working")
        print("   → Check server logs for errors")
        print("   → Verify route is properly defined")

def main():
    """Main diagnostic function"""
    print("🔧 Chat System Diagnostic")
    print("=" * 80)
    print("🎯 Diagnosing 'Error: undefined' in chat")
    print("=" * 80)
    
    test_results = {}
    
    # Run all tests
    test_results['env_vars'] = test_environment_variables()
    test_results['gemini_direct'] = test_gemini_directly()
    test_results['chat_manager'] = test_chat_manager()
    test_results['api_endpoint'] = test_chat_api_endpoint()
    test_results['websocket'] = test_websocket_chat()
    
    # Check frontend
    check_frontend_javascript()
    
    # Provide specific fix
    provide_specific_fix(test_results)
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"✅ Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All chat tests passed! Issue might be in frontend JavaScript.")
    else:
        print("❌ Issues found in chat system backend.")
        
        failing_tests = [test for test, result in test_results.items() if not result]
        print(f"🔧 Failing components: {', '.join(failing_tests)}")
    
    print("\n📋 Next Steps:")
    print("1. Fix any failing components above")
    print("2. Restart server: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("3. Test chat again")

if __name__ == "__main__":
    main()