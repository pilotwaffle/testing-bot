"""
File: simple_chat_diagnostic.py
Location: E:\Trade Chat Bot\G Trading Bot\simple_chat_diagnostic.py

Simple Chat Diagnostic
Focuses specifically on diagnosing the chat API error
"""

import requests
import json
from datetime import datetime

def test_chat_endpoint():
    """Test the chat endpoint that's failing"""
    print("🧪 Testing Chat Endpoint")
    print("=" * 50)
    
    test_messages = ["help", "status", "hello"]
    
    for message in test_messages:
        print(f"\n💬 Testing: '{message}'")
        
        try:
            response = requests.post(
                "http://localhost:8000/api/chat",
                json={"message": message},
                timeout=10
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ Success: {data.get('response', 'No response field')[:100]}...")
                except:
                    print(f"❌ Invalid JSON: {response.text[:200]}...")
            else:
                print(f"❌ Error {response.status_code}: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

def test_other_endpoints():
    """Test other endpoints to see what's working"""
    print("\n🔗 Testing Other Endpoints")
    print("=" * 50)
    
    endpoints = [
        ("/health", "Health Check"),
        ("/api/ml/status", "ML Status"),
        ("/", "Dashboard")
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"{status} {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: Failed - {e}")

def create_chat_test_page():
    """Create a simple test page for chat"""
    print("\n🔧 Creating Chat Test Page")
    print("=" * 50)
    
    test_html = '''<!DOCTYPE html>
<html>
<head>
    <title>Chat API Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .test-area { margin: 20px 0; padding: 20px; border: 1px solid #ccc; }
        button { padding: 10px 20px; margin: 5px; }
        .result { margin-top: 10px; padding: 10px; background: #f5f5f5; }
    </style>
</head>
<body>
    <h1>🧪 Chat API Test Page</h1>
    
    <div class="test-area">
        <h3>Quick Tests</h3>
        <button onclick="testMessage('help')">Test 'help'</button>
        <button onclick="testMessage('status')">Test 'status'</button>
        <button onclick="testMessage('hello')">Test 'hello'</button>
        <div id="result" class="result"></div>
    </div>
    
    <div class="test-area">
        <h3>Custom Test</h3>
        <input type="text" id="customMessage" placeholder="Type your message..." style="width: 300px; padding: 5px;">
        <button onclick="testCustomMessage()">Send</button>
    </div>

    <script>
        async function testMessage(message) {
            const result = document.getElementById('result');
            result.innerHTML = `🧪 Testing: "${message}"...`;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                result.innerHTML = `
                    <strong>Status:</strong> ${response.status}<br>
                    <strong>Message:</strong> "${message}"<br>
                `;
                
                if (response.ok) {
                    const data = await response.json();
                    result.innerHTML += `<strong>✅ Response:</strong> ${data.response || 'No response field'}<br>`;
                    result.innerHTML += `<strong>📊 Full Data:</strong> <pre>${JSON.stringify(data, null, 2)}</pre>`;
                } else {
                    const errorText = await response.text();
                    result.innerHTML += `<strong>❌ Error:</strong> ${errorText}<br>`;
                }
                
            } catch (error) {
                result.innerHTML += `<strong>❌ Exception:</strong> ${error.message}<br>`;
            }
        }
        
        function testCustomMessage() {
            const message = document.getElementById('customMessage').value;
            if (message.trim()) {
                testMessage(message);
            }
        }
        
        // Auto-test on page load
        setTimeout(() => {
            testMessage('help');
        }, 1000);
    </script>
</body>
</html>'''
    
    try:
        with open("templates/chat_test.html", 'w', encoding='utf-8') as f:
            f.write(test_html)
        print("✅ Created templates/chat_test.html")
        print("🌐 Visit: http://localhost:8000/chat_test.html")
        print("   (You may need to add route to main.py)")
        return True
    except Exception as e:
        print(f"❌ Failed to create test page: {e}")
        return False

def check_server_console():
    """Guide for checking server console"""
    print("\n📋 Server Console Check Guide")
    print("=" * 50)
    
    print("🔍 In your server terminal, when you send a chat message, look for:")
    print()
    print("✅ GOOD - Should see:")
    print("   INFO: 127.0.0.1:xxxxx - \"POST /api/chat HTTP/1.1\" 200 OK")
    print()
    print("❌ BAD - Probably seeing:")
    print("   ERROR: Exception in ASGI application")
    print("   INFO: 127.0.0.1:xxxxx - \"POST /api/chat HTTP/1.1\" 500 Internal Server Error")
    print()
    print("🔍 Error types to look for:")
    print("   • AttributeError: 'EnhancedChatManager' object has no attribute...")
    print("   • TypeError: process_message() missing required argument...")
    print("   • JSONDecodeError: ...")
    print("   • Any Gemini AI related errors")
    print()
    print("💡 Copy and paste any error stack trace for analysis!")

def main():
    """Main diagnostic function"""
    print("🔧 Simple Chat Diagnostic")
    print("=" * 60)
    
    print("🎯 Goal: Find out why chat returns 'Error: Could not get response from server'")
    print()
    print("📊 Known status:")
    print("   ✅ ML Training: Working perfectly (4 models)")
    print("   ✅ Enhanced Chat Manager: Initialized with dependencies")
    print("   ✅ Gemini AI: Initialized with gemini-1.5-flash")
    print("   ❌ Chat API: Failing with server error")
    print()
    
    # Run focused tests
    test_other_endpoints()
    test_chat_endpoint()
    create_chat_test_page()
    check_server_console()
    
    print("\n🎯 SUMMARY")
    print("=" * 60)
    print()
    print("If other endpoints work but chat fails, the issue is specifically")
    print("in the /api/chat endpoint or EnhancedChatManager.process_message()")
    print()
    print("🔧 Next steps:")
    print("1. Check server terminal for error stack trace when sending 'help'")
    print("2. Visit http://localhost:8000/chat_test.html for detailed testing")
    print("3. Share any error messages for targeted fix")
    print()
    print("💡 Your engines are loaded correctly - this is just an API integration issue!")

if __name__ == "__main__":
    main()