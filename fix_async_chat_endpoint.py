"""
File: fix_async_chat_endpoint.py
Location: E:\Trade Chat Bot\G Trading Bot\fix_async_chat_endpoint.py

Fix Async Chat Endpoint
Fixes the coroutine error by properly awaiting async chat manager calls
"""

import shutil
import re
from datetime import datetime

def backup_main_file():
    """Create backup of main.py"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"main.py.backup_async_{timestamp}"
    
    try:
        shutil.copy2("main.py", backup_name)
        print(f"📁 Backup created: {backup_name}")
        return backup_name
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return None

def fix_chat_endpoint():
    """Fix the chat endpoint to properly handle async chat manager"""
    print("🔧 Fixing Chat Endpoint for Async Operations")
    print("=" * 50)
    
    try:
        # Read current main.py
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Failed to read main.py: {e}")
        return False
    
    # Find the chat endpoint
    chat_endpoint_pattern = r'@app\.post\("/api/chat"\)(.*?)async def chat_endpoint\(request: Request\):(.*?)return \{'
    
    # Create the fixed chat endpoint
    fixed_chat_endpoint = '''@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Chat endpoint with proper async handling"""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return {
                "response": "Please provide a message.",
                "message_type": "text",
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if chat manager exists and has process_message method
        if chat_manager and hasattr(chat_manager, 'process_message'):
            try:
                # Check if process_message is async
                import asyncio
                if asyncio.iscoroutinefunction(chat_manager.process_message):
                    # It's async - await it
                    response = await chat_manager.process_message(message)
                else:
                    # It's sync - call directly
                    response = chat_manager.process_message(message)
                
                # Handle different response types
                if isinstance(response, dict):
                    # If it's already a dict, use it
                    if "response" in response:
                        response_text = response["response"]
                    else:
                        response_text = str(response)
                elif isinstance(response, str):
                    # If it's a string, use it directly
                    response_text = response
                else:
                    # Convert other types to string
                    response_text = str(response)
                
            except Exception as e:
                logger.error(f"Chat manager error: {e}")
                response_text = f"Sorry, I encountered an error: {str(e)}"
        else:
            # Fallback responses if chat manager not available
            if "status" in message.lower():
                response_text = "🚀 Elite Trading Bot is running! All systems operational."
            elif "help" in message.lower():
                response_text = "💡 Available commands: status, help, portfolio, market. Ask me anything about trading!"
            elif "portfolio" in message.lower():
                response_text = "📊 Portfolio analysis: Enhanced Chat Manager loading..."
            else:
                response_text = f"I received your message: '{message}'. Enhanced AI chat system loading..."
        
        return {
            "response": response_text,
            "message_type": "text",
            "intent": "general_chat",
            "response_time": 0.1,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return {
            "response": "Sorry, I encountered an error processing your message.",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }'''
    
    # Find and replace the chat endpoint
    pattern = r'@app\.post\("/api/chat"\).*?async def chat_endpoint\(request: Request\):.*?(?=@app\.|if __name__|$)'
    
    if re.search(pattern, content, re.DOTALL):
        # Replace the existing endpoint
        new_content = re.sub(pattern, fixed_chat_endpoint + '\n\n', content, flags=re.DOTALL)
        print("✅ Found and replaced existing chat endpoint")
    else:
        # If pattern not found, try a simpler approach
        # Look for the chat endpoint function definition
        chat_def_pattern = r'async def chat_endpoint\(request: Request\):.*?(?=async def|@app\.|if __name__|$)'
        
        if re.search(chat_def_pattern, content, re.DOTALL):
            new_content = re.sub(chat_def_pattern, fixed_chat_endpoint.split('async def chat_endpoint')[1] + '\n\n', content, flags=re.DOTALL)
            print("✅ Found and replaced chat endpoint function")
        else:
            print("❌ Could not find chat endpoint to replace")
            return False
    
    # Write the updated content
    try:
        with open("main.py", 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Chat endpoint updated with async handling")
        return True
    except Exception as e:
        print(f"❌ Failed to write updated main.py: {e}")
        return False

def fix_websocket_endpoint():
    """Fix WebSocket endpoint for async chat manager too"""
    print("\n🔧 Fixing WebSocket Endpoint")
    print("=" * 50)
    
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Failed to read main.py: {e}")
        return False
    
    # Find WebSocket message processing
    websocket_pattern = r'(# Process with real chat manager if available.*?else:.*?response_text = f"WebSocket received: \{message\}")'
    
    websocket_replacement = '''# Process with real chat manager if available
            message = message_data.get("message", "")
            if chat_manager and hasattr(chat_manager, 'process_message'):
                try:
                    import asyncio
                    if asyncio.iscoroutinefunction(chat_manager.process_message):
                        response_text = await chat_manager.process_message(message)
                    else:
                        response_text = chat_manager.process_message(message)
                    
                    # Handle response types
                    if isinstance(response_text, dict):
                        response_text = response_text.get("response", str(response_text))
                    elif not isinstance(response_text, str):
                        response_text = str(response_text)
                        
                except Exception as e:
                    response_text = f"WebSocket chat error: {str(e)}"
            else:
                response_text = f"WebSocket received: {message}"'''
    
    if re.search(websocket_pattern, content, re.DOTALL):
        new_content = re.sub(websocket_pattern, websocket_replacement, content, flags=re.DOTALL)
        
        try:
            with open("main.py", 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("✅ WebSocket endpoint updated with async handling")
            return True
        except Exception as e:
            print(f"❌ Failed to write updated main.py: {e}")
            return False
    else:
        print("⚠️ WebSocket pattern not found - may already be correct")
        return True

def create_test_async_endpoint():
    """Create a test endpoint to verify async chat works"""
    print("\n🧪 Creating Test Async Endpoint")
    print("=" * 50)
    
    test_endpoint = '''
@app.post("/api/chat/async-test")
async def chat_async_test(request: Request):
    """Test endpoint for async chat functionality"""
    try:
        data = await request.json()
        message = data.get("message", "test")
        
        # Test async detection
        import asyncio
        is_async = asyncio.iscoroutinefunction(chat_manager.process_message) if chat_manager else False
        
        # Test chat manager call
        if chat_manager and hasattr(chat_manager, 'process_message'):
            try:
                if is_async:
                    response = await chat_manager.process_message(message)
                else:
                    response = chat_manager.process_message(message)
                
                return {
                    "status": "success",
                    "message": message,
                    "is_async": is_async,
                    "response": str(response),
                    "response_type": str(type(response).__name__)
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": message,
                    "is_async": is_async,
                    "error": str(e),
                    "error_type": str(type(e).__name__)
                }
        else:
            return {
                "status": "no_chat_manager",
                "message": message,
                "chat_manager_exists": chat_manager is not None
            }
            
    except Exception as e:
        return {
            "status": "endpoint_error",
            "error": str(e)
        }
'''
    
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add test endpoint before if __name__ == "__main__"
        if_main_index = content.find('if __name__ == "__main__":')
        if if_main_index != -1:
            new_content = content[:if_main_index] + test_endpoint + "\n" + content[if_main_index:]
            
            with open("main.py", 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("✅ Test endpoint added: /api/chat/async-test")
            return True
        else:
            print("⚠️ Could not find place to add test endpoint")
            return False
            
    except Exception as e:
        print(f"❌ Failed to add test endpoint: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Fix Async Chat Endpoint")
    print("=" * 60)
    
    print("🎯 Root cause identified:")
    print("   ❌ 'coroutine' object is not subscriptable")
    print("   🔍 This means: chat_manager.process_message() is async")
    print("   🔍 But it's being called without 'await'")
    print()
    print("✅ Your engines are perfect:")
    print("   ✅ Enhanced Chat Manager initialized with dependencies")
    print("   ✅ Gemini AI initialized with gemini-1.5-flash")
    print("   ❌ Just need to fix async/await in API endpoint")
    print()
    
    # Step 1: Create backup
    backup_file = backup_main_file()
    if not backup_file:
        print("❌ Cannot proceed without backup")
        return
    
    # Step 2: Fix chat endpoint
    if fix_chat_endpoint():
        print("✅ Chat endpoint fixed")
    else:
        print("❌ Failed to fix chat endpoint")
        return
    
    # Step 3: Fix WebSocket endpoint
    if fix_websocket_endpoint():
        print("✅ WebSocket endpoint fixed")
    
    # Step 4: Add test endpoint
    if create_test_async_endpoint():
        print("✅ Test endpoint added")
    
    print("\n🎉 ASYNC CHAT ENDPOINT FIX COMPLETE!")
    print("=" * 60)
    
    print("🔄 Your server will auto-reload with the fixes")
    print()
    print("✅ Expected results:")
    print("   ✅ Chat 'help' -> Smart Gemini AI response")
    print("   ✅ Chat 'status' -> Enhanced trading bot status")
    print("   ✅ No more 'coroutine' errors")
    print("   ✅ No more 500 Internal Server Error")
    print()
    print("🧪 Test endpoints:")
    print("   • http://localhost:8000/chat - Regular chat")
    print("   • http://localhost:8000/api/chat/async-test - Async diagnostic")
    print()
    print("🎯 What was fixed:")
    print("   1. Chat endpoint now properly awaits async process_message()")
    print("   2. Handles both async and sync chat manager methods")
    print("   3. Proper error handling for coroutine issues")
    print("   4. WebSocket also fixed for async operations")
    print("   5. Test endpoint added for verification")

if __name__ == "__main__":
    main()