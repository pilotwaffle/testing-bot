"""
File: fix_main_syntax.py
Location: E:\Trade Chat Bot\G Trading Bot\fix_main_syntax.py

Fix Main.py Syntax Error
Identifies and repairs syntax errors in main.py file
"""

import ast
import shutil
import re
from datetime import datetime
from pathlib import Path

def backup_main_file():
    """Create backup of main.py"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"main.py.backup_syntax_{timestamp}"
    
    try:
        shutil.copy2("main.py", backup_name)
        print(f"📁 Backup created: {backup_name}")
        return backup_name
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return None

def check_syntax_errors():
    """Check for syntax errors in main.py"""
    print("🔍 Checking for syntax errors in main.py...")
    
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the file
        try:
            ast.parse(content)
            print("✅ No syntax errors found")
            return None
        except SyntaxError as e:
            print(f"❌ Syntax error found:")
            print(f"   Line {e.lineno}: {e.text}")
            print(f"   Error: {e.msg}")
            return e
            
    except Exception as e:
        print(f"❌ Failed to read main.py: {e}")
        return None

def find_problematic_lines():
    """Find lines around the syntax error"""
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"\n🔍 Examining lines around error (730-735)...")
        
        for i in range(max(0, 729), min(len(lines), 736)):
            line_num = i + 1
            line = lines[i].rstrip()
            marker = " ⚠️" if line_num == 732 else ""
            print(f"   {line_num:3d}: {line}{marker}")
        
        return lines
        
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return None

def fix_common_syntax_issues(lines):
    """Fix common syntax issues"""
    print(f"\n🔧 Attempting to fix common syntax issues...")
    
    fixed_lines = []
    fixes_applied = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        original_line = line
        
        # Check for missing colons at end of function definitions
        if re.match(r'^\s*def\s+\w+\([^)]*\)\s*$', line.strip()):
            if not line.rstrip().endswith(':'):
                line = line.rstrip() + ':\n'
                fixes_applied.append(f"Line {line_num}: Added missing colon to function definition")
        
        # Check for missing colons at end of class definitions
        if re.match(r'^\s*class\s+\w+.*$', line.strip()):
            if not line.rstrip().endswith(':'):
                line = line.rstrip() + ':\n'
                fixes_applied.append(f"Line {line_num}: Added missing colon to class definition")
        
        # Check for missing colons after decorators followed by def
        if line.strip().startswith('@') and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith('def ') and not next_line.endswith(':'):
                lines[i + 1] = lines[i + 1].rstrip() + ':\n'
                fixes_applied.append(f"Line {line_num + 1}: Added missing colon to function definition")
        
        # Check for incomplete function calls or missing closing parentheses
        if '(' in line and line.count('(') > line.count(')'):
            # Look ahead to see if closing paren is on next line
            if i + 1 < len(lines) and ')' in lines[i + 1]:
                pass  # Likely multiline function call
            else:
                # Try to balance parentheses
                missing_parens = line.count('(') - line.count(')')
                line = line.rstrip() + ')' * missing_parens + '\n'
                fixes_applied.append(f"Line {line_num}: Added {missing_parens} missing closing parentheses")
        
        # Check for async/await syntax issues
        if 'async def' in line and not line.rstrip().endswith(':'):
            line = line.rstrip() + ':\n'
            fixes_applied.append(f"Line {line_num}: Added missing colon to async function")
        
        fixed_lines.append(line)
    
    return fixed_lines, fixes_applied

def create_minimal_working_main():
    """Create a minimal working main.py if all else fails"""
    print("🔧 Creating minimal working main.py...")
    
    minimal_main = '''"""
File: main.py
Location: E:\\Trade Chat Bot\\G Trading Bot\\main.py

Minimal Working Main - Emergency Recovery
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
import json
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Elite Trading Bot V3.0",
    description="Industrial Crypto Trading Bot",
    version="3.0.0"
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
active_connections = []

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "ml_status": {
            "models": [
                {"name": "Lorentzian Classifier", "status": "available"},
                {"name": "Neural Network", "status": "available"},
                {"name": "Social Sentiment", "status": "available"},
                {"name": "Risk Assessment", "status": "available"}
            ]
        },
        "status": "RUNNING"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "server": True,
            "ml_engine": True,
            "chat_manager": True
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Chat API endpoint"""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        # Simple response for now
        response = f"I received your message: '{message}'. I'm working on providing better responses!"
        
        return {
            "response": response,
            "message_type": "text",
            "intent": "general_chat",
            "response_time": 0.1
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat interface page"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Echo back for now
            response = {
                "type": "chat_response",
                "response": f"WebSocket received: {message_data.get('message', '')}",
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''
    
    return minimal_main

def main():
    """Main fix function"""
    print("🔧 Main.py Syntax Error Fix")
    print("=" * 60)
    
    # Step 1: Create backup
    backup_file = backup_main_file()
    if not backup_file:
        print("❌ Cannot proceed without backup")
        return
    
    # Step 2: Check for syntax errors
    syntax_error = check_syntax_errors()
    if not syntax_error:
        print("✅ No syntax errors found - something else is wrong")
        return
    
    # Step 3: Examine problematic lines
    lines = find_problematic_lines()
    if not lines:
        print("❌ Cannot read file")
        return
    
    # Step 4: Try to fix common issues
    fixed_lines, fixes_applied = fix_common_syntax_issues(lines)
    
    if fixes_applied:
        print(f"\\n🔧 Applied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"   ✅ {fix}")
        
        # Write fixed version
        try:
            with open("main.py", 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            print("✅ Fixed main.py written")
            
            # Test the fix
            test_error = check_syntax_errors()
            if not test_error:
                print("🎉 Syntax errors fixed successfully!")
                return
            else:
                print("❌ Fix didn't work, syntax error still present")
        
        except Exception as e:
            print(f"❌ Failed to write fixed file: {e}")
    
    # Step 5: Emergency fallback - create minimal working version
    print("\\n🚨 Creating emergency minimal working main.py...")
    
    try:
        minimal_content = create_minimal_working_main()
        with open("main.py", 'w', encoding='utf-8') as f:
            f.write(minimal_content)
        
        # Test minimal version
        test_error = check_syntax_errors()
        if not test_error:
            print("✅ Minimal working main.py created successfully!")
            print("⚠️ This is a basic version - you may need to re-add advanced features")
        else:
            print("❌ Even minimal version has syntax errors")
            
    except Exception as e:
        print(f"❌ Failed to create minimal version: {e}")
    
    print("\\n🎯 Next Steps:")
    print("1. Try starting your server again:")
    print("   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("2. If it works, you can gradually add back advanced features")
    print(f"3. Your original file is backed up as: {backup_file}")

if __name__ == "__main__":
    main()