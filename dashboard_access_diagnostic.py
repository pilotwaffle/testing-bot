"""
File: dashboard_access_diagnostic.py
Location: E:\Trade Chat Bot\G Trading Bot\dashboard_access_diagnostic.py

Dashboard Access Diagnostic
Investigates why dashboard shows 99.5/100 but user can't access it
"""

import requests
import json
from datetime import datetime

def test_dashboard_content():
    """Test what the dashboard actually returns"""
    print("🔍 Dashboard Content Analysis")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:8000/", timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📊 Content Length: {len(response.content)} bytes")
        print(f"📊 Content Type: {response.headers.get('content-type')}")
        print()
        
        if response.status_code == 200:
            content = response.text
            print("📄 ACTUAL CONTENT RECEIVED:")
            print("=" * 40)
            print(content)
            print("=" * 40)
            print()
            
            # Analyze what's actually in the content
            if len(content) < 1000:
                print("❌ ISSUE: Content is too short for a full dashboard!")
                print("🔍 This looks like an error page or minimal response")
            
            # Check for key dashboard elements
            dashboard_indicators = [
                ("ML Training", ["ml", "train", "model", "lorentzian"]),
                ("Chat Interface", ["chat", "message", "send"]),
                ("Portfolio", ["portfolio", "trading", "balance"]),
                ("Navigation", ["nav", "menu", "dashboard"]),
                ("JavaScript", ["<script", "function"]),
                ("CSS", ["<style", "class="]),
                ("Template Variables", ["{{", "}}"]),
                ("Buttons", ["<button", "onclick"])
            ]
            
            print("🔍 Content Analysis:")
            for name, keywords in dashboard_indicators:
                found = any(keyword.lower() in content.lower() for keyword in keywords)
                status = "✅" if found else "❌"
                print(f"   {status} {name}: {'Found' if found else 'Missing'}")
            
            # Check if it looks like an error
            error_indicators = ["error", "exception", "traceback", "500", "404"]
            has_errors = any(indicator.lower() in content.lower() for indicator in error_indicators)
            
            if has_errors:
                print("\n❌ Content appears to contain error information!")
            
        else:
            print(f"❌ Dashboard returned status: {response.status_code}")
            print(f"📄 Error content: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed to fetch dashboard: {e}")

def test_different_urls():
    """Test different ways to access the dashboard"""
    print("\n🌐 Testing Different Dashboard URLs")
    print("=" * 60)
    
    urls_to_test = [
        "http://localhost:8000/",
        "http://localhost:8000",
        "http://127.0.0.1:8000/",
        "http://127.0.0.1:8000",
        "http://localhost:8000/dashboard",
        "http://localhost:8000/index"
    ]
    
    for url in urls_to_test:
        try:
            response = requests.get(url, timeout=5)
            print(f"✅ {url}: {response.status_code} ({len(response.content)} bytes)")
        except Exception as e:
            print(f"❌ {url}: Failed - {e}")

def test_template_rendering():
    """Test if templates are rendering correctly"""
    print("\n📄 Template Rendering Test")
    print("=" * 60)
    
    # Check if template file exists and what it contains
    try:
        with open("templates/dashboard.html", 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        print(f"✅ Template file exists: {len(template_content)} characters")
        
        # Check for template variables
        template_vars = []
        import re
        vars_found = re.findall(r'\{\{\s*(\w+)', template_content)
        template_vars = list(set(vars_found))
        
        print(f"📋 Template variables: {template_vars}")
        
        # Now test if the endpoint is providing these variables
        try:
            response = requests.get("http://localhost:8000/", timeout=10)
            content = response.text
            
            print("\n🔍 Checking if template variables are being replaced:")
            for var in template_vars:
                if f"{{{{{var}" in content:
                    print(f"❌ {var}: Not replaced (still shows {{{{ {var} }}}})")
                else:
                    print(f"✅ {var}: Replaced with actual data")
                    
        except Exception as e:
            print(f"❌ Could not test template rendering: {e}")
            
    except Exception as e:
        print(f"❌ Could not read template file: {e}")

def test_server_logs_guidance():
    """Provide guidance on checking server logs"""
    print("\n📋 Server Log Analysis Guide")
    print("=" * 60)
    
    print("🔍 In your server terminal, when you visit http://localhost:8000, look for:")
    print()
    print("✅ GOOD - Should see:")
    print("   INFO: 127.0.0.1:xxxxx - \"GET / HTTP/1.1\" 200 OK")
    print()
    print("❌ BAD - Might see:")
    print("   ERROR: Exception in ASGI application")
    print("   INFO: 127.0.0.1:xxxxx - \"GET / HTTP/1.1\" 500 Internal Server Error")
    print()
    print("🔍 Template errors to look for:")
    print("   • jinja2.exceptions.UndefinedError: 'variable_name' is undefined")
    print("   • jinja2.exceptions.TemplateSyntaxError:")
    print("   • KeyError: 'some_key'")
    print()
    print("💡 Copy any error stack traces and share them!")

def create_simple_test_route():
    """Create a simple test route to bypass template issues"""
    print("\n🔧 Creating Simple Test Route")
    print("=" * 60)
    
    test_route_code = '''
# Add this to your main.py to test if routing works

@app.get("/test-simple")
async def test_simple():
    """Simple test route that doesn't use templates"""
    return HTMLResponse("""
    <html>
    <head><title>Simple Test</title></head>
    <body>
        <h1>🎉 Simple Route Working!</h1>
        <p>If you see this, the server routing is working.</p>
        <p>The issue is in the dashboard template rendering.</p>
        <a href="/health">Health Check</a> | 
        <a href="/api/ml/status">ML Status</a> |
        <a href="/chat">Chat Page</a>
    </body>
    </html>
    """)
'''
    
    print("🔧 Test route code to add to main.py:")
    print(test_route_code)
    print()
    print("After adding this route, test:")
    print("   http://localhost:8000/test-simple")
    print()
    print("If this works but dashboard doesn't, the issue is template-related")

def main():
    """Main diagnostic function"""
    print("🔧 Dashboard Access Diagnostic")
    print("=" * 80)
    
    print("🎯 Issue: Diagnostic shows 99.5/100 but user can't access dashboard")
    print("🔍 Suspicion: 240 bytes is too small - might be error page or minimal response")
    print()
    
    # Run focused tests
    test_dashboard_content()
    test_different_urls()
    test_template_rendering()
    test_server_logs_guidance()
    create_simple_test_route()
    
    print("\n🎯 SUMMARY")
    print("=" * 80)
    print()
    print("The diagnostic showed 99.5/100 but with red flags:")
    print("   ❌ Only 240 bytes content (should be 5000+)")
    print("   ❌ Missing chat widget") 
    print("   ❌ Missing interactive elements")
    print()
    print("🔧 Most likely causes:")
    print("   1. Template rendering error (variables not provided)")
    print("   2. FastAPI returning error page that looks like success")
    print("   3. Template file corruption or missing sections")
    print("   4. Dashboard route not properly configured")
    print()
    print("📋 Next steps:")
    print("   1. Check what content is actually returned above")
    print("   2. Look at server logs when visiting dashboard")
    print("   3. Try the test route if routing is the issue")
    print("   4. Share any error messages for targeted fix")

if __name__ == "__main__":
    main()