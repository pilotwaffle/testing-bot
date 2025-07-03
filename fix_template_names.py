#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\fix_template_names.py
Location: E:\Trade Chat Bot\G Trading Bot\fix_template_names.py

🔧 Elite Trading Bot V3.0 - Template File Fix Script
Ensures your templates work regardless of naming convention
"""

import os
import shutil
from pathlib import Path

def fix_template_files():
    """Fix template file naming issues"""
    print("🔧 Elite Trading Bot V3.0 - Template Fix")
    print("="*50)
    
    base_dir = Path.cwd()
    templates_dir = base_dir / "templates"
    
    # Ensure templates directory exists
    templates_dir.mkdir(exist_ok=True)
    
    # Check which template files exist
    index_html = templates_dir / "index.html"
    dashboard_html = templates_dir / "dashboard.html"
    
    print("📄 Checking existing template files...")
    
    if index_html.exists():
        print("✅ index.html found")
        
        # Copy index.html to dashboard.html if dashboard.html doesn't exist
        if not dashboard_html.exists():
            shutil.copy2(index_html, dashboard_html)
            print("✅ Created dashboard.html (copy of index.html)")
        else:
            print("✅ dashboard.html already exists")
            
    elif dashboard_html.exists():
        print("✅ dashboard.html found")
        
        # Copy dashboard.html to index.html if index.html doesn't exist
        if not index_html.exists():
            shutil.copy2(dashboard_html, index_html)
            print("✅ Created index.html (copy of dashboard.html)")
        else:
            print("✅ index.html already exists")
            
    else:
        print("❌ No template files found!")
        print("💡 Please create the industrial dashboard HTML file first")
        return False
    
    # Check main.py to see which template it's expecting
    main_py = base_dir / "main.py"
    if main_py.exists():
        with open(main_py, "r", encoding="utf-8") as f:
            content = f.read()
            
        if 'templates.TemplateResponse("dashboard.html"' in content:
            print("📋 main.py expects: dashboard.html")
        elif 'templates.TemplateResponse("index.html"' in content:
            print("📋 main.py expects: index.html")
        else:
            print("⚠️ Could not determine which template main.py expects")
    
    print("\n✅ Template files are now compatible!")
    print("📁 Available templates:")
    
    for template_file in templates_dir.glob("*.html"):
        file_size = template_file.stat().st_size
        print(f"   - {template_file.name} ({file_size:,} bytes)")
    
    return True

def check_main_py_template_route():
    """Check and optionally fix the main.py template route"""
    print("\n🔍 Checking main.py template route...")
    
    main_py = Path("main.py")
    if not main_py.exists():
        print("❌ main.py not found!")
        return
    
    with open(main_py, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Look for template response patterns
    if 'templates.TemplateResponse("dashboard.html"' in content:
        print("✅ main.py is configured for dashboard.html")
    elif 'templates.TemplateResponse("index.html"' in content:
        print("✅ main.py is configured for index.html")
    else:
        print("⚠️ main.py template route may need updating")
        print("\n💡 Add this route to your main.py if missing:")
        print('''
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if templates is None:
        raise HTTPException(status_code=500, detail="Template engine not initialized")
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "service_name": "Elite Trading Bot V3.0"
    })
''')

def main():
    """Main function"""
    print("🚀 Starting template file fix...")
    
    if fix_template_files():
        check_main_py_template_route()
        
        print("\n" + "="*50)
        print("🎉 TEMPLATE FIX COMPLETED!")
        print("="*50)
        print("✅ Both index.html and dashboard.html are now available")
        print("✅ Your dashboard will work regardless of which one main.py expects")
        print("\n🚀 Next steps:")
        print("1. Start your server: python main.py")
        print("2. Open browser: http://localhost:8000")
        print("3. Enjoy your industrial dashboard! 🎨")
    else:
        print("\n❌ Template fix failed. Please create the HTML file first.")

if __name__ == "__main__":
    main()