"""
File: ml_diagnostic.py
Location: E:\Trade Chat Bot\G Trading Bot\ml_diagnostic.py

ML Status Diagnostic Script
Checks why ML models aren't showing up on dashboard
"""

import os
import sys
import json
from pathlib import Path

def check_ml_status():
    print("🔍 ML Status Diagnostic")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Look for ML-related files
    ml_files = [
        "core/ml_engine.py",
        "core/ml_models.py", 
        "core/enhanced_ml_engine.py",
        "ai/ml_engine.py",
        "ml/models.py",
        "strategies/ml_strategy.py"
    ]
    
    print("\n🔍 Checking for ML files:")
    found_ml_files = []
    for file_path in ml_files:
        if Path(file_path).exists():
            found_ml_files.append(file_path)
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
    
    # Check main.py or app.py for dashboard route
    print("\n🔍 Checking dashboard routes:")
    main_files = ["main.py", "app.py", "server.py"]
    for main_file in main_files:
        if Path(main_file).exists():
            print(f"✅ Found: {main_file}")
            try:
                with open(main_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'ml_status' in content:
                        print(f"   📊 Contains 'ml_status' reference")
                    if '/dashboard' in content or '@app.get("/")' in content:
                        print(f"   🌐 Contains dashboard route")
            except Exception as e:
                print(f"   ❌ Error reading {main_file}: {e}")
    
    # Check for ML engine initialization
    print("\n🔍 Checking for ML engine initialization:")
    init_patterns = [
        "ml_engine",
        "MLEngine", 
        "ml_status",
        "ml_models",
        "train_model"
    ]
    
    for pattern in init_patterns:
        found_in = []
        for main_file in main_files:
            if Path(main_file).exists():
                try:
                    with open(main_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if pattern in content:
                            found_in.append(main_file)
                except:
                    pass
        
        if found_in:
            print(f"✅ '{pattern}' found in: {', '.join(found_in)}")
        else:
            print(f"❌ '{pattern}' not found in main files")
    
    # Check requirements/dependencies
    print("\n🔍 Checking ML dependencies:")
    req_files = ["requirements.txt", "pyproject.toml", "Pipfile"]
    ml_packages = ["scikit-learn", "tensorflow", "torch", "numpy", "pandas"]
    
    for req_file in req_files:
        if Path(req_file).exists():
            print(f"✅ Found: {req_file}")
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for package in ml_packages:
                        if package in content:
                            print(f"   📦 {package} listed")
            except Exception as e:
                print(f"   ❌ Error reading {req_file}: {e}")
    
    # Generate recommendations
    print("\n💡 Recommendations:")
    print("=" * 50)
    
    if not found_ml_files:
        print("❌ No ML files found - ML engine might not be implemented")
        print("   → Need to create ML engine and models")
    
    # Check if dashboard template exists
    if Path("templates/dashboard.html").exists():
        print("✅ Dashboard template exists")
        # Check if it contains ml_status
        try:
            with open("templates/dashboard.html", 'r', encoding='utf-8') as f:
                content = f.read()
                if "ml_status" in content:
                    print("   📊 Template expects ml_status variable")
                    print("   → Backend needs to provide ml_status to template")
        except Exception as e:
            print(f"   ❌ Error reading template: {e}")
    
    print("\n🔧 Quick fixes to try:")
    print("1. Check if ML engine is imported in main.py")
    print("2. Verify ml_status is passed to dashboard template")
    print("3. Check browser console for JavaScript errors")
    print("4. Restart the bot to reinitialize ML system")

def check_browser_console_issues():
    print("\n🌐 Browser Console Check Instructions:")
    print("=" * 50)
    print("1. Open dashboard in browser")
    print("2. Press F12 to open developer tools")
    print("3. Go to Console tab")
    print("4. Look for red error messages")
    print("5. Look for failed network requests in Network tab")
    
    print("\nCommon issues to look for:")
    print("- Failed to load dashboard.js")
    print("- WebSocket connection errors") 
    print("- 404 errors for missing endpoints")
    print("- JavaScript syntax errors")

if __name__ == "__main__":
    check_ml_status()
    check_browser_console_issues()
    
    print(f"\n📋 Summary:")
    print("Run this script in your bot directory to diagnose ML issues")
    print("Then check the browser console when viewing the dashboard")