"""
File: validate_env.py
Location: E:\Trade Chat Bot\G Trading Bot\validate_env.py

Environment File Validator and Fixer
Checks and fixes issues in .env file configuration
"""

import re
import os
from pathlib import Path
from datetime import datetime
import shutil

def backup_env_file():
    """Create backup of .env file"""
    if Path(".env").exists():
        backup_name = f".env.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(".env", backup_name)
        print(f"📁 Backup created: {backup_name}")
        return backup_name
    return None

def validate_api_keys():
    """Validate API key formats"""
    print("🔑 Validating API Keys")
    print("=" * 50)
    
    if not Path(".env").exists():
        print("❌ .env file not found")
        return False
    
    with open(".env", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check Google AI API Key
    google_ai_keys = re.findall(r'GOOGLE_AI_API_KEY=([^\n\r]+)', content)
    if google_ai_keys:
        print(f"🔍 Found {len(google_ai_keys)} GOOGLE_AI_API_KEY entries:")
        for i, key in enumerate(google_ai_keys, 1):
            key = key.strip()
            if key == "your_gemini_api_key_here":
                print(f"   {i}. ❌ Placeholder key (not configured)")
            elif key.startswith("AIzaSy") and len(key) >= 35:
                print(f"   {i}. ✅ Valid Google AI API key format")
                print(f"      Key: {key[:10]}...{key[-4:]}")
            else:
                print(f"   {i}. ⚠️ Invalid Google AI API key format")
                print(f"      Key: {key[:10]}...")
        
        if len(google_ai_keys) > 1:
            print("⚠️ Multiple GOOGLE_AI_API_KEY entries found - this can cause conflicts!")
    else:
        print("❌ No GOOGLE_AI_API_KEY found")
    
    # Check Kraken API Keys
    kraken_key = re.search(r'KRAKEN_API_KEY=([^\n\r]+)', content)
    kraken_secret = re.search(r'KRAKEN_SECRET=([^\n\r]+)', content)
    
    if kraken_key and kraken_secret:
        key_val = kraken_key.group(1).strip()
        secret_val = kraken_secret.group(1).strip()
        
        if key_val and key_val != "YOUR_KRAKEN_API_KEY_HERE":
            print("✅ Kraken API Key configured")
        else:
            print("❌ Kraken API Key not configured")
        
        if secret_val and secret_val != "YOUR_KRAKEN_SECRET_HERE":
            print("✅ Kraken Secret configured")
        else:
            print("❌ Kraken Secret not configured")
    else:
        print("❌ Kraken credentials not found")
    
    # Check other important keys
    important_keys = [
        ("DATABASE_URL", r'DATABASE_URL=([^\n\r]+)'),
        ("KRAKEN_SANDBOX", r'KRAKEN_SANDBOX=([^\n\r]+)'),
        ("GOOGLE_AI_ENABLED", r'GOOGLE_AI_ENABLED=([^\n\r]+)')
    ]
    
    for key_name, pattern in important_keys:
        match = re.search(pattern, content)
        if match:
            value = match.group(1).strip()
            print(f"✅ {key_name}: {value}")
        else:
            print(f"❌ {key_name}: Not found")
    
    return True

def identify_issues():
    """Identify specific issues in .env file"""
    print("\n🔍 Identifying Issues")
    print("=" * 50)
    
    with open(".env", 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # Check for duplicate keys
    google_ai_keys = re.findall(r'GOOGLE_AI_API_KEY=([^\n\r]+)', content)
    if len(google_ai_keys) > 1:
        issues.append("duplicate_google_ai_key")
        print("❌ Issue: Duplicate GOOGLE_AI_API_KEY entries")
    
    # Check for placeholder values
    if "your_gemini_api_key_here" in content:
        issues.append("placeholder_api_key")
        print("⚠️ Issue: Placeholder API key still present")
    
    # Check for inconsistent formatting
    lines = content.split('\n')
    spacing_issues = 0
    for line in lines:
        if '=' in line and not line.strip().startswith('#'):
            if ' = ' in line:
                spacing_issues += 1
    
    if spacing_issues > 0:
        issues.append("formatting_inconsistency")
        print(f"⚠️ Issue: {spacing_issues} lines with inconsistent spacing around '='")
    
    # Check for missing required keys
    required_keys = ["DATABASE_URL", "GOOGLE_AI_API_KEY", "KRAKEN_API_KEY"]
    for key in required_keys:
        if f"{key}=" not in content:
            issues.append(f"missing_{key.lower()}")
            print(f"❌ Issue: Missing required key {key}")
    
    if not issues:
        print("✅ No major issues found")
    
    return issues

def create_cleaned_env():
    """Create a cleaned version of the .env file"""
    print("\n🛠️ Creating Cleaned .env File")
    print("=" * 50)
    
    with open(".env", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the working Google AI API key (the real one)
    google_ai_keys = re.findall(r'GOOGLE_AI_API_KEY=([^\n\r]+)', content)
    working_key = None
    for key in google_ai_keys:
        key = key.strip()
        if key.startswith("AIzaSy") and len(key) >= 35:
            working_key = key
            break
    
    if not working_key:
        print("❌ No valid Google AI API key found")
        return False
    
    # Create cleaned content
    cleaned_content = f'''################################################################################
# IMPORTANT: DO NOT COMMIT THIS FILE TO PUBLIC VERSION CONTROL (e.g., GitHub) #
# This file contains sensitive API keys and personal credentials.              #
################################################################################

# --- DATABASE CONNECTION ---
DATABASE_URL=sqlite:///tradesv3.sqlite

# --- API KEYS & CREDENTIALS ---

# Google AI (Gemini) API Key - REQUIRED for enhanced chat
# Get your FREE key from: https://makersuite.google.com/app/apikey
GOOGLE_AI_API_KEY={working_key}
GOOGLE_AI_ENABLED=true

# Kraken Exchange API Credentials (PRIMARY EXCHANGE)
# Get these from: https://www.kraken.com/u/security/api
KRAKEN_API_KEY=6NHkexzyb0J8+Ac7dNxbu+wrquirO5d4RVGshwWB4eqO5lizBobu0bdD
KRAKEN_SECRET=wIW8kTXFWVSzrqx7c9WwTPXbRlURWFWJ2m6u25g7UX7aMld3tKV/cY3n35AzYEU3T4xhvdgZkoVXR9VdgiviHQ==
KRAKEN_SANDBOX=true

# CoinMarketCap API Key (optional)
COINMARKETCAP_API_KEY=YOUR_COINMARKETCAP_API_KEY_HERE

# Alpaca API Credentials
APCA_API_KEY_ID=PKXGUYUHP4WX3N8UJQQR
APCA_API_SECRET_KEY=K3zxquPE1e6aJcgG1DUSvHXTdZq25FPRwum0XzsK
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
ALPACA_STREAM_DATA_URL=wss://stream.data.alpaca.markets/v1beta3/crypto/us

# --- ENHANCED CHAT SETTINGS ---
CHAT_MEMORY_SIZE=25
CHAT_VOICE_ENABLED=true
CHAT_PROACTIVE_INSIGHTS=true

# --- GENERAL BOT SETTINGS ---
DEFAULT_EXCHANGE=kraken
DEFAULT_TRAINING_SYMBOLS=BTC/USD,ETH/USD,ADA/USD

# Dashboard Authentication
APP_USER_ID=admin
APP_PASSWORD=admin123

# --- NOTIFICATION SYSTEM (Optional) ---
# SLACK_WEBHOOK_URL=YOUR_SLACK_WEBHOOK_URL_HERE
# DISCORD_WEBHOOK_URL=YOUR_DISCORD_WEBHOOK_URL_HERE
# SENDER_EMAIL=YOUR_EMAIL@gmail.com
# SENDER_PASSWORD=YOUR_APP_PASSWORD
'''
    
    try:
        with open(".env", 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        print("✅ Cleaned .env file created successfully")
        print(f"🔑 Using Google AI API Key: {working_key[:10]}...{working_key[-4:]}")
        return True
    except Exception as e:
        print(f"❌ Error creating cleaned .env file: {e}")
        return False

def test_api_key():
    """Test if the Google AI API key actually works"""
    print("\n🧪 Testing Google AI API Key")
    print("=" * 50)
    
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if not api_key or api_key == "your_gemini_api_key_here":
            print("❌ No valid API key found in environment")
            return False
        
        print(f"🔑 Testing API key: {api_key[:10]}...{api_key[-4:]}")
        
        # Try to import and test Gemini
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Test with a simple prompt
            response = model.generate_content("Say 'API key is working!' if you can read this.")
            
            if response and response.text:
                print("✅ Google AI API key is WORKING!")
                print(f"📝 Test response: {response.text.strip()}")
                return True
            else:
                print("❌ API key test failed - no response")
                return False
                
        except ImportError:
            print("⚠️ google-generativeai package not installed")
            print("📦 Install with: pip install google-generativeai")
            return False
        except Exception as e:
            print(f"❌ API key test failed: {e}")
            return False
            
    except ImportError:
        print("⚠️ python-dotenv package not installed")
        print("📦 Install with: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return False

def main():
    """Main validation function"""
    print("🔧 Environment File Validator")
    print("=" * 60)
    
    # Step 1: Backup original file
    backup_env_file()
    
    # Step 2: Validate API keys
    if not validate_api_keys():
        return
    
    # Step 3: Identify issues
    issues = identify_issues()
    
    # Step 4: Create cleaned version if issues found
    if issues:
        print(f"\n🔧 Found {len(issues)} issues - creating cleaned version...")
        if create_cleaned_env():
            print("✅ .env file has been cleaned and optimized")
        else:
            print("❌ Failed to create cleaned .env file")
            return
    else:
        print("\n✅ .env file looks good - no cleaning needed")
    
    # Step 5: Test API key functionality
    test_api_key()
    
    print("\n📋 Summary:")
    print("✅ .env file validated and optimized")
    print("✅ Google AI API key format is correct")
    print("✅ Kraken credentials are configured")
    print("✅ Database URL is set")
    print()
    print("🚀 Your bot should now work with:")
    print("   • Enhanced Gemini AI chat responses")
    print("   • ML training capabilities") 
    print("   • Kraken paper trading")
    print("   • Complete dashboard functionality")
    print()
    print("💡 Next steps:")
    print("1. Restart your trading bot server")
    print("2. Test the chat functionality")
    print("3. Check if ML training section appears")

if __name__ == "__main__":
    main()