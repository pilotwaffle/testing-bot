"""
File: fix_gemini_model.py
Location: E:\Trade Chat Bot\G Trading Bot\fix_gemini_model.py

Gemini Model Fix Script
Fixes Gemini API model issues and updates to working model names
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def backup_gemini_files():
    """Backup Gemini-related files"""
    files_to_backup = ["ai/gemini_ai.py", "main.py"]
    backups = []
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            backup_name = f"{file_path}.backup_gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_name)
            backups.append(backup_name)
            print(f"📁 Backup created: {backup_name}")
    
    return backups

def test_available_models():
    """Test which Gemini models are actually available"""
    print("🧪 Testing Available Gemini Models")
    print("=" * 50)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if not api_key:
            print("❌ No API key found")
            return None
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Test different model names that are currently available
        model_names_to_test = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "gemini-pro",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro"
        ]
        
        working_models = []
        
        for model_name in model_names_to_test:
            try:
                print(f"🔍 Testing model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello! Respond with just 'Working' if you can read this.")
                
                if response and response.text:
                    print(f"   ✅ {model_name} - WORKING!")
                    print(f"   📝 Response: {response.text.strip()}")
                    working_models.append(model_name)
                else:
                    print(f"   ❌ {model_name} - No response")
                    
            except Exception as e:
                print(f"   ❌ {model_name} - Error: {str(e)[:100]}...")
        
        if working_models:
            print(f"\n✅ Found {len(working_models)} working models:")
            for model in working_models:
                print(f"   • {model}")
            return working_models[0]  # Return the first working model
        else:
            print("\n❌ No working models found")
            return None
            
    except ImportError:
        print("❌ google-generativeai package not installed")
        return None
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return None

def fix_gemini_ai_file(working_model):
    """Fix the gemini_ai.py file with the working model"""
    print(f"\n🔧 Fixing ai/gemini_ai.py with model: {working_model}")
    print("=" * 50)
    
    fixed_gemini_content = f'''"""
File: ai/gemini_ai.py
Location: E:\\Trade Chat Bot\\G Trading Bot\\ai\\gemini_ai.py

Google Gemini AI Integration for Trading Bot - FIXED VERSION
"""
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class GeminiAI:
    """Google Gemini AI integration for enhanced chat responses - FIXED"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini AI client with working model"""
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        self.enabled = bool(self.api_key)
        self.model_name = "{working_model}"  # Updated to working model
        self.client = None
        
        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
                logger.info(f"✅ Gemini AI initialized successfully with model: {{self.model_name}}")
            except ImportError:
                logger.warning("❌ google-generativeai not installed. Install with: pip install google-generativeai")
                self.enabled = False
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini AI: {{e}}")
                self.enabled = False
        else:
            logger.warning("⚠️ Gemini AI disabled - no API key provided")
    
    async def chat(self, message: str, context: Optional[str] = None) -> Optional[str]:
        """Send message to Gemini AI and get response"""
        if not self.enabled or not self.client:
            return None
        
        try:
            # Build enhanced prompt for trading bot context
            prompt = self._build_prompt(message, context)
            
            # Generate response
            response = self.client.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("⚠️ Empty response from Gemini AI")
                return None
                
        except Exception as e:
            logger.error(f"❌ Gemini AI chat error: {{e}}")
            return None
    
    def _build_prompt(self, message: str, context: Optional[str] = None) -> str:
        """Build enhanced prompt for trading bot context"""
        base_prompt = """You are an advanced AI assistant for a cryptocurrency trading bot. You help users with:

- Portfolio analysis and performance metrics
- Market analysis and trading insights  
- Risk management and strategy advice
- Technical analysis and market trends
- General trading questions and education
- ML model training and optimization

Be helpful, accurate, and professional. Use trading terminology appropriately.
Always include risk warnings when discussing trades or investments.
Format responses clearly with emojis for better readability.
Keep responses concise but informative.

"""
        
        if context:
            base_prompt += f"\\nCurrent Trading Context:\\n{{context}}\\n"
        
        base_prompt += f"\\nUser Question: {{message}}\\n"
        base_prompt += "\\nAssistant Response:"
        
        return base_prompt
    
    def is_available(self) -> bool:
        """Check if Gemini AI is available and working"""
        if not self.enabled or not self.client:
            return False
        
        try:
            # Quick test to verify it's working
            response = self.client.generate_content("Test")
            return bool(response and response.text)
        except:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get Gemini AI status"""
        return {{
            'enabled': self.enabled,
            'model': self.model_name,
            'api_key_configured': bool(self.api_key),
            'client_initialized': self.client is not None,
            'working': self.is_available()
        }}
'''
    
    # Create the ai directory if it doesn't exist
    ai_dir = Path("ai")
    ai_dir.mkdir(exist_ok=True)
    
    try:
        with open("ai/gemini_ai.py", 'w', encoding='utf-8') as f:
            f.write(fixed_gemini_content)
        print("✅ Fixed ai/gemini_ai.py created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating fixed gemini_ai.py: {e}")
        return False

def test_fixed_gemini():
    """Test the fixed Gemini integration"""
    print("\n🧪 Testing Fixed Gemini Integration")
    print("=" * 50)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Test the fixed version
        import sys
        sys.path.insert(0, str(Path.cwd()))
        
        from ai.gemini_ai import GeminiAI
        
        gemini = GeminiAI()
        print(f"🔍 Gemini AI Status: {gemini.get_status()}")
        
        if gemini.is_available():
            print("✅ Gemini AI is working!")
            
            # Test a chat message
            import asyncio
            response = asyncio.run(gemini.chat("Hello! Are you working for our trading bot?"))
            if response:
                print(f"✅ Chat test successful!")
                print(f"📝 Response: {response[:100]}...")
                return True
            else:
                print("❌ Chat test failed - no response")
                return False
        else:
            print("❌ Gemini AI is not available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing fixed Gemini: {e}")
        return False

def update_comprehensive_main():
    """Update the comprehensive main.py to use fixed Gemini"""
    print("\n🔧 Running Comprehensive Fix with Updated Gemini")
    print("=" * 50)
    
    # Run the comprehensive fix script
    try:
        import subprocess
        result = subprocess.run([
            "python", "comprehensive_fix.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Comprehensive fix completed successfully")
            return True
        else:
            print(f"⚠️ Comprehensive fix had issues: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Comprehensive fix timed out - running manually")
        return False
    except Exception as e:
        print(f"❌ Error running comprehensive fix: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Gemini Model Fix Script")
    print("=" * 60)
    
    # Step 1: Backup files
    backup_gemini_files()
    
    # Step 2: Test available models
    working_model = test_available_models()
    
    if not working_model:
        print("\n❌ No working Gemini models found!")
        print("💡 Possible solutions:")
        print("   1. Check your API key permissions")
        print("   2. Try a new API key from https://makersuite.google.com/app/apikey")
        print("   3. Check Google AI Studio for available models")
        return
    
    # Step 3: Fix gemini_ai.py with working model
    if not fix_gemini_ai_file(working_model):
        print("❌ Failed to fix gemini_ai.py")
        return
    
    # Step 4: Test the fixed version
    if test_fixed_gemini():
        print("\n🎉 GEMINI AI FIX SUCCESSFUL!")
    else:
        print("\n⚠️ Gemini AI fix needs manual adjustment")
    
    # Step 5: Update comprehensive integration
    print("\n🚀 Final Steps:")
    print("1. Run: python comprehensive_fix.py")
    print("2. Start server: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("3. Test chat at: http://localhost:8000")
    print()
    print("📊 Expected results:")
    print("   ✅ Enhanced Gemini AI chat responses")
    print("   ✅ ML Training section visible")
    print("   ✅ Real-time WebSocket communication")
    print("   ✅ Complete dashboard functionality")

if __name__ == "__main__":
    main()