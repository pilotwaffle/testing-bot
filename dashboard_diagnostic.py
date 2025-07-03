"""
File: dashboard_diagnostic.py
Location: E:\Trade Chat Bot\G Trading Bot\dashboard_diagnostic.py

Dashboard Comprehensive Diagnostic
Tests all aspects of the dashboard functionality, loading, and performance
"""

import requests
import time
import json
from datetime import datetime
from pathlib import Path
import re

class DashboardDiagnostic:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "dashboard_loading": {},
            "template_analysis": {},
            "ml_functionality": {},
            "chat_integration": {},
            "static_files": {},
            "api_connectivity": {},
            "performance": {},
            "overall_score": 0
        }
        
    def print_header(self, title):
        """Print formatted header"""
        print(f"\n{'='*80}")
        print(f"🔍 {title}")
        print(f"{'='*80}")
        
    def print_subheader(self, title):
        """Print formatted subheader"""
        print(f"\n{'-'*60}")
        print(f"📊 {title}")
        print(f"{'-'*60}")

    def test_dashboard_loading(self):
        """Test dashboard loading performance and response"""
        self.print_header("Dashboard Loading Tests")
        
        loading_tests = {}
        
        # Test 1: Basic connectivity
        self.print_subheader("Basic Dashboard Connectivity")
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/", timeout=30)
            end_time = time.time()
            
            loading_time = end_time - start_time
            
            loading_tests["basic_connectivity"] = {
                "success": True,
                "status_code": response.status_code,
                "loading_time": loading_time,
                "content_length": len(response.content),
                "content_type": response.headers.get('content-type', '')
            }
            
            print(f"✅ Dashboard loaded successfully")
            print(f"📊 Status Code: {response.status_code}")
            print(f"⏱️ Loading Time: {loading_time:.2f} seconds")
            print(f"📄 Content Length: {len(response.content):,} bytes")
            print(f"📋 Content Type: {response.headers.get('content-type', 'Unknown')}")
            
            # Check if it's HTML or JSON
            if response.status_code == 200:
                if 'text/html' in response.headers.get('content-type', ''):
                    print("✅ Correct: Dashboard serving HTML")
                    loading_tests["serves_html"] = True
                elif 'application/json' in response.headers.get('content-type', ''):
                    print("❌ Issue: Dashboard serving JSON instead of HTML")
                    loading_tests["serves_html"] = False
                else:
                    print("⚠️ Unknown content type")
                    loading_tests["serves_html"] = None
                    
                # Store content for analysis
                loading_tests["content_sample"] = response.text[:500]
                
            else:
                print(f"❌ Dashboard returned status code: {response.status_code}")
                loading_tests["serves_html"] = False
                
        except requests.exceptions.Timeout:
            print("❌ Dashboard loading timed out (>30 seconds)")
            loading_tests["basic_connectivity"] = {
                "success": False,
                "error": "Timeout after 30 seconds"
            }
        except Exception as e:
            print(f"❌ Dashboard loading failed: {e}")
            loading_tests["basic_connectivity"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Multiple load test
        self.print_subheader("Dashboard Load Performance")
        if loading_tests.get("basic_connectivity", {}).get("success", False):
            load_times = []
            for i in range(3):
                try:
                    start = time.time()
                    response = requests.get(f"{self.base_url}/", timeout=15)
                    end = time.time()
                    load_times.append(end - start)
                    print(f"Load {i+1}: {end - start:.2f}s")
                except:
                    print(f"Load {i+1}: Failed")
            
            if load_times:
                avg_load_time = sum(load_times) / len(load_times)
                loading_tests["performance"] = {
                    "average_load_time": avg_load_time,
                    "fastest_load": min(load_times),
                    "slowest_load": max(load_times),
                    "stability": "stable" if max(load_times) - min(load_times) < 2 else "unstable"
                }
                print(f"📊 Average load time: {avg_load_time:.2f}s")
                print(f"📊 Performance: {'✅ Good' if avg_load_time < 3 else '⚠️ Slow' if avg_load_time < 10 else '❌ Very Slow'}")
            
        self.results["dashboard_loading"] = loading_tests
        return loading_tests

    def analyze_dashboard_template(self):
        """Analyze dashboard template content and structure"""
        self.print_header("Dashboard Template Analysis")
        
        template_analysis = {}
        
        # Test 1: Template file existence
        self.print_subheader("Template File Analysis")
        template_path = Path("templates/dashboard.html")
        
        if template_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                
                print(f"✅ Template file found: {template_path}")
                print(f"📄 Template size: {len(template_content):,} characters")
                
                # Analyze template content
                template_analysis["file_exists"] = True
                template_analysis["file_size"] = len(template_content)
                
                # Check for key sections
                key_sections = {
                    "ml_training_section": ["ml-training", "ML Training", "train"],
                    "chat_interface": ["chat", "Chat", "chat-"],
                    "status_section": ["status", "Status", "health"],
                    "portfolio_section": ["portfolio", "Portfolio", "trading"],
                    "javascript": ["<script", "function", "addEventListener"],
                    "css_styling": ["<style", "class=", "."],
                    "api_calls": ["fetch", "XMLHttpRequest", "/api/"]
                }
                
                sections_found = {}
                for section, keywords in key_sections.items():
                    found = any(keyword.lower() in template_content.lower() for keyword in keywords)
                    sections_found[section] = found
                    status = "✅" if found else "❌"
                    print(f"{status} {section.replace('_', ' ').title()}: {'Found' if found else 'Missing'}")
                
                template_analysis["sections"] = sections_found
                
                # Check for template variables
                template_vars = re.findall(r'\{\{\s*(\w+)', template_content)
                template_analysis["template_variables"] = list(set(template_vars))
                
                if template_vars:
                    print(f"📋 Template variables found: {', '.join(set(template_vars))}")
                else:
                    print("⚠️ No template variables found")
                
            except Exception as e:
                print(f"❌ Error reading template file: {e}")
                template_analysis["file_exists"] = False
                template_analysis["error"] = str(e)
        else:
            print(f"❌ Template file not found: {template_path}")
            template_analysis["file_exists"] = False
        
        # Test 2: Rendered content analysis
        self.print_subheader("Rendered Dashboard Content Analysis")
        try:
            response = requests.get(f"{self.base_url}/", timeout=15)
            if response.status_code == 200:
                content = response.text
                
                # Look for key dashboard elements
                dashboard_elements = {
                    "ml_training_buttons": ["train", "ml", "model"],
                    "chat_widget": ["chat", "message", "send"],
                    "status_indicators": ["status", "health", "running"],
                    "navigation": ["nav", "menu", "dashboard"],
                    "interactive_elements": ["button", "input", "form"],
                    "data_display": ["table", "chart", "metric"]
                }
                
                elements_found = {}
                for element, keywords in dashboard_elements.items():
                    found = any(keyword in content.lower() for keyword in keywords)
                    elements_found[element] = found
                    status = "✅" if found else "❌"
                    print(f"{status} {element.replace('_', ' ').title()}: {'Present' if found else 'Missing'}")
                
                template_analysis["rendered_elements"] = elements_found
                template_analysis["rendered_content_length"] = len(content)
                
            else:
                print(f"❌ Could not fetch rendered content: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error analyzing rendered content: {e}")
        
        self.results["template_analysis"] = template_analysis
        return template_analysis

    def test_ml_functionality(self):
        """Test ML functionality accessible from dashboard"""
        self.print_header("ML Functionality Tests")
        
        ml_tests = {}
        
        # Test 1: ML status endpoint
        self.print_subheader("ML Engine Integration")
        try:
            response = requests.get(f"{self.base_url}/api/ml/status", timeout=10)
            if response.status_code == 200:
                ml_status = response.json()
                print("✅ ML Status endpoint working")
                print(f"📊 ML Engine Status: {ml_status.get('status', 'Unknown')}")
                
                if 'models' in ml_status:
                    models = ml_status['models']
                    print(f"🤖 Models available: {len(models)}")
                    for model in models[:4]:  # Show first 4
                        name = model.get('name', 'Unknown')
                        status = model.get('status', 'Unknown')
                        print(f"   • {name}: {status}")
                        
                ml_tests["ml_status"] = {
                    "success": True,
                    "status": ml_status.get('status'),
                    "models_count": len(ml_status.get('models', [])),
                    "models": ml_status.get('models', [])
                }
            else:
                print(f"❌ ML Status endpoint failed: {response.status_code}")
                ml_tests["ml_status"] = {"success": False, "error": f"Status {response.status_code}"}
                
        except Exception as e:
            print(f"❌ ML Status test failed: {e}")
            ml_tests["ml_status"] = {"success": False, "error": str(e)}
        
        # Test 2: ML training endpoints
        self.print_subheader("ML Training Functionality")
        models_to_test = ["lorentzian", "neural", "sentiment", "risk"]
        
        for model in models_to_test:
            try:
                response = requests.post(
                    f"{self.base_url}/api/ml/train/{model}",
                    json={"test_mode": True},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {model.title()} training endpoint: Working")
                    ml_tests[f"train_{model}"] = {
                        "success": True,
                        "status": result.get('status'),
                        "message": result.get('message', '')[:50] + "..."
                    }
                else:
                    print(f"❌ {model.title()} training endpoint: Failed ({response.status_code})")
                    ml_tests[f"train_{model}"] = {"success": False, "status_code": response.status_code}
                    
            except Exception as e:
                print(f"❌ {model.title()} training test failed: {e}")
                ml_tests[f"train_{model}"] = {"success": False, "error": str(e)}
        
        self.results["ml_functionality"] = ml_tests
        return ml_tests

    def test_chat_integration(self):
        """Test chat functionality from dashboard"""
        self.print_header("Chat Integration Tests")
        
        chat_tests = {}
        
        # Test 1: Chat API endpoint
        self.print_subheader("Chat API Functionality")
        test_messages = ["help", "status", "dashboard test"]
        
        for message in test_messages:
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={"message": message},
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', 'No response')
                    print(f"✅ Chat '{message}': {response_text[:60]}...")
                    
                    chat_tests[f"message_{message.replace(' ', '_')}"] = {
                        "success": True,
                        "response_length": len(response_text),
                        "response_preview": response_text[:100],
                        "has_response_field": 'response' in data
                    }
                else:
                    print(f"❌ Chat '{message}': Failed ({response.status_code})")
                    chat_tests[f"message_{message.replace(' ', '_')}"] = {
                        "success": False,
                        "status_code": response.status_code
                    }
                    
            except Exception as e:
                print(f"❌ Chat '{message}': Error - {e}")
                chat_tests[f"message_{message.replace(' ', '_')}"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test 2: Chat page accessibility
        self.print_subheader("Chat Page Integration")
        try:
            response = requests.get(f"{self.base_url}/chat", timeout=10)
            if response.status_code == 200:
                print("✅ Chat page accessible")
                chat_tests["chat_page"] = {
                    "success": True,
                    "status_code": response.status_code,
                    "content_length": len(response.content)
                }
            else:
                print(f"❌ Chat page failed: {response.status_code}")
                chat_tests["chat_page"] = {"success": False, "status_code": response.status_code}
                
        except Exception as e:
            print(f"❌ Chat page test failed: {e}")
            chat_tests["chat_page"] = {"success": False, "error": str(e)}
        
        self.results["chat_integration"] = chat_tests
        return chat_tests

    def test_static_files(self):
        """Test static file loading (CSS, JS)"""
        self.print_header("Static Files Tests")
        
        static_tests = {}
        
        # Test static file paths
        static_files = [
            "/static/js/chat.js",
            "/static/js/dashboard.js", 
            "/static/css/style.css",
            "/static/css/dashboard.css"
        ]
        
        for file_path in static_files:
            self.print_subheader(f"Testing {file_path}")
            try:
                response = requests.get(f"{self.base_url}{file_path}", timeout=10)
                
                if response.status_code == 200:
                    print(f"✅ {file_path}: Available ({len(response.content)} bytes)")
                    static_tests[file_path.replace('/', '_')] = {
                        "success": True,
                        "size": len(response.content),
                        "content_type": response.headers.get('content-type', '')
                    }
                elif response.status_code == 404:
                    print(f"⚠️ {file_path}: Not found (404)")
                    static_tests[file_path.replace('/', '_')] = {
                        "success": False,
                        "status_code": 404,
                        "note": "File may not exist or path incorrect"
                    }
                else:
                    print(f"❌ {file_path}: Error {response.status_code}")
                    static_tests[file_path.replace('/', '_')] = {
                        "success": False,
                        "status_code": response.status_code
                    }
                    
            except Exception as e:
                print(f"❌ {file_path}: Failed - {e}")
                static_tests[file_path.replace('/', '_')] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.results["static_files"] = static_tests
        return static_tests

    def test_api_connectivity(self):
        """Test API endpoints used by dashboard"""
        self.print_header("API Connectivity Tests")
        
        api_tests = {}
        
        # Key API endpoints dashboard might use
        endpoints = [
            ("/health", "System Health"),
            ("/api/ml/status", "ML Status"),
            ("/api/portfolio", "Portfolio Data"),
            ("/api/system/info", "System Information"),
            ("/status", "Bot Status")
        ]
        
        for endpoint, name in endpoints:
            self.print_subheader(f"Testing {name}")
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"✅ {name}: Working (JSON response)")
                        api_tests[endpoint.replace('/', '_')] = {
                            "success": True,
                            "response_type": "json",
                            "data_keys": list(data.keys()) if isinstance(data, dict) else "non-dict"
                        }
                    except:
                        print(f"✅ {name}: Working (Non-JSON response)")
                        api_tests[endpoint.replace('/', '_')] = {
                            "success": True,
                            "response_type": "text",
                            "content_length": len(response.content)
                        }
                else:
                    print(f"❌ {name}: Failed ({response.status_code})")
                    api_tests[endpoint.replace('/', '_')] = {
                        "success": False,
                        "status_code": response.status_code
                    }
                    
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
                api_tests[endpoint.replace('/', '_')] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.results["api_connectivity"] = api_tests
        return api_tests

    def calculate_overall_score(self):
        """Calculate overall dashboard health score"""
        self.print_header("Overall Dashboard Health Score")
        
        scores = {
            "dashboard_loading": 0,
            "template_analysis": 0,
            "ml_functionality": 0,
            "chat_integration": 0,
            "static_files": 0,
            "api_connectivity": 0
        }
        
        # Dashboard loading score (25 points)
        loading = self.results.get("dashboard_loading", {})
        if loading.get("basic_connectivity", {}).get("success", False):
            scores["dashboard_loading"] += 15
            if loading.get("serves_html", False):
                scores["dashboard_loading"] += 10
        
        # Template analysis score (15 points)
        template = self.results.get("template_analysis", {})
        if template.get("file_exists", False):
            scores["template_analysis"] += 5
        sections = template.get("sections", {})
        scores["template_analysis"] += sum(2 for found in sections.values() if found)
        
        # ML functionality score (20 points)
        ml = self.results.get("ml_functionality", {})
        if ml.get("ml_status", {}).get("success", False):
            scores["ml_functionality"] += 10
        train_tests = [k for k in ml.keys() if k.startswith("train_")]
        scores["ml_functionality"] += min(10, len([k for k in train_tests if ml[k].get("success", False)]) * 2.5)
        
        # Chat integration score (20 points)
        chat = self.results.get("chat_integration", {})
        chat_tests = [k for k in chat.keys() if k.startswith("message_")]
        working_chat = len([k for k in chat_tests if chat[k].get("success", False)])
        scores["chat_integration"] += min(15, working_chat * 5)
        if chat.get("chat_page", {}).get("success", False):
            scores["chat_integration"] += 5
        
        # Static files score (10 points)
        static = self.results.get("static_files", {})
        static_working = len([k for k in static.keys() if static[k].get("success", False)])
        scores["static_files"] = min(10, static_working * 2.5)
        
        # API connectivity score (10 points)
        api = self.results.get("api_connectivity", {})
        api_working = len([k for k in api.keys() if api[k].get("success", False)])
        scores["api_connectivity"] = min(10, api_working * 2)
        
        total_score = sum(scores.values())
        self.results["overall_score"] = total_score
        self.results["score_breakdown"] = scores
        
        print(f"📊 Score Breakdown:")
        for category, score in scores.items():
            category_name = category.replace('_', ' ').title()
            print(f"   {category_name}: {score}")
        
        print(f"\n🎯 Overall Dashboard Score: {total_score}/100")
        
        if total_score >= 90:
            grade = "🎉 Excellent"
            status = "Dashboard is working perfectly!"
        elif total_score >= 75:
            grade = "✅ Good"  
            status = "Dashboard is working well with minor issues"
        elif total_score >= 60:
            grade = "⚠️ Fair"
            status = "Dashboard has some issues that need attention"
        else:
            grade = "❌ Poor"
            status = "Dashboard has significant issues"
            
        print(f"📋 Grade: {grade}")
        print(f"💡 Status: {status}")
        
        return total_score, scores

    def save_detailed_report(self):
        """Save detailed diagnostic report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_diagnostic_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n📄 Detailed report saved: {filename}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")

    def run_full_diagnostic(self):
        """Run complete dashboard diagnostic"""
        print("🔧 Dashboard Comprehensive Diagnostic")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Testing dashboard: {self.base_url}")
        
        try:
            # Run all test suites
            self.test_dashboard_loading()
            self.analyze_dashboard_template()
            self.test_ml_functionality()
            self.test_chat_integration()
            self.test_static_files()
            self.test_api_connectivity()
            
            # Calculate final score
            score, breakdown = self.calculate_overall_score()
            
            # Save report
            self.save_detailed_report()
            
            print(f"\n🏁 Dashboard diagnostic completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return score, self.results
            
        except KeyboardInterrupt:
            print("\n🛑 Diagnostic interrupted by user")
        except Exception as e:
            print(f"\n❌ Diagnostic failed: {e}")

def main():
    """Main function to run the diagnostic"""
    print("🔧 Dashboard Comprehensive Diagnostic")
    print("=" * 80)
    print("This script will thoroughly test your Elite Trading Bot dashboard:")
    print("• 🌐 Dashboard loading and performance")
    print("• 📄 Template structure and content")
    print("• 🤖 ML functionality integration")
    print("• 💬 Chat system integration")
    print("• 📁 Static file loading")
    print("• 🔗 API connectivity")
    print("• 📊 Overall health score")
    print()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server detected at http://localhost:8000")
        else:
            print("⚠️ Server responding but may have issues")
    except:
        print("❌ Server not detected at http://localhost:8000")
        print("Please start your server first:")
        print("python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    # Run diagnostic
    diagnostic = DashboardDiagnostic()
    score, results = diagnostic.run_full_diagnostic()
    
    print(f"\n🎯 Final Dashboard Health Score: {score}/100")

if __name__ == "__main__":
    main()