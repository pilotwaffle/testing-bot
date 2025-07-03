"""
File: comprehensive_diagnostic.py
Location: E:\Trade Chat Bot\G Trading Bot\comprehensive_diagnostic.py

Comprehensive Functionality Diagnostic
Tests ML Training, Chat, Status Button, and Market Data Button functionality.

This corrected version properly validates HTTP status codes to ensure that
a '404 Not Found' or other error code is treated as a failure, providing
an accurate and honest assessment of the system's health.
"""

import requests
import json
import time
import websocket
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

class ComprehensiveDiagnostic:
    """
    Runs a suite of diagnostic tests against the trading bot's API and frontend.
    """
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "details": {
                "ml_training": {},
                "chat": {},
                "status_button": {},
                "market_data": {},
                "frontend": {},
                "overall_health": {}
            }
        }
        self.session = requests.Session()
        # Common headers for requests
        self.session.headers.update({"Accept": "application/json"})

    def print_header(self, title):
        """Prints a formatted header to the console."""
        print(f"\n{'='*80}")
        print(f"🔍 {title}")
        print(f"{'='*80}")

    def print_subheader(self, title):
        """Prints a formatted subheader to the console."""
        print(f"\n{'-'*60}")
        print(f"📊 {title}")
        print(f"{'-'*60}")

    def test_endpoint(self, name, endpoint, method="GET", data=None, expected_status=200):
        """
        Tests a single API endpoint and validates the response.

        Args:
            name (str): A descriptive name for the test.
            endpoint (str): The API endpoint path (e.g., "/api/ml/status").
            method (str): The HTTP method to use ("GET" or "POST").
            data (dict, optional): The JSON payload for POST requests.
            expected_status (int): The expected HTTP status code for a successful test.

        Returns:
            dict: A dictionary containing detailed test results.
        """
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        result = {
            "name": name,
            "passed": False,
            "status_code": None,
            "response_time": None,
            "error": None,
            "data": None
        }
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            result["response_time"] = time.time() - start_time
            result["status_code"] = response.status_code

            if response.status_code == expected_status:
                result["passed"] = True
                print(f"✅ PASSED: {name} ({response.status_code} OK)")
            else:
                result["error"] = f"Expected status {expected_status}, but got {response.status_code}."
                print(f"❌ FAILED: {name} ({result['error']})")

            # Attempt to parse JSON, fall back to text if it fails
            try:
                result["data"] = response.json()
            except json.JSONDecodeError:
                result["data"] = response.text[:500] # Store a snippet of non-JSON response

        except requests.exceptions.RequestException as e:
            result["response_time"] = time.time() - start_time
            result["error"] = str(e)
            print(f"❌ FAILED: {name} (Connection Error: {e})")

        return result

    def run_ml_training_tests(self):
        """Tests all ML Training functionality."""
        self.print_header("ML Training Functionality Tests")
        ml_results = []

        # Define all ML tests
        tests_to_run = [
            {"name": "ML Engine Status", "endpoint": "/api/ml/status"},
            {"name": "ML Models List", "endpoint": "/api/ml/models"},
            {"name": "ML Test Endpoint", "endpoint": "/api/ml/test"},
            {"name": "Train Lorentzian Classifier", "endpoint": "/api/ml/train/lorentzian", "method": "POST"},
            {"name": "Train Neural Network", "endpoint": "/api/ml/train/neural", "method": "POST"},
            {"name": "Train Social Sentiment", "endpoint": "/api/ml/train/sentiment", "method": "POST"},
            {"name": "Train Risk Assessment", "endpoint": "/api/ml/train/risk", "method": "POST"},
            {"name": "Train All Models", "endpoint": "/api/ml/train/all", "method": "POST"},
        ]

        for test in tests_to_run:
            self.print_subheader(f"Testing {test['name']}")
            result = self.test_endpoint(
                name=test['name'],
                endpoint=test['endpoint'],
                method=test.get('method', 'GET'),
                data={"test_mode": True} if test.get('method') == 'POST' else None
            )
            if result.get("passed"):
                if result['data'] and isinstance(result['data'], dict):
                    if 'models_available' in result['data']:
                        print(f"   -> Models Available: {result['data']['models_available']}")
                    if 'models' in result['data']:
                        print(f"   -> Found {len(result['data']['models'])} models.")
            ml_results.append(result)

        self.results["details"]["ml_training"] = ml_results

    def run_chat_tests(self):
        """Tests all Chat functionality, including API and WebSockets."""
        self.print_header("Chat Functionality Tests")
        chat_results = []

        # Test 1: Chat API Endpoint
        self.print_subheader("Testing Chat API Endpoint")
        chat_messages = ["Hello", "status", "help", "market data"]
        for msg in chat_messages:
            result = self.test_endpoint(
                name=f"Chat message '{msg}'",
                endpoint="/api/chat",
                method="POST",
                data={"message": msg}
            )
            if result.get("passed") and isinstance(result.get("data"), dict):
                response_text = result["data"].get("response", "")
                print(f"   -> Bot Response: {response_text[:100]}")
                if not response_text or len(response_text) < 5:
                    result["passed"] = False # Downgrade to failure if response is empty
                    result["error"] = "Received an empty or meaningless response."
                    print("   -> ⚠️ Response quality is poor.")
            chat_results.append(result)

        # Test 2: WebSocket Connection
        self.print_subheader("Testing WebSocket Connection")
        chat_results.append(self.test_websocket())

        self.results["details"]["chat"] = chat_results

    def test_websocket(self):
        """Tests the WebSocket connection for real-time chat."""
        ws_url = f"ws://{self.base_url.split('//')[1]}/ws"
        result = {"name": "WebSocket Connection", "passed": False, "error": None}
        print(f"🔌 Connecting to WebSocket at {ws_url}...")
        try:
            ws = websocket.create_connection(ws_url, timeout=10)
            print("   -> Connection successful.")
            
            # Send a test message
            ws.send(json.dumps({"type": "chat", "message": "WebSocket test"}))
            print("   -> Sent test message.")
            
            # Wait for a response
            response = ws.recv()
            print(f"   -> Received response: {response[:120]}")
            ws.close()
            
            if response:
                result["passed"] = True
                print("✅ PASSED: WebSocket test")
            else:
                result["error"] = "Received an empty response from WebSocket."
                print("❌ FAILED: WebSocket test (Empty response)")

        except Exception as e:
            result["error"] = str(e)
            print(f"❌ FAILED: WebSocket test ({e})")
        
        return result

    def run_status_and_market_data_tests(self):
        """Tests status and market data endpoints."""
        self.print_header("Status & Market Data Tests")
        combined_results = []
        
        tests_to_run = [
            # Status Endpoints
            {"name": "Main Status", "endpoint": "/status"},
            {"name": "Health Check", "endpoint": "/health"},
            {"name": "System Info", "endpoint": "/api/system/info"},
            {"name": "Performance Metrics", "endpoint": "/api/performance"},
            # Market Data Endpoints
            {"name": "General Market Data", "endpoint": "/api/market-data"},
            {"name": "Specific Symbol (BTC/USD)", "endpoint": "/api/market-data/BTC/USD"},
            {"name": "Historical Data (BTC/USD)", "endpoint": "/api/market-data/historical?symbol=BTC/USD&period=1d"},
            {"name": "Real-time Prices", "endpoint": "/api/prices/realtime"},
        ]

        for test in tests_to_run:
            self.print_subheader(f"Testing {test['name']}")
            result = self.test_endpoint(name=test['name'], endpoint=test['endpoint'])
            combined_results.append(result)
        
        self.results["details"]["status_button"] = [r for r in combined_results if "Status" in r["name"] or "Health" in r["name"]]
        self.results["details"]["market_data"] = [r for r in combined_results if "Market" in r["name"] or "Prices" in r["name"]]

    def run_frontend_tests(self):
        """Tests basic frontend rendering using Selenium."""
        self.print_header("Frontend Functionality Tests")
        frontend_results = []
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("window-size=1920,1080")

        try:
            print("🌐 Starting headless Chrome browser...")
            driver = webdriver.Chrome(options=chrome_options)
        except WebDriverException as e:
            print(f"❌ FAILED: Could not start Selenium WebDriver. Skipping frontend tests. ({e})")
            print("   -> Make sure you have chromedriver installed and in your PATH.")
            self.results["details"]["frontend"] = [{"name": "Setup", "passed": False, "error": str(e)}]
            return

        try:
            # Test 1: Dashboard Page
            self.print_subheader("Testing Dashboard Page")
            url = f"{self.base_url}/"
            print(f"   -> Navigating to {url}")
            driver.get(url)
            
            elements_to_find = {
                "ML Training Section": (By.ID, "ml-training-section"),
                "Status Button": (By.ID, "status-button"),
                "Market Data Button": (By.ID, "market-data-button")
            }

            for name, (by, selector) in elements_to_find.items():
                result = {"name": f"Find '{name}' on Dashboard", "passed": False}
                try:
                    WebDriverWait(driver, 5).until(EC.presence_of_element_located((by, selector)))
                    result["passed"] = True
                    print(f"   -> ✅ Found element: {name}")
                except TimeoutException:
                    result["error"] = f"Element not found using {by}='{selector}'"
                    print(f"   -> ❌ Missing element: {name}")
                frontend_results.append(result)

        except Exception as e:
            error_msg = f"An unexpected error occurred during frontend tests: {e}"
            print(f"❌ FAILED: {error_msg}")
            frontend_results.append({"name": "General Frontend Test", "passed": False, "error": error_msg})
        
        finally:
            print("🌐 Closing browser...")
            driver.quit()
        
        self.results["details"]["frontend"] = frontend_results

    def calculate_scores_and_generate_report(self):
        """Calculates final scores and prints a summary report."""
        self.print_header("COMPREHENSIVE DIAGNOSTIC REPORT")

        scores = {}
        for section, section_results in self.results["details"].items():
            if not section_results:
                scores[section] = 0.0
                continue
            passed_count = sum(1 for r in section_results if r.get("passed"))
            total_count = len(section_results)
            scores[section] = (passed_count / total_count) * 100 if total_count > 0 else 0.0
        
        self.results["summary"]["scores"] = scores
        
        print("📊 FUNCTIONALITY SCORES:")
        print(f"   🤖 ML Training:   {scores.get('ml_training', 0.0):.1f}%")
        print(f"   💬 Chat System:   {scores.get('chat', 0.0):.1f}%")
        print(f"   📊 Status & Health: {scores.get('status_button', 0.0):.1f}%")
        print(f"   📈 Market Data:   {scores.get('market_data', 0.0):.1f}%")
        print(f"   🌐 Frontend:      {scores.get('frontend', 0.0):.1f}%")

        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        self.results["summary"]["overall_score"] = overall_score
        print(f"   ------------------------------------")
        print(f"   🎯 OVERALL SCORE: {overall_score:.1f}%")

        # Provide Recommendations
        print("\n🔧 RECOMMENDATIONS:")
        if overall_score == 100.0:
            print("   🎉 Excellent! All systems are fully operational.")
        else:
            if scores.get('ml_training', 0) < 100:
                print("   -> ⚠️ ML TRAINING: One or more ML API endpoints are failing. Check the server logs for errors related to '/api/ml/...'.")
            if scores.get('chat', 0) < 100:
                print("   -> ⚠️ CHAT SYSTEM: The chat API or WebSocket is not fully functional. Verify the '/api/chat' endpoint and WebSocket connection handler.")
            if scores.get('status_button', 0) < 100:
                 print("   -> ⚠️ STATUS & HEALTH: Core status endpoints like '/health' or '/status' are failing. This is critical for monitoring.")
            if scores.get('market_data', 0) < 100:
                print("   -> ⚠️ MARKET DATA: The bot cannot fetch market data. Check all '/api/market-data/...' endpoints and their upstream data sources.")
            if scores.get('frontend', 0) < 100:
                 print("   -> ⚠️ FRONTEND: The web interface is missing critical elements. Check your HTML templates and JavaScript to ensure all components are rendered correctly.")
            if overall_score < 50:
                 print("\n   -> 🚨 CRITICAL: The system has major failures and is not operational.")

        self._save_detailed_report()

    def _save_detailed_report(self):
        """Saves the detailed results dictionary to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostic_report_{timestamp}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=4)
            print(f"\n📄 Detailed report saved to: {filename}")
        except IOError as e:
            print(f"\n❌ Failed to save detailed report: {e}")

    def run_full_diagnostic(self):
        """Runs the entire diagnostic suite in order."""
        print("🚀 Starting Comprehensive Functionality Diagnostic")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Testing server: {self.base_url}")

        try:
            self.run_ml_training_tests()
            self.run_chat_tests()
            self.run_status_and_market_data_tests()
            self.run_frontend_tests()
            self.calculate_scores_and_generate_report()

        except KeyboardInterrupt:
            print("\n\n🛑 Diagnostic interrupted by user.")
        except Exception as e:
            print(f"\n\n❌ A critical error occurred during the diagnostic: {e}")
        
        finally:
            print(f"\n🏁 Diagnostic completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to initialize and run the diagnostic."""
    print("🔧 Comprehensive Functionality Diagnostic")
    print("=" * 80)
    print("This script will accurately test all major components of the trading bot.")
    print()

    # Pre-flight check to see if the server is reachable
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server detected at http://localhost:8000\n")
        else:
            print(f"⚠️ Server responded with status {response.status_code}. Issues may exist.\n")
    except requests.ConnectionError:
        print("❌ Server not detected at http://localhost:8000.")
        print("   Please ensure the bot's server is running before starting the diagnostic.")
        print("   You can typically start it with a command like:")
        print("   uvicorn main:app --reload")
        return
    
    diagnostic = ComprehensiveDiagnostic()
    diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    main()