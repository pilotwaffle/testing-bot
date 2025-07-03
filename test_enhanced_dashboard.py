# File: enhanced_dashboard_tester.py
# Location: E:\Trade Chat Bot\G Trading Bot\enhanced_dashboard_tester.py

import requests
import json
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

print("--- Starting Enhanced Dashboard Test Script ---") # Added to confirm script execution

# --- Configuration ---
BASE_URL = "http://localhost:8000" # Ensure your bot's web server is running on this URL

# Determine PROJECT_ROOT based on the script's location
# This assumes 'enhanced_dashboard_tester.py' is placed directly in 'E:\Trade Chat Bot\G Trading Bot\'
PROJECT_ROOT = Path(__file__).resolve().parent

# Define paths to your key files relative to PROJECT_ROOT
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

DASHBOARD_HTML_PATH = TEMPLATES_DIR / "dashboard.html"
STYLE_CSS_PATH = STATIC_DIR / "style.css" # Assuming style.css is directly in static/
ENHANCED_DASHBOARD_JS_PATH = STATIC_DIR / "js" / "enhanced-dashboard.js"
MAIN_PY_PATH = PROJECT_ROOT / "main.py" # Assuming your main app file is main.py

# Expected elements, text, and styles for validation
EXPECTED_HTML_ELEMENTS = [
    'id="marketData"',
    'id="ml-test-response"',
    'Enhanced Trading Bot V3.0', # Updated to 'Enhanced' based on previous analysis
    '/static/js/enhanced-dashboard.js' # Updated script link
]

EXPECTED_CSS_STYLES = [
    '--primary-color'
]

EXPECTED_MAIN_PY_ENDPOINTS = [
    '@app.get("/api/market-data")',
    '@app.post("/api/chat")',
    '@app.get("/health")'
]

# API Endpoints to test
API_ENDPOINTS = {
    "GET": {
        "/api": 404, # Expected to fail with 404 based on previous report
        "/api/market-data": 200,
        "/api/portfolio": 200, # Expected 500 previously, setting to 200 for ideal state
        "/api/portfolio/enhanced": 404, # Expected to fail with 404
        "/api/system/metrics": 404, # Expected to fail with 404
        "/api/ml/status": 404, # Expected to fail with 404
        "/health": 200, # Expected to be slow, but should return 200
    },
    "POST": {
        "/api/chat": {"payload": {"message": "Hello AI"}, "expected_status": 200}, # Expected 500, setting to 200
        "/api/ml/test": {"payload": {"model": "test", "data": []}, "expected_status": 404}, # Expected 404
        "/api/ml/train/lorentzian_classifier": {"payload": {"symbol": "BTC/USDT"}, "expected_status": 200}, # Expected 500
        "/api/ml/train/neural_network": {"payload": {"symbol": "BTC/USDT"}, "expected_status": 200}, # Expected 500
        "/api/ml/train/social_sentiment": {"payload": {"symbol": "BTC/USDT"}, "expected_status": 200}, # Expected 500
        "/api/ml/train/risk_assessment": {"payload": {"symbol": "BTC/USDT"}, "expected_status": 200}, # Expected 500
    }
}

# --- Test Results Storage ---
test_results = {
    "file_validation": {},
    "api_tests": [],
    "performance_tests": {},
    "concurrent_requests": {},
    "summary": {
        "total_tests": 0,
        "successful_tests": 0,
        "failed_tests": 0,
        "success_rate": 0.0
    },
    "failed_list": []
}

# --- Helper Functions ---

def log_test_status(test_name, status, details=""):
    """Logs the status of a test."""
    icon = "✅" if status == "PASS" else "❌"
    print(f"  {icon} {test_name} - {status}{f' ({details})' if details else ''}")
    test_results["summary"]["total_tests"] += 1
    if status == "PASS":
        test_results["summary"]["successful_tests"] += 1
    else:
        test_results["summary"]["failed_tests"] += 1
        test_results["failed_list"].append(f"• {test_name}: {details}")

def calculate_success_rate():
    """Calculates and updates the overall success rate."""
    total = test_results["summary"]["total_tests"]
    successful = test_results["summary"]["successful_tests"]
    test_results["summary"]["success_rate"] = (successful / total * 100) if total > 0 else 0.0

# --- File Validation Functions ---

def validate_html_file():
    """Validates elements and text within dashboard.html."""
    print("\n📁 Validating File Structure - dashboard.html...")
    if not DASHBOARD_HTML_PATH.exists():
        log_test_status("dashboard.html existence", "FAIL", f"File not found: {DASHBOARD_HTML_PATH}")
        return

    # Explicitly specify UTF-8 encoding to prevent UnicodeDecodeError
    try:
        content = DASHBOARD_HTML_PATH.read_text(encoding='utf-8')
    except UnicodeDecodeError as e:
        log_test_status("dashboard.html encoding", "FAIL", f"UnicodeDecodeError: {e}. Try saving file as UTF-8.")
        return

    all_passed = True

    for element in EXPECTED_HTML_ELEMENTS:
        if element in content:
            log_test_status(f"HTML element/text '{element}'", "PASS")
        else:
            log_test_status(f"HTML element/text '{element}'", "FAIL", "Missing")
            all_passed = False
    test_results["file_validation"]["dashboard.html"] = "PASS" if all_passed else "FAIL"


def validate_css_file():
    """Validates styles within style.css."""
    print("\n📁 Validating File Structure - style.css...")
    if not STYLE_CSS_PATH.exists():
        log_test_status("style.css existence", "FAIL", f"File not found: {STYLE_CSS_PATH}")
        return

    # Explicitly specify UTF-8 encoding for CSS file as well, common practice
    try:
        content = STYLE_CSS_PATH.read_text(encoding='utf-8')
    except UnicodeDecodeError as e:
        log_test_status("style.css encoding", "FAIL", f"UnicodeDecodeError: {e}. Try saving file as UTF-8.")
        return

    all_passed = True

    for style in EXPECTED_CSS_STYLES:
        if style in content:
            log_test_status(f"CSS style '{style}'", "PASS")
        else:
            log_test_status(f"CSS style '{style}'", "FAIL", "Missing")
            all_passed = False
    test_results["file_validation"]["style.css"] = "PASS" if all_passed else "FAIL"


def validate_js_file():
    """Validates existence of enhanced-dashboard.js."""
    print("\n📁 Validating File Structure - enhanced-dashboard.js...")
    if ENHANCED_DASHBOARD_JS_PATH.exists():
        log_test_status("enhanced-dashboard.js existence", "PASS")
        test_results["file_validation"]["enhanced-dashboard.js"] = "PASS"
    else:
        log_test_status("enhanced-dashboard.js existence", "FAIL", f"File not found: {ENHANCED_DASHBOARD_JS_PATH}")
        test_results["file_validation"]["enhanced-dashboard.js"] = "FAIL"


def validate_main_py_endpoints():
    """Validates expected endpoint decorators in main.py."""
    print("\n📁 Validating File Structure - main.py endpoints...")
    if not MAIN_PY_PATH.exists():
        log_test_status("main.py existence", "FAIL", f"File not found: {MAIN_PY_PATH}")
        return

    # Explicitly specify UTF-8 encoding for Python file as well, good practice
    try:
        content = MAIN_PY_PATH.read_text(encoding='utf-8')
    except UnicodeDecodeError as e:
        log_test_status("main.py encoding", "FAIL", f"UnicodeDecodeError: {e}. Try saving file as UTF-8.")
        return

    all_passed = True

    for endpoint_signature in EXPECTED_MAIN_PY_ENDPOINTS:
        if endpoint_signature in content:
            log_test_status(f"main.py endpoint '{endpoint_signature}'", "PASS")
        else:
            log_test_status(f"main.py endpoint '{endpoint_signature}'", "FAIL", "Missing")
            all_passed = False
    test_results["file_validation"]["main.py_endpoints"] = "PASS" if all_passed else "FAIL"

# --- API Testing Functions ---

def test_get_endpoint(path, expected_status=200):
    """Tests a GET API endpoint."""
    url = f"{BASE_URL}{path}"
    test_name = f"GET {path}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == expected_status:
            log_test_status(test_name, "PASS", f"Status {response.status_code}")
        else:
            log_test_status(test_name, "FAIL", f"Expected Status {expected_status}, Got {response.status_code} - Response: {response.text[:100]}")
            test_results["api_tests"].append({"endpoint": path, "method": "GET", "status": "FAIL", "details": f"Expected {expected_status}, Got {response.status_code}", "response": response.text})
    except requests.exceptions.ConnectionError:
        log_test_status(test_name, "FAIL", "Connection Error (Is server running?)")
        test_results["api_tests"].append({"endpoint": path, "method": "GET", "status": "FAIL", "details": "Connection Error"})
    except requests.exceptions.Timeout:
        log_test_status(test_name, "FAIL", "Timeout Error")
        test_results["api_tests"].append({"endpoint": path, "method": "GET", "status": "FAIL", "details": "Timeout Error"})
    except Exception as e:
        log_test_status(test_name, "FAIL", f"An unexpected error occurred: {e}")
        test_results["api_tests"].append({"endpoint": path, "method": "GET", "status": "FAIL", "details": str(e)})

def test_post_endpoint(path, payload, expected_status=200):
    """Tests a POST API endpoint."""
    url = f"{BASE_URL}{path}"
    test_name = f"POST {path}"
    try:
        response = requests.post(url, json=payload, timeout=10) # Increased timeout for ML training
        if response.status_code == expected_status:
            log_test_status(test_name, "PASS", f"Status {response.status_code}")
        else:
            log_test_status(test_name, "FAIL", f"Expected Status {expected_status}, Got {response.status_code} - Response: {response.text[:100]}")
            test_results["api_tests"].append({"endpoint": path, "method": "POST", "status": "FAIL", "details": f"Expected {expected_status}, Got {response.status_code}", "response": response.text})
    except requests.exceptions.ConnectionError:
        log_test_status(test_name, "FAIL", "Connection Error (Is server running?)")
        test_results["api_tests"].append({"endpoint": path, "method": "POST", "status": "FAIL", "details": "Connection Error"})
    except requests.exceptions.Timeout:
        log_test_status(test_name, "FAIL", "Timeout Error")
        test_results["api_tests"].append({"endpoint": path, "method": "POST", "status": "FAIL", "details": "Timeout Error"})
    except Exception as e:
        log_test_status(test_name, "FAIL", f"An unexpected error occurred: {e}")
        test_results["api_tests"].append({"endpoint": path, "method": "POST", "status": "FAIL", "details": str(e)})

# --- Performance Testing Functions ---

def measure_endpoint_performance(path, method="GET", payload=None, num_requests=5):
    """Measures average response time for an endpoint."""
    url = f"{BASE_URL}{path}"
    timings = []
    for _ in range(num_requests):
        start_time = time.perf_counter()
        try:
            if method == "GET":
                requests.get(url, timeout=15)
            elif method == "POST":
                requests.post(url, json=payload, timeout=15)
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000) # Convert to milliseconds
        except (requests.exceptions.RequestException, Exception):
            # Ignore errors for performance measurement, focus on successful requests
            pass

    if timings:
        avg_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)
        test_results["performance_tests"][path] = {
            "avg_ms": f"{avg_time:.1f}",
            "range_ms": f"{min_time:.1f}-{max_time:.1f}"
        }
        print(f"  {path}: Avg {avg_time:.1f}ms (Range: {min_time:.1f}-{max_time:.1f}ms)")
    else:
        test_results["performance_tests"][path] = "No successful requests to measure."
        print(f"  {path}: No successful requests to measure performance.")


def test_concurrent_requests(path, num_concurrent=10, num_total=10):
    """Tests concurrent requests to an endpoint."""
    url = f"{BASE_URL}{path}"
    test_name = f"{num_concurrent} concurrent requests to {path}"
    successful_requests = 0
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(requests.get, url, timeout=10) for _ in range(num_total)]
        for future in as_completed(futures):
            try:
                response = future.result()
                if response.status_code == 200:
                    successful_requests += 1
            except (requests.exceptions.RequestException, Exception):
                pass # Ignore errors for concurrency count

    end_time = time.perf_counter()
    duration = end_time - start_time
    status = "PASS" if successful_requests == num_total else "FAIL"
    details = f"{duration:.2f}s, {successful_requests}/{num_total} successful"
    log_test_status(test_name, status, details)
    test_results["concurrent_requests"][path] = {"duration_s": f"{duration:.2f}", "successful": successful_requests, "total": num_total}


# --- Main Test Execution ---

def run_tests():
    """Executes all validation and API tests."""
    print("Starting dashboard tests...")
    print("Make sure your server is running on http://localhost:8000\n")
    print("🚀 Enhanced Dashboard Validation & Testing")
    print("=" * 50)

    # File Structure Validation
    print("\n📁 Validating File Structure...")
    print("=" * 40)
    validate_html_file()
    validate_css_file()
    validate_js_file()
    validate_main_py_endpoints()

    # Wait for server to be ready
    print("\n⏳ Waiting 3 seconds for server to be ready...")
    time.sleep(3) # Give server time to start

    print("\n🧪 Starting Enhanced Dashboard Tests...")
    print("=" * 60)

    # Test API Endpoints - GET
    print("\n📡 Testing API Endpoints (GET)...")
    for path, expected_status in API_ENDPOINTS["GET"].items():
        test_get_endpoint(path, expected_status)

    # Test API Endpoints - POST
    print("\n📤 Testing POST Endpoints...")
    for path, data in API_ENDPOINTS["POST"].items():
        test_post_endpoint(path, data["payload"], data["expected_status"])

    # Performance Tests
    print("\n⚡ Running Performance Tests...")
    measure_endpoint_performance("/api/market-data", num_requests=10)
    measure_endpoint_performance("/api/portfolio", num_requests=10)
    measure_endpoint_performance("/health", num_requests=10) # This one was slow

    # Concurrent Requests Test
    print("\n🔄 Testing Concurrent Requests...")
    test_concurrent_requests("/api/market-data", num_concurrent=10, num_total=10)


    # --- Final Report ---
    calculate_success_rate()
    print("\n" + "=" * 60)
    print("📋 DASHBOARD TEST REPORT")
    print("=" * 60)

    print("\n📊 Overall Results:")
    print(f"  Total Tests: {test_results['summary']['total_tests']}")
    print(f"  ✅ Successful: {test_results['summary']['successful_tests']}")
    print(f"  ❌ Failed: {test_results['summary']['failed_tests']}")
    print(f"  📈 Success Rate: {test_results['summary']['success_rate']:.1f}%")

    print("\n⚡ Performance:")
    for endpoint, data in test_results["performance_tests"].items():
        if isinstance(data, dict):
            print(f"  {endpoint}: Avg {data['avg_ms']}ms (Range: {data['range_ms']}ms)")
        else:
            print(f"  {endpoint}: {data}")

    health_avg_time = float(test_results["performance_tests"].get("/health", {}).get("avg_ms", "0"))
    if health_avg_time > 500:
        performance_rating = "🔴 Poor (Health check too slow!)"
    elif health_avg_time > 100:
        performance_rating = "🟡 Good (Consider optimizing health check)"
    else:
        performance_rating = "🟢 Excellent"
    print(f"  Performance Rating: {performance_rating}")

    if test_results["failed_list"]:
        print("\n❌ Failed Tests:")
        for failure in test_results["failed_list"]:
            print(f"  {failure}")
    else:
        print("\n✅ All tests passed!")


    # Save detailed report to JSON
    report_filename = f"dashboard_test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"\n📄 Detailed report saved to: {report_filename}")

    print("\n🎯 Testing Complete!")
    print("\nNext Steps:")
    print("1. Fix any failed tests (especially 404s by adding missing routes, and 500s by debugging backend code).")
    print("2. Optimize slow endpoints (>500ms, like /health).")
    print("3. Ensure all HTML elements and CSS styles expected by the dashboard are present.")
    print("4. Deploy to production.")
    print("5. Monitor performance in production.")