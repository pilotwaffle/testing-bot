import requests
import json
import sys
import time

# --- Configuration ---
# Base URL of your FastAPI application
# Make sure this matches where your FastAPI app is running (e.g., http://localhost:8000)
BASE_URL = "http://localhost:8000"

# List of endpoints to check, along with their expected content type and specific validation logic
# 'json' for API endpoints, 'html' for rendered pages
ENDPOINTS_TO_CHECK = [
    {"path": "/", "name": "Dashboard Page", "expected_type": "html"},
    {"path": "/health", "name": "Health Check API", "expected_type": "json", "validator": lambda d: d.get("status") == "healthy"},
    {"path": "/api", "name": "API Info Endpoint", "expected_type": "json", "validator": lambda d: "name" in d and "endpoints" in d},
    {"path": "/api/market-data", "name": "Market Data API", "expected_type": "json", "validator": lambda d: d.get("success") is True and "symbols" in d},
    {"path": "/api/trading-pairs", "name": "Trading Pairs API", "expected_type": "json", "validator": lambda d: d.get("success") is True and "pairs" in d},
    {"path": "/api/market-overview", "name": "Market Overview API", "expected_type": "json", "validator": lambda d: d.get("success") is True and "overview" in d},
    {"path": "/api/portfolio", "name": "Portfolio API", "expected_type": "json", "validator": lambda d: d.get("status") == "success" or d.get("status") == "error"}, # Can be success or error with message
    {"path": "/api/portfolio/enhanced", "name": "Enhanced Portfolio API", "expected_type": "json", "validator": lambda d: d.get("success") is True and "portfolio" in d},
    {"path": "/api/chat", "name": "Chat API (GET check)", "expected_type": "json", "method": "GET", "expected_status_codes": [405]}, # Expect 405 Method Not Allowed for GET on POST endpoint
    {"path": "/api/ml/status", "name": "ML Status API", "expected_type": "json", "validator": lambda d: d.get("status") == "success" and "models" in d},
    {"path": "/api/ml/models", "name": "ML Models API", "expected_type": "json", "validator": lambda d: d.get("status") == "success" and "models" in d},
    {"path": "/api/ml/test", "name": "ML Test API (GET check)", "expected_type": "json", "method": "GET", "expected_status_codes": [405]}, # Expect 405 for GET on POST endpoint
    {"path": "/api/ml/train/test_model", "name": "ML Train API (GET check)", "expected_type": "json", "method": "GET", "expected_status_codes": [405]}, # Expect 405 for GET on POST endpoint
    {"path": "/api/trading/start", "name": "Trading Start API (GET check)", "expected_type": "json", "method": "GET", "expected_status_codes": [405]}, # Expect 405 for GET on POST endpoint
    {"path": "/api/trading/stop", "name": "Trading Stop API (GET check)", "expected_type": "json", "method": "GET", "expected_status_codes": [405]}, # Expect 405 for GET on POST endpoint
]

# --- Helper Functions ---

def print_status(message, status_type="info"):
    """Prints a formatted message with a status indicator."""
    if status_type == "success":
        print(f"✅ {message}")
    elif status_type == "error":
        print(f"❌ {message}")
    elif status_type == "warning":
        print(f"⚠️ {message}")
    else:
        print(f"➡️ {message}")

def check_endpoint(endpoint_config):
    """
    Checks a single API endpoint and reports its status.
    Args:
        endpoint_config (dict): A dictionary containing 'path', 'name', 'expected_type',
                                optional 'validator' (for JSON), 'method', and 'expected_status_codes'.
    Returns:
        bool: True if the check passed, False otherwise.
    """
    path = endpoint_config["path"]
    name = endpoint_config["name"]
    expected_type = endpoint_config["expected_type"]
    method = endpoint_config.get("method", "GET")
    expected_status_codes = endpoint_config.get("expected_status_codes", [200])
    full_url = f"{BASE_URL}{path}"

    print_status(f"Checking {name} ({full_url})...")

    try:
        if method == "GET":
            response = requests.get(full_url, timeout=10)
        elif method == "POST":
            # For POST, send a minimal JSON body if applicable, or empty for just status check
            response = requests.post(full_url, json={}, timeout=10)
        else:
            print_status(f"  Unsupported method: {method}", "error")
            return False
        
        print_status(f"  Status Code: {response.status_code}", "info")
        
        # Check if status code is as expected
        if response.status_code not in expected_status_codes:
            print_status(f"  Endpoint returned unexpected status code: {response.status_code}. Expected one of: {expected_status_codes}", "error")
            print_status(f"  Raw Response (first 500 chars): {response.text[:500]}...", "error")
            return False

        # If we expect a non-200 but it's in expected_status_codes, consider it a pass for that aspect
        if response.status_code != 200 and response.status_code in expected_status_codes:
             print_status(f"  Response: Expected status code {response.status_code} received.", "success")
             return True # For 405, etc., just confirm the status code

        # Proceed with content validation for 200 OK responses
        if expected_type == "json":
            try:
                data = response.json()
                validator = endpoint_config.get("validator")
                if validator:
                    if validator(data):
                        print_status(f"  Response: Valid JSON and passed custom validation.", "success")
                        return True
                    else:
                        print_status(f"  Response: Valid JSON but failed custom validation.", "warning")
                        print_status(f"  Raw JSON: {json.dumps(data, indent=2)}", "info")
                        return False
                else:
                    print_status(f"  Response: Valid JSON (no specific validation provided).", "success")
                    print_status(f"  Raw JSON: {json.dumps(data, indent=2)}", "info")
                    return True
            except json.JSONDecodeError:
                print_status(f"  Response: Expected JSON but received non-JSON content!", "error")
                print_status(f"  Raw Response (first 500 chars): {response.text[:500]}...", "error")
                print_status(f"  HINT: This often means your backend returned an HTML error page. Check backend logs!", "error")
                return False
        elif expected_type == "html":
            if "<!DOCTYPE html>" in response.text.lower():
                print_status(f"  Response: Valid HTML page", "success")
                return True
            else:
                print_status(f"  Response: Expected HTML but content does not look like HTML.", "warning")
                print_status(f"  Raw Response (first 500 chars): {response.text[:500]}...", "warning")
                return False
        
        return True # Default to true if no specific validation failed

    except requests.exceptions.ConnectionError as e:
        print_status(f"  Connection Error: Could not connect to the server at {BASE_URL}. Is the FastAPI app running?", "error")
        print_status(f"  Error details: {e}", "error")
        return False
    except requests.exceptions.Timeout:
        print_status(f"  Timeout Error: Request to {full_url} timed out after 10 seconds.", "error")
        print_status(f"  HINT: The server might be overloaded or the endpoint is very slow.", "error")
        return False
    except Exception as e:
        print_status(f"  An unexpected error occurred: {e}", "error")
        return False

# --- Main Execution ---

def run_all_checks():
    """Runs all defined endpoint checks and provides a summary."""
    print("\n--- Starting API Endpoint Checks ---")
    all_passed = True
    
    for endpoint in ENDPOINTS_TO_CHECK:
        # For POST endpoints, we're doing a GET check to see if they return 405.
        # If you want to test actual POST requests, you'd need to define payload and method.
        # For now, this is sufficient to verify endpoint existence.
        if not check_endpoint(endpoint):
            all_passed = False
        print("-" * 40) # Separator for readability
        time.sleep(0.5) # Small delay between checks

    print("\n--- Check Summary ---")
    if all_passed:
        print_status("🎉 All critical endpoints passed their checks!", "success")
        print("Your FastAPI application appears to be responding correctly.")
    else:
        print_status("⚠️ Some endpoints failed or returned warnings.", "warning")
        print("Please review the errors above and check your FastAPI server logs (`main.py` console output) for more details.")
        print("\n--- Common Fixes ---")
        print("1.  **Backend Not Running:** Ensure your `main.py` script is actively running.")
        print("    (e.g., `python main.py` or `uvicorn main:app --reload`)")
        print("2.  **Port Mismatch:** Verify `BASE_URL` in this script matches the port FastAPI is listening on (default 8000).")
        print("3.  **Firewall/Network:** Check if a firewall is blocking connections to the server's port.")
        print("4.  **Backend Logic Errors:** If an API returns HTML when JSON is expected, it often means an unhandled error occurred in FastAPI before it could format a JSON response. Review `main.py` for recent changes, especially in the `/api/market-data` endpoint and global exception handlers.")
        print("5.  **Dependencies:** Ensure all Python dependencies are installed (`pip install fastapi uvicorn[standard] python-dotenv aiohttp psutil`).")

if __name__ == "__main__":
    run_all_checks()