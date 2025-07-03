#!/bin/bash
# File: start_safe.sh
# Location: E:\Trade Chat Bot\G Trading Bot/start_safe.sh
# Description: Elite Trading Bot V3.0 - Safe Startup Script  
# Purpose: Start bot with proper Unicode and encoding support

echo "Elite Trading Bot V3.0 - Safe Startup"
echo "====================================="

# Set UTF-8 encoding
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8

# Navigate to bot directory
cd "E:\Trade Chat Bot\G Trading Bot"

echo "Starting server with Unicode support..."
echo "Server will be available at: http://localhost:8000"

# Start server
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level warning

echo "Server stopped."
