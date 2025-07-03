#!/bin/bash
echo "Starting Web Dashboard..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Start dashboard
echo "Dashboard will be available at http://localhost:8050"
python web_dashboard.py --host localhost --port 8050
