#!/bin/bash
echo "Running Backtest..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Default parameters
START_DATE=${1:-"2024-01-01"}
END_DATE=${2:-"2024-12-31"}

echo "Backtesting period: $START_DATE to $END_DATE"
python backtesting_engine.py --start-date "$START_DATE" --end-date "$END_DATE"
