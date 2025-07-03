#!/bin/bash
echo "Starting Enhanced Trading Bot..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "WARNING: No trained models found. Training models first..."
    python enhanced_model_trainer.py --full-train
fi

# Start the bot
echo "Starting trading bot..."
python main_trading_bot.py
