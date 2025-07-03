#!/bin/bash
echo "Training ML Models..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Run training
python enhanced_model_trainer.py --full-train --verbose
