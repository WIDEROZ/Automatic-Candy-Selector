#!/bin/bash

REQUIREMENT_FILE="requirements.txt"

# Check if venv exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
    source venv/bin/activate || { echo "Failed to activate virtual environment" >&2; exit 1; }
else
    echo "Creating new virtual environment..."
    python3 -m venv venv || { echo "Failed to create virtual environment" >&2; exit 1; }
    source venv/bin/activate || { echo "Failed to activate virtual environment" >&2; exit 1; }
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing requirements..."
python3 -m pip install -r "$REQUIREMENT_FILE"


