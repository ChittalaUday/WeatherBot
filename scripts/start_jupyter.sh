#!/bin/bash

# Navigate to project root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "Error: .venv not found. Run 'python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt' first."
    exit 1
fi

# Launch independent Jupyter Notebook server
echo "Starting independent Jupyter Notebook server..."
jupyter notebook --ip=127.0.0.1 --port=8888 --notebook-dir="$PROJECT_DIR"
