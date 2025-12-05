#!/bin/bash

# Ensure ANTHROPIC_API_KEY is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY environment variable is not set."
    echo "Please export it before running this script."
    exit 1
fi

# Install dependencies if needed (quietly)
echo "Checking dependencies..."
python3 -m pip install --user -q -r requirements.txt

# Kill any existing process on port 8000
lsof -t -i:8000 | xargs kill -9 2>/dev/null || true

# Start the server
echo "Starting Prompt Engineering Pro..."
echo "Open http://localhost:8000 in your browser"

# Open browser in background after a short delay
(sleep 2 && open http://localhost:8000) &

python3 -m uvicorn app:app --reload --port 8000
