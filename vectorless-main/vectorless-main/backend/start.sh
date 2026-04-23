#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp env_example.txt .env
    echo ""
    echo "⚠️  Please edit the .env file and add your OpenAI API key:"
    echo "   OPENAI_API_KEY=your_actual_api_key_here"
    echo ""
fi

# Start the server
echo "Starting FastAPI server..."
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 