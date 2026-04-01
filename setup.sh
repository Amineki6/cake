#!/bin/bash

# Stop execution if any command fails
set -e

echo "Starting environment setup..."

# 1. Check if uv is installed; if not, install it
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Downloading and installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed."
fi

# Ensure uv is in the PATH for this current script session
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
fi

# 2. Clean up any existing virtual environment to avoid conflicts
if [ -d ".venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf .venv
fi

# 3. Pin Python version and create the new virtual environment
echo "Pinning Python to 3.10.19 and creating virtual environment..."
uv python pin 3.10.19
uv venv

# 4. Activate the environment and install dependencies
echo "Activating virtual environment and installing requirements..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo "Setup complete."
echo "Run 'source .venv/bin/activate' to start working."