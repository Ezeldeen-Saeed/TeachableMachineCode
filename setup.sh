#!/bin/bash

# Look for the first match of mu_venv-* in the expected directories
# Common locations for Mu venv on Linux include ~/.local/share/mu and ~/.cache/mu
FOUND=$(find ~/.local/share/mu ~/.cache/mu -name "mu_venv-*" -type d -print -quit 2>/dev/null)

if [ -z "$FOUND" ]; then
    echo "Mu virtual environment not found."
    echo "Searched in ~/.local/share/mu and ~/.cache/mu"
    exit 1
fi

echo "Found Mu environment: $FOUND"

PYTHON_PATH="$FOUND/bin/python3"
PIP_PATH="$FOUND/bin/pip3"

# Fallback to 'python'/'pip' if 'python3'/'pip3' aren't explicitly named
if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH="$FOUND/bin/python"
    PIP_PATH="$FOUND/bin/pip"
fi

echo "Using Python from: $PYTHON_PATH"

# Upgrade pip using Mu's Python
echo "Upgrading pip..."
"$PYTHON_PATH" -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to upgrade pip."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
"$PIP_PATH" install tensorflow opencv-python numpy
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

echo "Setup complete and all dependencies installed correctly!"
exit 0
