#!/bin/bash

# Usage:
#   experiment --run   # Run the full pipeline: train, generate, synthetic training
#   experiment --clean # Remove data, results, img, models directories

if [ ! -f ".venv/bin/activate" ]; then
    echo "Creating virtual environment (.venv)..."
    uv venv --python 3.12
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "Installing requirements..."
    uv pip install -r requirements.txt
else
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

if [[ "$1" == "--run" ]]; then
    python train_vae.py
    python generate_synthetic.py
    python synthetic_training.py
elif [[ "$1" == "--clean" ]]; then
    rm -rf data results img models
    echo "Clean completed."
else
    echo "Usage:"
    echo "  experiment --run   # Run the full pipeline"
    echo "  experiment --clean # Remove data, results, img, models directories"
    exit 1
fi
