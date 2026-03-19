#!/bin/bash

# Simple launcher for Linux GPU nodes.
# 1) Activates the local virtual environment
# 2) Runs the experiment pipeline end-to-end
# 3) Writes outputs to results/experiment_results.csv and results/experiment_summary.csv

echo "Activating environment..."
source venv/bin/activate

echo "Running pipeline..."
python -m src.runner

echo "Done!"
