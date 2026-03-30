#!/bin/bash
set -e  # Stop on any error

cd "$(dirname "$0")/.."
echo "Project root: $(pwd)"

echo "=== Step 1: Install requirements ==="
pip install -r requirements.txt

echo "=== Step 2: Download data ==="
kaggle competitions download -c plant-seedlings-classification -p data/
unzip -q data/plant-seedlings-classification.zip -d data/
rm data/plant-seedlings-classification.zip

# Verify expected structure
echo "Checking data structure..."
ls data/train/ | head -3
ls data/test/ | head -3

echo "=== Step 3: Install project ==="
pip install -e .

echo "=== Step 4: Train ==="
python scripts/main.py

echo "=== Done ==="
echo "Submission file: output/submission/submission.csv"
