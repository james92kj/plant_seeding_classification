#!/bin/bash
set -e  # Stop on any error

echo "=== Step 1: Install requirements ==="
pip install -r requirements.txt

echo "=== Step 2: Download data ==="
kaggle competitions download -c plant-seedlings-classification -p data/
unzip -q data/plant-seedlings-classification.zip -d data/
# Clean up zip
rm data/plant-seedlings-classification.zip

echo "=== Step 3: Install project ==="
pip install -e .

echo "=== Step 4: Train ==="
python scripts/main.py

echo "=== Done ==="
echo "Submission file: outputs/submission/submission.csv"
