#!/bin/bash
# Quick setup script for PUPA dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
PAPILLON_DIR="/tmp/PAPILLON"

echo "========================================="
echo "PUPA Dataset Setup"
echo "========================================="

# Step 1: Clone PAPILLON repo if needed
if [ ! -d "$PAPILLON_DIR" ]; then
    echo "Cloning PAPILLON repository..."
    git clone https://github.com/Columbia-NLP-Lab/PAPILLON/ "$PAPILLON_DIR"
    echo "✓ Repository cloned"
else
    echo "✓ PAPILLON repository already exists at $PAPILLON_DIR"
fi

# Step 2: Check if CSV exists
CSV_PATH="$PAPILLON_DIR/pupa/PUPA_New.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: PUPA_New.csv not found at $CSV_PATH"
    exit 1
fi
echo "✓ Found PUPA_New.csv"

# Step 3: Convert to JSON
echo "Converting CSV to JSON splits..."
python3 "$SCRIPT_DIR/convert_pupa_data.py" \
    --csv_path "$CSV_PATH" \
    --output_dir "$DATA_DIR" \
    --seed 42

echo ""
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo "Data files created in: $DATA_DIR"
echo "  - train.json (111 examples)"
echo "  - dev.json (111 examples)"
echo "  - test.json (221 examples)"
echo ""
echo "You can now run GEPA training with:"
echo "  --work_dir $DATA_DIR"
echo ""
echo "Or submit the SLURM job:"
echo "  sbatch $SCRIPT_DIR/job.pupa_compare.sbatch"
echo "========================================="
