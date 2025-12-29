# PUPA Dataset Setup - FIXED

## Problem
When running `sbatch job.pupa_compare.sbatch`, you got:
```
FileNotFoundError: PUPA dataset not found
```

## Root Cause
The PUPA dataset wasn't available in the expected location. The dataset exists in the PAPILLON GitHub repository as a CSV file, but your code expected JSON files (train.json, dev.json, test.json).

## Solution
I've created a complete data pipeline to download and convert the PUPA dataset:

### What I Created

1. **`convert_pupa_data.py`** - Python script that:
   - Loads PUPA data from CSV (`PUPA_New.csv` - 664 examples)
   - Converts to the expected format with fields:
     - `user_query`: Original query with PII
     - `reference_response`: Ground truth response
     - `private_info`: List of PII entities
   - Splits into 111 train / 111 dev / 221 test (as per paper)
   - Saves as JSON files

2. **`setup_pupa_data.sh`** - Automated setup script that:
   - Clones the PAPILLON repository
   - Runs the conversion script
   - Creates all necessary files

3. **Updated `job.pupa_compare.sbatch`** - Added `--work_dir` parameter to point to data directory

## How to Use

### Quick Setup (One Command)

```bash
cd /work1/krishnamurthy/arvind/adBO/pupa
./setup_pupa_data.sh
```

That's it! This will:
- Download the PAPILLON repo
- Convert PUPA_New.csv to JSON format
- Create train.json, dev.json, test.json in `./data/`

### Verify Setup

```bash
python3 -c "
from pupa_data import load_pupa_splits
train, dev, test = load_pupa_splits(seed=0, data_dir='./data')
print(f'✓ Loaded: {len(train)} train, {len(dev)} dev, {len(test)} test')
"
```

Expected output:
```
[PUPA] Loading from local directory: ./data
[PUPA] Loaded 111 train, 111 dev, 221 test examples
✓ Loaded: 111 train, 111 dev, 221 test
```

### Run Your Job

Now you can submit your SLURM job:

```bash
sbatch /work1/krishnamurthy/arvind/adBO/pupa/job.pupa_compare.sbatch
```

## Data Location

All data files are now in:
```
/work1/krishnamurthy/arvind/adBO/pupa/data/
  ├── train.json  (111 examples)
  ├── dev.json    (111 examples)
  └── test.json   (221 examples)
```

## What the Data Looks Like

Each example has:
```json
{
  "user_query": "Create a resume for John Smith, email john@example.com...",
  "reference_response": "Here is a professional resume...",
  "private_info": ["john smith", "john@example.com"],
  "redacted_query": "Create a resume for [REDACTED], email [REDACTED]...",
  "conversation_hash": "abc123...",
  "category": "professional information"
}
```

## Dataset Statistics

- **Total examples**: 664 (from PUPA_New.csv)
- **Train**: 111 examples (16.7%)
- **Dev**: 111 examples (16.7%)
- **Test**: 221 examples (33.3%)
- **Unused**: 221 examples (33.3%)

The split matches the PUPA paper specification: 111/111/221

## Troubleshooting

### If setup_pupa_data.sh fails:

1. **Manual clone**:
   ```bash
   git clone https://github.com/Columbia-NLP-Lab/PAPILLON/ /tmp/PAPILLON
   ```

2. **Manual conversion**:
   ```bash
   python3 convert_pupa_data.py \
     --csv_path /tmp/PAPILLON/pupa/PUPA_New.csv \
     --output_dir ./data
   ```

### If data loading fails in your job:

Check the `--work_dir` parameter in your run command points to the data directory:
```bash
--work_dir /work1/krishnamurthy/arvind/adBO/pupa/data
```

## Source

- **PAPILLON Repo**: https://github.com/Columbia-NLP-Lab/PAPILLON/
- **Original CSV**: `PAPILLON/pupa/PUPA_New.csv`
- **PUPA Paper**: https://aclanthology.org/2025.naacl-long.173.pdf
