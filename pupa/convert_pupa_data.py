#!/usr/bin/env python3
"""
Convert PUPA CSV data from PAPILLON repo to JSON format for GEPA training.

Downloads and converts the PUPA dataset from the PAPILLON GitHub repository
into train.json, dev.json, test.json with 111/111/221 split as per the paper.
"""
import json
import csv
import random
from pathlib import Path
import argparse


def load_pupa_csv(csv_path: Path):
    """Load PUPA CSV file and convert to list of dictionaries."""
    data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse PII units (separated by ||)
            pii_units_str = row.get('pii_units', '')
            if pii_units_str:
                pii_units = [p.strip() for p in pii_units_str.split('||') if p.strip()]
            else:
                pii_units = []

            # Convert to our expected format
            example = {
                'user_query': row.get('user_query', ''),
                'reference_response': row.get('target_response', ''),
                'private_info': pii_units,
                'redacted_query': row.get('redacted_query', ''),
                'conversation_hash': row.get('conversation_hash', ''),
                'category': row.get('predicted_category', ''),
            }

            # Only include if we have both query and response
            if example['user_query'] and example['reference_response']:
                data.append(example)

    return data


def main():
    parser = argparse.ArgumentParser(description='Convert PUPA CSV to JSON splits')
    parser.add_argument(
        '--csv_path',
        type=str,
        default='/tmp/PAPILLON/pupa/PUPA_New.csv',
        help='Path to PUPA CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/work1/krishnamurthy/arvind/adBO/pupa/data',
        help='Output directory for JSON files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling'
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=111,
        help='Number of training examples'
    )
    parser.add_argument(
        '--dev_size',
        type=int,
        default=111,
        help='Number of dev examples'
    )
    parser.add_argument(
        '--test_size',
        type=int,
        default=221,
        help='Number of test examples'
    )

    args = parser.parse_args()

    # Load CSV data
    print(f"Loading data from {args.csv_path}...")
    data = load_pupa_csv(Path(args.csv_path))
    print(f"Loaded {len(data)} examples")

    # Check if we have enough data
    total_needed = args.train_size + args.dev_size + args.test_size
    if len(data) < total_needed:
        print(f"WARNING: Only {len(data)} examples available, need {total_needed}")
        print("Will create splits with available data...")

    # Shuffle with seed
    random.seed(args.seed)
    random.shuffle(data)

    # Create splits
    train_data = data[:args.train_size]
    dev_data = data[args.train_size:args.train_size + args.dev_size]
    test_data = data[args.train_size + args.dev_size:args.train_size + args.dev_size + args.test_size]

    print(f"Split sizes: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON files
    splits = {
        'train.json': train_data,
        'dev.json': dev_data,
        'test.json': test_data,
    }

    for filename, split_data in splits.items():
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(split_data)} examples to {output_path}")

    # Print sample
    print("\n" + "="*60)
    print("Sample from training set:")
    print("="*60)
    sample = train_data[0]
    print(f"User Query: {sample['user_query'][:200]}...")
    print(f"Private Info: {sample['private_info'][:5]}..." if len(sample['private_info']) > 5 else f"Private Info: {sample['private_info']}")
    print(f"Reference Response: {sample['reference_response'][:200]}...")
    print("="*60)

    print(f"\n✓ Successfully created PUPA dataset splits in {output_dir}")
    print(f"\nYou can now run your GEPA training with:")
    print(f"  --work_dir {output_dir}")


if __name__ == '__main__':
    main()
