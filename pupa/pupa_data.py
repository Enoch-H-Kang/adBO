# pupa_data.py
"""
Data loader for PUPA (Privacy-preserving User Prompts with Annotations) dataset.

From the PUPA paper:
- 111 training examples
- 111 validation examples
- 221 test examples
- Total: 443 examples

Task: Privacy-conscious delegation using trusted + untrusted models
"""
import dspy
import random
from datasets import load_dataset


def load_pupa_splits(seed: int = 0, data_dir: str = None):
    """
    Loads the PUPA dataset splits.

    Args:
        seed: Random seed for shuffling
        data_dir: Optional directory containing PUPA dataset files

    Returns:
        train, dev, test splits as lists of dspy.Example
    """
    # Try multiple data sources in order
    dataset = None

    # 1. Try loading from local directory if provided
    if data_dir:
        import json
        from pathlib import Path
        data_path = Path(data_dir)

        try:
            # Look for JSON files with train/dev/test splits
            train_file = data_path / "train.json"
            dev_file = data_path / "dev.json"
            test_file = data_path / "test.json"

            if train_file.exists() and test_file.exists():
                print(f"[PUPA] Loading from local directory: {data_dir}")
                with open(train_file) as f:
                    train_split = json.load(f)
                with open(dev_file if dev_file.exists() else train_file) as f:
                    dev_split = json.load(f) if dev_file.exists() else train_split[:111]
                with open(test_file) as f:
                    test_split = json.load(f)

                dataset = {
                    'train': train_split,
                    'validation': dev_split,
                    'test': test_split
                }
        except Exception as e:
            print(f"Warning: Could not load from local directory: {e}")

    # 2. Try loading from HuggingFace
    if dataset is None:
        try:
            print("[PUPA] Attempting to load from HuggingFace...")
            # Try different possible dataset names
            dataset_names = [
                "Columbia-NLP-Lab/PUPA",
                "papillon-pupa/pupa",
                "pupa-dataset/pupa"
            ]

            for name in dataset_names:
                try:
                    dataset = load_dataset(name)
                    print(f"[PUPA] Successfully loaded from {name}")
                    break
                except:
                    continue

        except Exception as e:
            print(f"Warning: Could not load PUPA dataset from HuggingFace: {e}")

    # 3. Extract splits
    if dataset and 'train' in dataset:
        train_split = dataset['train']
        dev_split = dataset.get('validation', dataset.get('dev', []))
        test_split = dataset['test']
    else:
        # Dataset not available - raise clear error
        raise FileNotFoundError(
            "PUPA dataset not found. Please either:\n"
            "1. Download the dataset from https://github.com/Columbia-NLP-Lab/PAPILLON/\n"
            "2. Place train.json, dev.json, test.json in the data_dir\n"
            "3. Or specify the correct HuggingFace dataset name in pupa_data.py\n"
            f"Current data_dir: {data_dir}"
        )

    def convert_to_dspy_examples(data_split):
        """
        Convert PUPA examples to DSPy format.

        Expected fields in PUPA dataset:
        - user_query: Original user query (may contain PII)
        - reference_response: High-quality response (ground truth)
        - private_info: List of PII entities in the query
        """
        examples = []
        for example in data_split:
            # Extract fields from PUPA dataset
            user_query = example.get('user_query', example.get('query', ''))
            reference_response = example.get('reference_response',
                                            example.get('response', ''))
            private_info = example.get('private_info',
                                      example.get('pii_entities', []))

            examples.append(
                dspy.Example(
                    user_query=user_query,
                    reference_response=reference_response,
                    private_info=private_info,
                ).with_inputs("user_query")
            )
        return examples

    train = convert_to_dspy_examples(train_split)
    dev = convert_to_dspy_examples(dev_split)
    test = convert_to_dspy_examples(test_split)

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    print(f"[PUPA] Loaded {len(train)} train, {len(dev)} dev, {len(test)} test examples")

    # Validate dataset sizes (PUPA paper specifies 111/111/221)
    expected_train, expected_dev, expected_test = 111, 111, 221
    if len(train) != expected_train or len(dev) != expected_dev or len(test) != expected_test:
        print(f"[PUPA] WARNING: Dataset sizes don't match paper specification!")
        print(f"  Expected: {expected_train}/{expected_dev}/{expected_test}")
        print(f"  Got:      {len(train)}/{len(dev)}/{len(test)}")
        print(f"  Proceeding with available data...")

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            "PUPA dataset is empty. Please check the data source and format.\n"
            "Expected fields: user_query, reference_response, private_info"
        )

    return train, dev, test
