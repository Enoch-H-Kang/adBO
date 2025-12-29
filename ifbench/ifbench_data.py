# ifbench_data.py
import dspy
import random
from datasets import load_dataset

def load_ifbench_splits(seed, data_dir):
    """
    Loads the IFBench dataset splits.
    Dataset has 300 examples total, so we split as: 100/100/100
    """
    dataset = load_dataset("allenai/IFBench_test")

    train_split = dataset['train'].select(range(100))
    dev_split = dataset['train'].select(range(100, 200))
    test_split = dataset['train'].select(range(200, 300))

    def convert_to_dspy_examples(data_split):
        examples = []
        for example in data_split:
            # IFBench contains instruction-following constraints, not ground truth answers
            # Store the constraints for evaluation
            examples.append(
                dspy.Example(
                    prompt=example['prompt'],
                    instruction_id_list=example.get('instruction_id_list', []),
                    kwargs=example.get('kwargs', []),
                    key=example.get('key', '')
                ).with_inputs("prompt")
            )
        return examples

    train = convert_to_dspy_examples(train_split)
    dev = convert_to_dspy_examples(dev_split)
    test = convert_to_dspy_examples(test_split)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    return train, dev, test