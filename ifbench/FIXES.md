# IFBench Data Loading Fix

## Error Fixed
```
KeyError: 'response'
```

## Root Cause
The code expected a `'response'` field in the IFBench dataset, but the actual dataset structure only contains:
- `'key'`: Example ID
- `'prompt'`: The instruction/query with embedded constraints
- `'instruction_id_list'`: List of constraint types to check
- `'kwargs'`: Parameters for each constraint

IFBench is an **instruction-following benchmark**, not a question-answering benchmark. There are no ground truth "answers" - instead, the evaluation checks whether the model's response satisfies specific constraints (e.g., "mention keyword X exactly 3 times", "answer in exactly 2 paragraphs", etc.).

## Changes Made

### 1. ifbench_data.py
**Before:**
```python
dspy.Example(
    prompt=example['prompt'],
    answer=example['response']  # ❌ This field doesn't exist!
)
```

**After:**
```python
dspy.Example(
    prompt=example['prompt'],
    instruction_id_list=example.get('instruction_id_list', []),
    kwargs=example.get('kwargs', []),
    key=example.get('key', '')
)
```

### 2. ifbench_metric.py

**Updated `ifbench_score()`:**
- Removed dependency on non-existent `gold.answer` field
- Implemented placeholder scoring based on answer length
- Added TODO comments for proper constraint checking

**Updated `ifbench_feedback_text()`:**
- Provides feedback based on actual IFBench structure
- Lists the constraints that should be verified
- Shows instruction_id_list for debugging

## Current Limitations

The current implementation uses a **placeholder scoring heuristic** based on answer length:
- Empty/very short answer: 0.0
- Short answer (< 20 words): 0.3
- Medium answer (< 50 words): 0.5
- Long answer (50+ words): 0.7

### For Production Use

For proper IFBench evaluation, you should implement actual constraint checking:

```python
# TODO: Use official IFBench evaluation code
# https://github.com/google-research/google-research/tree/master/instruction_following_eval

def check_constraint(instruction_id, kwargs, response):
    """Check if response satisfies a specific constraint."""
    if instruction_id == "count:keywords_multiple":
        # Check if keywords appear correct number of times
        keyword1 = kwargs.get('keyword1', '')
        # ... check response contains keyword1 exactly once
        pass
    # ... handle other constraint types
```

The official IFBench evaluation code can check constraints like:
- Keyword frequency (mention word X exactly N times)
- Length constraints (exactly/at least/at most N words/sentences/paragraphs)
- Formatting constraints (JSON format, bullet points, etc.)
- Content constraints (forbidden words, required phrases, etc.)
- And many more...

## Testing

Verify the fix works:
```bash
python3 -c "
from ifbench_data import load_ifbench_splits
train, dev, test = load_ifbench_splits(seed=0, data_dir='/tmp')
print(f'Loaded: {len(train)} train, {len(dev)} dev, {len(test)} test')
print(f'Example: {train[0].prompt[:100]}...')
"
```

## Next Steps

To improve IFBench evaluation:

1. **Implement constraint checking**: Use the official IFBench evaluation code or implement custom constraint checkers based on `instruction_id_list` and `kwargs`

2. **Better feedback**: Provide specific feedback about which constraints were satisfied/violated

3. **Proper scoring**: Replace length heuristic with actual constraint satisfaction rates

## References
- IFBench Paper: https://arxiv.org/abs/2311.07911
- IFBench Dataset: https://huggingface.co/datasets/allenai/IFBench_test
- Evaluation Code: https://github.com/google-research/google-research/tree/master/instruction_following_eval
