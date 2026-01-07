# PAPILLON Original Code

This directory contains code copied from the official PAPILLON implementation:
https://github.com/Columbia-NLP-Lab/PAPILLON

## Citation

If you use this code, please cite the original PAPILLON paper:

```bibtex
@inproceedings{papillon2025,
  title={PAPILLON: PrivAcy Preservation in Large Language models by Integrating Locally-trained OptiONs},
  author={Columbia NLP Lab},
  booktitle={NAACL},
  year={2025}
}
```

## License

MIT License - see LICENSE_PAPILLON.txt

## Changes from Original

### ✓ Exact Matches (Preserved from Original)

1. **Signature Field Names**: Exact camelCase names from original
   - `CreateOnePrompt`: `userQuery` → `createdPrompt`
   - `InfoAggregator`: `userQuery`, `modelExampleResponses` → `finalOutput`

2. **Signature Docstrings**: Copied verbatim from original

3. **PAPILLON Class Structure**:
   - `prompt_creater = dspy.ChainOfThought(CreateOnePrompt)` ✓
   - `info_aggregator = dspy.Predict(InfoAggregator)` ✓ (NOT ChainOfThought)
   - Takes `untrusted_model` parameter ✓

4. **Forward Method Logic**: Exact implementation
   - Calls `self.prompt_creater(userQuery=user_query).createdPrompt`
   - Calls `self.untrusted_model(prompt)[0]`
   - Calls `self.info_aggregator(userQuery=..., modelExampleResponses=...)`
   - Returns `dspy.Prediction(prompt=..., output=..., gptResponse=...)`

5. **3-Stage Architecture**: Preserved exactly
   - Stage 1: CreateOnePrompt (trusted model)
   - Stage 2: Untrusted model call
   - Stage 3: InfoAggregator (trusted model)

### ⚙️ Intentional Adaptations for GEPA

1. **Modularization**: Extracted into separate files for maintainability
   - `run_llama_dspy.py` → `papillon_signatures.py` + `papillon_pipeline.py`

2. **GEPA Integration**: Added wrapper functions for compatibility
   - `papillon_quality_score()` - Works with dspy.Example
   - `papillon_leakage_count()` - Works with dspy.Example
   - `papillon_prompt_quality()` - Works with dspy.Example
   - `papillon_aggregate_score()` - Works with dspy.Example

3. **Untrusted Model Wrapper**: Created callable wrapper for DSPy LM
   - Original expects `untrusted_model(prompt) -> [response]`
   - Wrapper adapts DSPy's LM to this interface

4. **Backward Compatibility**: Metric functions support both old and new field names
   - Tries `prompt` first, falls back to `rewritten_query`
   - Tries `output` first, falls back to `final_response`/`response`

**Verification**: Run `python test_papillon_match.py` to confirm all matches.

## Original PAPILLON Architecture

```
User Query (with PII)
    ↓
[Stage 1: CreateOnePrompt]  ← Trusted Model
    → Rewritten Query (PII removed)
    ↓
[Stage 2: Untrusted Model Call]
    → Model Response
    ↓
[Stage 3: InfoAggregator]  ← Trusted Model
    → Final Response
```

## Metrics

- **Quality**: Pairwise comparison with position bias handling
- **Leakage**: Count of PII pieces in rewritten query
- **Prompt Quality**: Binary validation of rewritten query
- **Aggregate**: (quality - leakage/num_pii + prompt_quality) / 2
