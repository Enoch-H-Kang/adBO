# sanity_pupa.py
"""
Sanity check script for PUPA benchmark using original PAPILLON pipeline.

Tests the privacy-preserving delegation system on a few examples
to verify the implementation is working correctly.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import dspy

from pupa_data import load_pupa_splits
from papillon.llm_judge import (
    papillon_quality_score,
    papillon_leakage_count,
    papillon_prompt_quality,
    papillon_aggregate_score,
)
from pupa_program import create_pupa_program


def configure_dspy_lm_from_vllm(
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    cache: bool = False,
    num_retries: int = 3,
):
    """
    vLLM OpenAI-compatible endpoint configuration.
    Defaults match the GEPA paper decoding settings for Qwen3-8B.
    """
    lm = dspy.LM(
        f"openai/{model}",
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=num_retries,
    )
    dspy.configure(lm=lm)
    return lm


def main():
    ap = argparse.ArgumentParser()

    # vLLM endpoint
    ap.add_argument("--api_base", type=str, default=os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api_key", type=str, default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", type=str, default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"))

    # Decoding (defaults align with GEPA config)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=8192)

    # Data
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n_examples", type=int, default=3)
    ap.add_argument("--data_dir", type=str, default=None, help="Directory containing PUPA dataset files (defaults to ./data)")

    ap.add_argument("--dump_jsonl", type=str, default=None, help="Optional path to write per-example results.jsonl")

    args = ap.parse_args()

    # Configure LM
    lm = configure_dspy_lm_from_vllm(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        cache=False,
        num_retries=3,
    )

    # Load data
    try:
        train, dev, test = load_pupa_splits(seed=args.seed, data_dir=args.data_dir)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nTo run this sanity check, you need the PUPA dataset.")
        print("Please download it from: https://github.com/Columbia-NLP-Lab/PAPILLON/")
        return 1

    split_map = {"train": train, "dev": dev, "test": test}
    examples = split_map[args.split][: args.n_examples]

    if len(examples) == 0:
        print(f"[ERROR] No examples found in {args.split} split!")
        return 1

    # Create program (Original PAPILLON)
    prog = create_pupa_program()

    dump_path = Path(args.dump_jsonl) if args.dump_jsonl else None
    if dump_path:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        f_dump = dump_path.open("w", encoding="utf-8")
    else:
        f_dump = None

    try:
        for i, ex in enumerate(examples, 1):
            t0 = time.time()
            pred = prog(user_query=ex.user_query)
            dt = time.time() - t0

            # Compute metrics using original PAPILLON methodology
            quality = papillon_quality_score(ex, pred)  # Binary: 0 or 1
            leakage_count = papillon_leakage_count(ex, pred)  # Count of leaked PII
            prompt_qual = papillon_prompt_quality(ex, pred)  # Binary: 0 or 1
            aggregate = papillon_aggregate_score(ex, pred)

            # Get PII count for display
            private_info = ex.private_info if hasattr(ex, 'private_info') else []
            num_pii = len(private_info) if private_info else 1

            print(f"\n=== EXAMPLE {i}/{len(examples)} (Original PAPILLON) ===")
            print("USER QUERY:", ex.user_query)
            if hasattr(ex, 'private_info') and ex.private_info:
                print("PII ENTITIES:", ex.private_info)

            print("\n--- STAGE 1: CreateOnePrompt (Trusted) ---")
            # Original PAPILLON returns 'prompt' field (the privacy-preserving prompt)
            rewritten_query = getattr(pred, 'prompt', getattr(pred, 'rewritten_query', ''))
            print("REWRITTEN QUERY:", rewritten_query)

            print("\n--- STAGE 2: Untrusted Model Response ---")
            # Original PAPILLON returns 'gptResponse' field
            model_response = getattr(pred, 'gptResponse', getattr(pred, 'model_response', ''))
            print("RESPONSE:", model_response[:200], "..." if len(model_response) > 200 else "")

            print("\n--- STAGE 3: InfoAggregator (Trusted) ---")
            # Original PAPILLON returns 'output' field
            final_response = getattr(pred, 'output', getattr(pred, 'final_response', getattr(pred, 'response', '')))
            print("FINAL RESPONSE:", final_response[:200], "..." if len(final_response) > 200 else "")

            if hasattr(ex, 'reference_response') and ex.reference_response:
                print("\n--- REFERENCE ---")
                print("REFERENCE RESPONSE:", ex.reference_response[:200], "..." if len(ex.reference_response) > 200 else "")

            print("\n--- METRICS (Original PAPILLON from GitHub) ---")
            print(f"Quality Score:      {quality} (binary: 0=poor, 1=good)")
            print(f"PII Leakage:        {leakage_count}/{num_pii} pieces leaked")
            print(f"Prompt Quality:     {prompt_qual} (binary: 0=invalid, 1=valid)")
            print(f"Aggregate Score:    {aggregate:.3f}")
            print(f"Formula: (quality - leakage/num_pii + prompt_qual) / 2")
            print(f"Latency: {dt:.2f}s")

            # Check if PII leaked
            if hasattr(ex, 'private_info') and ex.private_info:
                leaked_entities = []
                for pii in ex.private_info:
                    # Use the rewritten_query variable we extracted above
                    if isinstance(pii, str) and pii.lower() in rewritten_query.lower():
                        leaked_entities.append(pii)

                if leaked_entities:
                    print(f"\n[WARNING] PII LEAKED to untrusted model: {leaked_entities}")
                else:
                    print("\n[SUCCESS] No PII leaked to untrusted model!")

            if f_dump:
                rec = {
                    "idx": i,
                    "user_query": ex.user_query,
                    "private_info": getattr(ex, 'private_info', []),
                    "rewritten_query": rewritten_query,  # Using extracted variable
                    "model_response": model_response,    # Using extracted variable
                    "final_response": final_response,    # Using extracted variable
                    "reference_response": getattr(ex, 'reference_response', ''),
                    "quality_score": int(quality),  # Binary 0/1
                    "leakage_count": int(leakage_count),  # Count
                    "num_pii": num_pii,
                    "prompt_quality_score": int(prompt_qual),  # Binary 0/1
                    "aggregate_score": float(aggregate),
                    "latency_sec": float(dt),
                }
                f_dump.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        if f_dump:
            f_dump.close()
            print(f"\nWrote JSONL dump to: {dump_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
