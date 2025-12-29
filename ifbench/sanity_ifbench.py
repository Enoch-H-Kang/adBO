# sanity_ifbench.py
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import dspy

from ifbench_data import load_ifbench_splits
from ifbench_metric import ifbench_score, ifbench_feedback_text
from ifbench_program import create_ifbench_program


def configure_dspy_lm_from_vllm(
    *, 
    api_base: str,
    api_key: str,
    model: str,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int | None = None,
    max_tokens: int = 8192,
    cache: bool = False,
    num_retries: int = 3,
):
    """
    vLLM OpenAI-compatible endpoint configuration.
    """
    lm_kwargs = dict(
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=num_retries,
    )
    if top_k is not None:
        lm_kwargs["top_k"] = top_k

    lm = dspy.LM(f"openai/{model}", **lm_kwargs)
    dspy.configure(lm=lm)


def main():
    ap = argparse.ArgumentParser()

    # vLLM endpoint
    ap.add_argument("--api_base", type=str, default=os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api_key", type=str, default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", type=str, default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"))

    # Decoding
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=1024)

    # Data
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n_examples", type=int, default=3)
    ap.add_argument("--work_dir", type=str, default=os.environ.get("WORK", "/tmp/ifbench_workdir"))
    ap.add_argument("--dump_jsonl", type=str, default=None, help="Optional path to write per-example results.jsonl")

    # Program
    ap.add_argument(
        "--program_variant",
        type=str,
        default="two_stage",
        choices=["two_stage", "single_stage", "iterative"],
        help="Which IFBench program variant to use.",
    )

    args = ap.parse_args()

    configure_dspy_lm_from_vllm(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        cache=False,
        num_retries=3,
    )

    train, dev, test = load_ifbench_splits(seed=args.seed, data_dir=args.work_dir)
    split_map = {"train": train, "dev": dev, "test": test}
    examples = split_map[args.split][: args.n_examples]

    prog = create_ifbench_program(variant=args.program_variant)

    dump_path = Path(args.dump_jsonl) if args.dump_jsonl else None
    if dump_path:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        f_dump = dump_path.open("w", encoding="utf-8")
    else:
        f_dump = None

    try:
        for i, ex in enumerate(examples, 1):
            t0 = time.time()
            pred = prog(prompt=ex.prompt, constraint_text=ex.constraint_text)
            dt = time.time() - t0

            score = ifbench_score(ex, pred)
            feedback = ifbench_feedback_text(ex, pred)

            print(f"\n=== EXAMPLE {i}/{len(examples)} ===")
            print("PROMPT:", ex.prompt)
            print("CONSTRAINT:", ex.constraint_text)
            print("GOLD ANSWER:", ex.answer)
            print("PRED ANSWER:", pred.answer)
            print(f"SCORE: {score:.3f} | LATENCY: {dt:.2f}s")

            if "initial_answer" in pred:
                print("INITIAL ANSWER:", pred.initial_answer)
            
            print("\n[FEEDBACK]:\n", feedback)

            if f_dump:
                rec = {
                    "idx": i,
                    "prompt": ex.prompt,
                    "constraint_text": ex.constraint_text,
                    "gold_answer": ex.answer,
                    "pred_answer": pred.answer,
                    "score": float(score),
                    "initial_answer": pred.initial_answer if "initial_answer" in pred else None,
                    "feedback": feedback,
                    "latency_sec": float(dt),
                }
                f_dump.write(json.dumps(rec, ensure_ascii=False) + "\n")

finally:
    if f_dump:
        f_dump.close()
        print(f"\nWrote JSONL dump to: {dump_path}")


if __name__ == "__main__":
    main()
