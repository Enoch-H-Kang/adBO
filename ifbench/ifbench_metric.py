# ifbench_metric.py
from __future__ import annotations

import re
import string
from typing import Optional

import dspy

# -------------------------
# Normalization helpers (from hotpot_metric.py, can be adapted for IFBench if needed)
# -------------------------
def _normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

# -------------------------
# Scores
# -------------------------
def ifbench_score(gold, pred, trace=None) -> float:
    """
    Instruction-following score for IFBench.

    This is a placeholder that returns a score based on whether the response
    exists and has reasonable length. For proper IFBench evaluation, you would
    need to check constraint satisfaction using the instruction_id_list and kwargs.

    TODO: Implement actual constraint checking using IFBench evaluation code
    from https://github.com/google-research/google-research/tree/master/instruction_following_eval
    """
    # Get the prediction answer
    pred_answer = getattr(pred, "answer", "") or getattr(pred, "final_answer", "")

    if not pred_answer or len(pred_answer.strip()) < 10:
        return 0.0

    # Basic heuristic: non-empty substantial answer gets partial credit
    # Real implementation should check actual constraints
    # For now, return a score based on answer length as a proxy for effort
    answer_len = len(pred_answer.split())
    if answer_len < 20:
        return 0.3
    elif answer_len < 50:
        return 0.5
    else:
        return 0.7  # Base score for substantial answer

    # TODO: Add actual constraint checking here based on:
    # - gold.instruction_id_list (list of constraint types)
    # - gold.kwargs (parameters for each constraint)
    # - pred_answer (the generated response)

# -------------------------
# Feedback
# -------------------------
def ifbench_feedback_text(gold, pred, pred_name: Optional[str] = None) -> str:
    """
    Provides feedback on instruction-following quality.

    This is a simplified version that provides basic feedback.
    A complete implementation would check each constraint from instruction_id_list.
    """
    pred_answer = getattr(pred, "answer", "") or getattr(pred, "final_answer", "")
    instruction_ids = getattr(gold, "instruction_id_list", [])
    prompt = getattr(gold, "prompt", "")

    feedback_parts = []

    # Basic feedback about answer quality
    if not pred_answer or len(pred_answer.strip()) < 10:
        feedback_parts.append("⚠️ Answer is too short or empty.")
    else:
        answer_len = len(pred_answer.split())
        feedback_parts.append(f"✓ Generated answer with {answer_len} words.")

    # List the constraints that should be checked
    if instruction_ids:
        feedback_parts.append(f"\nConstraints to verify: {', '.join(instruction_ids)}")

    # Basic instruction reminder
    feedback_parts.append(f"\nOriginal prompt (first 200 chars): {prompt[:200]}...")

    # TODO: Add actual constraint verification here
    feedback_parts.append("\n[NOTE: Detailed constraint checking not yet implemented. "
                         "Score is based on answer length heuristic.]")

    return "\n".join(feedback_parts)

# -------------------------
# GEPA metric (must accept 5 args)
# -------------------------
def ifbench_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    DSPy GEPA requires metric(gold, pred, trace, pred_name, pred_trace).
    Return score + feedback; score must remain consistent across pred_name requests.
    """
    score = ifbench_score(gold, pred, trace=None)
    feedback = ifbench_feedback_text(gold, pred, pred_name=pred_name)
    return dspy.Prediction(score=score, feedback=feedback)

# -------------------------
# Learning curve helpers (copied from hotpot_metric.py)
# -------------------------
def build_best_so_far_curve_from_detailed_results(detailed_results, baseline_score: float | None = None):
    """
    Extract a best-so-far curve vs "rollouts" (= metric calls).
    DSPy exposes `detailed_results` when track_stats=True.
    """
    scores = list(getattr(detailed_results, "val_aggregate_scores", []) or [])
    eval_counts = getattr(detailed_results, "discovery_eval_counts", None)

    if eval_counts is None:
        eval_counts = list(range(1, len(scores) + 1))
    else:
        eval_counts = list(eval_counts)

    triples = [(i, int(eval_counts[i]), float(scores[i])) for i in range(len(scores))]
    triples.sort(key=lambda x: x[1])

    curve = []
    best = float("-inf")

    if baseline_score is not None:
        best = float(baseline_score)
        curve.append(dict(rollouts=0, candidate_idx=None, candidate_val_score=baseline_score, best_val_score=best))

    for cand_idx, rollouts, cand_score in triples:
        if cand_score > best:
            best = cand_score
        curve.append(
            dict(
                rollouts=int(rollouts),
                candidate_idx=int(cand_idx),
                candidate_val_score=float(cand_score),
                best_val_score=float(best),
            )
        )
    return curve