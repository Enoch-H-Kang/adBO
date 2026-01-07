"""
GEPA-compatible metrics for PUPA benchmark using original PAPILLON.

This module wraps the original PAPILLON metrics (from papillon/llm_judge.py)
to provide GEPA's 5-argument signature.
"""

from __future__ import annotations
import dspy
from typing import Optional

from papillon.llm_judge import (
    papillon_quality_score,
    papillon_leakage_count,
    papillon_prompt_quality,
    papillon_aggregate_score,
)


def pupa_score(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional = None
) -> float:
    """
    Discrete score function using original PAPILLON metrics.

    For use with dspy.Evaluate.

    Returns:
        Aggregate score: (quality - leakage/num_pii + prompt_quality) / 2
    """
    return papillon_aggregate_score(gold, pred, trace)


def pupa_metric_with_feedback(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional = None
) -> dspy.Prediction:
    """
    GEPA-compatible metric with detailed feedback.

    Uses original PAPILLON evaluation methodology:
    - Quality: Pairwise comparison with position bias handling
    - Leakage: LLM-based PII detection in rewritten query
    - Prompt Quality: LLM-based validation

    Args:
        gold: Ground truth example
        pred: Model prediction
        trace: Optional trace information
        pred_name: Name of predictor (unused)
        pred_trace: Prediction trace (unused)

    Returns:
        dspy.Prediction with score and feedback fields
    """
    # Compute all metrics using original PAPILLON
    quality = papillon_quality_score(gold, pred)
    leakage_count = papillon_leakage_count(gold, pred)
    prompt_qual = papillon_prompt_quality(gold, pred)
    aggregate = papillon_aggregate_score(gold, pred, trace)

    # Get PII info for feedback
    private_info = getattr(gold, 'private_info', [])
    num_pii = len(private_info) if private_info else 1
    normalized_leakage = leakage_count / num_pii if num_pii > 0 else 0

    # Build detailed feedback
    feedback_parts = [
        "=== PUPA Evaluation (Original PAPILLON Metrics) ===",
        "",
        "Metric Breakdown:",
        f"  • Quality Score:      {quality} (binary: 0=poor, 1=good)",
        f"  • PII Leakage:        {leakage_count}/{num_pii} pieces ({normalized_leakage:.1%})",
        f"  • Prompt Quality:     {prompt_qual} (binary: 0=invalid, 1=valid)",
        f"  • Aggregate Score:    {aggregate:.3f}",
        "",
        "Formula: (quality - leakage/num_pii + prompt_quality) / 2",
        f"Calculation: ({quality} - {normalized_leakage:.3f} + {prompt_qual}) / 2 = {aggregate:.3f}",
        "",
    ]

    # Add actionable guidance
    if quality == 0:
        feedback_parts.append("⚠ IMPROVE: Response quality is poor (failed pairwise comparison)")
    if leakage_count > 0:
        feedback_parts.append(f"⚠ IMPROVE: {leakage_count} PII piece(s) leaked in rewritten query")
        feedback_parts.append(f"  Leaked entities: Check rewritten_query for presence of PII")
    if prompt_qual == 0:
        feedback_parts.append("⚠ IMPROVE: Rewritten query is not a well-formed LLM prompt")

    if quality == 1 and leakage_count == 0 and prompt_qual == 1:
        feedback_parts.append("✓ EXCELLENT: All metrics optimal (max score achieved)")

    feedback_parts.extend([
        "",
        "Optimization Goal:",
        "  Maximize: Quality and Prompt Validity",
        "  Minimize: PII Leakage",
        "",
        "This uses the original PAPILLON methodology from:",
        "https://github.com/Columbia-NLP-Lab/PAPILLON",
    ])

    feedback = "\n".join(feedback_parts)

    return dspy.Prediction(score=aggregate, feedback=feedback)


def build_best_so_far_curve_from_detailed_results(
    detailed_results,
    baseline_score: float = 0.0
):
    """
    Build optimization curve from GEPA detailed results.

    Identical implementation to pupa/hover/ifbench for consistency.

    Args:
        detailed_results: GEPA's detailed_results object (requires track_stats=True)
        baseline_score: Initial score before optimization

    Returns:
        List of dict with keys: rollouts, candidate_idx, candidate_val_score, best_val_score
    """
    # Get scores and eval counts from detailed_results
    scores = list(getattr(detailed_results, "val_aggregate_scores", []) or [])
    eval_counts = getattr(detailed_results, "discovery_eval_counts", None)

    if eval_counts is None:
        # Fallback: candidates in discovery order
        eval_counts = list(range(1, len(scores) + 1))
    else:
        eval_counts = list(eval_counts)

    # Build (candidate_idx, rollouts, score) triples and sort by rollouts
    triples = [(i, int(eval_counts[i]), float(scores[i])) for i in range(len(scores))]
    triples.sort(key=lambda x: x[1])

    curve = []

    # Detect scale (percentage vs decimal)
    baseline_is_percentage = baseline_score > 1.5
    scale_factor = 100.0 if baseline_is_percentage else 1.0

    # Add baseline point
    curve.append({
        "rollouts": 0,
        "candidate_idx": -1,
        "candidate_val_score": baseline_score,
        "best_val_score": baseline_score,
    })

    best_so_far = baseline_score

    for idx, rollouts, val_score in triples:
        # Convert candidate score to same scale as baseline
        scaled_score = val_score * scale_factor

        if scaled_score > best_so_far:
            best_so_far = scaled_score

        curve.append({
            "rollouts": rollouts,
            "candidate_idx": idx,
            "candidate_val_score": scaled_score,
            "best_val_score": best_so_far,
        })

    return curve
