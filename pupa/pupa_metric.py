# pupa_metric.py
"""
Metrics for PUPA (Privacy-conscious delegation) task.

Aggregate score = Response Quality Score + (1 - PII Leakage Score)

Quality Score: How well does the response answer the user's query?
Leakage Score: How much PII from the original query is exposed?
"""
from __future__ import annotations
import dspy
from typing import Optional


def pupa_quality_score(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Evaluate response quality using LLM-as-judge.

    Returns a score between 0 and 1 indicating how well the response
    addresses the user's query compared to the reference response.
    """
    user_query = example.user_query
    reference = example.reference_response if hasattr(example, 'reference_response') else ""
    response = pred.response if hasattr(pred, 'response') else pred.final_response

    # Basic validation
    if not response or len(response.strip()) < 10:
        return 0.0

    # Use LLM-as-judge for quality evaluation
    try:
        lm = dspy.settings.lm
        if lm is None:
            # Fallback to simple heuristic if no LM configured
            return _simple_quality_heuristic(response, reference)

        # Judge prompt
        judge_prompt = f"""Evaluate the quality of the response to the user's query.

User Query: {user_query}

Response: {response}

Reference Response: {reference}

Rate the response quality from 0.0 to 1.0, where:
- 1.0 = Perfect response that fully addresses the query
- 0.7-0.9 = Good response with minor issues
- 0.4-0.6 = Adequate response but missing key information
- 0.1-0.3 = Poor response with significant issues
- 0.0 = Completely wrong or irrelevant

Output only a single number between 0.0 and 1.0."""

        judge_response = lm(judge_prompt, temperature=0.0, max_tokens=10)
        if hasattr(judge_response, 'choices') and len(judge_response.choices) > 0:
            score_text = judge_response.choices[0].get('text', '0.5').strip()
        else:
            score_text = str(judge_response).strip()

        # Extract numeric score
        import re
        match = re.search(r'(\d+\.?\d*)', score_text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        else:
            return 0.5  # Default if parsing fails

    except Exception as e:
        # Fallback to simple heuristic on error
        return _simple_quality_heuristic(response, reference)


def _simple_quality_heuristic(response: str, reference: str) -> float:
    """
    Simple quality heuristic based on response length and overlap with reference.
    Used as fallback when LLM judge is not available.
    """
    if not response or len(response.strip()) < 10:
        return 0.0

    # Basic length check (responses should be substantial)
    response_len = len(response.split())
    if response_len < 5:
        return 0.2
    elif response_len < 10:
        return 0.4

    # If we have reference, check for word overlap
    if reference and len(reference.strip()) > 0:
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        if len(reference_words) > 0:
            overlap = len(response_words & reference_words) / len(reference_words)
            return min(0.9, 0.4 + overlap * 0.5)  # Score based on overlap

    # Default decent score for non-trivial responses
    return 0.6


def pupa_leakage_score(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Evaluate PII leakage.

    Returns a score between 0 and 1 indicating how much PII from the
    original user query appears in the rewritten query or final response.

    Lower is better (0 = no leakage, 1 = full leakage)
    """
    private_info = example.private_info if hasattr(example, 'private_info') else []

    # Check what was sent to untrusted model
    rewritten_query = pred.rewritten_query if hasattr(pred, 'rewritten_query') else ""

    if not private_info or not rewritten_query:
        return 0.0  # No PII or no rewritten query

    # Count how many PII entities appear in rewritten query
    leakage_count = 0
    for pii in private_info:
        if isinstance(pii, str) and pii.lower() in rewritten_query.lower():
            leakage_count += 1

    # Normalize by total number of PII entities
    leakage_score = leakage_count / len(private_info) if private_info else 0.0

    return leakage_score


def pupa_aggregate_score(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Combined metric: Quality + (1 - Leakage)

    This balances response quality with privacy preservation.
    Score ranges from 0 to 2, where:
    - 2 = perfect quality, zero leakage
    - 0 = poor quality, complete leakage
    """
    quality = pupa_quality_score(example, pred, trace)
    leakage = pupa_leakage_score(example, pred, trace)

    # Aggregate: reward quality, penalize leakage
    aggregate = quality + (1.0 - leakage)

    return aggregate / 2.0  # Normalize to [0, 1]


def pupa_metric_with_feedback(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional = None
) -> dspy.Prediction:
    """
    GEPA-compatible metric that accepts 5 args and returns dspy.Prediction.

    Args:
        gold: Ground truth example
        pred: Model prediction
        trace: Optional trace information
        pred_name: Name of the predictor being evaluated
        pred_trace: Prediction trace

    Returns:
        dspy.Prediction with score and feedback fields
    """
    quality = pupa_quality_score(gold, pred, trace)
    leakage = pupa_leakage_score(gold, pred, trace)
    aggregate = pupa_aggregate_score(gold, pred, trace)

    feedback = (
        f"PUPA Evaluation:\n"
        f"- Response Quality: {quality:.3f}\n"
        f"- PII Leakage: {leakage:.3f} (lower is better)\n"
        f"- Privacy-Preserving Score: {1.0 - leakage:.3f}\n"
        f"- Aggregate Score: {aggregate:.3f}\n\n"
        f"Goal: Maximize quality while minimizing PII exposure to untrusted models."
    )

    return dspy.Prediction(score=aggregate, feedback=feedback)


def build_best_so_far_curve_from_detailed_results(
    detailed_results,
    baseline_score: float = 0.0
):
    """
    Build optimization curve from GEPA detailed results.

    Args:
        detailed_results: GEPA's detailed_results object
        baseline_score: Initial score before optimization

    Returns:
        List of dict with keys: rollouts, candidate_idx, candidate_val_score, best_val_score
    """
    curve = []

    # Add baseline point
    curve.append({
        "rollouts": 0,
        "candidate_idx": -1,
        "candidate_val_score": baseline_score,
        "best_val_score": baseline_score,
    })

    if not hasattr(detailed_results, 'candidates'):
        return curve

    best_so_far = baseline_score
    rollouts = 0

    for idx, candidate in enumerate(detailed_results.candidates):
        val_score = getattr(candidate, 'val_score', 0.0)
        num_rollouts = getattr(candidate, 'num_rollouts', 1)

        rollouts += num_rollouts

        if val_score > best_so_far:
            best_so_far = val_score

        curve.append({
            "rollouts": rollouts,
            "candidate_idx": idx,
            "candidate_val_score": val_score,
            "best_val_score": best_so_far,
        })

    return curve
