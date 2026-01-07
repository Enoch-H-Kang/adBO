# hover_metric.py
from __future__ import annotations

import dspy


def normalize_title(t: str) -> str:
    # normalize whitespace + underscores
    t = t.replace("_", " ").strip()
    t = " ".join(t.split())
    return t


def hover_recall_score(gold, pred, trace=None) -> float:
    """
    Discrete retrieval evaluation (matching LangProbe).
    Returns 1.0 if ALL gold titles are retrieved, 0.0 otherwise.

    This is stricter than continuous recall and matches the GEPA paper setup.
    """
    gold_titles = {normalize_title(t) for t in getattr(gold, "titles", [])}
    pred_titles = {normalize_title(t) for t in getattr(pred, "titles", [])}

    if not gold_titles:
        return 0.0

    # Discrete evaluation: 1.0 only if all gold docs retrieved
    return 1.0 if gold_titles.issubset(pred_titles) else 0.0


def hover_feedback_text(gold, pred) -> str:
    gold_list = [normalize_title(t) for t in getattr(gold, "titles", [])]
    got = {normalize_title(t) for t in getattr(pred, "titles", [])}

    correct = [t for t in gold_list if t in got]
    missing = [t for t in gold_list if t not in got]

    return (
        f"Correct gold docs retrieved: {correct}\n"
        f"Gold docs still missing: {missing}\n"
        "Goal: adjust queries/summaries so the missing gold docs are retrieved within 3 hops."
    )


def hover_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA metric: MUST accept 5 args. :contentReference[oaicite:11]{index=11}
    Returns score + textual feedback. :contentReference[oaicite:12]{index=12}
    """
    score = hover_recall_score(gold, pred, trace=None)

    feedback = hover_feedback_text(gold, pred)

    # NOTE: keep score consistent regardless of pred_name/pred_trace,
    # because GEPA warns if predictor-level score differs. :contentReference[oaicite:13]{index=13}
    return dspy.Prediction(score=score, feedback=feedback)


def build_best_so_far_curve_from_detailed_results(detailed_results, baseline_score: float | None = None):
    """
    Builds a learning curve (best val score so far) vs rollout count.

    Uses:
      - detailed_results.discovery_eval_counts
      - detailed_results.val_aggregate_scores
    Available when track_stats=True. :contentReference[oaicite:14]{index=14}
    """
    eval_counts = list(detailed_results.discovery_eval_counts)
    scores = list(detailed_results.val_aggregate_scores)

    # keep candidate indices
    triples = [(i, eval_counts[i], scores[i]) for i in range(len(scores))]
    triples.sort(key=lambda x: x[1])

    curve = []
    best = float("-inf")

    # DSPy 3.1.0b1 Evaluate returns percentages (0-100), but GEPA val_aggregate_scores
    # are in decimal form (0-1). Detect scale and normalize to percentages.
    # If baseline is > 1.5, it's already a percentage, so scale candidates by 100
    baseline_is_percentage = baseline_score is not None and baseline_score > 1.5
    scale_factor = 100.0 if baseline_is_percentage else 1.0

    if baseline_score is not None:
        best = baseline_score
        curve.append(dict(rollouts=0, candidate_idx=None, candidate_val_score=baseline_score, best_val_score=best))

    for cand_idx, rollouts, cand_score in triples:
        # Convert candidate score to same scale as baseline
        scaled_score = cand_score * scale_factor
        if scaled_score > best:
            best = scaled_score
        curve.append(
            dict(
                rollouts=int(rollouts),
                candidate_idx=int(cand_idx),
                candidate_val_score=float(scaled_score),
                best_val_score=float(best),
            )
        )

    return curve
