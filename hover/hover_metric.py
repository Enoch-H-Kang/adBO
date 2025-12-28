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
    Fraction of gold titles present anywhere in pred.titles.
    """
    gold_titles = {normalize_title(t) for t in getattr(gold, "titles", [])}
    pred_titles = {normalize_title(t) for t in getattr(pred, "titles", [])}

    if not gold_titles:
        return 0.0

    return len(gold_titles & pred_titles) / len(gold_titles)


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

    if baseline_score is not None:
        best = baseline_score
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
