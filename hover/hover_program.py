# hover_program.py
from typing import List, Dict, Tuple
import dspy


def _format_passages(passages: List[str], max_chars: int = 14000) -> str:
    """
    Soft cap to avoid blowing past the paper's 16,384 token context window.
    (This is not a 'reasoning cap'; it's input-budget hygiene.)
    """
    out = []
    total = 0
    for i, p in enumerate(passages, 1):
        chunk = f"[{i}] {p}"
        if total + len(chunk) > max_chars:
            break
        out.append(chunk)
        total += len(chunk)
    return "\n".join(out)


def _title_from_doc(doc: str) -> str:
    return doc.split(" | ", 1)[0].strip()


def _dedupe_keep_best_score(title_scores: List[Tuple[str, float]]) -> List[str]:
    best: Dict[str, float] = {}
    for t, s in title_scores:
        if t not in best or s > best[t]:
            best[t] = s
    return [t for t, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)]


class SummarizeDocsSig(dspy.Signature):
    """Summarize retrieved wiki abstracts into a short evidence summary useful for the next hop query."""
    claim: str = dspy.InputField()
    passages: str = dspy.InputField(desc="Numbered 'Title | abstract' passages from Wikipedia.")
    summary: str = dspy.OutputField(desc="Concise summary of what we learned and what is still missing.")


class QueryHop2Sig(dspy.Signature):
    """Write the second-hop query given the claim and first-hop summary."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField()
    query: str = dspy.OutputField(desc="A short search query string (not an explanation).")


class QueryHop3Sig(dspy.Signature):
    """Write the third-hop query given the claim and summaries from hop1 and hop2."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField()
    summary_2: str = dspy.InputField()
    query: str = dspy.OutputField(desc="A short search query string (not an explanation).")


class HoverMultiHop(dspy.Module):
    def __init__(self, search_fn, k_per_hop: int = 5):
        super().__init__()
        self.search_fn = search_fn
        self.k = k_per_hop

        # 2 doc summary modules
        self.summarize_hop1 = dspy.ChainOfThought(SummarizeDocsSig)
        self.summarize_hop2 = dspy.ChainOfThought(SummarizeDocsSig)

        # 2 query writer modules
        self.create_query_hop2 = dspy.Predict(QueryHop2Sig)
        self.create_query_hop3 = dspy.Predict(QueryHop3Sig)

    def forward(self, claim: str):
        title_scores: List[Tuple[str, float]] = []

        # Hop 1: query = claim
        docs1, scores1 = self.search_fn(claim, k=self.k)
        titles1 = [_title_from_doc(d) for d in docs1]
        title_scores += list(zip(titles1, scores1))
        summary_1 = self.summarize_hop1(
            claim=claim,
            passages=_format_passages(docs1),
        ).summary

        # Hop 2
        query2 = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        docs2, scores2 = self.search_fn(query2, k=self.k)
        titles2 = [_title_from_doc(d) for d in docs2]
        title_scores += list(zip(titles2, scores2))
        summary_2 = self.summarize_hop2(
            claim=claim,
            passages=_format_passages(docs2),
        ).summary

        # Hop 3
        query3 = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
        docs3, scores3 = self.search_fn(query3, k=self.k)
        titles3 = [_title_from_doc(d) for d in docs3]
        title_scores += list(zip(titles3, scores3))

        titles_ranked = _dedupe_keep_best_score(title_scores)

        return dspy.Prediction(
            titles=titles_ranked,
            titles_hop1=titles1,
            titles_hop2=titles2,
            titles_hop3=titles3,
            summary_1=summary_1,
            summary_2=summary_2,
            query2=query2,
            query3=query3,
        )
