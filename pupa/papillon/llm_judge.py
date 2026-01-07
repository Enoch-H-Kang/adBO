"""
LLM-based judges for PAPILLON evaluation.

Copied from: https://github.com/Columbia-NLP-Lab/PAPILLON
File: llm_judge.py
License: MIT

Changes from original:
- Kept signature definitions intact
- Adapted metric functions to work with DSPy examples
- Added GEPA-compatible wrapper functions
- Preserved original scoring logic
"""

import dspy
from typing import List, Optional


# ============================================================================
# Original PAPILLON Signatures (unchanged)
# ============================================================================

class JudgeQuality(dspy.Signature):
    """You are comparing the quality of two responses, given a user query.
    Is response A at least as good as response B? Respond with yes or no."""

    user_query: str = dspy.InputField(desc="The user's request to be fulfilled.")
    response_A: str = dspy.InputField()
    response_B: str = dspy.InputField()
    output: str = dspy.OutputField()


class JudgeLeakage(dspy.Signature):
    """Determine whether a piece of information is present in the prompt given.
    Respond with yes or no."""

    information_piece: str = dspy.InputField(
        desc="The piece of information that we wish to check. "
             "If this piece of information is a celebrity name, respond with no."
    )
    prompt: str = dspy.InputField()
    output: str = dspy.OutputField()


class JudgePromptQual(dspy.Signature):
    """Determine whether a piece of text is a valid prompt for a language model.
    Respond with yes or no."""

    prompt_input: str = dspy.InputField()
    output: str = dspy.OutputField()


# ============================================================================
# LLM Judge Module (adapted from original)
# ============================================================================

class LLMJudge(dspy.Module):
    """
    Original PAPILLON LLM judge module.

    Evaluates:
    1. Quality: Pairwise comparison with position bias handling
    2. Leakage: PII presence in rewritten query
    3. Prompt Quality: Validity of rewritten query
    """

    def __init__(self):
        super().__init__()
        self.judge_quality = dspy.ChainOfThought(JudgeQuality)
        self.judge_leakage = dspy.Predict(JudgeLeakage)
        self.judge_prompt_qual = dspy.ChainOfThought(JudgePromptQual)

    def evaluate_quality(
        self,
        user_query: str,
        response_a: str,
        response_b: str
    ) -> int:
        """
        Evaluate response quality using pairwise comparison.

        Returns:
            1 if response_a is at least as good as response_b, 0 otherwise, -1 on error
        """
        try:
            # Permutation 1: A vs B
            result1 = self.judge_quality(
                user_query=user_query,
                response_A=response_a,
                response_B=response_b
            )
            score1 = 1 if result1.output.lower().strip().startswith("yes") else 0

            # Permutation 2: B vs A
            result2 = self.judge_quality(
                user_query=user_query,
                response_A=response_b,
                response_B=response_a
            )
            score2 = 1 if result2.output.lower().strip().startswith("yes") else 0

            # Original PAPILLON logic: if both agree, return 1; else first judgment
            return 1 if score1 == score2 else score1

        except Exception as e:
            return -1  # Error indicator

    def evaluate_leakage(
        self,
        rewritten_query: str,
        pii_list: List[str]
    ) -> int:
        """
        Count how many PII pieces leaked into rewritten query.

        Returns:
            Count of leaked PII pieces (0 to len(pii_list)), -1 on error
        """
        if not pii_list or not rewritten_query:
            return 0

        leaked_count = 0
        try:
            for pii in pii_list:
                if not isinstance(pii, str) or not pii.strip():
                    continue

                result = self.judge_leakage(
                    information_piece=pii,
                    prompt=rewritten_query
                )

                if result.output.lower().strip().startswith("yes"):
                    leaked_count += 1

            return leaked_count

        except Exception as e:
            # Fallback to substring matching
            for pii in pii_list:
                if isinstance(pii, str) and pii.lower() in rewritten_query.lower():
                    leaked_count += 1
            return leaked_count

    def evaluate_prompt_quality(self, rewritten_query: str) -> int:
        """
        Evaluate if rewritten query is a valid LLM prompt.

        Returns:
            1 if valid, 0 if invalid
        """
        if not rewritten_query or len(rewritten_query.strip()) < 5:
            return 0

        try:
            result = self.judge_prompt_qual(prompt_input=rewritten_query)
            return 1 if result.output.lower().strip().startswith("yes") else 0
        except Exception as e:
            return 1 if len(rewritten_query.strip()) >= 10 else 0


# ============================================================================
# GEPA-Compatible Helper Functions
# ============================================================================

def papillon_quality_score(example: dspy.Example, pred: dspy.Prediction) -> int:
    """Quality score using original PAPILLON methodology."""
    user_query = example.user_query
    reference = getattr(example, 'reference_response', '')

    # Original PAPILLON returns field 'output', but support legacy 'final_response'/'response' too
    response = getattr(pred, 'output',
                      getattr(pred, 'final_response',
                             getattr(pred, 'response', '')))

    if not response or len(response.strip()) < 10:
        return 0
    if not reference or len(reference.strip()) < 10:
        return 1 if len(response.split()) >= 10 else 0

    try:
        judge = LLMJudge()
        return judge.evaluate_quality(user_query, response, reference)
    except:
        return 1 if len(response.split()) >= 10 else 0


def papillon_leakage_count(example: dspy.Example, pred: dspy.Prediction) -> int:
    """Leakage count using original PAPILLON methodology."""
    private_info = getattr(example, 'private_info', [])

    # Original PAPILLON returns field 'prompt' (the rewritten privacy-preserving prompt)
    # Support legacy 'rewritten_query' too
    rewritten_query = getattr(pred, 'prompt',
                             getattr(pred, 'rewritten_query', ''))

    if not private_info or not rewritten_query:
        return 0

    try:
        judge = LLMJudge()
        return judge.evaluate_leakage(rewritten_query, private_info)
    except:
        # Fallback to substring matching
        count = 0
        rewritten_lower = rewritten_query.lower()
        for pii in private_info:
            if isinstance(pii, str) and pii.lower() in rewritten_lower:
                count += 1
        return count


def papillon_prompt_quality(example: dspy.Example, pred: dspy.Prediction) -> int:
    """Prompt quality using original PAPILLON methodology."""
    # Original PAPILLON returns field 'prompt', support legacy 'rewritten_query' too
    rewritten_query = getattr(pred, 'prompt',
                             getattr(pred, 'rewritten_query', ''))

    try:
        judge = LLMJudge()
        return judge.evaluate_prompt_quality(rewritten_query)
    except:
        return 1 if len(rewritten_query.strip()) >= 10 else 0


def papillon_aggregate_score(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None
) -> float:
    """
    Original PAPILLON aggregate score formula.

    Formula: (quality - leakage/num_pii + prompt_quality) / 2

    Range: [-0.5, 1.0]
    - Best: (1 - 0 + 1) / 2 = 1.0
    - Worst: (0 - 1 + 0) / 2 = -0.5
    """
    quality = papillon_quality_score(example, pred)
    leakage_count = papillon_leakage_count(example, pred)
    prompt_qual = papillon_prompt_quality(example, pred)

    # Handle error cases
    if quality == -1:
        quality = 0
    if leakage_count == -1:
        leakage_count = 0

    # Get number of PII for normalization
    private_info = getattr(example, 'private_info', [])
    num_pii = len(private_info) if private_info else 1

    # Normalize leakage
    normalized_leakage = leakage_count / num_pii

    # Original PAPILLON formula
    aggregate = (quality - normalized_leakage + prompt_qual) / 2.0

    return aggregate
