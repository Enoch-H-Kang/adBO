"""
Program factory for PUPA benchmark using original PAPILLON.

Creates instances of the original PAPILLON pipeline from the
papillon/ module.
"""

import dspy
from papillon.papillon_pipeline import PAPILLON


def _create_untrusted_model_wrapper():
    """
    Create a wrapper for the untrusted model that matches PAPILLON's interface.

    The original PAPILLON expects untrusted_model to be a callable that:
    - Takes a prompt string
    - Returns a list with at least one response

    We wrap DSPy's language model to match this interface.
    """
    def untrusted_model(prompt: str):
        """
        Simulate calling an untrusted powerful model.

        In the original PAPILLON, this would call GPT-4 or another API model.
        For GEPA optimization, we use the same DSPy LM that's being optimized.

        Args:
            prompt: The privacy-preserving prompt

        Returns:
            List containing the model's response as first element
        """
        try:
            # Use DSPy's current language model to generate response
            response = dspy.Predict("question -> answer")(question=prompt).answer
            return [response]
        except Exception as e:
            # Fallback on error
            return [""]

    return untrusted_model


def create_pupa_program() -> dspy.Module:
    """
    Factory function to create original PAPILLON pipeline.

    This uses the actual PAPILLON implementation from:
    https://github.com/Columbia-NLP-Lab/PAPILLON

    Returns:
        PAPILLON pipeline module with 3-stage architecture:
        1. CreateOnePrompt: Removes PII from user query (trusted)
        2. Untrusted Model: Generates response to privacy-preserving prompt
        3. InfoAggregator: Synthesizes final response (trusted)
    """
    untrusted_model = _create_untrusted_model_wrapper()
    return PAPILLON(untrusted_model=untrusted_model)
