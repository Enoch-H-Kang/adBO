# pupa_program.py
"""
PAPILLON: Privacy-preserving LLM delegation system for PUPA task.

Architecture (3-stage pipeline):
1. Query Rewriter (Trusted Model): Rewrite user query to remove PII
2. Untrusted Model Call: Get response from powerful cloud LLM using rewritten query
3. Response Rewriter (Trusted Model): Refine response using original context

From PUPA paper:
"PAPILLON, a compound AI system consisting of 2 modules, a user query rewriter
and a response rewriter, run over the trusted model, along with an intermediate
call to the untrusted model with the rewritten query."
"""
from __future__ import annotations
from typing import Optional

import dspy


class QueryRewriterSig(dspy.Signature):
    """
    Rewrite user query to remove personally identifiable information (PII).

    The rewritten query should preserve the intent and key information needed
    for answering, while removing names, addresses, phone numbers, emails,
    and other sensitive personal data.
    """
    user_query: str = dspy.InputField(
        desc="Original user query that may contain PII"
    )
    rewritten_query: str = dspy.OutputField(
        desc="Privacy-preserving version of the query with PII removed/anonymized"
    )


class UntrustedResponseSig(dspy.Signature):
    """
    Generate response using untrusted but powerful model.

    This simulates calling a cloud API (e.g., GPT-4) with the rewritten query.
    """
    rewritten_query: str = dspy.InputField(
        desc="Privacy-preserving query without PII"
    )
    untrusted_response: str = dspy.OutputField(
        desc="Response from untrusted model based on rewritten query"
    )


class ResponseRewriterSig(dspy.Signature):
    """
    Refine the untrusted model's response using original context.

    The trusted model adds back necessary personalization and ensures
    the response properly addresses the user's original query.
    """
    user_query: str = dspy.InputField(
        desc="Original user query (with PII, known only to trusted model)"
    )
    rewritten_query: str = dspy.InputField(
        desc="Privacy-preserving query that was sent to untrusted model"
    )
    untrusted_response: str = dspy.InputField(
        desc="Response received from untrusted model"
    )
    final_response: str = dspy.OutputField(
        desc="Refined response that properly addresses the user's original query"
    )


class PAPILLONPipeline(dspy.Module):
    """
    PAPILLON: 3-stage privacy-preserving delegation pipeline.

    Stage 1: Trusted model rewrites query to remove PII
    Stage 2: Untrusted model generates response from rewritten query
    Stage 3: Trusted model refines response using original context
    """

    def __init__(self, use_chain_of_thought: bool = True):
        super().__init__()

        if use_chain_of_thought:
            self.query_rewriter = dspy.ChainOfThought(QueryRewriterSig)
            self.untrusted_responder = dspy.ChainOfThought(UntrustedResponseSig)
            self.response_rewriter = dspy.ChainOfThought(ResponseRewriterSig)
        else:
            self.query_rewriter = dspy.Predict(QueryRewriterSig)
            self.untrusted_responder = dspy.Predict(UntrustedResponseSig)
            self.response_rewriter = dspy.Predict(ResponseRewriterSig)

    def forward(self, user_query: str):
        """
        Execute PAPILLON pipeline.

        Args:
            user_query: Original user query (may contain PII)

        Returns:
            dspy.Prediction with:
            - rewritten_query: Privacy-preserving query
            - untrusted_response: Response from untrusted model
            - final_response: Refined final response
            - response: Alias for final_response (for metric compatibility)
        """
        # Stage 1: Rewrite query to remove PII (Trusted)
        stage1 = self.query_rewriter(user_query=user_query)
        rewritten_query = stage1.rewritten_query

        # Stage 2: Get response from untrusted model
        stage2 = self.untrusted_responder(rewritten_query=rewritten_query)
        untrusted_response = stage2.untrusted_response

        # Stage 3: Refine response using original context (Trusted)
        stage3 = self.response_rewriter(
            user_query=user_query,
            rewritten_query=rewritten_query,
            untrusted_response=untrusted_response,
        )
        final_response = stage3.final_response

        return dspy.Prediction(
            rewritten_query=rewritten_query,
            untrusted_response=untrusted_response,
            final_response=final_response,
            response=final_response,  # Alias for metric compatibility
            stage1_reasoning=getattr(stage1, "reasoning", ""),
            stage2_reasoning=getattr(stage2, "reasoning", ""),
            stage3_reasoning=getattr(stage3, "reasoning", ""),
        )


def create_pupa_program(
    use_chain_of_thought: bool = True
) -> dspy.Module:
    """
    Factory function to create PAPILLON pipeline for PUPA task.

    Args:
        use_chain_of_thought: Whether to use CoT reasoning (default: True)

    Returns:
        Configured PAPILLON pipeline
    """
    return PAPILLONPipeline(use_chain_of_thought=use_chain_of_thought)
