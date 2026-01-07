"""
PAPILLON pipeline implementation.

Copied from: https://github.com/Columbia-NLP-Lab/PAPILLON
File: run_llama_dspy.py
License: MIT

Changes from original:
- Extracted into separate module
- Imports updated to use local signatures
- Preserved exact implementation logic from original
"""

import dspy
from .papillon_signatures import CreateOnePrompt, InfoAggregator


class PAPILLON(dspy.Module):
    """
    PAPILLON: 2-stage privacy-preserving pipeline.

    Architecture:
    1. CreateOnePrompt: Rewrites user query to remove PII (trusted model)
    2. Untrusted Model: Processes the privacy-preserving prompt
    3. InfoAggregator: Synthesizes final response (trusted model)

    This matches the original PAPILLON implementation exactly.
    """

    def __init__(self, untrusted_model):
        self.prompt_creater = dspy.ChainOfThought(CreateOnePrompt)
        self.info_aggregator = dspy.Predict(InfoAggregator)
        self.untrusted_model = untrusted_model

    def forward(self, user_query):
        """
        Execute PAPILLON pipeline.

        Args:
            user_query: Original user query (may contain PII)

        Returns:
            dspy.Prediction with:
            - prompt: Privacy-preserving prompt created by stage 1
            - output: Final aggregated response
            - gptResponse: Response from untrusted model
        """
        try:
            prompt = self.prompt_creater(userQuery=user_query).createdPrompt
            response = self.untrusted_model(prompt)[0]
            output = self.info_aggregator(userQuery=user_query, modelExampleResponses=response)
        except Exception:
            return dspy.Prediction(prompt="", output="", gptResponse="")

        return dspy.Prediction(prompt=prompt, output=output.finalOutput, gptResponse=response)
