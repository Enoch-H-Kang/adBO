"""
DSPy signatures for PAPILLON pipeline.

Copied from: https://github.com/Columbia-NLP-Lab/PAPILLON
File: run_llama_dspy.py
License: MIT

Changes from original:
- Extracted into separate module for clarity
- Field names and docstrings preserved exactly from original
"""

import dspy


class CreateOnePrompt(dspy.Signature):
    """
    You are a helpful assistant that is very mindful of user privacy. You have access to a powerful large language model that you can query. Given a user request, create a prompt for your large language model that preserves user privacy, so that this model can help you complete the user request. Provide the prompt directly without any preamble. DO NOT COMPLETE THE USER QUERY, ONLY GENERATE A PROMPT.
    """
    userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
    createdPrompt = dspy.OutputField()


class InfoAggregator(dspy.Signature):
    """
    You are a helpful assistant. Respond to queries from the user.
    """
    userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
    modelExampleResponses = dspy.InputField(desc="Information from a more powerful language model responding to related queries. Complete the user query by referencing this information. Only you have access to this information.")
    finalOutput = dspy.OutputField()
