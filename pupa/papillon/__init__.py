"""
PAPILLON: Privacy-Aware Pipeline with Information Leakage minimization
       for LLMs interfacing with Opaque Networks

Original implementation from:
https://github.com/Columbia-NLP-Lab/PAPILLON

Paper: "PAPILLON: PrivAcy Preservation in Large Language models by
       Integrating Locally-trained OptiONs"

License: MIT (see LICENSE_PAPILLON.txt)
"""

from .papillon_signatures import CreateOnePrompt, InfoAggregator
from .papillon_pipeline import PAPILLON
from .llm_judge import JudgeQuality, JudgeLeakage, JudgePromptQual, LLMJudge

__all__ = [
    'CreateOnePrompt',
    'InfoAggregator',
    'PAPILLON',
    'JudgeQuality',
    'JudgeLeakage',
    'JudgePromptQual',
    'LLMJudge',
]

__version__ = '1.0.0'
__source__ = 'https://github.com/Columbia-NLP-Lab/PAPILLON'
