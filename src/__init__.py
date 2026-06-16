"""
Syablit - Abliteration Toolkit for LLM Red-Teaming
Remove refusal behavior from LLMs via activation steering.
"""

__version__ = "0.1.0"
__author__ = "suvijya"

from .abliterator import Abliterator
from .data import get_harmful_instructions, get_harmless_instructions
