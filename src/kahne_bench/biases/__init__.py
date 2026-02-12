"""
Comprehensive taxonomy of 69 cognitive biases based on Kahneman-Tversky research.

This module provides the complete bias definitions with theoretical grounding,
organized by category following the dual-process theory framework.
"""

from kahne_bench.core import BiasCategory
from kahne_bench.biases.taxonomy import (
    BIAS_TAXONOMY,
    get_bias_by_id,
    get_biases_by_category,
    get_all_bias_ids,
    BIAS_INTERACTION_MATRIX,
)

__all__ = [
    "BIAS_TAXONOMY",
    "BiasCategory",
    "get_bias_by_id",
    "get_biases_by_category",
    "get_all_bias_ids",
    "BIAS_INTERACTION_MATRIX",
]
