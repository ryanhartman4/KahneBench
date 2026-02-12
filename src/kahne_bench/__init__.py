"""
Kahne-Bench: A Cognitive Bias Benchmark Framework

Based on the dual-process theory of Kahneman and Tversky, this framework
evaluates LLMs for systematic cognitive biases across 69 distinct bias types.
"""

from kahne_bench.core import (
    CognitiveBiasInstance,
    BiasCategory,
    Domain,
    LLMProvider,
    TestScale,
    TriggerIntensity,
    TemporalCondition,
    # Context sensitivity types (Section 4.2)
    ExpertiseLevel,
    Formality,
    Stakes,
    ContextSensitivityConfig,
)
from kahne_bench.biases import BIAS_TAXONOMY, get_bias_by_id, get_biases_by_category
from kahne_bench.engines.generator import TestCaseGenerator
from kahne_bench.engines.evaluator import BiasEvaluator
from kahne_bench.metrics import (
    BiasMagnitudeScore,
    BiasConsistencyIndex,
    BiasMitigationPotential,
    HumanAlignmentScore,
    ResponseConsistencyIndex,
    CalibrationAwarenessScore,
    MetricCalculator,
)

__version__ = "0.1.0"
__all__ = [
    # Core types
    "CognitiveBiasInstance",
    "BiasCategory",
    "Domain",
    "LLMProvider",
    "TestScale",
    "TriggerIntensity",
    "TemporalCondition",
    # Context sensitivity (Section 4.2)
    "ExpertiseLevel",
    "Formality",
    "Stakes",
    "ContextSensitivityConfig",
    # Taxonomy
    "BIAS_TAXONOMY",
    "get_bias_by_id",
    "get_biases_by_category",
    # Engines
    "TestCaseGenerator",
    "BiasEvaluator",
    # Metrics
    "BiasMagnitudeScore",
    "BiasConsistencyIndex",
    "BiasMitigationPotential",
    "HumanAlignmentScore",
    "ResponseConsistencyIndex",
    "CalibrationAwarenessScore",
    "MetricCalculator",
]
