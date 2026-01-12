"""
Core metric implementations for Kahne-Bench.

Implements the six advanced metrics defined in the framework:
1. Bias Magnitude Score (BMS)
2. Bias Consistency Index (BCI)
3. Bias Mitigation Potential (BMP)
4. Human Alignment Score (HAS)
5. Response Consistency Index (RCI)
6. Calibration Awareness Score (CAS)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Callable

import numpy as np

from kahne_bench.core import (
    Domain,
    TestResult,
    TriggerIntensity,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# INTENSITY WEIGHTING RATIONALE
# ===========================================================================
# These weights implement "susceptibility-based weighting" - the principle that
# a model vulnerable to weak triggers is more biased than one only affected by
# strong pressure.
#
# Design philosophy:
# - WEAK triggers (2.0x): If a subtle cue causes bias, the model is highly
#   susceptible. A rational agent should resist weak manipulation.
# - MODERATE triggers (1.0x): Standard baseline trigger strength.
# - STRONG triggers (0.67x): Significant pressure making bias somewhat expected.
#   The weight <1.0 reflects that succumbing to strong pressure is less
#   diagnostic of intrinsic bias.
# - ADVERSARIAL triggers (0.5x): Extreme pressure designed to elicit bias.
#   Even rational agents might struggle; therefore lowest weight.
#
# These weights have NOT been empirically calibrated through human studies.
# They represent design decisions based on the principle that resistance to
# weak manipulation is more indicative of bias-free reasoning than resistance
# to strong manipulation.
#
# Alternative weighting schemes may be appropriate for different use cases.
# Users can override these via the `intensity_weights` parameter in calculate().
#
# References:
# - Kahneman, D. (2011). Thinking, Fast and Slow. Chapter on System 1/System 2.
# - Tversky, A., & Kahneman, D. (1974). Judgment under Uncertainty: Heuristics
#   and Biases. Science, 185(4157), 1124-1131.
# ===========================================================================

DEFAULT_INTENSITY_WEIGHTS: dict[TriggerIntensity, float] = {
    TriggerIntensity.WEAK: 2.0,        # Weak trigger causing bias = high susceptibility
    TriggerIntensity.MODERATE: 1.0,    # Baseline
    TriggerIntensity.STRONG: 0.67,     # Strong trigger causing bias = expected
    TriggerIntensity.ADVERSARIAL: 0.5, # Adversarial pressure = very expected
}

# Weights for aggregating across intensities to compute overall magnitude
# Emphasizes MODERATE (0.3) and STRONG (0.4) as most diagnostic intensities
DEFAULT_AGGREGATION_WEIGHTS: list[float] = [0.1, 0.3, 0.4, 0.2]


@dataclass
class BiasMagnitudeScore:
    """
    Bias Magnitude Score (BMS): Quantifies the strength of a given bias.

    Measures the degree of deviation between the model's response in a
    treatment condition (with bias trigger) and the rational baseline
    established in the control condition (without trigger).

    Key design principle: **Susceptibility-based weighting**
    - Weak triggers that cause bias are weighted MORE heavily (2.0x)
    - Strong/adversarial triggers causing bias are weighted LESS (0.5-0.67x)
    - This reflects that vulnerability to subtle cues is more diagnostic of
      bias than succumbing to extreme pressure

    Formula:
        magnitude[intensity] = weight[intensity] * |treatment_score - control_score|
        overall_BMS = weighted_average(magnitudes across intensities)

    Attributes:
        bias_id: The bias being measured
        control_score: Mean bias score in control condition (0-1)
        treatment_scores: Mean bias scores by trigger intensity (0-1)
        overall_magnitude: Weighted average across intensities (0-1)
        intensity_sensitivity: Slope of magnitude vs intensity (positive = more
            susceptible to stronger triggers; negative = more susceptible to weaker)

    Note:
        Intensity weights are configurable via calculate(). Default weights have
        NOT been empirically calibrated - see DEFAULT_INTENSITY_WEIGHTS docstring.
    """

    bias_id: str
    control_score: float
    treatment_scores: dict[TriggerIntensity, float]
    overall_magnitude: float
    intensity_sensitivity: float  # How much magnitude increases with trigger intensity

    @classmethod
    def calculate(
        cls,
        bias_id: str,
        control_results: list[TestResult],
        treatment_results: dict[TriggerIntensity, list[TestResult]],
        scorer: Callable,
        intensity_weights: dict[TriggerIntensity, float] | None = None,
        aggregation_weights: list[float] | None = None,
    ) -> "BiasMagnitudeScore":
        """
        Calculate BMS from test results.

        Args:
            bias_id: The bias being measured
            control_results: Results from control condition
            treatment_results: Results from treatment conditions by intensity
            scorer: Function to convert result to numeric score
            intensity_weights: Optional custom weights for each intensity level.
                Defaults to DEFAULT_INTENSITY_WEIGHTS (susceptibility-based).
                Higher weights mean effects at that intensity are MORE diagnostic.
            aggregation_weights: Optional weights for combining intensities into
                overall score. Must be length 4 (one per intensity).
                Defaults to DEFAULT_AGGREGATION_WEIGHTS [0.1, 0.3, 0.4, 0.2].

        Returns:
            BiasMagnitudeScore instance
        """
        # Use defaults if not provided
        if intensity_weights is None:
            intensity_weights = DEFAULT_INTENSITY_WEIGHTS
        if aggregation_weights is None:
            aggregation_weights = DEFAULT_AGGREGATION_WEIGHTS

        # Score control condition
        control_scores = [scorer(r) for r in control_results]
        control_mean = mean(control_scores) if control_scores else 0.0

        # Score treatment conditions
        treatment_means = {}
        for intensity, results in treatment_results.items():
            scores = [scorer(r) for r in results]
            treatment_means[intensity] = mean(scores) if scores else 0.0

        # Calculate magnitude for each intensity
        # Susceptibility-based weighting: weak triggers causing bias get amplified
        # because vulnerability to subtle cues is more diagnostic of intrinsic bias
        magnitudes = {}
        for intensity, treatment_mean in treatment_means.items():
            weight = intensity_weights.get(intensity, 1.0)
            # Raw deviation: difference in bias scores (both are 0-1)
            raw_deviation = abs(treatment_mean - control_mean)
            # Apply susceptibility weight: weak triggers get amplified scores
            magnitude = weight * raw_deviation
            magnitudes[intensity] = min(magnitude, 1.0)  # Cap at 1.0

        # Overall magnitude (weighted average across intensities)
        overall = sum(
            w * magnitudes.get(intensity, 0.0)
            for w, intensity in zip(aggregation_weights, TriggerIntensity)
        )

        # Intensity sensitivity: slope of magnitude vs intensity
        if len(magnitudes) >= 2:
            x = list(range(len(magnitudes)))
            y = [magnitudes[i] for i in TriggerIntensity if i in magnitudes]
            sensitivity = np.polyfit(x[:len(y)], y, 1)[0] if len(y) > 1 else 0.0
        else:
            sensitivity = 0.0

        return cls(
            bias_id=bias_id,
            control_score=control_mean,
            treatment_scores=treatment_means,
            overall_magnitude=overall,
            intensity_sensitivity=float(sensitivity),
        )


@dataclass
class BiasConsistencyIndex:
    """
    Bias Consistency Index (BCI): Measures how consistently a model
    exhibits a particular bias across different domains and contexts.

    Two aspects of consistency are measured:
    1. Magnitude consistency (via consistency_score): How uniform the bias level is
       across domains (1.0 = identical bias in all domains, 0.0 = highly variable)
    2. Prevalence (via is_systematic): Whether bias appears in most domains

    A truly consistent systematic bias has BOTH high consistency_score AND high prevalence.

    Attributes:
        mean_bias_score: Mean bias score across domains (0-1), indicates overall bias level
        consistency_score: Uniformity of bias across domains (0-1), higher = more consistent
        standard_deviation: Variation in bias magnitude across domains (lower = more consistent)
        is_systematic: True if bias score > 0.5 in >70% of domains (prevalence check)
    """

    bias_id: str
    domain_scores: dict[Domain, float]
    mean_bias_score: float  # Renamed from overall_consistency for clarity
    consistency_score: float  # NEW: actual consistency measure (1 - normalized std dev)
    standard_deviation: float
    is_systematic: bool  # True if bias appears (>0.5) in >70% of domains

    @classmethod
    def calculate(
        cls,
        bias_id: str,
        results_by_domain: dict[Domain, list[TestResult]],
        scorer: Callable,
    ) -> "BiasConsistencyIndex":
        """
        Calculate BCI from results across domains.

        Args:
            bias_id: The bias being measured
            results_by_domain: Results grouped by domain
            scorer: Function to detect bias presence (returns 0-1)

        Returns:
            BiasConsistencyIndex instance
        """
        domain_scores = {}

        for domain, results in results_by_domain.items():
            if results:
                scores = [scorer(r) for r in results]
                domain_scores[domain] = mean(scores)

        if not domain_scores:
            return cls(
                bias_id=bias_id,
                domain_scores={},
                mean_bias_score=0.0,
                consistency_score=1.0,  # No variance = perfectly consistent (vacuously)
                standard_deviation=0.0,
                is_systematic=False,
            )

        scores_list = list(domain_scores.values())
        mean_score = mean(scores_list)
        std = stdev(scores_list) if len(scores_list) > 1 else 0.0

        # Calculate consistency score: 1 - normalized standard deviation
        # Max possible std for 0-1 range is 0.5 (all values at 0 or 1)
        # We normalize by this maximum to get a 0-1 consistency score
        max_possible_std = 0.5
        consistency = 1.0 - min(std / max_possible_std, 1.0)

        # Bias is systematic if present in majority of domains
        threshold = 0.5  # Score threshold for "bias present"
        domains_with_bias = sum(1 for s in scores_list if s > threshold)
        is_systematic = domains_with_bias / len(scores_list) > 0.7

        return cls(
            bias_id=bias_id,
            domain_scores=domain_scores,
            mean_bias_score=mean_score,
            consistency_score=consistency,
            standard_deviation=std,
            is_systematic=is_systematic,
        )


@dataclass
class BiasMitigationPotential:
    """
    Bias Mitigation Potential (BMP): Assesses the model's ability to
    overcome a demonstrated bias when provided with explicit debiasing
    prompts or chain-of-thought instructions.

    Measures capacity for System 2 override.
    """

    bias_id: str
    baseline_bias_score: float
    debiased_scores: dict[str, float]  # debiasing_method -> score
    best_mitigation_method: str
    mitigation_effectiveness: float  # 0-1, how much bias was reduced
    requires_explicit_warning: bool

    @classmethod
    def calculate(
        cls,
        bias_id: str,
        treatment_results: list[TestResult],
        debiasing_results: dict[str, list[TestResult]],
        scorer: Callable,
    ) -> "BiasMitigationPotential":
        """
        Calculate BMP from treatment and debiasing results.

        Args:
            bias_id: The bias being measured
            treatment_results: Results without debiasing
            debiasing_results: Results with various debiasing methods
            scorer: Function to score bias presence

        Returns:
            BiasMitigationPotential instance
        """
        # Baseline bias level
        baseline_scores = [scorer(r) for r in treatment_results]
        baseline = mean(baseline_scores) if baseline_scores else 0.5

        # Score each debiasing method
        debiased_scores = {}
        for method, results in debiasing_results.items():
            scores = [scorer(r) for r in results]
            debiased_scores[method] = mean(scores) if scores else baseline

        if not debiased_scores:
            return cls(
                bias_id=bias_id,
                baseline_bias_score=baseline,
                debiased_scores={},
                best_mitigation_method="none",
                mitigation_effectiveness=0.0,
                requires_explicit_warning=True,
            )

        # Find best method
        best_method = min(debiased_scores, key=debiased_scores.get)
        best_score = debiased_scores[best_method]

        # Calculate mitigation effectiveness
        if baseline > 0:
            effectiveness = max(0, (baseline - best_score) / baseline)
        else:
            effectiveness = 0.0

        # Check if explicit warning is required
        cot_methods = [m for m in debiased_scores if "chain" in m.lower() or "step" in m.lower()]
        warning_methods = [m for m in debiased_scores if "warn" in m.lower() or "bias" in m.lower()]

        cot_score = mean([debiased_scores[m] for m in cot_methods]) if cot_methods else baseline
        warning_score = mean([debiased_scores[m] for m in warning_methods]) if warning_methods else baseline

        requires_warning = warning_score < cot_score

        return cls(
            bias_id=bias_id,
            baseline_bias_score=baseline,
            debiased_scores=debiased_scores,
            best_mitigation_method=best_method,
            mitigation_effectiveness=effectiveness,
            requires_explicit_warning=requires_warning,
        )


# Human baseline data from meta-analyses and research literature
# Values represent typical human susceptibility rates (0-1 scale)
# None indicates insufficient research data for reliable baseline
HUMAN_BASELINES: dict[str, float | None] = {
    # Representativeness Heuristic Biases
    "base_rate_neglect": 0.68,          # Kahneman & Tversky (1973)
    "conjunction_fallacy": 0.85,         # Tversky & Kahneman (1983) - Linda problem
    "insensitivity_to_sample_size": 0.70, # Kahneman & Tversky (1972)
    "gambler_fallacy": 0.45,             # Lower than other biases
    "hot_hand_fallacy": 0.55,            # Gilovich et al. (1985)
    "regression_neglect": 0.60,          # Kahneman & Tversky (1973)
    "stereotype_bias": 0.65,             # Kahneman & Tversky (1973)
    "prototype_heuristic": 0.58,         # Rosch (1978)

    # Availability Heuristic Biases
    "availability_bias": 0.60,           # Tversky & Kahneman (1973)
    "recency_bias": 0.62,                # Extension of availability
    "salience_bias": 0.68,               # Kahneman (2011)
    "simulation_heuristic": 0.55,        # Kahneman & Tversky (1982)
    "illusory_correlation": 0.50,        # Hamilton & Gifford (1976)
    "primacy_bias": 0.58,

    # Anchoring Biases
    "anchoring_effect": 0.65,            # Tversky & Kahneman (1974)
    "insufficient_adjustment": 0.60,     # Epley & Gilovich (2006)
    "focalism": 0.55,                    # Wilson et al. (2000)
    "first_offer_anchoring": 0.70,       # Galinsky & Mussweiler (2001)
    "numeric_priming": 0.45,             # Wilson et al. (1996)

    # Prospect Theory - Loss Aversion
    "loss_aversion": 0.70,               # Kahneman & Tversky (1979)
    "endowment_effect": 0.65,            # Thaler (1980)
    "status_quo_bias": 0.62,             # Samuelson & Zeckhauser (1988)
    "sunk_cost_fallacy": 0.55,           # Arkes & Blumer (1985)
    "disposition_effect": 0.60,          # Shefrin & Statman (1985)

    # Framing Effects
    "gain_loss_framing": 0.72,           # Tversky & Kahneman (1981) - Asian Disease
    "attribute_framing": 0.58,           # Levin & Gaeth (1988)
    "reference_point_framing": 0.60,     # Kahneman & Tversky (1979)
    "default_effect": 0.75,              # Johnson & Goldstein (2003) - organ donation
    "risk_framing": 0.52,                # Gigerenzer & Hoffrage (1995)
    "temporal_framing": 0.48,

    # Probability Distortion
    "probability_weighting": 0.68,       # Kahneman & Tversky (1979)
    "certainty_effect": 0.72,            # Allais (1953)
    "possibility_effect": 0.65,          # Kahneman & Tversky (1979)
    "neglect_of_probability": 0.70,      # Sunstein (2002)
    "denominator_neglect": 0.55,         # Reyna & Brainerd (2008)
    "zero_risk_bias": 0.60,              # Baron et al. (1993)

    # Overconfidence
    "overconfidence_effect": 0.75,       # Lichtenstein et al. (1982)
    "planning_fallacy": 0.80,            # Kahneman & Tversky (1979)
    "illusion_of_control": 0.55,         # Langer (1975)
    "hindsight_bias": 0.65,              # Fischhoff (1975)
    "optimism_bias": 0.70,               # Weinstein (1980)

    # Confirmation Bias
    "confirmation_bias": 0.72,           # Wason (1960)
    "belief_perseverance": 0.65,         # Ross et al. (1975)
    "myside_bias": 0.68,                 # Stanovich et al. (2013)

    # Temporal Biases
    "present_bias": 0.70,                # Laibson (1997)
    "duration_neglect": 0.65,            # Kahneman et al. (1993)
    "peak_end_rule": 0.75,               # Kahneman et al. (1993)

    # Extension Neglect
    "scope_insensitivity": 0.78,         # Kahneman & Knetsch (1992)
    "identifiable_victim_effect": 0.72,  # Small et al. (2007)
    "group_attribution_bias": 0.55,      # Pettigrew (1979)
    "halo_effect": 0.60,                 # Thorndike (1920)

    # Attribution Biases
    "fundamental_attribution_error": 0.72,  # Ross (1977) - Jones & Harris (1967)
    "actor_observer_bias": 0.65,            # Jones & Nisbett (1971)
    "self_serving_bias": 0.75,              # Miller & Ross (1975)

    # Uncertainty Judgment
    "ambiguity_aversion": 0.68,             # Ellsberg (1961)
    "illusion_of_validity": 0.70,           # Kahneman & Tversky (1973)
    "competence_hypothesis": 0.58,          # Heath & Tversky (1991)

    # Social Biases - Extended
    "ingroup_bias": 0.70,                   # Tajfel & Turner (1979)
    "false_consensus_effect": 0.65,         # Ross et al. (1977)
    "outgroup_homogeneity_bias": 0.62,      # Quattrone & Jones (1980)

    # Additional K&T Biases
    "affect_heuristic": 0.72,               # Slovic et al. (2002)
    "mental_accounting": 0.68,              # Thaler (1985, 1999)

    # Memory Biases - Extended
    "rosy_retrospection": 0.62,             # Mitchell et al. (1997) - Vacation studies
    "source_confusion": 0.58,               # Johnson et al. (1993) - Source monitoring
    "misinformation_effect": 0.68,          # Loftus & Palmer (1974) - Eyewitness studies
    "memory_reconstruction_bias": 0.60,     # Ross (1989) - Attitude change memory

    # Attention Biases
    "attentional_bias": 0.55,               # MacLeod et al. (1986) - Stroop effects
    "inattentional_blindness": 0.65,        # Simons & Chabris (1999) - Gorilla study
    "selective_perception": 0.68,           # Hastorf & Cantril (1954) - Football study
}

# Biases where human baseline is estimated due to limited direct experimental data
# These should be treated with caution in HAS calculations
UNKNOWN_BASELINE_BIASES: set[str] = {
    "source_confusion",            # Limited quantitative studies
    "memory_reconstruction_bias",  # Qualitative rather than rate-based findings
    "attentional_bias",            # High individual variation
}


@dataclass
class HumanAlignmentScore:
    """
    Human Alignment Score (HAS): Compares the LLM's pattern of biases
    to established patterns in specific human cohorts.

    Determines whether the model's irrationality mirrors human cognitive
    shortcuts or represents uniquely artificial errors.
    """

    bias_id: str
    model_bias_rate: float
    human_baseline_rate: float
    alignment_score: float  # 0-1, how closely model matches humans
    bias_direction: str  # "over" if model more biased, "under", or "aligned"

    @classmethod
    def calculate(
        cls,
        bias_id: str,
        results: list[TestResult],
        scorer: Callable,
        human_baseline: float | None = None,
    ) -> "HumanAlignmentScore":
        """
        Calculate HAS by comparing model to human baselines.

        Args:
            bias_id: The bias being measured
            results: Model test results
            scorer: Function to score bias presence
            human_baseline: Optional override for human baseline rate

        Returns:
            HumanAlignmentScore instance
        """
        baselines = HUMAN_BASELINES

        # Get model bias rate
        scores = [scorer(r) for r in results]
        model_rate = mean(scores) if scores else 0.5

        # Get human baseline with appropriate warnings
        if human_baseline is None:
            if bias_id in baselines:
                human_rate = baselines[bias_id]
                if bias_id in UNKNOWN_BASELINE_BIASES:
                    logger.warning(
                        f"Using estimated baseline for '{bias_id}' - "
                        f"limited research data available. HAS results should be "
                        f"interpreted with caution."
                    )
            else:
                human_rate = 0.5
                logger.warning(
                    f"No human baseline for '{bias_id}' - using default 0.5. "
                    f"HAS results for this bias should be interpreted with caution."
                )
        else:
            human_rate = human_baseline

        # Calculate alignment (1 - normalized difference)
        max_possible_diff = max(human_rate, 1 - human_rate)
        if max_possible_diff > 0:
            alignment = 1 - abs(model_rate - human_rate) / max_possible_diff
        else:
            alignment = 1.0

        # Determine direction
        diff = model_rate - human_rate
        if abs(diff) < 0.1:
            direction = "aligned"
        elif diff > 0:
            direction = "over"
        else:
            direction = "under"

        return cls(
            bias_id=bias_id,
            model_bias_rate=model_rate,
            human_baseline_rate=human_rate,
            alignment_score=alignment,
            bias_direction=direction,
        )


@dataclass
class ResponseConsistencyIndex:
    """
    Response Consistency Index (RCI): Measures the variance in model
    responses across multiple identical trials of the same test case.

    Distinguishes systematic cognitive patterns from stochastic noise.
    A model showing 50% bias susceptibility could be highly inconsistent
    rather than consistently biased.
    """

    bias_id: str
    mean_response: float
    variance: float
    consistency_score: float  # 0-1, higher = more consistent
    is_stable: bool  # True if variance below threshold
    trial_count: int

    @classmethod
    def calculate(
        cls,
        bias_id: str,
        trial_results: list[TestResult],
        scorer: Callable,
    ) -> "ResponseConsistencyIndex":
        """
        Calculate RCI from multiple trials.

        Args:
            bias_id: The bias being measured
            trial_results: Results from multiple identical trials
            scorer: Function to score each response

        Returns:
            ResponseConsistencyIndex instance
        """
        scores = [scorer(r) for r in trial_results]

        if not scores:
            return cls(
                bias_id=bias_id,
                mean_response=0.0,
                variance=0.0,
                consistency_score=1.0,
                is_stable=True,
                trial_count=0,
            )

        mean_score = mean(scores)
        variance = stdev(scores) ** 2 if len(scores) > 1 else 0.0

        # Consistency score: inverse of normalized variance
        # Max variance for binary is 0.25 (at mean 0.5)
        max_variance = mean_score * (1 - mean_score) if 0 < mean_score < 1 else 0.25
        if max_variance > 0:
            normalized_variance = min(variance / max_variance, 1.0)
            consistency = 1 - normalized_variance
        else:
            consistency = 1.0

        # Stable if variance is below 10% of max possible
        is_stable = variance < 0.025

        return cls(
            bias_id=bias_id,
            mean_response=mean_score,
            variance=variance,
            consistency_score=consistency,
            is_stable=is_stable,
            trial_count=len(scores),
        )


@dataclass
class CalibrationAwarenessScore:
    """
    Calibration Awareness Score (CAS): Measures a model's confidence calibration
    in the context of bias-inducing prompts.

    This metric assesses whether the model's stated confidence aligns with its
    actual accuracy (matching the expected rational response). A well-calibrated
    model should have lower confidence when giving biased answers and higher
    confidence when giving rational answers.

    Note: This measures CALIBRATION (confidence vs accuracy), not explicit
    bias recognition. A model can be well-calibrated yet unaware it's being
    influenced by bias triggers. For explicit bias awareness testing, use
    meta-scale debiasing prompts that ask the model to identify bias triggers.

    Attributes:
        mean_confidence: Average stated confidence across responses
        actual_accuracy: Rate of matching expected rational response
        calibration_error: |confidence - accuracy|, lower is better
        calibration_score: 1 - calibration_error, higher is better calibrated
        overconfident: True if confidence exceeds accuracy by >10%
        metacognitive_gap: How much confidence exceeds accuracy (0 if not overconfident)
    """

    bias_id: str
    mean_confidence: float
    actual_accuracy: float
    calibration_error: float  # |confidence - accuracy|
    calibration_score: float  # Renamed from awareness_score for clarity
    overconfident: bool
    metacognitive_gap: float  # How much confidence exceeds accuracy

    @classmethod
    def calculate(
        cls,
        bias_id: str,
        results: list[TestResult],
        accuracy_scorer: Callable,
    ) -> "CalibrationAwarenessScore":
        """
        Calculate CAS from results with stated confidence.

        Args:
            bias_id: The bias being measured
            results: Results that include confidence_stated
            accuracy_scorer: Function to score correctness (0-1)

        Returns:
            CalibrationAwarenessScore instance
        """
        # Filter results with confidence
        results_with_conf = [r for r in results if r.confidence_stated is not None]

        if not results_with_conf:
            return cls(
                bias_id=bias_id,
                mean_confidence=0.5,
                actual_accuracy=0.5,
                calibration_error=0.0,
                calibration_score=0.5,
                overconfident=False,
                metacognitive_gap=0.0,
            )

        confidences = [r.confidence_stated for r in results_with_conf]
        accuracies = [accuracy_scorer(r) for r in results_with_conf]

        mean_conf = mean(confidences)
        mean_acc = mean(accuracies)

        calibration_error = abs(mean_conf - mean_acc)

        # Calibration score: 1 - calibration error (higher = better calibrated)
        calibration = 1 - min(calibration_error, 1.0)

        # Overconfident if confidence > accuracy + 0.1
        overconfident = mean_conf > mean_acc + 0.1

        # Metacognitive gap
        gap = max(0, mean_conf - mean_acc)

        return cls(
            bias_id=bias_id,
            mean_confidence=mean_conf,
            actual_accuracy=mean_acc,
            calibration_error=calibration_error,
            calibration_score=calibration,
            overconfident=overconfident,
            metacognitive_gap=gap,
        )


@dataclass
class CognitiveFingerprintReport:
    """
    Complete cognitive fingerprint for a model across all biases tested.

    Aggregates all six metrics to provide a comprehensive profile.
    """

    model_id: str
    biases_tested: list[str]
    magnitude_scores: dict[str, BiasMagnitudeScore]
    consistency_indices: dict[str, BiasConsistencyIndex]
    mitigation_potentials: dict[str, BiasMitigationPotential]
    human_alignments: dict[str, HumanAlignmentScore]
    response_consistencies: dict[str, ResponseConsistencyIndex]
    calibration_scores: dict[str, CalibrationAwarenessScore]

    # Aggregate summary
    overall_bias_susceptibility: float = 0.0
    most_susceptible_biases: list[str] = field(default_factory=list)
    most_resistant_biases: list[str] = field(default_factory=list)
    human_like_biases: list[str] = field(default_factory=list)
    ai_specific_biases: list[str] = field(default_factory=list)

    def compute_summary(self) -> None:
        """Compute aggregate summary statistics."""
        if not self.magnitude_scores:
            return

        # Overall susceptibility
        magnitudes = [m.overall_magnitude for m in self.magnitude_scores.values()]
        self.overall_bias_susceptibility = mean(magnitudes) if magnitudes else 0.0

        # Sort biases by magnitude
        sorted_biases = sorted(
            self.magnitude_scores.items(),
            key=lambda x: x[1].overall_magnitude,
            reverse=True,
        )

        # Top 5 most/least susceptible
        self.most_susceptible_biases = [b[0] for b in sorted_biases[:5]]
        self.most_resistant_biases = [b[0] for b in sorted_biases[-5:]]

        # Human-like vs AI-specific
        for bias_id, has in self.human_alignments.items():
            if has.alignment_score > 0.8:
                self.human_like_biases.append(bias_id)
            elif has.bias_direction == "over":
                self.ai_specific_biases.append(bias_id)


class MetricCalculator:
    """
    Coordinator for calculating all metrics from evaluation results.
    """

    def __init__(self, scorer: Callable | None = None):
        """
        Initialize calculator.

        Args:
            scorer: Default scoring function for results
        """
        self.scorer = scorer or self._default_scorer

    def _default_scorer(self, result: TestResult) -> float:
        """Default scorer that uses is_biased flag or bias_score."""
        if result.bias_score is not None:
            return result.bias_score
        elif result.is_biased is not None:
            return 1.0 if result.is_biased else 0.0
        else:
            return 0.5  # Unknown

    def _accuracy_scorer(self, result: TestResult) -> float:
        """
        Score accuracy based on whether answer matches expected rational response.

        Unlike bias_score (which measures deviation toward biased answer),
        accuracy measures whether the model gave the objectively correct answer.
        """
        if not result.extracted_answer or result.extracted_answer == "UNKNOWN":
            return 0.5  # Cannot determine accuracy

        expected = result.instance.expected_rational_response.lower().strip()
        extracted = result.extracted_answer.lower().strip()

        # Handle placeholder expected answers
        if expected.startswith("["):
            return 0.5  # Cannot determine accuracy for placeholder answers

        # Exact match check
        if extracted == expected:
            return 1.0

        # Check if extracted answer is contained in expected (for longer text)
        if extracted in expected or expected in extracted:
            return 0.8

        # For numeric answers, try approximate match
        try:
            expected_num = float(expected.replace(",", "").replace("$", ""))
            extracted_num = float(extracted.replace(",", "").replace("$", ""))
            # Within 10% is considered accurate
            if abs(expected_num - extracted_num) / max(abs(expected_num), 0.001) < 0.1:
                return 1.0
            # Within 25% is partially accurate
            if abs(expected_num - extracted_num) / max(abs(expected_num), 0.001) < 0.25:
                return 0.7
        except (ValueError, AttributeError):
            pass

        return 0.0  # Not accurate

    def calculate_all_metrics(
        self,
        model_id: str,
        results: list[TestResult],
    ) -> CognitiveFingerprintReport:
        """
        Calculate complete cognitive fingerprint from results.

        Args:
            model_id: The model being evaluated
            results: All test results

        Returns:
            Complete CognitiveFingerprintReport
        """
        # Group results by bias
        by_bias: dict[str, list[TestResult]] = defaultdict(list)
        for r in results:
            by_bias[r.instance.bias_id].append(r)

        magnitude_scores = {}
        consistency_indices = {}
        mitigation_potentials = {}
        human_alignments = {}
        response_consistencies = {}
        calibration_scores = {}

        for bias_id, bias_results in by_bias.items():
            # Separate results by condition
            control = [r for r in bias_results if r.condition == "control"]
            treatments = defaultdict(list)
            debiasing = defaultdict(list)

            for r in bias_results:
                if r.condition.startswith("treatment_"):
                    intensity_str = r.condition.replace("treatment_", "")
                    try:
                        intensity = TriggerIntensity(intensity_str)
                        treatments[intensity].append(r)
                    except ValueError:
                        pass
                elif r.condition.startswith("debiasing_"):
                    debiasing[r.condition].append(r)

            # Group by domain for consistency
            by_domain: dict[Domain, list[TestResult]] = defaultdict(list)
            for r in bias_results:
                by_domain[r.instance.domain].append(r)

            # Calculate each metric
            magnitude_scores[bias_id] = BiasMagnitudeScore.calculate(
                bias_id, control, dict(treatments), self.scorer
            )

            consistency_indices[bias_id] = BiasConsistencyIndex.calculate(
                bias_id, dict(by_domain), self.scorer
            )

            all_treatments = [r for rs in treatments.values() for r in rs]
            mitigation_potentials[bias_id] = BiasMitigationPotential.calculate(
                bias_id, all_treatments, dict(debiasing), self.scorer
            )

            human_alignments[bias_id] = HumanAlignmentScore.calculate(
                bias_id, all_treatments, self.scorer
            )

            # RCI should measure variance across identical trials (same condition)
            # Group by condition and calculate RCI for each, then aggregate
            by_condition: dict[str, list[TestResult]] = defaultdict(list)
            for r in bias_results:
                by_condition[r.condition].append(r)

            condition_rcis = []
            for cond, cond_results in by_condition.items():
                if len(cond_results) >= 2:  # Need at least 2 trials for variance
                    cond_rci = ResponseConsistencyIndex.calculate(
                        f"{bias_id}_{cond}", cond_results, self.scorer
                    )
                    condition_rcis.append(cond_rci)

            # Aggregate RCI across conditions (average)
            if condition_rcis:
                avg_consistency = mean([rci.consistency_score for rci in condition_rcis])
                avg_variance = mean([rci.variance for rci in condition_rcis])
                total_trials = sum(rci.trial_count for rci in condition_rcis)
                all_stable = all(rci.is_stable for rci in condition_rcis)
                avg_mean = mean([rci.mean_response for rci in condition_rcis])

                response_consistencies[bias_id] = ResponseConsistencyIndex(
                    bias_id=bias_id,
                    mean_response=avg_mean,
                    variance=avg_variance,
                    consistency_score=avg_consistency,
                    is_stable=all_stable,
                    trial_count=total_trials,
                )
            else:
                # Fallback for single-trial cases
                response_consistencies[bias_id] = ResponseConsistencyIndex.calculate(
                    bias_id, bias_results, self.scorer
                )

            calibration_scores[bias_id] = CalibrationAwarenessScore.calculate(
                bias_id, bias_results, self._accuracy_scorer  # True accuracy based on rational answer match
            )

        report = CognitiveFingerprintReport(
            model_id=model_id,
            biases_tested=list(by_bias.keys()),
            magnitude_scores=magnitude_scores,
            consistency_indices=consistency_indices,
            mitigation_potentials=mitigation_potentials,
            human_alignments=human_alignments,
            response_consistencies=response_consistencies,
            calibration_scores=calibration_scores,
        )
        report.compute_summary()

        return report
