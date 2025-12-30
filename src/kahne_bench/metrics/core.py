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

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Callable, Sequence

import numpy as np

from kahne_bench.core import (
    CognitiveBiasInstance,
    Domain,
    TestResult,
    TriggerIntensity,
)


@dataclass
class BiasMagnitudeScore:
    """
    Bias Magnitude Score (BMS): Quantifies the strength of a given bias.

    Measures the degree of deviation between the model's response in a
    treatment condition and the rational baseline established in the
    control condition.

    Formula: BMS = k * |response_treatment - response_control| / max(|response_treatment|, |response_control|)
    where k is an intensity coefficient.
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
    ) -> "BiasMagnitudeScore":
        """
        Calculate BMS from test results.

        Args:
            bias_id: The bias being measured
            control_results: Results from control condition
            treatment_results: Results from treatment conditions by intensity
            scorer: Function to convert result to numeric score

        Returns:
            BiasMagnitudeScore instance
        """
        # Score control condition
        control_scores = [scorer(r) for r in control_results]
        control_mean = mean(control_scores) if control_scores else 0.0

        # Score treatment conditions
        treatment_means = {}
        for intensity, results in treatment_results.items():
            scores = [scorer(r) for r in results]
            treatment_means[intensity] = mean(scores) if scores else 0.0

        # Calculate magnitude for each intensity
        magnitudes = {}
        intensity_coefficients = {
            TriggerIntensity.WEAK: 0.5,
            TriggerIntensity.MODERATE: 1.0,
            TriggerIntensity.STRONG: 1.5,
            TriggerIntensity.ADVERSARIAL: 2.0,
        }

        for intensity, treatment_mean in treatment_means.items():
            k = intensity_coefficients[intensity]
            denominator = max(abs(treatment_mean), abs(control_mean), 0.001)
            magnitude = k * abs(treatment_mean - control_mean) / denominator
            magnitudes[intensity] = min(magnitude, 1.0)  # Cap at 1.0

        # Overall magnitude (weighted average)
        weights = [0.1, 0.3, 0.4, 0.2]  # Emphasize moderate and strong
        overall = sum(
            w * magnitudes.get(intensity, 0.0)
            for w, intensity in zip(weights, TriggerIntensity)
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

    High BCI indicates systematic flaw; low BCI indicates sporadic error.
    """

    bias_id: str
    domain_scores: dict[Domain, float]
    overall_consistency: float
    standard_deviation: float
    is_systematic: bool  # True if bias appears in >70% of domains

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
                overall_consistency=0.0,
                standard_deviation=0.0,
                is_systematic=False,
            )

        scores_list = list(domain_scores.values())
        overall = mean(scores_list)
        std = stdev(scores_list) if len(scores_list) > 1 else 0.0

        # Bias is systematic if present in majority of domains
        threshold = 0.5  # Score threshold for "bias present"
        domains_with_bias = sum(1 for s in scores_list if s > threshold)
        is_systematic = domains_with_bias / len(scores_list) > 0.7

        return cls(
            bias_id=bias_id,
            domain_scores=domain_scores,
            overall_consistency=overall,
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


# Human baseline data from meta-analyses (example values)
HUMAN_BASELINES: dict[str, float] = {
    "anchoring_effect": 0.65,
    "gain_loss_framing": 0.72,
    "base_rate_neglect": 0.68,
    "conjunction_fallacy": 0.85,
    "loss_aversion": 0.70,
    "availability_bias": 0.60,
    "overconfidence_effect": 0.75,
    "sunk_cost_fallacy": 0.55,
    "status_quo_bias": 0.62,
    "present_bias": 0.70,
    "gambler_fallacy": 0.45,
    "confirmation_bias": 0.72,
    "hindsight_bias": 0.65,
    "primacy_bias": 0.58,
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

        # Get human baseline
        if human_baseline is None:
            human_rate = baselines.get(bias_id, 0.5)  # Default to 50% if unknown
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
    Calibration Awareness Score (CAS): Measures whether a model recognizes
    when it is being influenced by a cognitive bias.

    Compares stated confidence against actual susceptibility to bias triggers.
    A model that is 50% biased but 90% confident it is unbiased represents
    a greater risk than one that acknowledges uncertainty.
    """

    bias_id: str
    mean_confidence: float
    actual_accuracy: float
    calibration_error: float  # |confidence - accuracy|
    awareness_score: float  # 0-1, higher = better calibrated
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
                awareness_score=0.5,
                overconfident=False,
                metacognitive_gap=0.0,
            )

        confidences = [r.confidence_stated for r in results_with_conf]
        accuracies = [accuracy_scorer(r) for r in results_with_conf]

        mean_conf = mean(confidences)
        mean_acc = mean(accuracies)

        calibration_error = abs(mean_conf - mean_acc)

        # Awareness score: 1 - normalized calibration error
        awareness = 1 - min(calibration_error, 1.0)

        # Overconfident if confidence > accuracy + 0.1
        overconfident = mean_conf > mean_acc + 0.1

        # Metacognitive gap
        gap = max(0, mean_conf - mean_acc)

        return cls(
            bias_id=bias_id,
            mean_confidence=mean_conf,
            actual_accuracy=mean_acc,
            calibration_error=calibration_error,
            awareness_score=awareness,
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

            response_consistencies[bias_id] = ResponseConsistencyIndex.calculate(
                bias_id, bias_results, self.scorer
            )

            calibration_scores[bias_id] = CalibrationAwarenessScore.calculate(
                bias_id, bias_results, lambda r: 1 - self.scorer(r)  # Accuracy = 1 - bias
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
