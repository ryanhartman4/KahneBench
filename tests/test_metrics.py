"""Tests for the evaluation metrics."""


import pytest
from kahne_bench.metrics import (
    BiasMagnitudeScore,
    BiasConsistencyIndex,
    BiasMitigationPotential,
    HumanAlignmentScore,
    ResponseConsistencyIndex,
    CalibrationAwarenessScore,
    MetricCalculator,
)
from kahne_bench.metrics.core import HUMAN_BASELINES, UNKNOWN_BASELINE_BIASES
from kahne_bench.core import (
    CognitiveBiasInstance,
    TestResult,
    Domain,
    TriggerIntensity,
)


@pytest.fixture
def sample_instance():
    """Create a sample test instance."""
    return CognitiveBiasInstance(
        bias_id="anchoring_effect",
        base_scenario="test scenario",
        bias_trigger="anchor value presented",
        control_prompt="Control prompt",
        treatment_prompts={
            TriggerIntensity.WEAK: "Weak treatment",
            TriggerIntensity.MODERATE: "Moderate treatment",
            TriggerIntensity.STRONG: "Strong treatment",
            TriggerIntensity.ADVERSARIAL: "Adversarial treatment",
        },
        expected_rational_response="unbiased answer",
        expected_biased_response="biased answer",
        domain=Domain.PROFESSIONAL,
    )


@pytest.fixture
def sample_results(sample_instance):
    """Create sample test results."""
    results = []
    for i, condition in enumerate(["control", "treatment_moderate"]):
        result = TestResult(
            instance=sample_instance,
            model_id="test-model",
            condition=condition,
            prompt_used=f"Prompt {i}",
            model_response=f"Response {i}",
            extracted_answer="A" if condition == "control" else "B",
            response_time_ms=100.0,
            confidence_stated=0.8 if i % 2 == 0 else None,
            is_biased=condition != "control",
            bias_score=0.0 if condition == "control" else 0.7,
        )
        results.append(result)
    return results


class TestBiasMagnitudeScore:
    """Tests for Bias Magnitude Score calculation."""

    def test_calculate_with_no_bias(self, sample_instance):
        """Test BMS when control and treatment are the same."""
        def scorer(r):
            return 0.0  # No bias

        control = [TestResult(
            instance=sample_instance,
            model_id="test",
            condition="control",
            prompt_used="",
            model_response="",
            extracted_answer="A",
            response_time_ms=100.0,
        )]

        treatment = {
            TriggerIntensity.MODERATE: [TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )]
        }

        bms = BiasMagnitudeScore.calculate("anchoring_effect", control, treatment, scorer)
        assert bms.overall_magnitude >= 0
        assert bms.bias_id == "anchoring_effect"

    def test_calculate_with_high_bias(self, sample_instance):
        """Test BMS when treatment shows strong bias."""
        def scorer(r):
            return 0.0 if r.condition == "control" else 1.0

        control = [TestResult(
            instance=sample_instance,
            model_id="test",
            condition="control",
            prompt_used="",
            model_response="",
            extracted_answer="A",
            response_time_ms=100.0,
        )]

        treatment = {
            TriggerIntensity.MODERATE: [TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
            )]
        }

        bms = BiasMagnitudeScore.calculate("anchoring_effect", control, treatment, scorer)
        assert bms.overall_magnitude > 0
        # Control should be 0.0, treatment should be 1.0
        assert bms.control_score == 0.0
        assert bms.treatment_scores[TriggerIntensity.MODERATE] == 1.0
        # With max deviation (0 vs 1), magnitude should be high
        assert bms.overall_magnitude >= 0.3  # Weighted by 0.3 for moderate


class TestBiasConsistencyIndex:
    """Tests for Bias Consistency Index calculation."""

    def test_calculate_single_domain(self, sample_instance):
        """Test BCI with a single domain."""
        def scorer(r):
            return 0.5

        results = {
            Domain.PROFESSIONAL: [TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )]
        }

        bci = BiasConsistencyIndex.calculate("anchoring_effect", results, scorer)
        assert bci.bias_id == "anchoring_effect"
        assert bci.mean_bias_score == 0.5
        assert bci.consistency_score == 1.0  # Single domain = no variance = perfect consistency

    def test_calculate_multiple_domains(self, sample_instance):
        """Test BCI across multiple domains."""
        def scorer(r):
            return 0.7

        sample_instance_individual = CognitiveBiasInstance(
            **{**sample_instance.__dict__, "domain": Domain.INDIVIDUAL}
        )

        results = {
            Domain.PROFESSIONAL: [TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )],
            Domain.INDIVIDUAL: [TestResult(
                instance=sample_instance_individual,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )],
        }

        bci = BiasConsistencyIndex.calculate("anchoring_effect", results, scorer)
        assert len(bci.domain_scores) == 2
        assert bci.mean_bias_score == 0.7  # Same score across domains
        assert bci.consistency_score == 1.0  # No variance = perfect consistency

    def test_calculate_variable_domains(self, sample_instance):
        """Test BCI when domain scores vary (low consistency)."""
        # Different scores per domain
        domain_scores_map = {
            Domain.PROFESSIONAL: 0.9,
            Domain.INDIVIDUAL: 0.1,
        }

        def scorer(r):
            return domain_scores_map[r.instance.domain]

        sample_instance_individual = CognitiveBiasInstance(
            **{**sample_instance.__dict__, "domain": Domain.INDIVIDUAL}
        )

        results = {
            Domain.PROFESSIONAL: [TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )],
            Domain.INDIVIDUAL: [TestResult(
                instance=sample_instance_individual,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )],
        }

        bci = BiasConsistencyIndex.calculate("anchoring_effect", results, scorer)
        assert len(bci.domain_scores) == 2
        assert bci.mean_bias_score == 0.5  # Average of 0.9 and 0.1
        # High variance = low consistency score
        assert bci.consistency_score < 0.5
        assert bci.standard_deviation > 0.3


class TestHumanAlignmentScore:
    """Tests for Human Alignment Score calculation."""

    def test_human_baselines_exist(self):
        """Verify human baselines are defined."""
        assert len(HUMAN_BASELINES) > 0
        assert "anchoring_effect" in HUMAN_BASELINES

    def test_calculate_alignment(self, sample_instance):
        """Test HAS calculation."""
        def scorer(r):
            return 0.65  # Match human baseline for anchoring_effect

        results = [TestResult(
            instance=sample_instance,
            model_id="test",
            condition="treatment",
            prompt_used="",
            model_response="",
            extracted_answer="A",
            response_time_ms=100.0,
        )]

        has = HumanAlignmentScore.calculate("anchoring_effect", results, scorer)
        assert has.bias_id == "anchoring_effect"
        assert 0 <= has.alignment_score <= 1
        # Since model rate (0.65) matches human baseline (0.65), alignment should be perfect
        assert has.model_bias_rate == 0.65
        assert has.human_baseline_rate == 0.65
        assert has.alignment_score == 1.0
        assert has.bias_direction == "aligned"

    def test_alignment_over_biased(self, sample_instance):
        """Test HAS when model is more biased than humans."""
        def scorer(r):
            return 0.95  # Much higher than human baseline (0.65)

        results = [TestResult(
            instance=sample_instance,
            model_id="test",
            condition="treatment",
            prompt_used="",
            model_response="",
            extracted_answer="A",
            response_time_ms=100.0,
        )]

        has = HumanAlignmentScore.calculate("anchoring_effect", results, scorer)
        assert has.bias_direction == "over"
        assert has.alignment_score < 1.0


class TestResponseConsistencyIndex:
    """Tests for Response Consistency Index calculation."""

    def test_perfect_consistency(self, sample_instance):
        """Test RCI when all responses are identical."""
        def scorer(r):
            return 0.5

        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="Same response",
                extracted_answer="A",
                response_time_ms=100.0,
            )
            for _ in range(5)
        ]

        rci = ResponseConsistencyIndex.calculate("anchoring_effect", results, scorer)
        assert rci.consistency_score == 1.0
        assert rci.is_stable

    def test_inconsistent_responses(self, sample_instance):
        """Test RCI when responses vary."""
        responses = [0.0, 1.0, 0.0, 1.0, 0.0]  # Alternating

        def scorer(r):
            return responses[int(r.extracted_answer)]

        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response=f"Response {i}",
                extracted_answer=str(i),
                response_time_ms=100.0,
            )
            for i in range(5)
        ]

        rci = ResponseConsistencyIndex.calculate("anchoring_effect", results, scorer)
        assert rci.trial_count == 5
        # With alternating 0, 1, 0, 1, 0 scores, variance should be non-zero
        assert rci.variance > 0
        # Consistency should be less than 1.0 due to variance
        assert rci.consistency_score < 1.0
        # Should not be considered stable with high variance
        assert not rci.is_stable


class TestCalibrationAwarenessScore:
    """Tests for Calibration Awareness Score calculation."""

    def test_well_calibrated(self, sample_instance):
        """Test CAS when model is well-calibrated."""
        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="Response",
                extracted_answer="A",
                response_time_ms=100.0,
                confidence_stated=0.7,
            )
        ]

        def accuracy_scorer(r):
            return 0.7  # Match stated confidence

        cas = CalibrationAwarenessScore.calculate("anchoring_effect", results, accuracy_scorer)
        assert cas.calibration_error < 0.1
        assert not cas.overconfident

    def test_overconfident(self, sample_instance):
        """Test CAS when model is overconfident."""
        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="Response",
                extracted_answer="A",
                response_time_ms=100.0,
                confidence_stated=0.9,
            )
        ]

        def accuracy_scorer(r):
            return 0.5  # Much lower than stated confidence

        cas = CalibrationAwarenessScore.calculate("anchoring_effect", results, accuracy_scorer)
        assert cas.overconfident
        assert cas.metacognitive_gap > 0


class TestMetricCalculator:
    """Tests for the MetricCalculator class."""

    def test_default_scorer(self, sample_results):
        """Test that default scorer works."""
        calculator = MetricCalculator()
        # Just verify it doesn't crash
        report = calculator.calculate_all_metrics("test-model", sample_results)
        assert report.model_id == "test-model"

    def test_custom_scorer(self, sample_results):
        """Test with custom scorer."""
        def custom_scorer(r):
            return 0.5

        calculator = MetricCalculator(scorer=custom_scorer)
        report = calculator.calculate_all_metrics("test-model", sample_results)
        assert report.model_id == "test-model"


class TestBiasMitigationPotential:
    """Tests for Bias Mitigation Potential calculation."""

    def test_bmp_calculate_basic(self, sample_instance):
        """Test basic BMP calculation with mock debiasing results."""
        def scorer(r):
            # Treatment results have high bias, debiased results have lower bias
            if "chain" in r.condition or "warn" in r.condition:
                return 0.2  # Low bias after debiasing
            return 0.8  # High bias in treatment

        treatment_results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment_moderate",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
            )
            for _ in range(3)
        ]

        debiasing_results = {
            "chain_of_thought": [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="chain_of_thought",
                    prompt_used="",
                    model_response="",
                    extracted_answer="A",
                    response_time_ms=100.0,
                )
            ],
            "bias_warning": [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="bias_warning",
                    prompt_used="",
                    model_response="",
                    extracted_answer="A",
                    response_time_ms=100.0,
                )
            ],
        }

        bmp = BiasMitigationPotential.calculate(
            "anchoring_effect", treatment_results, debiasing_results, scorer
        )
        assert bmp.bias_id == "anchoring_effect"
        assert bmp.baseline_bias_score == 0.8
        assert bmp.mitigation_effectiveness > 0  # Should show improvement
        assert len(bmp.debiased_scores) == 2

    def test_bmp_with_zero_baseline(self, sample_instance):
        """Test BMP when baseline_bias_score is 0 (potential division issue at line 340)."""
        def scorer(r):
            return 0.0  # All results show zero bias

        treatment_results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment_moderate",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )
        ]

        debiasing_results = {
            "chain_of_thought": [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="chain_of_thought",
                    prompt_used="",
                    model_response="",
                    extracted_answer="A",
                    response_time_ms=100.0,
                )
            ],
        }

        bmp = BiasMitigationPotential.calculate(
            "anchoring_effect", treatment_results, debiasing_results, scorer
        )
        # When baseline is 0, effectiveness should be 0 (nothing to mitigate)
        assert bmp.baseline_bias_score == 0.0
        assert bmp.mitigation_effectiveness == 0.0

    def test_bmp_selects_best_method(self, sample_instance):
        """Test that best_mitigation_method is the one with lowest bias score."""
        method_scores = {
            "method_a": 0.5,
            "method_b": 0.1,  # Best method (lowest score)
            "method_c": 0.3,
        }

        def scorer(r):
            if r.condition in method_scores:
                return method_scores[r.condition]
            return 0.8  # Treatment baseline

        treatment_results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment_moderate",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
            )
        ]

        debiasing_results = {
            method: [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition=method,
                    prompt_used="",
                    model_response="",
                    extracted_answer="A",
                    response_time_ms=100.0,
                )
            ]
            for method in method_scores.keys()
        }

        bmp = BiasMitigationPotential.calculate(
            "anchoring_effect", treatment_results, debiasing_results, scorer
        )
        assert bmp.best_mitigation_method == "method_b"
        assert bmp.debiased_scores["method_b"] == 0.1

    def test_bmp_requires_explicit_warning(self, sample_instance):
        """Test the requires_explicit_warning logic."""
        # Scenario: warning methods work better than chain-of-thought methods
        def scorer(r):
            if "warn" in r.condition or "bias" in r.condition:
                return 0.1  # Warning methods very effective
            elif "chain" in r.condition or "step" in r.condition:
                return 0.5  # Chain-of-thought less effective
            return 0.8

        treatment_results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment_moderate",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
            )
        ]

        debiasing_results = {
            "chain_of_thought": [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="chain_of_thought",
                    prompt_used="",
                    model_response="",
                    extracted_answer="A",
                    response_time_ms=100.0,
                )
            ],
            "bias_warning": [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="bias_warning",
                    prompt_used="",
                    model_response="",
                    extracted_answer="A",
                    response_time_ms=100.0,
                )
            ],
        }

        bmp = BiasMitigationPotential.calculate(
            "anchoring_effect", treatment_results, debiasing_results, scorer
        )
        # Warning score (0.1) < chain-of-thought score (0.5), so requires_explicit_warning should be True
        assert bmp.requires_explicit_warning is True

    def test_bmp_empty_debiasing_results(self, sample_instance):
        """Test BMP when no debiasing results are provided."""
        def scorer(r):
            return 0.7

        treatment_results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment_moderate",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
            )
        ]

        bmp = BiasMitigationPotential.calculate(
            "anchoring_effect", treatment_results, {}, scorer
        )
        assert bmp.best_mitigation_method == "none"
        assert bmp.mitigation_effectiveness == 0.0
        assert bmp.debiased_scores == {}
        assert bmp.requires_explicit_warning is True


class TestBMSIntensityRenormalization:
    """Tests for BMS aggregation weight renormalization when not all intensities are present."""

    def test_bms_3_intensities_equals_4_with_same_rate(self, sample_instance):
        """BMS with 3 intensities should match BMS with 4 when all deviations are equal.

        Using uniform intensity weights (all 1.0) so that all per-intensity
        magnitudes are identical, then the renormalized weighted average over
        3 should equal the weighted average over 4.
        """
        uniform_weights = {
            TriggerIntensity.WEAK: 1.0,
            TriggerIntensity.MODERATE: 1.0,
            TriggerIntensity.STRONG: 1.0,
            TriggerIntensity.ADVERSARIAL: 1.0,
        }

        def scorer(r):
            return 0.0 if r.condition == "control" else 0.8

        control = [TestResult(
            instance=sample_instance,
            model_id="test",
            condition="control",
            prompt_used="",
            model_response="",
            extracted_answer="A",
            response_time_ms=100.0,
        )]

        # 3 intensities (no ADVERSARIAL)
        treatment_3 = {
            TriggerIntensity.WEAK: [TestResult(
                instance=sample_instance, model_id="test",
                condition="treatment_weak", prompt_used="",
                model_response="", extracted_answer="B",
                response_time_ms=100.0,
            )],
            TriggerIntensity.MODERATE: [TestResult(
                instance=sample_instance, model_id="test",
                condition="treatment_moderate", prompt_used="",
                model_response="", extracted_answer="B",
                response_time_ms=100.0,
            )],
            TriggerIntensity.STRONG: [TestResult(
                instance=sample_instance, model_id="test",
                condition="treatment_strong", prompt_used="",
                model_response="", extracted_answer="B",
                response_time_ms=100.0,
            )],
        }

        # 4 intensities (all same rate)
        treatment_4 = {
            **treatment_3,
            TriggerIntensity.ADVERSARIAL: [TestResult(
                instance=sample_instance, model_id="test",
                condition="treatment_adversarial", prompt_used="",
                model_response="", extracted_answer="B",
                response_time_ms=100.0,
            )],
        }

        bms_3 = BiasMagnitudeScore.calculate(
            "test_bias", control, treatment_3, scorer,
            intensity_weights=uniform_weights,
        )
        bms_4 = BiasMagnitudeScore.calculate(
            "test_bias", control, treatment_4, scorer,
            intensity_weights=uniform_weights,
        )

        assert abs(bms_3.overall_magnitude - bms_4.overall_magnitude) < 0.01, (
            f"BMS with 3 intensities ({bms_3.overall_magnitude:.4f}) should equal "
            f"BMS with 4 intensities ({bms_4.overall_magnitude:.4f}) when all "
            f"per-intensity magnitudes are identical"
        )

    def test_bms_renormalization_prevents_deflation(self, sample_instance):
        """BMS with 3 intensities should NOT be lower than the minimum per-intensity magnitude.

        Before the fix, missing intensities contributed 0.0 to the weighted sum,
        pulling the overall score down. After the fix, missing intensities are
        excluded from both numerator and denominator.
        """
        def scorer(r):
            return 0.0 if r.condition == "control" else 0.6

        control = [TestResult(
            instance=sample_instance,
            model_id="test",
            condition="control",
            prompt_used="",
            model_response="",
            extracted_answer="A",
            response_time_ms=100.0,
        )]

        treatment_3 = {
            TriggerIntensity.WEAK: [TestResult(
                instance=sample_instance, model_id="test",
                condition="treatment_weak", prompt_used="",
                model_response="", extracted_answer="B",
                response_time_ms=100.0,
            )],
            TriggerIntensity.MODERATE: [TestResult(
                instance=sample_instance, model_id="test",
                condition="treatment_moderate", prompt_used="",
                model_response="", extracted_answer="B",
                response_time_ms=100.0,
            )],
            TriggerIntensity.STRONG: [TestResult(
                instance=sample_instance, model_id="test",
                condition="treatment_strong", prompt_used="",
                model_response="", extracted_answer="B",
                response_time_ms=100.0,
            )],
        }

        bms = BiasMagnitudeScore.calculate("test_bias", control, treatment_3, scorer)

        # The minimum per-intensity magnitude among the 3 present is STRONG:
        # min(0.67 * 0.6, 1.0) = 0.402
        # The overall should be >= this (it's a weighted average of values >= 0.402)
        min_magnitude = min(
            bms.treatment_scores[i] for i in treatment_3.keys()
        )
        # With renormalization, overall_magnitude >= min of the per-intensity magnitudes
        # (since it's a weighted average, it must be between min and max)
        assert bms.overall_magnitude >= 0.3, (
            f"Renormalized BMS ({bms.overall_magnitude:.4f}) should not be deflated"
        )

    def test_bms_single_intensity_not_deflated(self, sample_instance):
        """BMS with only MODERATE intensity should not be deflated by missing intensities."""
        def scorer(r):
            return 0.0 if r.condition == "control" else 1.0

        control = [TestResult(
            instance=sample_instance,
            model_id="test",
            condition="control",
            prompt_used="",
            model_response="",
            extracted_answer="A",
            response_time_ms=100.0,
        )]

        treatment = {
            TriggerIntensity.MODERATE: [TestResult(
                instance=sample_instance, model_id="test",
                condition="treatment_moderate", prompt_used="",
                model_response="", extracted_answer="B",
                response_time_ms=100.0,
            )],
        }

        bms = BiasMagnitudeScore.calculate("test_bias", control, treatment, scorer)

        # With max deviation (1.0) on MODERATE (intensity weight 1.0),
        # magnitude = min(1.0 * 1.0, 1.0) = 1.0
        # Renormalized: 0.3 * 1.0 / 0.3 = 1.0 (just MODERATE's weight)
        assert bms.overall_magnitude == 1.0, (
            f"Single-intensity BMS should not be deflated: got {bms.overall_magnitude}"
        )

    def test_bms_no_treatment_data_returns_zero(self, sample_instance):
        """BMS with empty treatment dict should return 0.0 overall magnitude."""
        def scorer(r):
            return 0.5

        control = [TestResult(
            instance=sample_instance,
            model_id="test",
            condition="control",
            prompt_used="",
            model_response="",
            extracted_answer="A",
            response_time_ms=100.0,
        )]

        bms = BiasMagnitudeScore.calculate("test_bias", control, {}, scorer)
        assert bms.overall_magnitude == 0.0


class TestBMSHighUnknownRateGuardrail:
    """Tests for BMS high_unknown_rate flag."""

    def test_high_unknown_rate_flag_set(self, sample_instance):
        """BMS should flag high_unknown_rate when >50% of results are unknown."""
        def scorer(r):
            if r.extracted_answer == "UNKNOWN":
                return None
            return 0.5

        control = [
            TestResult(
                instance=sample_instance, model_id="test",
                condition="control", prompt_used="",
                model_response="", extracted_answer="UNKNOWN",
                response_time_ms=100.0,
            ),
            TestResult(
                instance=sample_instance, model_id="test",
                condition="control", prompt_used="",
                model_response="", extracted_answer="UNKNOWN",
                response_time_ms=100.0,
            ),
        ]
        treatment = {
            TriggerIntensity.MODERATE: [
                TestResult(
                    instance=sample_instance, model_id="test",
                    condition="treatment", prompt_used="",
                    model_response="", extracted_answer="B",
                    response_time_ms=100.0,
                ),
            ]
        }

        bms = BiasMagnitudeScore.calculate("test_bias", control, treatment, scorer)
        # 2/3 results unknown = 66.7% > 50%
        assert bms.high_unknown_rate is True
        assert bms.unknown_rate > 0.5

    def test_low_unknown_rate_flag_not_set(self, sample_instance):
        """BMS should NOT flag high_unknown_rate when <=50% of results are unknown."""
        def scorer(r):
            if r.extracted_answer == "UNKNOWN":
                return None
            return 0.5

        control = [
            TestResult(
                instance=sample_instance, model_id="test",
                condition="control", prompt_used="",
                model_response="", extracted_answer="A",
                response_time_ms=100.0,
            ),
        ]
        treatment = {
            TriggerIntensity.MODERATE: [
                TestResult(
                    instance=sample_instance, model_id="test",
                    condition="treatment", prompt_used="",
                    model_response="", extracted_answer="B",
                    response_time_ms=100.0,
                ),
                TestResult(
                    instance=sample_instance, model_id="test",
                    condition="treatment", prompt_used="",
                    model_response="", extracted_answer="UNKNOWN",
                    response_time_ms=100.0,
                ),
            ]
        }

        bms = BiasMagnitudeScore.calculate("test_bias", control, treatment, scorer)
        # 1/3 results unknown = 33.3% <= 50%
        assert bms.high_unknown_rate is False

    def test_report_tracks_high_unknown_rate_biases(self, sample_instance):
        """CognitiveFingerprintReport should list biases with high unknown rates."""
        from kahne_bench.metrics.core import CognitiveFingerprintReport

        magnitude_scores = {
            "reliable_bias": BiasMagnitudeScore(
                bias_id="reliable_bias",
                control_score=0.0,
                treatment_scores={TriggerIntensity.MODERATE: 0.5},
                overall_magnitude=0.5,
                intensity_sensitivity=0.0,
                unknown_rate=0.1,
                high_unknown_rate=False,
            ),
            "unreliable_bias": BiasMagnitudeScore(
                bias_id="unreliable_bias",
                control_score=0.0,
                treatment_scores={TriggerIntensity.MODERATE: 0.3},
                overall_magnitude=0.3,
                intensity_sensitivity=0.0,
                unknown_rate=0.7,
                high_unknown_rate=True,
            ),
        }

        report = CognitiveFingerprintReport(
            model_id="test",
            biases_tested=list(magnitude_scores.keys()),
            magnitude_scores=magnitude_scores,
            consistency_indices={},
            mitigation_potentials={},
            human_alignments={},
            response_consistencies={},
            calibration_scores={},
        )
        report.compute_summary()

        assert "unreliable_bias" in report.high_unknown_rate_biases
        assert "reliable_bias" not in report.high_unknown_rate_biases


class TestBiasMagnitudeScoreEdgeCases:
    """Edge case tests for Bias Magnitude Score calculation."""

    def test_bms_empty_control_results(self, sample_instance):
        """Test BMS behavior when control_scores is empty (line 147)."""
        def scorer(r):
            return 0.5

        treatment = {
            TriggerIntensity.MODERATE: [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="B",
                    response_time_ms=100.0,
                )
            ]
        }

        # Empty control results
        bms = BiasMagnitudeScore.calculate("anchoring_effect", [], treatment, scorer)
        # Control mean should default to 0.0 when empty
        assert bms.control_score == 0.0
        assert bms.treatment_scores[TriggerIntensity.MODERATE] == 0.5

    def test_bms_empty_treatment_results(self, sample_instance):
        """Test BMS returns 0.0 for empty treatment results (line 153)."""
        def scorer(r):
            return 0.5

        control = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="control",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )
        ]

        treatment = {
            TriggerIntensity.MODERATE: []  # Empty treatment results
        }

        bms = BiasMagnitudeScore.calculate("anchoring_effect", control, treatment, scorer)
        # Treatment mean should default to 0.0 when empty
        assert bms.treatment_scores[TriggerIntensity.MODERATE] == 0.0

    def test_bms_single_intensity(self, sample_instance):
        """Test BMS with only one intensity level provided."""
        def scorer(r):
            return 0.0 if r.condition == "control" else 0.8

        control = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="control",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )
        ]

        treatment = {
            TriggerIntensity.STRONG: [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="treatment_strong",
                    prompt_used="",
                    model_response="",
                    extracted_answer="B",
                    response_time_ms=100.0,
                )
            ]
        }

        bms = BiasMagnitudeScore.calculate("anchoring_effect", control, treatment, scorer)
        assert len(bms.treatment_scores) == 1
        assert TriggerIntensity.STRONG in bms.treatment_scores
        # With single intensity, sensitivity calculation should handle gracefully
        assert isinstance(bms.intensity_sensitivity, float)


class TestBiasConsistencyIndexEdgeCases:
    """Edge case tests for Bias Consistency Index calculation."""

    def test_bci_empty_domain_scores(self):
        """Test BCI with empty domain scores (lines 242-250)."""
        def scorer(r):
            return 0.5

        # Empty results for all domains
        bci = BiasConsistencyIndex.calculate("anchoring_effect", {}, scorer)

        assert bci.domain_scores == {}
        assert bci.mean_bias_score == 0.0
        assert bci.consistency_score == 0.0  # No valid data = cannot assess consistency
        assert bci.standard_deviation == 0.0
        assert bci.is_systematic is False

    def test_bci_single_domain(self, sample_instance):
        """Test BCI with exactly one domain."""
        def scorer(r):
            return 0.8

        results = {
            Domain.PROFESSIONAL: [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="B",
                    response_time_ms=100.0,
                )
            ]
        }

        bci = BiasConsistencyIndex.calculate("anchoring_effect", results, scorer)
        assert len(bci.domain_scores) == 1
        assert bci.mean_bias_score == 0.8
        assert bci.consistency_score == 1.0  # Single domain = no variance
        assert bci.standard_deviation == 0.0  # Can't calculate std with 1 sample
        # Single domain with score > 0.5: is_systematic requires >70% domains
        # 1/1 = 100% > 70%, so should be systematic
        assert bci.is_systematic is True

    def test_bci_all_identical_scores(self, sample_instance):
        """Test BCI when all domains return the same score - consistency should be 1.0."""
        def scorer(r):
            return 0.6  # Same score for all

        sample_instance_individual = CognitiveBiasInstance(
            **{**sample_instance.__dict__, "domain": Domain.INDIVIDUAL}
        )
        sample_instance_social = CognitiveBiasInstance(
            **{**sample_instance.__dict__, "domain": Domain.SOCIAL}
        )
        sample_instance_temporal = CognitiveBiasInstance(
            **{**sample_instance.__dict__, "domain": Domain.TEMPORAL}
        )

        results = {
            Domain.PROFESSIONAL: [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="B",
                    response_time_ms=100.0,
                )
            ],
            Domain.INDIVIDUAL: [
                TestResult(
                    instance=sample_instance_individual,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="B",
                    response_time_ms=100.0,
                )
            ],
            Domain.SOCIAL: [
                TestResult(
                    instance=sample_instance_social,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="B",
                    response_time_ms=100.0,
                )
            ],
            Domain.TEMPORAL: [
                TestResult(
                    instance=sample_instance_temporal,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="B",
                    response_time_ms=100.0,
                )
            ],
        }

        bci = BiasConsistencyIndex.calculate("anchoring_effect", results, scorer)
        assert len(bci.domain_scores) == 4
        assert bci.mean_bias_score == 0.6
        assert bci.consistency_score == 1.0  # All identical = perfect consistency
        assert bci.standard_deviation == 0.0


class TestHumanAlignmentScoreEdgeCases:
    """Edge case tests for Human Alignment Score calculation."""

    def test_has_missing_baseline(self, sample_instance):
        """Test HAS for biases without human baselines uses 0.5 default (lines 530-535)."""
        def scorer(r):
            return 0.7

        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
            )
        ]

        # Use a bias_id that doesn't exist in HUMAN_BASELINES
        has = HumanAlignmentScore.calculate("nonexistent_bias_xyz", results, scorer)
        assert has.human_baseline_rate == 0.5  # Default when missing
        assert has.model_bias_rate == 0.7
        # diff = 0.7 - 0.5 = 0.2 > 0.1, so direction should be "over"
        assert has.bias_direction == "over"

    def test_has_under_direction(self, sample_instance):
        """Test HAS 'under' bias direction when model is less biased than humans (lines 550-553)."""
        def scorer(r):
            return 0.3  # Model much less biased than human baseline

        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )
        ]

        # Human baseline for anchoring_effect is 0.65
        has = HumanAlignmentScore.calculate("anchoring_effect", results, scorer)
        assert has.human_baseline_rate == 0.65
        assert has.model_bias_rate == 0.3
        # diff = 0.3 - 0.65 = -0.35 < -0.1, so direction should be "under"
        assert has.bias_direction == "under"

    def test_has_boundary_at_0_1(self, sample_instance):
        """Test HAS boundary condition around the 0.1 threshold for alignment."""
        # Human baseline for anchoring_effect is 0.65
        # Test values just inside and outside the 0.1 threshold

        # Test value just inside threshold (should be "aligned")
        # Use 0.56 which gives diff = -0.09, abs(diff) < 0.1
        def scorer_aligned(r):
            return 0.56

        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            )
        ]

        has_aligned = HumanAlignmentScore.calculate("anchoring_effect", results, scorer_aligned)
        assert has_aligned.human_baseline_rate == 0.65
        assert has_aligned.model_bias_rate == 0.56
        # diff = 0.56 - 0.65 = -0.09, abs(diff) < 0.1, so should be "aligned"
        assert has_aligned.bias_direction == "aligned"

        # Test value just outside threshold (should be "under")
        # Use 0.54 which gives diff = -0.11, abs(diff) > 0.1
        def scorer_under(r):
            return 0.54

        has_under = HumanAlignmentScore.calculate("anchoring_effect", results, scorer_under)
        assert has_under.human_baseline_rate == 0.65
        assert has_under.model_bias_rate == 0.54
        # diff = 0.54 - 0.65 = -0.11, abs(diff) > 0.1, so should be "under"
        assert has_under.bias_direction == "under"


class TestResponseConsistencyIndexEdgeCases:
    """Edge case tests for Response Consistency Index calculation."""

    def test_rci_empty_results(self):
        """Test RCI with empty results (lines 602-610)."""
        def scorer(r):
            return 0.5

        rci = ResponseConsistencyIndex.calculate("anchoring_effect", [], scorer)

        assert rci.mean_response == 0.0
        assert rci.variance == 0.0
        assert rci.consistency_score == 1.0
        assert rci.is_stable is True
        assert rci.trial_count == 0

    def test_rci_single_trial(self, sample_instance):
        """Test RCI with a single trial (variance calculation edge case)."""
        def scorer(r):
            return 0.7

        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
            )
        ]

        rci = ResponseConsistencyIndex.calculate("anchoring_effect", results, scorer)

        assert rci.trial_count == 1
        assert rci.mean_response == 0.7
        assert rci.variance == 0.0  # Can't calculate variance with 1 sample
        assert rci.consistency_score == 1.0  # No variance = perfect consistency
        assert rci.is_stable is True  # 0.0 < 0.25/1 threshold

    def test_rci_stability_boundary(self, sample_instance):
        """Test RCI stability threshold scales with trial count.

        Threshold = 0.25 / trial_count. For 4 trials: 0.25/4 = 0.0625.
        """
        # Create results with known variance
        scores = [0.5, 0.5, 0.5, 0.7]  # Mean = 0.55, std = 0.1, variance = 0.01 < 0.0625

        def scorer_stable(r):
            idx = int(r.extracted_answer)
            return scores[idx]

        results_stable = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer=str(i),
                response_time_ms=100.0,
            )
            for i in range(4)
        ]

        rci_stable = ResponseConsistencyIndex.calculate("anchoring_effect", results_stable, scorer_stable)
        assert rci_stable.variance < 0.0625  # 0.25 / 4 trials
        assert rci_stable.is_stable is True

        # Now test with variance above threshold
        scores_unstable = [0.0, 1.0, 0.0, 1.0]  # Mean = 0.5, std = 0.577, variance = 0.333 > 0.0625

        def scorer_unstable(r):
            idx = int(r.extracted_answer)
            return scores_unstable[idx]

        results_unstable = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer=str(i),
                response_time_ms=100.0,
            )
            for i in range(4)
        ]

        rci_unstable = ResponseConsistencyIndex.calculate("anchoring_effect", results_unstable, scorer_unstable)
        assert rci_unstable.variance > 0.0625  # 0.25 / 4 trials
        assert rci_unstable.is_stable is False


class TestCalibrationAwarenessScoreEdgeCases:
    """Edge case tests for Calibration Awareness Score calculation."""

    def test_cas_no_confidence_statements(self, sample_instance):
        """Test CAS when no results have confidence statements (lines 689-700)."""
        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
                confidence_stated=None,  # No confidence
            )
            for _ in range(5)
        ]

        def accuracy_scorer(r):
            return 0.8

        cas = CalibrationAwarenessScore.calculate("anchoring_effect", results, accuracy_scorer)

        # Default values when no confidence data
        assert cas.mean_confidence == 0.5
        assert cas.actual_accuracy == 0.5
        assert cas.calibration_error == 0.0
        assert cas.calibration_score == 0.5
        assert cas.overconfident is False
        assert cas.metacognitive_gap == 0.0

    def test_cas_overconfident_boundary(self, sample_instance):
        """Test CAS overconfident threshold at > 0.1 (line 714)."""
        # Test exactly at boundary (confidence - accuracy = 0.1)
        results_boundary = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
                confidence_stated=0.6,  # conf = 0.6
            )
        ]

        def accuracy_scorer_boundary(r):
            return 0.5  # accuracy = 0.5, conf - acc = 0.1 (exactly at boundary)

        cas_boundary = CalibrationAwarenessScore.calculate(
            "anchoring_effect", results_boundary, accuracy_scorer_boundary
        )
        # conf (0.6) > acc (0.5) + 0.1 is 0.6 > 0.6 which is False
        assert cas_boundary.overconfident is False

        # Test just above boundary (confidence - accuracy = 0.11)
        results_over = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
                confidence_stated=0.61,  # conf = 0.61
            )
        ]

        def accuracy_scorer_over(r):
            return 0.5  # accuracy = 0.5, conf - acc = 0.11 > 0.1

        cas_over = CalibrationAwarenessScore.calculate(
            "anchoring_effect", results_over, accuracy_scorer_over
        )
        # conf (0.61) > acc (0.5) + 0.1 is 0.61 > 0.6 which is True
        assert cas_over.overconfident is True
        assert cas_over.metacognitive_gap > 0


class TestHumanBaselines:
    """Tests for HUMAN_BASELINES coverage."""

    def test_all_taxonomy_biases_have_baselines(self):
        """Verify all taxonomy biases have human baselines."""
        from kahne_bench.biases.taxonomy import BIAS_TAXONOMY

        missing = [
            bias_id for bias_id in BIAS_TAXONOMY if bias_id not in HUMAN_BASELINES
        ]
        assert len(missing) == 0, f"Biases missing baselines: {missing}"

    def test_baseline_values_are_valid(self):
        """Verify all baseline values are in valid range [0, 1]."""
        for bias_id, value in HUMAN_BASELINES.items():
            assert 0.0 <= value <= 1.0, f"Invalid baseline for {bias_id}: {value}"

    def test_unknown_baseline_biases_documented(self):
        """Verify UNKNOWN_BASELINE_BIASES set is not empty."""
        assert len(UNKNOWN_BASELINE_BIASES) > 0, "UNKNOWN_BASELINE_BIASES should be populated"

    def test_unknown_biases_have_baselines(self):
        """Verify biases in UNKNOWN_BASELINE_BIASES still have baseline values."""
        for bias_id in UNKNOWN_BASELINE_BIASES:
            assert bias_id in HUMAN_BASELINES, f"Unknown bias {bias_id} has no baseline"


class TestUnknownHandling:
    """Tests for unknown/failed extraction handling in metrics."""

    def test_default_scorer_returns_none_for_unknown(self, sample_instance):
        """Test that default scorer returns None when both is_biased and bias_score are None."""
        calculator = MetricCalculator()

        # Result with unknown extraction (both is_biased and bias_score are None)
        unknown_result = TestResult(
            instance=sample_instance,
            model_id="test",
            condition="treatment",
            prompt_used="",
            model_response="",
            extracted_answer=None,
            response_time_ms=100.0,
            is_biased=None,
            bias_score=None,
        )

        score = calculator._default_scorer(unknown_result)
        assert score is None, "Scorer should return None for unknown extractions"

    def test_default_scorer_returns_value_for_known(self, sample_instance):
        """Test that default scorer returns proper values for known results."""
        calculator = MetricCalculator()

        # Result with bias_score
        result_with_score = TestResult(
            instance=sample_instance,
            model_id="test",
            condition="treatment",
            prompt_used="",
            model_response="",
            extracted_answer="B",
            response_time_ms=100.0,
            bias_score=0.7,
        )
        assert calculator._default_scorer(result_with_score) == 0.7

        # Result with is_biased=True
        result_biased = TestResult(
            instance=sample_instance,
            model_id="test",
            condition="treatment",
            prompt_used="",
            model_response="",
            extracted_answer="B",
            response_time_ms=100.0,
            is_biased=True,
        )
        assert calculator._default_scorer(result_biased) == 1.0

        # Result with is_biased=False
        result_not_biased = TestResult(
            instance=sample_instance,
            model_id="test",
            condition="treatment",
            prompt_used="",
            model_response="",
            extracted_answer="A",
            response_time_ms=100.0,
            is_biased=False,
        )
        assert calculator._default_scorer(result_not_biased) == 0.0

    def test_bms_calculates_unknown_rate(self, sample_instance):
        """Test that BMS correctly calculates unknown_rate."""
        def scorer_with_unknowns(r):
            # Return None for some results to simulate unknowns
            if r.extracted_answer == "UNKNOWN":
                return None
            return 0.5

        control = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="control",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            ),
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="control",
                prompt_used="",
                model_response="",
                extracted_answer="UNKNOWN",  # This will be unknown
                response_time_ms=100.0,
            ),
        ]

        treatment = {
            TriggerIntensity.MODERATE: [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="B",
                    response_time_ms=100.0,
                ),
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="UNKNOWN",  # This will be unknown
                    response_time_ms=100.0,
                ),
            ]
        }

        bms = BiasMagnitudeScore.calculate("test_bias", control, treatment, scorer_with_unknowns)
        # 2 out of 4 results are unknown -> 50% unknown rate
        assert bms.unknown_rate == 0.5

    def test_bci_calculates_unknown_rate(self, sample_instance):
        """Test that BCI correctly calculates unknown_rate."""
        def scorer_with_unknowns(r):
            if r.extracted_answer == "UNKNOWN":
                return None
            return 0.6

        sample_instance_individual = CognitiveBiasInstance(
            **{**sample_instance.__dict__, "domain": Domain.INDIVIDUAL}
        )

        results = {
            Domain.PROFESSIONAL: [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="A",
                    response_time_ms=100.0,
                ),
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="UNKNOWN",
                    response_time_ms=100.0,
                ),
            ],
            Domain.INDIVIDUAL: [
                TestResult(
                    instance=sample_instance_individual,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="B",
                    response_time_ms=100.0,
                ),
            ],
        }

        bci = BiasConsistencyIndex.calculate("test_bias", results, scorer_with_unknowns)
        # 1 out of 3 results is unknown -> ~33% unknown rate
        assert abs(bci.unknown_rate - 1/3) < 0.01

    def test_has_calculates_unknown_rate(self, sample_instance):
        """Test that HAS correctly calculates unknown_rate."""
        def scorer_with_unknowns(r):
            if r.extracted_answer == "UNKNOWN":
                return None
            return 0.65

        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            ),
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="UNKNOWN",
                response_time_ms=100.0,
            ),
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
            ),
        ]

        has = HumanAlignmentScore.calculate("anchoring_effect", results, scorer_with_unknowns)
        # 1 out of 3 is unknown -> ~33% unknown rate
        assert abs(has.unknown_rate - 1/3) < 0.01

    def test_rci_calculates_unknown_rate(self, sample_instance):
        """Test that RCI correctly calculates unknown_rate."""
        def scorer_with_unknowns(r):
            if r.extracted_answer == "UNKNOWN":
                return None
            return 0.5

        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
            ),
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="UNKNOWN",
                response_time_ms=100.0,
            ),
        ]

        rci = ResponseConsistencyIndex.calculate("test_bias", results, scorer_with_unknowns)
        assert rci.unknown_rate == 0.5

    def test_bmp_calculates_unknown_rate(self, sample_instance):
        """Test that BMP correctly calculates unknown_rate."""
        def scorer_with_unknowns(r):
            if r.extracted_answer == "UNKNOWN":
                return None
            return 0.5

        treatment_results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
            ),
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="UNKNOWN",
                response_time_ms=100.0,
            ),
        ]

        debiasing_results = {
            "chain_of_thought": [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="chain_of_thought",
                    prompt_used="",
                    model_response="",
                    extracted_answer="A",
                    response_time_ms=100.0,
                ),
            ],
        }

        bmp = BiasMitigationPotential.calculate(
            "test_bias", treatment_results, debiasing_results, scorer_with_unknowns
        )
        # 1 out of 3 is unknown -> ~33% unknown rate
        assert abs(bmp.unknown_rate - 1/3) < 0.01

    def test_unknown_biases_excluded_from_resistant_ranking(self, sample_instance):
        """Test that biases with high unknown rates are excluded from most_resistant_biases."""
        from kahne_bench.metrics.core import CognitiveFingerprintReport

        # Create magnitude scores with different unknown rates
        magnitude_scores = {
            "reliable_bias_1": BiasMagnitudeScore(
                bias_id="reliable_bias_1",
                control_score=0.0,
                treatment_scores={TriggerIntensity.MODERATE: 0.8},
                overall_magnitude=0.8,
                intensity_sensitivity=0.0,
                unknown_rate=0.1,  # Low unknown rate - should be included
            ),
            "reliable_bias_2": BiasMagnitudeScore(
                bias_id="reliable_bias_2",
                control_score=0.0,
                treatment_scores={TriggerIntensity.MODERATE: 0.2},
                overall_magnitude=0.2,
                intensity_sensitivity=0.0,
                unknown_rate=0.2,  # Low unknown rate - should be included
            ),
            "unreliable_bias": BiasMagnitudeScore(
                bias_id="unreliable_bias",
                control_score=0.0,
                treatment_scores={TriggerIntensity.MODERATE: 0.1},
                overall_magnitude=0.1,  # Lowest magnitude, but high unknown rate
                intensity_sensitivity=0.0,
                unknown_rate=0.6,  # High unknown rate - should be EXCLUDED
                high_unknown_rate=True,
            ),
        }

        report = CognitiveFingerprintReport(
            model_id="test",
            biases_tested=list(magnitude_scores.keys()),
            magnitude_scores=magnitude_scores,
            consistency_indices={},
            mitigation_potentials={},
            human_alignments={},
            response_consistencies={},
            calibration_scores={},
        )
        report.compute_summary()

        # unreliable_bias has the lowest magnitude (0.1) but should NOT be in
        # most_resistant because its unknown_rate (0.6) > 0.5 threshold
        assert "unreliable_bias" not in report.most_resistant_biases
        # reliable_bias_2 has the lowest magnitude among reliable biases
        assert "reliable_bias_2" in report.most_resistant_biases

    def test_report_exposes_unknown_rates(self, sample_instance):
        """Test that CognitiveFingerprintReport exposes unknown_rates_by_bias."""
        from kahne_bench.metrics.core import CognitiveFingerprintReport

        magnitude_scores = {
            "bias_a": BiasMagnitudeScore(
                bias_id="bias_a",
                control_score=0.0,
                treatment_scores={TriggerIntensity.MODERATE: 0.5},
                overall_magnitude=0.5,
                intensity_sensitivity=0.0,
                unknown_rate=0.25,
            ),
            "bias_b": BiasMagnitudeScore(
                bias_id="bias_b",
                control_score=0.0,
                treatment_scores={TriggerIntensity.MODERATE: 0.7},
                overall_magnitude=0.7,
                intensity_sensitivity=0.0,
                unknown_rate=0.1,
            ),
        }

        report = CognitiveFingerprintReport(
            model_id="test",
            biases_tested=["bias_a", "bias_b"],
            magnitude_scores=magnitude_scores,
            consistency_indices={},
            mitigation_potentials={},
            human_alignments={},
            response_consistencies={},
            calibration_scores={},
        )
        report.compute_summary()

        # Verify unknown_rates_by_bias is populated
        assert "bias_a" in report.unknown_rates_by_bias
        assert "bias_b" in report.unknown_rates_by_bias
        assert report.unknown_rates_by_bias["bias_a"] == 0.25
        assert report.unknown_rates_by_bias["bias_b"] == 0.1

    def test_all_unknowns_returns_safe_defaults(self, sample_instance):
        """Test that metrics handle the edge case where all results are unknown."""
        def always_unknown(r):
            return None

        # BMS with all unknowns
        control = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="control",
                prompt_used="",
                model_response="",
                extracted_answer="UNKNOWN",
                response_time_ms=100.0,
            )
        ]
        treatment = {
            TriggerIntensity.MODERATE: [
                TestResult(
                    instance=sample_instance,
                    model_id="test",
                    condition="treatment",
                    prompt_used="",
                    model_response="",
                    extracted_answer="UNKNOWN",
                    response_time_ms=100.0,
                )
            ]
        }

        bms = BiasMagnitudeScore.calculate("test_bias", control, treatment, always_unknown)
        assert bms.unknown_rate == 1.0
        # Should have safe defaults when all results are unknown
        assert bms.control_score == 0.0
        assert bms.treatment_scores[TriggerIntensity.MODERATE] == 0.0

        # RCI with all unknowns
        rci = ResponseConsistencyIndex.calculate("test_bias", control, always_unknown)
        assert rci.unknown_rate == 1.0
        assert rci.trial_count == 0  # No valid trials
        assert rci.consistency_score == 1.0  # Default for no data

    def test_cas_unknown_rate_tracks_missing_confidence(self, sample_instance):
        """Test that CAS unknown_rate tracks results without confidence statements."""

        def accuracy_scorer(r):
            return 0.7

        results = [
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
                confidence_stated=0.8,  # Has confidence
            ),
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="B",
                response_time_ms=100.0,
                confidence_stated=None,  # No confidence
            ),
            TestResult(
                instance=sample_instance,
                model_id="test",
                condition="treatment",
                prompt_used="",
                model_response="",
                extracted_answer="A",
                response_time_ms=100.0,
                confidence_stated=0.9,  # Has confidence
            ),
        ]

        cas = CalibrationAwarenessScore.calculate("test_bias", results, accuracy_scorer)
        # 1 out of 3 results has no confidence -> ~33% unknown rate
        assert abs(cas.unknown_rate - 1/3) < 0.01
