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
from kahne_bench.metrics.core import HUMAN_BASELINES
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
        assert bci.overall_consistency == 0.5

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
