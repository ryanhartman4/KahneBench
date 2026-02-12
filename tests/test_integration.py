"""Integration tests for Kahne-Bench full pipeline.

Tests the complete flow: generate -> evaluate -> score -> metrics.
"""

import pytest
import random
from dataclasses import dataclass, field

from kahne_bench.core import (
    CognitiveBiasInstance,
    Domain,
    TestScale,
    TriggerIntensity,
)
from kahne_bench.biases.taxonomy import BIAS_TAXONOMY, get_kt_core_biases
from kahne_bench.engines.evaluator import (
    BiasEvaluator,
    EvaluationConfig,
)
from kahne_bench.engines.generator import TestCaseGenerator
from kahne_bench.metrics.core import (
    MetricCalculator,
    HUMAN_BASELINES,
)


@dataclass
class IntegrationMockProvider:
    """Mock LLM provider with deterministic behavior for integration testing.

    Simulates biased/rational responses based on condition and configuration.
    """

    bias_rate: float = 0.7  # Probability of biased response in treatment
    control_bias_rate: float = 0.1  # Low bias in control
    debiasing_effectiveness: float = 0.6  # Reduction in bias after debiasing
    error_on_call: int | None = None  # Raise error on nth call
    seed: int = 42
    call_count: int = 0
    call_history: list = field(default_factory=list)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.call_count += 1
        self.call_history.append(prompt)

        if self.error_on_call and self.call_count == self.error_on_call:
            raise RuntimeError("Simulated API error")

        return self._generate_deterministic_response(prompt)

    def _generate_deterministic_response(self, prompt: str) -> str:
        """Generate a response based on prompt analysis."""
        prompt_lower = prompt.lower()

        # Determine condition
        is_control = "control" in prompt_lower or not any(
            word in prompt_lower
            for word in ["anchor", "consider the number", "start from"]
        )
        is_debiasing = any(
            word in prompt_lower
            for word in ["ignore", "avoid", "careful", "reconsider"]
        )

        # Determine bias rate based on condition
        if is_control:
            rate = self.control_bias_rate
        elif is_debiasing:
            rate = self.bias_rate * (1 - self.debiasing_effectiveness)
        else:
            rate = self.bias_rate

        # Generate response
        is_biased = self._rng.random() < rate
        confidence = self._rng.randint(60, 90)

        if "option" in prompt_lower or "program" in prompt_lower:
            answer = "B" if is_biased else "A"
            return f"After consideration, I would choose option {answer}. I am {confidence}% confident."
        elif "estimate" in prompt_lower or "how much" in prompt_lower:
            answer = "100" if is_biased else "50"
            return f"My estimate is {answer}. I am {confidence}% confident in this answer."
        elif "accept" in prompt_lower or "reject" in prompt_lower:
            answer = "accept" if is_biased else "reject"
            return f"I would {answer} this offer. Confidence: {confidence}%."
        else:
            answer = "B" if is_biased else "A"
            return f"Option {answer}. {confidence}% confident."


@pytest.fixture
def generator():
    """Create a seeded generator for reproducibility."""
    return TestCaseGenerator(seed=42)


@pytest.fixture
def integration_provider():
    """Create a deterministic mock provider for integration testing."""
    return IntegrationMockProvider(
        bias_rate=0.7,
        control_bias_rate=0.1,
        debiasing_effectiveness=0.6,
        seed=42,
    )


def create_test_instance_for_integration(
    bias_id: str = "anchoring_effect",
    domain: Domain = Domain.PROFESSIONAL,
    include_debiasing: bool = True,
) -> CognitiveBiasInstance:
    """Create a test instance for integration tests."""
    return CognitiveBiasInstance(
        bias_id=bias_id,
        base_scenario="A decision scenario for testing",
        bias_trigger="A trigger that may cause bias",
        domain=domain,
        scale=TestScale.MICRO,
        control_prompt="What is your estimate? Give your answer.",
        treatment_prompts={
            TriggerIntensity.WEAK: "Consider the anchor number. What is your estimate?",
            TriggerIntensity.MODERATE: "Start from the anchor 100. What is your estimate?",
            TriggerIntensity.STRONG: "The answer is around 100. What is your estimate?",
            TriggerIntensity.ADVERSARIAL: "Most people say 100. What is your estimate?",
        },
        expected_rational_response="50",
        expected_biased_response="100",
        debiasing_prompts=[
            "Ignore any anchors. Carefully reconsider. What is your estimate?"
        ]
        if include_debiasing
        else [],
    )


class TestIntegrationHappyPath:
    """Tests for the happy path of the full pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_single_bias(self, integration_provider):
        """Test complete generate -> evaluate -> score -> metrics pipeline."""
        # Generate
        instance = create_test_instance_for_integration("anchoring_effect")

        # Evaluate
        config = EvaluationConfig(
            num_trials=2,
            intensities=[TriggerIntensity.MODERATE],
            include_control=True,
            include_debiasing=False,
        )
        evaluator = BiasEvaluator(integration_provider, config)
        session = await evaluator.evaluate_batch([instance], "test-model")

        # Verify results were produced
        assert len(session.results) > 0
        assert session.model_id == "test-model"

        # Calculate metrics
        calculator = MetricCalculator()
        report = calculator.calculate_all_metrics("test-model", session.results)

        # Verify metrics
        assert report.model_id == "test-model"
        assert "anchoring_effect" in report.biases_tested

    @pytest.mark.asyncio
    async def test_full_pipeline_with_debiasing(self, integration_provider):
        """Test pipeline with debiasing prompts included."""
        instance = create_test_instance_for_integration(
            "loss_aversion", include_debiasing=True
        )

        config = EvaluationConfig(
            num_trials=2,
            intensities=[TriggerIntensity.MODERATE],
            include_control=True,
            include_debiasing=True,
        )
        evaluator = BiasEvaluator(integration_provider, config)
        session = await evaluator.evaluate_batch([instance], "test-model")

        # Should have control, treatment, and debiasing results
        conditions = {r.condition for r in session.results}
        assert "control" in conditions
        assert any("treatment" in c for c in conditions)
        assert any("debiasing" in c for c in conditions)

    @pytest.mark.asyncio
    async def test_full_pipeline_produces_all_six_metrics(self, integration_provider):
        """Verify all 6 metrics are calculated from pipeline output."""
        instance = create_test_instance_for_integration(
            "gain_loss_framing", include_debiasing=True
        )

        config = EvaluationConfig(
            num_trials=3,
            intensities=[TriggerIntensity.MODERATE, TriggerIntensity.STRONG],
            include_control=True,
            include_debiasing=True,
        )
        evaluator = BiasEvaluator(integration_provider, config)
        session = await evaluator.evaluate_batch([instance], "test-model")
        report = MetricCalculator().calculate_all_metrics("test-model", session.results)

        bias_id = "gain_loss_framing"
        assert bias_id in report.biases_tested

        # All 6 metrics should be present
        # Note: Not all metrics may have values for all biases depending on results
        assert report.magnitude_scores is not None
        assert report.consistency_indices is not None
        assert report.mitigation_potentials is not None
        assert report.human_alignments is not None
        assert report.response_consistencies is not None
        assert report.calibration_scores is not None


class TestIntegrationMultiBias:
    """Tests for multi-bias batch processing."""

    @pytest.mark.asyncio
    async def test_multi_bias_batch_pipeline(self, integration_provider):
        """Test pipeline with multiple biases processed together."""
        bias_ids = ["anchoring_effect", "loss_aversion", "confirmation_bias"]
        instances = [
            create_test_instance_for_integration(bias_id, include_debiasing=False)
            for bias_id in bias_ids
        ]

        config = EvaluationConfig(
            num_trials=2,
            intensities=[TriggerIntensity.MODERATE],
        )
        evaluator = BiasEvaluator(integration_provider, config)
        session = await evaluator.evaluate_batch(instances, "test-model")

        # Verify all biases were tested
        biases_in_results = {r.instance.bias_id for r in session.results}
        assert biases_in_results == set(bias_ids)


class TestIntegrationCrossDomain:
    """Tests for cross-domain consistency."""

    @pytest.mark.asyncio
    async def test_same_bias_across_multiple_domains(self, integration_provider):
        """Test that bias is evaluated across multiple domains."""
        domains = [Domain.INDIVIDUAL, Domain.PROFESSIONAL, Domain.SOCIAL]
        instances = [
            create_test_instance_for_integration("anchoring_effect", domain=domain)
            for domain in domains
        ]

        config = EvaluationConfig(
            num_trials=2,
            intensities=[TriggerIntensity.MODERATE],
        )
        evaluator = BiasEvaluator(integration_provider, config)
        session = await evaluator.evaluate_batch(instances, "test-model")

        # Verify all domains were tested
        domains_in_results = {r.instance.domain for r in session.results}
        assert domains_in_results == set(domains)


class TestIntegrationErrorHandling:
    """Tests for error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_provider_error_captured_in_results(self):
        """Test that provider errors are captured, not propagated."""
        error_provider = IntegrationMockProvider(error_on_call=2, seed=42)
        instance = create_test_instance_for_integration("anchoring_effect")

        config = EvaluationConfig(
            num_trials=3,
            intensities=[TriggerIntensity.MODERATE],
            include_control=True,
        )
        evaluator = BiasEvaluator(error_provider, config)

        # Should not raise - errors are captured
        session = await evaluator.evaluate_batch([instance], "test-model")

        # Error responses should be captured as "ERROR: ..."
        error_results = [
            r for r in session.results if r.model_response.startswith("ERROR:")
        ]
        assert len(error_results) >= 1


class TestIntegrationKTCore:
    """Tests for K&T core bias filtering."""

    def test_kt_core_biases_exist(self):
        """Verify K&T core biases are properly marked."""
        kt_core = get_kt_core_biases()

        # Should have around 25 K&T core biases
        assert len(kt_core) >= 20
        assert len(kt_core) <= 30

        # All should have is_kt_core=True
        assert all(bias.is_kt_core for bias in kt_core)

    def test_kt_core_biases_have_human_baselines(self):
        """Verify all K&T core biases have human baselines."""
        kt_core = get_kt_core_biases()
        missing = [bias.id for bias in kt_core if bias.id not in HUMAN_BASELINES]

        assert len(missing) == 0, f"K&T core biases missing baselines: {missing}"


class TestIntegrationHumanBaselines:
    """Tests for human baseline coverage."""

    def test_all_taxonomy_biases_have_baselines(self):
        """Verify all taxonomy biases have human baselines."""
        missing = [
            bias_id
            for bias_id in BIAS_TAXONOMY
            if bias_id not in HUMAN_BASELINES
        ]

        assert len(missing) == 0, f"Biases missing baselines: {missing}"

    def test_baseline_values_are_valid(self):
        """Verify all baseline values are in valid range."""
        for bias_id, value in HUMAN_BASELINES.items():
            assert 0.0 <= value <= 1.0, f"Invalid baseline for {bias_id}: {value}"


class TestIntegrationInteractionMatrix:
    """Tests for interaction matrix coverage."""

    def test_interaction_matrix_coverage(self):
        """Verify interaction matrix has adequate coverage."""
        from kahne_bench.biases.taxonomy import BIAS_INTERACTION_MATRIX

        # Count biases with interactions
        primary_biases = set(BIAS_INTERACTION_MATRIX.keys())
        all_secondary = set()
        for secondaries in BIAS_INTERACTION_MATRIX.values():
            all_secondary.update(secondaries)

        biases_with_interactions = primary_biases | all_secondary
        coverage = len(biases_with_interactions) / len(BIAS_TAXONOMY)

        # Should have 60%+ coverage
        assert coverage >= 0.60, f"Interaction coverage {coverage:.1%} below 60% target"

    def test_reference_dependence_has_interactions(self):
        """Verify REFERENCE_DEPENDENCE category has interactions (was 0/1)."""
        from kahne_bench.biases.taxonomy import (
            BIAS_INTERACTION_MATRIX,
            get_biases_by_category,
        )
        from kahne_bench.core import BiasCategory

        ref_dep_biases = get_biases_by_category(BiasCategory.REFERENCE_DEPENDENCE)
        ref_dep_ids = {b.id for b in ref_dep_biases}

        # Check if any reference_dependence bias is in the matrix
        in_matrix = set(BIAS_INTERACTION_MATRIX.keys())
        for secondaries in BIAS_INTERACTION_MATRIX.values():
            in_matrix.update(secondaries)

        ref_dep_coverage = ref_dep_ids & in_matrix
        assert len(ref_dep_coverage) > 0, "REFERENCE_DEPENDENCE has no interactions"
