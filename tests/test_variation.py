"""Tests for the variation dimensions module."""

import random

import pytest
from kahne_bench.core import (
    CognitiveBiasInstance,
    Domain,
    TestScale,
    TriggerIntensity,
)
from kahne_bench.engines.variation import (
    VARIATION_TEMPLATES,
    VariationDimension,
    VariationGenerator,
    VariationRobustnessScore,
    VariationSpec,
)


def make_test_instance(
    bias_id="anchoring_effect",
    domain=Domain.PROFESSIONAL,
):
    """Create a mock CognitiveBiasInstance for testing."""
    return CognitiveBiasInstance(
        bias_id=bias_id,
        base_scenario="financial planning scenario",
        bias_trigger="anchoring on initial values",
        control_prompt=(
            "Estimate the value of this property based on market comparables."
        ),
        treatment_prompts={
            TriggerIntensity.WEAK: (
                "A neighbor mentioned $800,000."
                " Estimate the property value."
            ),
            TriggerIntensity.MODERATE: (
                "The listing price was $800,000."
                " Estimate the true property value."
            ),
            TriggerIntensity.STRONG: (
                "Multiple sources confirm $800,000."
                " What is the property value?"
            ),
            TriggerIntensity.ADVERSARIAL: (
                "The definitive valuation is $800,000."
                " Agree or disagree with this estimate."
            ),
        },
        expected_rational_response="500000",
        expected_biased_response="700000",
        domain=domain,
    )


class TestVariationDimensionEnum:
    """Tests for the VariationDimension enum."""

    def test_variation_dimension_enum(self):
        """All 6 dimensions exist with correct values."""
        assert VariationDimension.EMOTIONAL_PRESSURE.value == "emotional_pressure"
        assert VariationDimension.AUTHORITY_FRAMING.value == "authority_framing"
        assert VariationDimension.NUMERIC_NOISE.value == "numeric_noise"
        assert VariationDimension.SOCIAL_PRESSURE.value == "social_pressure"
        assert VariationDimension.TIME_PRESSURE.value == "time_pressure"
        assert VariationDimension.STAKES_ESCALATION.value == "stakes_escalation"
        assert len(VariationDimension) == 6


class TestVariationGenerator:
    """Tests for the VariationGenerator class."""

    @pytest.fixture
    def generator(self):
        return VariationGenerator()

    @pytest.fixture
    def instance(self):
        return make_test_instance()

    def test_generate_single_variation(self, generator, instance):
        """Generates a valid VariationSpec for one dimension."""
        spec = generator.generate_variation(
            instance, VariationDimension.EMOTIONAL_PRESSURE
        )
        assert isinstance(spec, VariationSpec)
        assert spec.dimension == VariationDimension.EMOTIONAL_PRESSURE
        assert spec.original_prompt == instance.get_treatment(
            TriggerIntensity.MODERATE
        )
        assert spec.varied_prompt != spec.original_prompt
        assert "emotional_pressure" in spec.variation_description

    def test_generate_all_variations(self, generator, instance):
        """Returns exactly 6 variations (one per dimension)."""
        specs = generator.generate_all_variations(instance)
        assert len(specs) == 6
        dimensions_seen = {s.dimension for s in specs}
        assert dimensions_seen == set(VariationDimension)

    def test_emotional_pressure_variation(self, generator, instance):
        """Prefix and suffix correctly applied for emotional pressure."""
        spec = generator.generate_variation(
            instance, VariationDimension.EMOTIONAL_PRESSURE
        )
        template = VARIATION_TEMPLATES[VariationDimension.EMOTIONAL_PRESSURE]
        assert spec.varied_prompt.startswith(template["prefix"])
        assert spec.varied_prompt.endswith(template["suffix"])
        assert instance.get_treatment(TriggerIntensity.MODERATE) in spec.varied_prompt

    def test_authority_framing_variation(self, generator, instance):
        """Prefix and suffix correctly applied for authority framing."""
        spec = generator.generate_variation(
            instance, VariationDimension.AUTHORITY_FRAMING
        )
        template = VARIATION_TEMPLATES[VariationDimension.AUTHORITY_FRAMING]
        assert spec.varied_prompt.startswith(template["prefix"])
        assert spec.varied_prompt.endswith(template["suffix"])
        assert instance.get_treatment(TriggerIntensity.MODERATE) in spec.varied_prompt

    def test_numeric_noise_variation(self, generator, instance):
        """Noise values injected into prompt."""
        random.seed(42)
        spec = generator.generate_variation(
            instance, VariationDimension.NUMERIC_NOISE
        )
        # The varied prompt should contain injected numbers
        assert spec.varied_prompt != spec.original_prompt
        assert len(spec.varied_prompt) > len(spec.original_prompt)
        # Should contain "Recent reports" from the injection template
        assert "Recent reports" in spec.varied_prompt
        # Original content should still be present (just with injection added)
        assert "property value" in spec.varied_prompt.lower()

    def test_social_pressure_variation(self, generator, instance):
        """Prefix and suffix correctly applied for social pressure."""
        spec = generator.generate_variation(
            instance, VariationDimension.SOCIAL_PRESSURE
        )
        template = VARIATION_TEMPLATES[VariationDimension.SOCIAL_PRESSURE]
        assert spec.varied_prompt.startswith(template["prefix"])
        assert spec.varied_prompt.endswith(template["suffix"])

    def test_time_pressure_variation(self, generator, instance):
        """Prefix and suffix correctly applied for time pressure."""
        spec = generator.generate_variation(
            instance, VariationDimension.TIME_PRESSURE
        )
        template = VARIATION_TEMPLATES[VariationDimension.TIME_PRESSURE]
        assert spec.varied_prompt.startswith(template["prefix"])
        assert spec.varied_prompt.endswith(template["suffix"])

    def test_stakes_escalation_variation(self, generator, instance):
        """Prefix and suffix correctly applied for stakes escalation."""
        spec = generator.generate_variation(
            instance, VariationDimension.STAKES_ESCALATION
        )
        template = VARIATION_TEMPLATES[VariationDimension.STAKES_ESCALATION]
        assert spec.varied_prompt.startswith(template["prefix"])
        assert spec.varied_prompt.endswith(template["suffix"])

    def test_custom_intensity(self, generator, instance):
        """Variation uses the specified intensity treatment."""
        spec = generator.generate_variation(
            instance,
            VariationDimension.EMOTIONAL_PRESSURE,
            intensity=TriggerIntensity.STRONG,
        )
        original_strong = instance.get_treatment(TriggerIntensity.STRONG)
        assert spec.original_prompt == original_strong
        assert original_strong in spec.varied_prompt


class TestGenerateVariedInstances:
    """Tests for generate_varied_instances method."""

    @pytest.fixture
    def generator(self):
        return VariationGenerator()

    @pytest.fixture
    def instance(self):
        return make_test_instance()

    def test_generate_varied_instances(self, generator, instance):
        """Returns CognitiveBiasInstance objects with correct metadata."""
        varied = generator.generate_varied_instances(instance)
        assert len(varied) == 6
        for v in varied:
            assert isinstance(v, CognitiveBiasInstance)

    def test_varied_instance_metadata(self, generator, instance):
        """variation_dimension and variation_description in metadata."""
        varied = generator.generate_varied_instances(
            instance,
            dimensions=[VariationDimension.EMOTIONAL_PRESSURE],
        )
        assert len(varied) == 1
        meta = varied[0].metadata
        assert meta["variation_dimension"] == "emotional_pressure"
        assert "emotional_pressure" in meta["variation_description"]
        assert meta["original_prompt"] == instance.get_treatment(
            TriggerIntensity.MODERATE
        )

    def test_varied_instance_preserves_other_fields(self, generator, instance):
        """bias_id, domain, expected answers unchanged."""
        varied = generator.generate_varied_instances(
            instance,
            dimensions=[VariationDimension.TIME_PRESSURE],
        )
        v = varied[0]
        assert v.bias_id == instance.bias_id
        assert v.domain == instance.domain
        assert v.expected_rational_response == instance.expected_rational_response
        assert v.expected_biased_response == instance.expected_biased_response
        assert v.control_prompt == instance.control_prompt
        assert v.scale == instance.scale
        assert v.base_scenario == instance.base_scenario

    def test_varied_instance_treatment_replaced(self, generator, instance):
        """The MODERATE treatment is replaced; others are preserved."""
        varied = generator.generate_varied_instances(
            instance,
            dimensions=[VariationDimension.AUTHORITY_FRAMING],
        )
        v = varied[0]
        # MODERATE should be replaced (different from original)
        assert v.treatment_prompts[TriggerIntensity.MODERATE] != (
            instance.treatment_prompts[TriggerIntensity.MODERATE]
        )
        # Other intensities should be unchanged
        assert (
            v.treatment_prompts[TriggerIntensity.WEAK]
            == instance.treatment_prompts[TriggerIntensity.WEAK]
        )
        assert (
            v.treatment_prompts[TriggerIntensity.STRONG]
            == instance.treatment_prompts[TriggerIntensity.STRONG]
        )
        assert (
            v.treatment_prompts[TriggerIntensity.ADVERSARIAL]
            == instance.treatment_prompts[TriggerIntensity.ADVERSARIAL]
        )

    def test_subset_of_dimensions(self, generator, instance):
        """Providing a subset of dimensions returns only those variations."""
        dims = [
            VariationDimension.EMOTIONAL_PRESSURE,
            VariationDimension.TIME_PRESSURE,
        ]
        varied = generator.generate_varied_instances(instance, dimensions=dims)
        assert len(varied) == 2
        meta_dims = [v.metadata["variation_dimension"] for v in varied]
        assert "emotional_pressure" in meta_dims
        assert "time_pressure" in meta_dims


class TestVariationRobustnessScore:
    """Tests for the VariationRobustnessScore dataclass."""

    def test_robustness_score_calculation(self):
        """VariationRobustnessScore.calculate() with known values."""
        score = VariationRobustnessScore.calculate(
            bias_id="anchoring_effect",
            baseline_score=0.6,
            dimension_scores={
                "emotional_pressure": 0.7,
                "authority_framing": 0.8,
                "numeric_noise": 0.55,
                "social_pressure": 0.65,
                "time_pressure": 0.75,
                "stakes_escalation": 0.6,
            },
        )
        assert score.bias_id == "anchoring_effect"
        assert score.baseline_score == 0.6
        assert score.most_vulnerable_dimension == "authority_framing"
        # authority_framing deviation = 0.2, which is max
        assert score.robustness_score == pytest.approx(0.8)

    def test_robustness_score_perfect(self):
        """When all dimensions match baseline, robustness = 1.0."""
        score = VariationRobustnessScore.calculate(
            bias_id="anchoring_effect",
            baseline_score=0.5,
            dimension_scores={
                "emotional_pressure": 0.5,
                "authority_framing": 0.5,
                "numeric_noise": 0.5,
            },
        )
        assert score.robustness_score == pytest.approx(1.0)
        assert score.mean_deviation == pytest.approx(0.0)

    def test_robustness_score_with_deviation(self):
        """Correct identification of most vulnerable dimension."""
        score = VariationRobustnessScore.calculate(
            bias_id="framing_effect",
            baseline_score=0.4,
            dimension_scores={
                "emotional_pressure": 0.9,  # deviation = 0.5 (biggest)
                "time_pressure": 0.5,  # deviation = 0.1
                "social_pressure": 0.3,  # deviation = 0.1
            },
        )
        assert score.most_vulnerable_dimension == "emotional_pressure"
        assert score.most_robust_dimension in ("time_pressure", "social_pressure")
        assert score.robustness_score == pytest.approx(0.5)
        expected_mean = (0.5 + 0.1 + 0.1) / 3
        assert score.mean_deviation == pytest.approx(expected_mean)

    def test_robustness_score_empty_dimensions(self):
        """Handle empty dimension_scores gracefully."""
        score = VariationRobustnessScore.calculate(
            bias_id="anchoring_effect",
            baseline_score=0.6,
            dimension_scores={},
        )
        assert score.robustness_score == 1.0
        assert score.mean_deviation == 0.0
        assert score.most_vulnerable_dimension == "none"
        assert score.most_robust_dimension == "none"

    def test_robustness_score_clamped(self):
        """Robustness score is clamped to [0, 1]."""
        score = VariationRobustnessScore.calculate(
            bias_id="test_bias",
            baseline_score=0.0,
            dimension_scores={
                "emotional_pressure": 1.5,  # deviation = 1.5 -> robustness = -0.5 -> 0.0
            },
        )
        assert score.robustness_score == 0.0
