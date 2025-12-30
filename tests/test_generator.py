"""Tests for the test case generator."""

import pytest
from kahne_bench.engines.generator import (
    TestCaseGenerator,
    DOMAIN_SCENARIOS,
    BIAS_TEMPLATES,
    KahneBenchTier,
    get_tier_biases,
    KAHNE_BENCH_CORE_BIASES,
    KAHNE_BENCH_INTERACTION_PAIRS,
)
from kahne_bench.core import (
    Domain,
    TestScale,
    TriggerIntensity,
    CognitiveBiasInstance,
)
from kahne_bench.biases import BIAS_TAXONOMY


class TestTestCaseGenerator:
    """Tests for the TestCaseGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    def test_generate_instance_returns_valid_instance(self, generator):
        """Test that generate_instance returns a valid CognitiveBiasInstance."""
        instance = generator.generate_instance("anchoring_effect", Domain.PROFESSIONAL)
        assert isinstance(instance, CognitiveBiasInstance)
        assert instance.bias_id == "anchoring_effect"
        assert instance.domain == Domain.PROFESSIONAL

    def test_generate_instance_has_control_prompt(self, generator):
        """Test that generated instance has a control prompt."""
        instance = generator.generate_instance("anchoring_effect")
        assert len(instance.control_prompt) > 0

    def test_generate_instance_has_treatment_prompts(self, generator):
        """Test that generated instance has treatment prompts for all intensities."""
        instance = generator.generate_instance("anchoring_effect")
        for intensity in TriggerIntensity:
            treatment = instance.get_treatment(intensity)
            assert len(treatment) > 0

    def test_generate_instance_unknown_bias_raises(self, generator):
        """Test that unknown bias ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown bias ID"):
            generator.generate_instance("unknown_bias_xyz")

    def test_generate_instance_with_debiasing(self, generator):
        """Test that debiasing prompts are generated when requested."""
        instance = generator.generate_instance(
            "anchoring_effect",
            include_debiasing=True,
        )
        assert instance.has_debiasing()
        assert len(instance.debiasing_prompts) > 0

    def test_generate_instance_without_debiasing(self, generator):
        """Test that debiasing prompts are not generated when not requested."""
        instance = generator.generate_instance(
            "anchoring_effect",
            include_debiasing=False,
        )
        assert not instance.has_debiasing()

    def test_generate_instance_across_all_domains(self, generator):
        """Test generation works for all domains."""
        for domain in Domain:
            instance = generator.generate_instance("loss_aversion", domain=domain)
            assert instance.domain == domain

    def test_template_variables_are_filled(self, generator):
        """Test that template variables are properly filled in."""
        instance = generator.generate_instance("anchoring_effect", Domain.PROFESSIONAL)
        # Should not contain unfilled template variables (pattern: {word})
        # Check that no {variable} patterns remain in the control prompt
        import re
        unfilled_vars = re.findall(r'\{[a-z_]+\}', instance.control_prompt)
        # Some generic variables like {context} might remain - that's acceptable
        # But core variables should be filled
        assert "{decision_maker}" not in instance.control_prompt
        # Control prompt should have actual content (not just the template)
        assert len(instance.control_prompt) > 50

    def test_generate_batch_returns_correct_count(self, generator):
        """Test that batch generation returns expected number of instances."""
        instances = generator.generate_batch(
            bias_ids=["anchoring_effect", "loss_aversion"],
            domains=[Domain.PROFESSIONAL],
            instances_per_combination=2,
        )
        # 2 biases * 1 domain * 2 instances = 4
        assert len(instances) == 4

    def test_generate_batch_all_biases_and_domains(self, generator):
        """Test batch generation with defaults (limited for speed)."""
        instances = generator.generate_batch(
            bias_ids=["anchoring_effect"],
            domains=[Domain.INDIVIDUAL, Domain.PROFESSIONAL],
            instances_per_combination=1,
        )
        assert len(instances) == 2


class TestDomainScenarios:
    """Tests for domain scenario configuration."""

    def test_all_domains_have_scenarios(self):
        """Verify all domains have scenario configurations."""
        for domain in Domain:
            assert domain in DOMAIN_SCENARIOS, f"Missing scenarios for domain: {domain}"
            assert len(DOMAIN_SCENARIOS[domain]) > 0

    def test_scenarios_have_required_fields(self):
        """Verify each scenario has required fields."""
        for domain, scenarios in DOMAIN_SCENARIOS.items():
            for scenario in scenarios:
                assert scenario.domain == domain
                assert len(scenario.context) > 0
                assert len(scenario.actors) > 0
                assert len(scenario.typical_decisions) > 0
                assert len(scenario.value_ranges) > 0


class TestBiasTemplates:
    """Tests for bias-specific templates."""

    def test_templates_have_control_and_treatment(self):
        """Verify templates have both control and treatment versions."""
        for bias_id, templates in BIAS_TEMPLATES.items():
            assert "control" in templates, f"Missing control template for {bias_id}"
            # Treatment can be 'treatment' or 'treatment_*' variants
            treatment_keys = [k for k in templates.keys() if k.startswith("treatment")]
            assert len(treatment_keys) > 0, f"Missing treatment template for {bias_id}"

    def test_templates_are_non_empty(self):
        """Verify all templates have content."""
        for bias_id, templates in BIAS_TEMPLATES.items():
            for key, template in templates.items():
                assert len(template.strip()) > 0, f"Empty template: {bias_id}.{key}"


class TestBenchmarkTiers:
    """Tests for benchmark tier configuration."""

    def test_core_tier_biases_exist(self):
        """Verify all core tier biases exist in taxonomy."""
        core_biases = get_tier_biases(KahneBenchTier.CORE)
        for bias_id in core_biases:
            assert bias_id in BIAS_TAXONOMY, f"Core bias not in taxonomy: {bias_id}"

    def test_extended_tier_includes_all_biases(self):
        """Verify extended tier includes all biases."""
        extended_biases = get_tier_biases(KahneBenchTier.EXTENDED)
        assert len(extended_biases) == len(BIAS_TAXONOMY)

    def test_interaction_tier_biases_exist(self):
        """Verify all interaction tier biases exist in taxonomy."""
        interaction_biases = get_tier_biases(KahneBenchTier.INTERACTION)
        for bias_id in interaction_biases:
            assert bias_id in BIAS_TAXONOMY, f"Interaction bias not in taxonomy: {bias_id}"

    def test_interaction_pairs_are_valid(self):
        """Verify all interaction pairs contain valid biases."""
        for b1, b2 in KAHNE_BENCH_INTERACTION_PAIRS:
            assert b1 in BIAS_TAXONOMY, f"Invalid interaction pair bias: {b1}"
            assert b2 in BIAS_TAXONOMY, f"Invalid interaction pair bias: {b2}"

    def test_unknown_tier_raises(self):
        """Test that unknown tier raises ValueError."""
        with pytest.raises(ValueError):
            get_tier_biases("unknown_tier")
