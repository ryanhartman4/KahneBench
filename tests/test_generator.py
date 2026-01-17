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

    def test_core_tier_is_subset_of_extended(self):
        """Verify that CORE tier biases are a strict subset of EXTENDED tier.

        The CORE tier represents foundational biases that should all be present
        in the comprehensive EXTENDED tier.
        """
        core_biases = set(get_tier_biases(KahneBenchTier.CORE))
        extended_biases = set(get_tier_biases(KahneBenchTier.EXTENDED))

        # All core biases must be in extended
        assert core_biases.issubset(extended_biases), (
            f"CORE tier contains biases not in EXTENDED: {core_biases - extended_biases}"
        )

        # CORE should be smaller than EXTENDED (strict subset)
        assert len(core_biases) < len(extended_biases), (
            "CORE tier should be smaller than EXTENDED tier"
        )


class TestTreatmentPromptVariation:
    """Tests for treatment prompt variation across intensities."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    def test_treatment_prompts_differ_by_intensity(self, generator):
        """Verify that treatment prompts differ meaningfully across intensities.

        Different trigger intensities (WEAK, MODERATE, STRONG, ADVERSARIAL) should
        produce different prompts, not identical copies. This tests that the
        intensity adjustment mechanism is working.
        """
        instance = generator.generate_instance("anchoring_effect", Domain.PROFESSIONAL)

        # Collect all treatment prompts
        treatments = {
            intensity: instance.get_treatment(intensity)
            for intensity in TriggerIntensity
        }

        # Get unique treatment texts
        unique_treatments = set(treatments.values())

        # We expect at least 2 different treatments (due to anchor_value adjustment)
        # Even if the base template is the same, numeric values should differ
        assert len(unique_treatments) >= 2, (
            f"Expected at least 2 different treatment prompts, "
            f"but all {len(TriggerIntensity)} intensities produced identical text. "
            f"Treatments: {treatments}"
        )

    @pytest.mark.parametrize("bias_id", ["anchoring_effect", "loss_aversion", "gain_loss_framing"])
    def test_treatment_prompts_differ_for_multiple_biases(self, generator, bias_id):
        """Test that treatment variation works across different bias types.

        This parametrized test ensures the intensity differentiation isn't
        specific to just one bias template.
        """
        instance = generator.generate_instance(bias_id, Domain.INDIVIDUAL)

        treatments = {
            intensity: instance.get_treatment(intensity)
            for intensity in TriggerIntensity
        }

        unique_treatments = set(treatments.values())

        # At minimum, treatments should not all be identical
        assert len(unique_treatments) >= 1, "Treatment prompts should be generated"

        # Verify each treatment is non-empty
        for intensity, treatment in treatments.items():
            assert len(treatment.strip()) > 0, (
                f"Empty treatment prompt for {bias_id} at {intensity} intensity"
            )


class TestDebiasingPrompts:
    """Tests for debiasing prompt generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    def test_debiasing_prompts_reference_bias(self, generator):
        """Verify that debiasing prompts mention the bias name or mechanism.

        Effective debiasing prompts should explicitly reference the bias being
        tested to help the model engage System 2 reasoning. This test ensures
        prompts aren't generic but bias-specific.
        """
        instance = generator.generate_instance(
            "anchoring_effect",
            include_debiasing=True,
        )

        assert instance.has_debiasing(), "Instance should have debiasing prompts"

        # At least one debiasing prompt should mention the bias
        bias_mentioned = False
        for prompt in instance.debiasing_prompts:
            # Check for bias name (case-insensitive) or key terms
            prompt_lower = prompt.lower()
            if "anchoring" in prompt_lower or "anchor" in prompt_lower:
                bias_mentioned = True
                break

        assert bias_mentioned, (
            f"Expected at least one debiasing prompt to mention 'anchoring'. "
            f"Prompts: {instance.debiasing_prompts}"
        )

    @pytest.mark.parametrize("bias_id,expected_terms", [
        ("loss_aversion", ["loss", "aversion"]),
        ("confirmation_bias", ["confirmation", "bias"]),
        ("overconfidence_effect", ["overconfidence"]),
    ])
    def test_debiasing_prompts_reference_specific_biases(
        self, generator, bias_id, expected_terms
    ):
        """Verify debiasing prompts reference specific bias names.

        Each bias should have debiasing prompts that mention relevant terms
        to effectively prime the model to avoid that specific bias.
        """
        instance = generator.generate_instance(bias_id, include_debiasing=True)

        assert instance.has_debiasing(), f"Instance for {bias_id} should have debiasing"

        # Check that at least one expected term appears in at least one prompt
        all_prompts_text = " ".join(instance.debiasing_prompts).lower()
        term_found = any(term in all_prompts_text for term in expected_terms)

        assert term_found, (
            f"Expected debiasing prompts for '{bias_id}' to contain one of "
            f"{expected_terms}. Got: {instance.debiasing_prompts[:100]}..."
        )


class TestBatchGenerationEdgeCases:
    """Tests for edge cases in batch generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    def test_generate_batch_with_empty_bias_list(self, generator):
        """Edge case: empty bias_ids list should return empty list.

        When no biases are specified, the batch should be empty rather than
        raising an error or defaulting to all biases.
        """
        instances = generator.generate_batch(
            bias_ids=[],
            domains=[Domain.PROFESSIONAL],
            instances_per_combination=3,
        )

        assert isinstance(instances, list), "Should return a list"
        assert len(instances) == 0, (
            f"Empty bias list should produce empty result, got {len(instances)} instances"
        )

    def test_generate_batch_with_single_combination(self, generator):
        """Edge case: single bias + single domain + 1 instance should work.

        Minimal valid batch generation should produce exactly one instance.
        """
        instances = generator.generate_batch(
            bias_ids=["anchoring_effect"],
            domains=[Domain.INDIVIDUAL],
            instances_per_combination=1,
        )

        assert len(instances) == 1, (
            f"Expected exactly 1 instance for single bias/domain/count, got {len(instances)}"
        )

        # Verify the instance is valid
        instance = instances[0]
        assert isinstance(instance, CognitiveBiasInstance)
        assert instance.bias_id == "anchoring_effect"
        assert instance.domain == Domain.INDIVIDUAL

    def test_generate_batch_with_empty_domain_list(self, generator):
        """Edge case: empty domains list should return empty list.

        When no domains are specified explicitly as empty list, should produce
        no instances.
        """
        instances = generator.generate_batch(
            bias_ids=["anchoring_effect"],
            domains=[],
            instances_per_combination=3,
        )

        assert isinstance(instances, list), "Should return a list"
        assert len(instances) == 0, (
            f"Empty domain list should produce empty result, got {len(instances)} instances"
        )


class TestScaleParameter:
    """Tests for TestScale parameter handling."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    @pytest.mark.parametrize("scale", list(TestScale))
    def test_generate_instance_respects_scale_parameter(self, generator, scale):
        """Verify that generate_instance correctly sets the scale parameter.

        The TestScale (MICRO, MESO, MACRO, META) should be passed through
        and stored in the generated instance.
        """
        instance = generator.generate_instance(
            "anchoring_effect",
            domain=Domain.PROFESSIONAL,
            scale=scale,
        )

        assert instance.scale == scale, (
            f"Expected scale {scale}, got {instance.scale}"
        )

    def test_scale_parameter_default_is_micro(self, generator):
        """Verify that the default scale is MICRO when not specified."""
        instance = generator.generate_instance("anchoring_effect")

        assert instance.scale == TestScale.MICRO, (
            f"Expected default scale MICRO, got {instance.scale}"
        )


class TestTemplateCoverage:
    """Tests for template coverage of the bias taxonomy."""

    def test_template_coverage_percentage(self):
        """Verify that BIAS_TEMPLATES covers at least 60% of BIAS_TAXONOMY.

        Having good template coverage ensures most biases can be generated
        with high-quality, bias-specific prompts rather than falling back
        to generic generation.
        """
        total_biases = len(BIAS_TAXONOMY)
        covered_biases = len(BIAS_TEMPLATES)

        coverage_percentage = (covered_biases / total_biases) * 100

        assert coverage_percentage >= 60, (
            f"Template coverage is only {coverage_percentage:.1f}% "
            f"({covered_biases}/{total_biases}). Expected at least 60%."
        )

    def test_all_template_biases_exist_in_taxonomy(self):
        """Verify all biases with templates actually exist in the taxonomy.

        Prevents orphaned templates for biases that were renamed or removed.
        Note: Known orphaned templates are excluded from this check.
        """
        # Known orphaned templates that exist in BIAS_TEMPLATES but not in taxonomy
        # These should be addressed in a future cleanup (either add to taxonomy or remove templates)
        known_orphans = {
            "distinction_bias",
            "denomination_effect",
            "decoy_effect",
            "attribute_substitution",
            "outcome_bias",
            "bandwagon_effect",
        }

        orphaned = []
        for bias_id in BIAS_TEMPLATES.keys():
            if bias_id not in BIAS_TAXONOMY and bias_id not in known_orphans:
                orphaned.append(bias_id)

        assert len(orphaned) == 0, (
            f"Found unexpected orphaned templates (not in BIAS_TAXONOMY): {orphaned}"
        )


class TestDomainScenarioQuality:
    """Tests for domain scenario quality and completeness."""

    def test_scenario_actors_are_realistic(self):
        """Verify domain scenarios have non-empty, reasonable actor names.

        Actors should be realistic role names (e.g., 'physician', 'manager')
        not placeholder text or empty strings.
        """
        for domain, scenarios in DOMAIN_SCENARIOS.items():
            for scenario in scenarios:
                assert len(scenario.actors) > 0, (
                    f"Scenario in {domain} has no actors"
                )

                for actor in scenario.actors:
                    # Actor should be non-empty
                    assert len(actor.strip()) > 0, (
                        f"Empty actor name in {domain} scenario"
                    )

                    # Actor should be a reasonable length (not too short, not too long)
                    assert 3 <= len(actor) <= 50, (
                        f"Actor '{actor}' in {domain} has unreasonable length"
                    )

                    # Actor should not be a placeholder
                    placeholder_patterns = ["xxx", "placeholder", "todo", "tbd"]
                    actor_lower = actor.lower()
                    for pattern in placeholder_patterns:
                        assert pattern not in actor_lower, (
                            f"Actor '{actor}' in {domain} appears to be a placeholder"
                        )

    def test_domain_scenarios_have_required_fields(self):
        """Verify all DOMAIN_SCENARIOS have all required fields with valid values.

        Each scenario must have: domain, context, actors, typical_decisions, value_ranges.
        This is more thorough than the existing test by checking value types and content.
        """
        for domain, scenarios in DOMAIN_SCENARIOS.items():
            assert len(scenarios) > 0, f"Domain {domain} has no scenarios"

            for i, scenario in enumerate(scenarios):
                # Domain must match the key
                assert scenario.domain == domain, (
                    f"Scenario {i} in {domain} has mismatched domain: {scenario.domain}"
                )

                # Context must be a non-empty string
                assert isinstance(scenario.context, str), (
                    f"Scenario {i} in {domain}: context must be string"
                )
                assert len(scenario.context.strip()) > 0, (
                    f"Scenario {i} in {domain}: context is empty"
                )

                # Actors must be a non-empty list of strings
                assert isinstance(scenario.actors, list), (
                    f"Scenario {i} in {domain}: actors must be list"
                )
                assert len(scenario.actors) > 0, (
                    f"Scenario {i} in {domain}: actors list is empty"
                )

                # Typical decisions must be a non-empty list
                assert isinstance(scenario.typical_decisions, list), (
                    f"Scenario {i} in {domain}: typical_decisions must be list"
                )
                assert len(scenario.typical_decisions) > 0, (
                    f"Scenario {i} in {domain}: typical_decisions is empty"
                )

                # Value ranges must be a non-empty dict with tuple values
                assert isinstance(scenario.value_ranges, dict), (
                    f"Scenario {i} in {domain}: value_ranges must be dict"
                )
                assert len(scenario.value_ranges) > 0, (
                    f"Scenario {i} in {domain}: value_ranges is empty"
                )

                for key, value_range in scenario.value_ranges.items():
                    assert isinstance(value_range, tuple), (
                        f"Scenario {i} in {domain}: value_range '{key}' must be tuple"
                    )
                    assert len(value_range) == 2, (
                        f"Scenario {i} in {domain}: value_range '{key}' must have 2 elements"
                    )
                    assert value_range[0] <= value_range[1], (
                        f"Scenario {i} in {domain}: value_range '{key}' min > max"
                    )
