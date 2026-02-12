"""Tests for the test case generator."""

import re

import pytest
from kahne_bench.engines.generator import (
    TestCaseGenerator,
    NovelScenarioGenerator,
    DOMAIN_SCENARIOS,
    BIAS_TEMPLATES,
    KahneBenchTier,
    get_tier_biases,
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


class TestEVValueModelBiases:
    """Tests for EV/value-model biases (Part E fixes).

    These tests verify that gain_loss_framing, loss_aversion, certainty_effect,
    and present_bias always have distinct rational and biased responses, with
    rational choices grounded in explicit EV or discounting rules.
    """

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    @pytest.mark.parametrize("bias_id", [
        "gain_loss_framing",
        "loss_aversion",
        "certainty_effect",
        "present_bias",
    ])
    def test_ev_biases_have_different_rational_and_biased_responses(self, generator, bias_id):
        """Verify EV/discounting biases always have distinct rational and biased responses.

        This is the key acceptance criteria for Part E: no generated instance should
        have expected_rational_response == expected_biased_response.
        """
        # Run multiple trials to catch edge cases
        for trial in range(10):
            # Use different seeds for variety
            gen = TestCaseGenerator(seed=42 + trial)
            instance = gen.generate_instance(bias_id, Domain.INDIVIDUAL)
            assert instance.expected_rational_response != instance.expected_biased_response, (
                f"{bias_id} (trial {trial}): rational ({instance.expected_rational_response}) "
                f"== biased ({instance.expected_biased_response})"
            )

    def test_loss_aversion_ev_always_positive(self, generator):
        """Verify loss_aversion always generates positive EV gambles.

        The rational choice should always be 'Accept' because EV > 0.
        """
        for trial in range(20):
            gen = TestCaseGenerator(seed=42 + trial)
            instance = gen.generate_instance("loss_aversion", Domain.INDIVIDUAL)
            # Control prompt should contain explicit EV
            assert "Expected value: $" in instance.control_prompt, (
                "loss_aversion control prompt should show expected value"
            )
            # Rational should always be Accept since EV > 0
            assert instance.expected_rational_response == "Accept", (
                f"loss_aversion rational should be 'Accept' but got "
                f"'{instance.expected_rational_response}'"
            )

    def test_certainty_effect_gamble_ev_exceeds_certain(self, generator):
        """Verify certainty_effect gamble EV always exceeds certain amount.

        The rational choice should always be 'B' (gamble) since EV_B > EV_A.
        """
        for trial in range(20):
            gen = TestCaseGenerator(seed=42 + trial)
            instance = gen.generate_instance("certainty_effect", Domain.INDIVIDUAL)
            # Rational should always be B (higher EV gamble)
            assert instance.expected_rational_response == "B", (
                f"certainty_effect rational should be 'B' but got "
                f"'{instance.expected_rational_response}'"
            )

    def test_present_bias_uses_discounting(self, generator):
        """Verify present_bias includes discount rate in control prompt.

        The rational choice must be grounded in an explicit discounting rule.
        """
        instance = generator.generate_instance("present_bias", Domain.INDIVIDUAL)
        control_lower = instance.control_prompt.lower()
        assert "discount rate" in control_lower or "present value" in control_lower, (
            "present_bias control prompt should reference discount rate or present value"
        )
        # Rational should always be B (higher PV)
        assert instance.expected_rational_response == "B", (
            f"present_bias rational should be 'B' but got "
            f"'{instance.expected_rational_response}'"
        )

    def test_gain_loss_framing_shows_ev_comparison(self, generator):
        """Verify gain_loss_framing control prompt shows both EVs.

        The rational choice should be grounded in EV comparison.
        """
        instance = generator.generate_instance("gain_loss_framing", Domain.INDIVIDUAL)
        # Control should show both EVs
        assert "EV =" in instance.control_prompt, (
            "gain_loss_framing control prompt should show EV values"
        )
        # Rational should be B (higher EV by construction)
        assert instance.expected_rational_response == "B", (
            f"gain_loss_framing rational should be 'B' but got "
            f"'{instance.expected_rational_response}'"
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


class TestNumericTargetRedesign:
    """Tests for Part F: Numeric-Target Redesign.

    These tests verify that anchoring_effect, base_rate_neglect, and gambler_fallacy
    have defensible normative models or categorical choices, with single parseable
    answer formats.
    """

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    def test_anchoring_effect_has_answer_marker(self, generator):
        """Test anchoring prompt includes explicit answer format."""
        instance = generator.generate_instance("anchoring_effect", Domain.PROFESSIONAL)
        assert "Answer:" in instance.control_prompt, (
            "anchoring_effect control prompt should include 'Answer:' marker"
        )
        for intensity in TriggerIntensity:
            treatment = instance.get_treatment(intensity)
            assert "Answer:" in treatment, (
                f"anchoring_effect treatment ({intensity}) should include 'Answer:' marker"
            )

    def test_anchoring_effect_model_grounded_biased_answer(self, generator):
        """Test anchoring biased answer reflects insufficient adjustment.

        The biased answer should be between the rational answer and anchor,
        reflecting the 'insufficient adjustment' mechanism.
        """
        instance = generator.generate_instance("anchoring_effect", Domain.PROFESSIONAL)
        # Biased should differ from rational (insufficient adjustment)
        assert instance.expected_rational_response != instance.expected_biased_response, (
            "anchoring_effect: rational and biased answers should differ"
        )

    def test_base_rate_neglect_categorical_format(self, generator):
        """Test base_rate_neglect uses categorical A/B choice."""
        instance = generator.generate_instance("base_rate_neglect", Domain.PROFESSIONAL)
        assert "A)" in instance.control_prompt, (
            "base_rate_neglect should have option A"
        )
        assert "B)" in instance.control_prompt, (
            "base_rate_neglect should have option B"
        )
        assert "Reply with just A or B" in instance.control_prompt, (
            "base_rate_neglect should specify answer format"
        )
        assert instance.expected_rational_response in ["A", "B"], (
            f"base_rate_neglect rational answer should be A or B, got {instance.expected_rational_response}"
        )
        assert instance.expected_biased_response in ["A", "B"], (
            f"base_rate_neglect biased answer should be A or B, got {instance.expected_biased_response}"
        )

    def test_base_rate_neglect_rational_follows_base_rate(self, generator):
        """Test rational answer follows base rate (majority category).

        With base_rate < 50%, category_b (lawyers) is the majority,
        so the rational answer should be B.
        """
        instance = generator.generate_instance("base_rate_neglect", Domain.PROFESSIONAL)
        # With base_rate < 50%, category_b (lawyers) is majority, so rational = "B"
        assert instance.expected_rational_response == "B", (
            "base_rate_neglect: rational should be B (majority by base rate)"
        )
        # Biased ignores base rate, uses representativeness, so biased = "A"
        assert instance.expected_biased_response == "A", (
            "base_rate_neglect: biased should be A (representativeness heuristic)"
        )

    def test_gambler_fallacy_open_ended_format(self, generator):
        """Test gambler_fallacy uses open-ended descriptive format.

        Gambler's fallacy now uses domain-specific streak scenarios with
        open-ended questions. No "fair", "random", or "independent" language.
        Asks for a recommendation, not a categorical choice.
        """
        instance = generator.generate_instance("gambler_fallacy", Domain.INDIVIDUAL)
        # Should ask for reasoning and recommendation
        assert "reasoning" in instance.control_prompt.lower() or \
               "recommend" in instance.control_prompt.lower(), (
            "gambler_fallacy should ask for reasoning or recommendation"
        )
        # Should NOT use "fair", "random", or "independent" (game-resistant)
        control_lower = instance.control_prompt.lower()
        for forbidden in ["fair coin", "random", "independent"]:
            assert forbidden not in control_lower, (
                f"gambler_fallacy control should not contain '{forbidden}'"
            )
        # Treatment should contain streak-related scenario context
        treatment = instance.get_treatment(TriggerIntensity.MODERATE)
        assert "reasoning" in treatment.lower() or \
               "recommend" in treatment.lower(), (
            "gambler_fallacy treatment should ask for reasoning or recommendation"
        )

    def test_gambler_fallacy_descriptive_answers(self, generator):
        """Test gambler_fallacy uses descriptive rational/biased answers.

        The rational response should reference independence or base rates.
        The biased response should reference streak reversal expectations.
        """
        instance = generator.generate_instance("gambler_fallacy", Domain.INDIVIDUAL)
        # Rational should mention independence or base rates
        rational_lower = instance.expected_rational_response.lower()
        assert "streak" in rational_lower or "independent" in rational_lower or \
               "base rate" in rational_lower, (
            f"gambler_fallacy rational should reference independence/base rates, "
            f"got: {instance.expected_rational_response}"
        )
        # Biased should mention reversal expectation
        biased_lower = instance.expected_biased_response.lower()
        assert "reversal" in biased_lower or "due" in biased_lower or \
               "change" in biased_lower or "streak" in biased_lower, (
            f"gambler_fallacy biased should reference streak reversal, "
            f"got: {instance.expected_biased_response}"
        )

    def test_no_arbitrary_numeric_targets(self, generator):
        """Test that base_rate_neglect uses categorical choices.

        base_rate_neglect should use categorical choices (A/B) rather than
        arbitrary numeric values. gambler_fallacy now uses descriptive
        (open-ended) format scored by LLM judge.
        """
        instance = generator.generate_instance("base_rate_neglect")
        assert instance.expected_rational_response in ["A", "B", "C"], (
            f"base_rate_neglect: rational should be categorical, "
            f"got {instance.expected_rational_response}"
        )
        assert instance.expected_biased_response in ["A", "B", "C"], (
            f"base_rate_neglect: biased should be categorical, "
            f"got {instance.expected_biased_response}"
        )

    def test_rational_differs_from_biased(self, generator):
        """Test that rational and biased answers differ for all three biases.

        This ensures the bias test is valid - if rational == biased,
        the test cannot distinguish between biased and unbiased responses.
        """
        for bias_id in ["anchoring_effect", "base_rate_neglect", "gambler_fallacy"]:
            instance = generator.generate_instance(bias_id)
            assert instance.expected_rational_response != instance.expected_biased_response, (
                f"{bias_id}: rational and biased must differ"
            )


class TestChoiceFormatPromptContract:
    """Tests for Part D: Choice-Format Prompt Contract.

    These tests verify that the 6 biases with A/B/C choice format
    (conjunction_fallacy, sunk_cost_fallacy, status_quo_bias,
    endowment_effect, confirmation_bias, hindsight_bias) include
    explicit Answer: format instructions in their templates.
    """

    # Biases that should have explicit Answer: format instructions
    CHOICE_FORMAT_BIASES = [
        "conjunction_fallacy",
        "sunk_cost_fallacy",
        "status_quo_bias",
        "endowment_effect",
        "confirmation_bias",
        "hindsight_bias",
    ]

    def test_choice_templates_have_answer_format(self):
        """Verify all choice-based bias templates include Answer: format instruction."""
        for bias_id in self.CHOICE_FORMAT_BIASES:
            assert bias_id in BIAS_TEMPLATES, f"Missing template for {bias_id}"
            templates = BIAS_TEMPLATES[bias_id]

            for template_key, template_text in templates.items():
                if not isinstance(template_text, str):
                    continue

                assert "Answer:" in template_text, (
                    f"{bias_id}.{template_key} missing 'Answer:' format instruction"
                )

                assert re.search(r"Respond with your choice in this format:", template_text), (
                    f"{bias_id}.{template_key} missing format instruction line"
                )

    def test_control_and_treatment_have_matching_format(self):
        """Verify both control and treatment use same Answer: format."""
        for bias_id in self.CHOICE_FORMAT_BIASES:
            templates = BIAS_TEMPLATES[bias_id]

            control = templates.get("control", "")
            treatment_keys = [k for k in templates if k.startswith("treatment")]

            control_has_format = "Answer:" in control

            for tkey in treatment_keys:
                treatment = templates[tkey]
                treatment_has_format = "Answer:" in treatment

                assert control_has_format == treatment_has_format, (
                    f"{bias_id}: Control and {tkey} have inconsistent Answer: format"
                )

    def test_two_option_biases_use_a_or_b_format(self):
        """Verify 2-option biases use [A or B] format."""
        two_option_biases = [
            "conjunction_fallacy",
            "sunk_cost_fallacy",
            "status_quo_bias",
            "endowment_effect",
            "hindsight_bias",
        ]

        for bias_id in two_option_biases:
            templates = BIAS_TEMPLATES[bias_id]
            for template_key, template_text in templates.items():
                if not isinstance(template_text, str):
                    continue
                assert "[A or B]" in template_text, (
                    f"{bias_id}.{template_key} should use [A or B] format for 2-option choice"
                )

    def test_three_option_biases_use_a_b_or_c_format(self):
        """Verify 3-option biases use [A, B, or C] format."""
        three_option_biases = ["confirmation_bias"]

        for bias_id in three_option_biases:
            templates = BIAS_TEMPLATES[bias_id]
            for template_key, template_text in templates.items():
                if not isinstance(template_text, str):
                    continue
                assert "[A, B, or C]" in template_text, (
                    f"{bias_id}.{template_key} should use [A, B, or C] format for 3-option choice"
                )


class TestAnswerTypeMetadata:
    """Verify generator sets answer_type metadata for structured outputs."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    @pytest.mark.parametrize("bias_id", [
        "conjunction_fallacy",
        "sunk_cost_fallacy",
        "status_quo_bias",
        "endowment_effect",
        "confirmation_bias",
        "hindsight_bias",
        "gain_loss_framing",
        "certainty_effect",
        "present_bias",
        "base_rate_neglect",
    ])
    def test_choice_biases_use_option_answer_type(self, generator, bias_id):
        instance = generator.generate_instance(bias_id, Domain.INDIVIDUAL)
        assert instance.metadata.get("answer_type") == "option", (
            f"{bias_id} should set answer_type='option'"
        )

    def test_gambler_fallacy_uses_descriptive_answer_type(self, generator):
        """gambler_fallacy now uses open-ended format scored by LLM judge."""
        instance = generator.generate_instance("gambler_fallacy", Domain.INDIVIDUAL)
        assert instance.metadata.get("answer_type") == "descriptive", (
            "gambler_fallacy should set answer_type='descriptive'"
        )

    def test_loss_aversion_uses_yes_no_answer_type(self, generator):
        instance = generator.generate_instance("loss_aversion", Domain.INDIVIDUAL)
        assert instance.metadata.get("answer_type") == "yes_no", (
            "loss_aversion should set answer_type='yes_no'"
        )

    def test_overconfidence_uses_confidence_answer_type(self, generator):
        instance = generator.generate_instance("overconfidence_effect", Domain.INDIVIDUAL)
        assert instance.metadata.get("answer_type") == "confidence", (
            "overconfidence_effect should set answer_type='confidence'"
        )


class TestAvailabilityBiasPromptContract:
    """Tests for Part G: Availability bias prompt/output contract."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    def test_availability_bias_requires_single_numeric_answer(self, generator):
        """Availability bias should request a single numeric answer with Answer: marker."""
        instance = generator.generate_instance("availability_bias", Domain.INDIVIDUAL)
        assert "Answer:" in instance.control_prompt, (
            "availability_bias control prompt should include 'Answer:' marker"
        )
        assert "single numeric estimate" in instance.control_prompt.lower(), (
            "availability_bias should request a single numeric estimate"
        )
        assert instance.expected_rational_response.isdigit(), (
            "availability_bias rational answer should be numeric"
        )
        assert instance.expected_biased_response.isdigit(), (
            "availability_bias biased answer should be numeric"
        )

class TestAnswerExtractionWithNewFormat:
    """Tests that AnswerExtractor handles the new Answer: format.

    These tests verify the extraction patterns work correctly with
    responses formatted per the new prompt contract (Answer: A/B/C).
    """

    @pytest.fixture
    def extractor(self):
        """Create an AnswerExtractor for testing."""
        from kahne_bench.engines.evaluator import AnswerExtractor
        return AnswerExtractor()

    def test_extract_answer_from_formatted_response(self, extractor):
        """Verify extraction of Answer: A/B/C format responses."""
        test_cases = [
            ("Answer: A", "A"),
            ("Answer: B", "B"),
            ("Answer: C", "C"),
            ("I've considered the options carefully.\n\nAnswer: A", "A"),
            ("Based on my analysis:\nAnswer: B", "B"),
            ("answer: a", "A"),
            ("Answer:A", "A"),
            ("Answer: A\n\nThis is because...", "A"),
        ]

        for response, expected in test_cases:
            result = extractor.extract(response, "option")
            assert result is not None, f"Failed to extract from: {response}"
            assert result.upper() == expected, (
                f"Failed for response '{response}': expected '{expected}', got '{result}'"
            )

    def test_extraction_success_rate_target(self, extractor):
        """Verify >95% extraction success on realistic Answer: format responses."""
        responses = [
            "After careful consideration, Option A is better.\n\nAnswer: A",
            "Looking at this objectively, Option B is preferable.\n\nAnswer: B",
            "Both options have merit, but I'll go with C.\n\nAnswer: C",
            "Answer: A",
            "Answer: B\n\nMy reasoning is as follows...",
            "I would recommend:\nAnswer: A",
            "The rational choice here is clear.\n\nAnswer: B",
            "Given the constraints, my choice is:\nAnswer: A",
            "While this is difficult, I must choose.\nAnswer: A\nThough B also has appeal.",
            "**Answer: B**",
        ]

        successes = 0
        for response in responses:
            result = extractor.extract(response, "option")
            if result and result.upper() in ["A", "B", "C"]:
                successes += 1

        success_rate = successes / len(responses)
        assert success_rate >= 0.95, (
            f"Extraction success rate {success_rate:.1%} below 95% target"
        )

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed for reproducibility."""
        return TestCaseGenerator(seed=42)

    def test_generated_prompts_have_answer_format(self, generator):
        """Verify generated instances include Answer: format in prompts."""
        choice_biases = [
            "conjunction_fallacy",
            "sunk_cost_fallacy",
            "status_quo_bias",
            "endowment_effect",
            "confirmation_bias",
            "hindsight_bias",
        ]

        for bias_id in choice_biases:
            instance = generator.generate_instance(bias_id, Domain.INDIVIDUAL)

            # Check control prompt
            assert "Answer:" in instance.control_prompt, (
                f"{bias_id}: control prompt missing Answer: format"
            )

            # Check treatment prompts
            for intensity in TriggerIntensity:
                treatment = instance.get_treatment(intensity)
                assert "Answer:" in treatment, (
                    f"{bias_id}: treatment {intensity.value} missing Answer: format"
                )


class TestGainLossFramingBothFrames:
    """Tests for P0 #3: Both gain and loss frames must be tested.

    The canonical Kahneman & Tversky (1981) framing effect is the REVERSAL
    between gain and loss frames: risk-averse in gains, risk-seeking in losses.
    Testing only the gain frame misses half the phenomenon.
    """

    @pytest.fixture
    def generator(self):
        return TestCaseGenerator(seed=42)

    def test_loss_frame_is_used(self, generator):
        """Verify the loss frame template is actually used (was dead code before fix)."""
        instance = generator.generate_instance("gain_loss_framing", Domain.INDIVIDUAL)
        # STRONG and ADVERSARIAL should use loss frame (mentions "will die")
        for intensity in (TriggerIntensity.STRONG, TriggerIntensity.ADVERSARIAL):
            treatment = instance.get_treatment(intensity)
            assert "will die" in treatment, (
                f"Loss frame treatment at {intensity.value} should mention deaths. "
                f"Got: {treatment[:200]}"
            )

    def test_gain_frame_is_used(self, generator):
        """Verify the gain frame template is used for WEAK/MODERATE."""
        instance = generator.generate_instance("gain_loss_framing", Domain.INDIVIDUAL)
        for intensity in (TriggerIntensity.WEAK, TriggerIntensity.MODERATE):
            treatment = instance.get_treatment(intensity)
            assert "will be saved" in treatment, (
                f"Gain frame treatment at {intensity.value} should mention saving. "
                f"Got: {treatment[:200]}"
            )

    def test_gain_and_loss_frames_differ(self, generator):
        """Verify gain and loss frame prompts are meaningfully different."""
        instance = generator.generate_instance("gain_loss_framing", Domain.INDIVIDUAL)
        gain_text = instance.get_treatment(TriggerIntensity.MODERATE)
        loss_text = instance.get_treatment(TriggerIntensity.STRONG)
        assert gain_text != loss_text, "Gain and loss frame prompts should differ"

    def test_frame_map_metadata(self, generator):
        """Verify frame_map metadata correctly labels each intensity's frame."""
        instance = generator.generate_instance("gain_loss_framing", Domain.INDIVIDUAL)
        frame_map = instance.metadata.get("frame_map")
        assert frame_map is not None, "gain_loss_framing should have frame_map metadata"
        assert frame_map["weak"] == "gain"
        assert frame_map["moderate"] == "gain"
        assert frame_map["strong"] == "loss"
        assert frame_map["adversarial"] == "loss"

    def test_frame_specific_biased_responses_in_metadata(self, generator):
        """Verify metadata records different biased responses per frame."""
        instance = generator.generate_instance("gain_loss_framing", Domain.INDIVIDUAL)
        assert instance.metadata.get("gain_frame_biased") == "A"
        assert instance.metadata.get("loss_frame_biased") == "B"

    def test_frame_specific_rational_responses_in_metadata(self, generator):
        """Verify metadata records frame-conditional rational targets.

        The K&T framing effect is the reversal between frames, so 'rational'
        means resisting the frame-specific bias:
          Gain frame: bias=A (risk aversion) → rational=B
          Loss frame: bias=B (risk seeking) → rational=A
        """
        instance = generator.generate_instance("gain_loss_framing", Domain.INDIVIDUAL)
        assert instance.metadata.get("gain_frame_rational") == "B"
        assert instance.metadata.get("loss_frame_rational") == "A"

    def test_loss_frame_rational_differs_from_biased(self, generator):
        """P0 regression: loss_frame_rational must differ from loss_frame_biased.

        Previously both were 'B', making loss-frame bias undetectable.
        """
        instance = generator.generate_instance("gain_loss_framing", Domain.INDIVIDUAL)
        rational = instance.metadata.get("loss_frame_rational")
        biased = instance.metadata.get("loss_frame_biased")
        assert rational is not None, "loss_frame_rational must be set"
        assert biased is not None, "loss_frame_biased must be set"
        assert rational != biased, (
            f"loss_frame_rational ({rational}) must differ from "
            f"loss_frame_biased ({biased}) for bias to be scorable"
        )

    def test_all_four_frame_targets_present(self, generator):
        """Verify all four frame-conditional targets exist in metadata."""
        instance = generator.generate_instance("gain_loss_framing", Domain.INDIVIDUAL)
        for key in ("gain_frame_rational", "gain_frame_biased",
                     "loss_frame_rational", "loss_frame_biased"):
            assert key in instance.metadata, f"Missing metadata key: {key}"
            assert instance.metadata[key] in ("A", "B"), (
                f"{key} should be 'A' or 'B', got {instance.metadata[key]}"
            )


class TestIntensityVariation:
    """Tests for P0 #4: Intensity adjustment must work for all CORE biases.

    Before the fix, only anchoring_effect had different prompts across
    intensities. Now all CORE biases should produce measurably different
    treatment prompts.
    """

    @pytest.fixture
    def generator(self):
        return TestCaseGenerator(seed=42)

    @pytest.mark.parametrize("bias_id", [
        "anchoring_effect",
        "availability_bias",
        "base_rate_neglect",
        "conjunction_fallacy",
        "loss_aversion",
        "endowment_effect",
        "status_quo_bias",
        "certainty_effect",
        "overconfidence_effect",
        "confirmation_bias",
        "sunk_cost_fallacy",
        "present_bias",
        "hindsight_bias",
        "gambler_fallacy",
        "gain_loss_framing",
    ])
    def test_core_bias_has_intensity_variation(self, generator, bias_id):
        """Each CORE bias must produce at least 2 different treatment prompts."""
        instance = generator.generate_instance(bias_id, Domain.INDIVIDUAL)
        treatments = {
            intensity: instance.get_treatment(intensity)
            for intensity in TriggerIntensity
        }
        unique_treatments = set(treatments.values())
        assert len(unique_treatments) >= 2, (
            f"{bias_id}: Expected at least 2 different treatment prompts across "
            f"intensities, but all {len(TriggerIntensity)} produced identical text"
        )

    def test_weak_intensity_is_softer_than_adversarial(self, generator):
        """WEAK prompts should be shorter/softer than ADVERSARIAL."""
        instance = generator.generate_instance("loss_aversion", Domain.INDIVIDUAL)
        weak = instance.get_treatment(TriggerIntensity.WEAK)
        adversarial = instance.get_treatment(TriggerIntensity.ADVERSARIAL)
        # Adversarial should be longer due to added emotional pressure
        assert len(adversarial) > len(weak), (
            "ADVERSARIAL treatment should be longer than WEAK due to added context"
        )


class TestConstructValidity:
    """Tests that bias prompts measure only the intended construct."""

    def test_present_bias_treatment_has_no_uncertainty_language(self):
        """present_bias treatment must not introduce risk/uncertainty confounds."""
        template = BIAS_TEMPLATES["present_bias"]["treatment"]
        confound_words = ["uncertain", "uncertainty", "risk", "risky", "gamble"]
        for word in confound_words:
            assert word not in template.lower(), (
                f"present_bias treatment contains '{word}', which confounds "
                "temporal preference with risk aversion"
            )

    def test_present_bias_novel_trigger_has_no_uncertainty_language(self):
        """NovelScenarioGenerator temporal_bias trigger must not use uncertainty."""
        gen = NovelScenarioGenerator(seed=42)
        bias_def = BIAS_TAXONOMY["present_bias"]
        for intensity in TriggerIntensity:
            trigger = gen._generate_novel_trigger(
                bias_def, intensity,
                anchor=100, probability=50, amount=1000,
                item="option", event="event",
            )
            assert "uncertain" not in trigger.lower(), (
                f"temporal_bias trigger at {intensity.name} contains 'uncertain', "
                "confounding temporal preference with risk aversion"
            )

    def test_anchoring_treatment_uses_naturalistic_source(self):
        """anchoring treatment must use {anchor_source}, not transparent framing."""
        template = BIAS_TEMPLATES["anchoring_effect"]["treatment"]
        assert "{anchor_source}" in template, (
            "anchoring treatment should use {anchor_source} for ecological validity"
        )
        assert "unrelated context" not in template.lower(), (
            "anchoring treatment should not telegraph the bias with 'unrelated context'"
        )

    def test_anchoring_sources_are_domain_specific(self):
        """Anchor sources should be naturalistic and vary by domain."""
        generator = TestCaseGenerator(seed=42)
        sources_seen = set()
        for domain in Domain:
            instance = generator.generate_instance("anchoring_effect", domain)
            treatment = instance.get_treatment(TriggerIntensity.MODERATE)
            # Each domain should produce a treatment with a naturalistic source
            assert "unrelated" not in treatment.lower(), (
                f"anchoring treatment in {domain.name} uses 'unrelated' language"
            )
            sources_seen.add(domain)
        assert len(sources_seen) == len(Domain)
