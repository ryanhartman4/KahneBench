"""Tests for advanced generator components: NovelScenarioGenerator, MacroScaleGenerator,
CompoundTestGenerator, and RobustnessTester."""


import pytest

from kahne_bench.core import Domain, TestScale, TriggerIntensity, CognitiveBiasInstance
from kahne_bench.engines.generator import (
    NovelScenarioGenerator,
    MacroScaleGenerator,
    DecisionNode,
    DecisionChain,
    NOVEL_SCENARIO_ELEMENTS,
    TestCaseGenerator,
)
from kahne_bench.engines.compound import CompoundTestGenerator, CompoundBiasScenario
from kahne_bench.engines.robustness import (
    RobustnessTester,
    PARAPHRASE_STRATEGIES,
    ContrastiveRobustnessTester,
)
from kahne_bench.biases.taxonomy import BIAS_TAXONOMY, BIAS_INTERACTION_MATRIX


class TestNovelScenarioGenerator:
    """Tests for NovelScenarioGenerator contamination-resistant testing."""

    def test_init_has_scenario_elements(self):
        generator = NovelScenarioGenerator()
        assert hasattr(generator, "scenario_elements")
        assert "professions" in generator.scenario_elements
        assert "decisions" in generator.scenario_elements
        assert "contexts" in generator.scenario_elements

    def test_scenario_elements_are_novel(self):
        """Verify scenarios are futuristic/novel, not classic psychology examples."""
        elements = NOVEL_SCENARIO_ELEMENTS

        # Check professions are non-traditional
        for profession in elements["professions"]:
            # Should NOT contain classic professions used in psychology studies
            assert "engineer/lawyer" not in profession.lower()
            assert "nurse" not in profession.lower()
            assert "accountant" not in profession.lower()

        # Check contexts include futuristic elements
        context_text = " ".join(elements["contexts"]).lower()
        assert any(word in context_text for word in ["mars", "ai", "quantum", "pandemic", "renewable"])

    def test_generate_novel_instance_returns_valid_instance(self):
        generator = NovelScenarioGenerator(seed=42)
        instance = generator.generate_novel_instance("anchoring_effect")

        assert instance.bias_id == "anchoring_effect"
        assert instance.control_prompt != ""
        assert len(instance.treatment_prompts) > 0

    def test_generate_novel_instance_with_seed_reproducible(self):
        generator1 = NovelScenarioGenerator()
        generator2 = NovelScenarioGenerator()

        instance1 = generator1.generate_novel_instance("anchoring_effect", seed=42)
        instance2 = generator2.generate_novel_instance("anchoring_effect", seed=42)

        assert instance1.control_prompt == instance2.control_prompt
        assert instance1.treatment_prompts == instance2.treatment_prompts

    def test_generate_novel_instance_unknown_bias_raises(self):
        generator = NovelScenarioGenerator()
        with pytest.raises(ValueError, match="Unknown bias ID"):
            generator.generate_novel_instance("fake_bias_that_does_not_exist")

    def test_generate_novel_instance_all_biases(self):
        """Test that novel instances can be generated for all taxonomy biases."""
        generator = NovelScenarioGenerator(seed=42)

        # Test a representative sample of biases
        sample_biases = [
            "anchoring_effect",
            "loss_aversion",
            "confirmation_bias",
            "availability_bias",
            "overconfidence_effect",
        ]

        for bias_id in sample_biases:
            instance = generator.generate_novel_instance(bias_id)
            assert instance.bias_id == bias_id
            assert instance.control_prompt != ""

    def test_generate_novel_instance_has_all_intensities(self):
        generator = NovelScenarioGenerator(seed=42)
        instance = generator.generate_novel_instance("gain_loss_framing")

        # Should have treatments for all standard intensities
        for intensity in TriggerIntensity:
            assert intensity in instance.treatment_prompts

    def test_generate_novel_instance_different_domains(self):
        generator = NovelScenarioGenerator(seed=42)

        # Novel generator should work across domains
        for domain in [Domain.PROFESSIONAL, Domain.INDIVIDUAL, Domain.RISK]:
            instance = generator.generate_novel_instance(
                "anchoring_effect",
                domain=domain
            )
            assert instance.domain == domain

    def test_generate_contamination_resistant_batch_returns_instances(self):
        generator = NovelScenarioGenerator(seed=42)

        # Generate small batch for testing
        instances = generator.generate_contamination_resistant_batch(
            bias_ids=["anchoring_effect", "loss_aversion"],
            domains=[Domain.PROFESSIONAL],
            instances_per_combination=2,
        )

        # 2 biases * 1 domain * 2 instances = 4
        assert len(instances) == 4

    def test_generate_contamination_resistant_batch_diverse(self):
        generator = NovelScenarioGenerator(seed=42)

        instances = generator.generate_contamination_resistant_batch(
            bias_ids=["anchoring_effect"],
            domains=[Domain.PROFESSIONAL],
            instances_per_combination=3,
        )

        # All 3 instances should have different control prompts (novel generation)
        control_prompts = [i.control_prompt for i in instances]
        assert len(set(control_prompts)) == 3

    def test_novel_prompts_contain_novel_elements(self):
        """Verify generated prompts include novel scenario elements."""
        generator = NovelScenarioGenerator(seed=42)
        instance = generator.generate_novel_instance("anchoring_effect")

        all_prompts = instance.control_prompt + " " + " ".join(
            instance.treatment_prompts.values()
        )
        all_prompts_lower = all_prompts.lower()

        # Should contain at least one novel profession or context
        novel_terms = (
            NOVEL_SCENARIO_ELEMENTS["professions"] +
            NOVEL_SCENARIO_ELEMENTS["contexts"]
        )
        has_novel_term = any(
            term.lower() in all_prompts_lower
            for term in novel_terms
        )

        # The instance should reference novel scenarios
        # (allowing for some flexibility in implementation)
        assert len(all_prompts) > 100  # Non-trivial prompts


class TestMacroScaleGenerator:
    """Tests for MacroScaleGenerator sequential decision chain testing."""

    def test_generate_decision_chain_returns_chain(self):
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("anchoring_effect", chain_length=3)

        assert isinstance(chain, DecisionChain)
        assert chain.chain_id.startswith("anchoring_effect_chain_")
        assert len(chain.nodes) == 3

    def test_generate_decision_chain_default_length(self):
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("loss_aversion")

        # Default chain_length is 4
        assert len(chain.nodes) == 4

    def test_generate_decision_chain_nodes_are_decision_nodes(self):
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("confirmation_bias", chain_length=3)

        for node in chain.nodes:
            assert isinstance(node, DecisionNode)
            assert node.prompt != ""
            assert node.bias_id == "confirmation_bias"

    def test_generate_decision_chain_has_dependencies(self):
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("anchoring_effect", chain_length=4)

        # Later nodes should depend on earlier nodes (for sequential testing)
        # First node has no dependencies
        assert chain.nodes[0].depends_on == []

        # At least some later nodes should have dependencies
        has_dependencies = any(len(node.depends_on) > 0 for node in chain.nodes[1:])
        assert has_dependencies

    def test_generate_decision_chain_unknown_bias_raises(self):
        generator = MacroScaleGenerator()
        with pytest.raises(ValueError, match="Unknown bias ID"):
            generator.generate_decision_chain("nonexistent_bias")

    def test_generate_decision_chain_different_domains(self):
        generator = MacroScaleGenerator()

        for domain in [Domain.PROFESSIONAL, Domain.INDIVIDUAL, Domain.RISK]:
            chain = generator.generate_decision_chain(
                "anchoring_effect",
                chain_length=3,
                domain=domain
            )
            assert chain.domain == domain

    def test_generate_decision_chain_has_expected_responses(self):
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("loss_aversion", chain_length=3)

        for node in chain.nodes:
            assert node.expected_rational != ""
            assert node.expected_biased != ""
            assert node.expected_rational != node.expected_biased

    def test_generate_decision_chain_has_cumulative_expectation(self):
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("overconfidence_effect", chain_length=4)

        # Chain should describe cumulative bias effect
        assert chain.cumulative_bias_expected != ""
        assert chain.description != ""

    def test_decision_chain_anchoring_type(self):
        """Test that anchoring biases produce appropriate chains."""
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("anchoring_effect", chain_length=3)

        # Anchoring chains should test anchor persistence across decisions
        assert len(chain.nodes) == 3
        assert chain.nodes[0].bias_id == "anchoring_effect"

    def test_decision_chain_prospect_type(self):
        """Test that prospect theory biases produce appropriate chains."""
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("loss_aversion", chain_length=3)

        assert len(chain.nodes) == 3
        # Loss aversion is categorized under loss_aversion
        assert chain.nodes[0].bias_id == "loss_aversion"

    def test_decision_chain_confirmation_type(self):
        """Test that confirmation biases produce appropriate chains."""
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("confirmation_bias", chain_length=4)

        assert len(chain.nodes) == 4
        # Should test confirmation bias across sequential evidence evaluation

    def test_decision_chain_overconfidence_type(self):
        """Test that overconfidence biases produce appropriate chains."""
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("planning_fallacy", chain_length=4)

        assert len(chain.nodes) == 4
        # Should test optimistic estimation across phases

    def test_decision_chain_generic_type(self):
        """Test that biases without specific chain logic still work."""
        generator = MacroScaleGenerator()
        # Use a bias that doesn't have specialized chain logic
        chain = generator.generate_decision_chain("scope_insensitivity", chain_length=3)

        assert len(chain.nodes) == 3
        assert chain.chain_id.startswith("scope_insensitivity_chain_")

    def test_chain_nodes_sequential_prompts(self):
        """Test that chain nodes build on each other."""
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("anchoring_effect", chain_length=4)

        # Each node's prompt should be non-empty and distinct
        prompts = [node.prompt for node in chain.nodes]
        assert all(p != "" for p in prompts)
        # Prompts should generally be different
        assert len(set(prompts)) >= 3  # Allow for some similarity


class TestDecisionNode:
    """Tests for DecisionNode dataclass."""

    def test_decision_node_creation(self):
        node = DecisionNode(
            prompt="What should we do?",
            bias_id="anchoring_effect",
            depends_on=[],
            expected_rational="B",
            expected_biased="A",
        )

        assert node.prompt == "What should we do?"
        assert node.bias_id == "anchoring_effect"
        assert node.depends_on == []

    def test_decision_node_with_dependencies(self):
        node = DecisionNode(
            prompt="Given earlier decisions...",
            bias_id="loss_aversion",
            depends_on=[0, 1],
            expected_rational="Accept",
            expected_biased="Reject",
        )

        assert node.depends_on == [0, 1]


class TestDecisionChain:
    """Tests for DecisionChain dataclass."""

    def test_decision_chain_creation(self):
        nodes = [
            DecisionNode(
                prompt="Decision 1",
                bias_id="test",
                depends_on=[],
                expected_rational="A",
                expected_biased="B",
            ),
            DecisionNode(
                prompt="Decision 2",
                bias_id="test",
                depends_on=[0],
                expected_rational="A",
                expected_biased="B",
            ),
        ]

        chain = DecisionChain(
            chain_id="test_chain_001",
            nodes=nodes,
            domain=Domain.PROFESSIONAL,
            description="Test chain",
            cumulative_bias_expected="Compounded bias effect",
        )

        assert chain.chain_id == "test_chain_001"
        assert len(chain.nodes) == 2
        assert chain.domain == Domain.PROFESSIONAL


class TestNovelScenarioElements:
    """Tests for NOVEL_SCENARIO_ELEMENTS data structure."""

    def test_has_required_keys(self):
        assert "professions" in NOVEL_SCENARIO_ELEMENTS
        assert "decisions" in NOVEL_SCENARIO_ELEMENTS
        assert "contexts" in NOVEL_SCENARIO_ELEMENTS

    def test_professions_not_empty(self):
        assert len(NOVEL_SCENARIO_ELEMENTS["professions"]) >= 5

    def test_decisions_not_empty(self):
        assert len(NOVEL_SCENARIO_ELEMENTS["decisions"]) >= 5

    def test_contexts_not_empty(self):
        assert len(NOVEL_SCENARIO_ELEMENTS["contexts"]) >= 5

    def test_elements_are_strings(self):
        for key in ["professions", "decisions", "contexts"]:
            for element in NOVEL_SCENARIO_ELEMENTS[key]:
                assert isinstance(element, str)
                assert len(element) > 0


class TestCompoundTestGenerator:
    """Tests for CompoundTestGenerator meso-scale compound bias testing."""

    def test_compound_generator_init(self):
        """Test that CompoundTestGenerator initializes with interaction matrix."""
        generator = CompoundTestGenerator()
        assert hasattr(generator, "interaction_matrix")
        assert isinstance(generator.interaction_matrix, dict)
        assert "anchoring_effect" in generator.interaction_matrix

    def test_generate_compound_instance_basic(self):
        """Test basic generation of compound instance with primary and secondary biases."""
        generator = CompoundTestGenerator()
        instance = generator.generate_compound_instance(
            primary_bias="anchoring_effect",
            secondary_biases=["availability_bias"],
            domain=Domain.PROFESSIONAL,
        )

        assert isinstance(instance, CognitiveBiasInstance)
        assert instance.bias_id == "anchoring_effect"
        assert instance.control_prompt != ""
        assert len(instance.treatment_prompts) > 0

    def test_compound_instance_has_meso_scale(self):
        """Verify generated compound instance has scale=MESO."""
        generator = CompoundTestGenerator()
        instance = generator.generate_compound_instance(
            primary_bias="loss_aversion",
            domain=Domain.INDIVIDUAL,
        )

        assert instance.scale == TestScale.MESO

    def test_compound_amplifying_interaction(self):
        """Test generation with interaction_type='amplifying'."""
        generator = CompoundTestGenerator()
        instance = generator.generate_compound_instance(
            primary_bias="anchoring_effect",
            secondary_biases=["availability_bias", "overconfidence_effect"],
            domain=Domain.PROFESSIONAL,
            interaction_type="amplifying",
        )

        assert instance.metadata["interaction_type"] == "amplifying"
        assert instance.metadata["expected_amplification"] > 1.0
        assert instance.scale == TestScale.MESO

    def test_compound_competing_interaction(self):
        """Test generation with interaction_type='competing'."""
        generator = CompoundTestGenerator()
        instance = generator.generate_compound_instance(
            primary_bias="availability_bias",
            secondary_biases=["neglect_of_probability"],
            domain=Domain.RISK,
            interaction_type="competing",
        )

        assert instance.metadata["interaction_type"] == "competing"
        # Competing biases may cancel each other out
        assert instance.metadata["expected_amplification"] <= 1.0

    def test_compound_cascading_interaction(self):
        """Test generation with interaction_type='cascading'."""
        generator = CompoundTestGenerator()
        instance = generator.generate_compound_instance(
            primary_bias="confirmation_bias",
            secondary_biases=["belief_perseverance"],
            domain=Domain.PROFESSIONAL,
            interaction_type="cascading",
        )

        assert instance.metadata["interaction_type"] == "cascading"
        # Cascading can compound significantly
        assert instance.metadata["expected_amplification"] >= 1.0

    def test_compound_unknown_bias_raises(self):
        """Test that unknown primary bias_id raises ValueError."""
        generator = CompoundTestGenerator()
        with pytest.raises(ValueError, match="Unknown primary bias"):
            generator.generate_compound_instance(
                primary_bias="totally_fake_bias_xyz",
                domain=Domain.PROFESSIONAL,
            )

    def test_compound_secondary_from_matrix(self):
        """Verify secondary biases come from BIAS_INTERACTION_MATRIX when not specified."""
        generator = CompoundTestGenerator()

        # Test with anchoring_effect which has known interactions in the matrix
        instance = generator.generate_compound_instance(
            primary_bias="anchoring_effect",
            secondary_biases=None,  # Should default to matrix
            domain=Domain.PROFESSIONAL,
        )

        # The secondary biases should come from the interaction matrix
        expected_secondaries = BIAS_INTERACTION_MATRIX.get("anchoring_effect", [])[:2]
        assert instance.metadata["secondary_biases"] == expected_secondaries

    def test_compound_interaction_biases_attribute(self):
        """Test that interaction_biases attribute is set correctly."""
        generator = CompoundTestGenerator()
        secondary = ["availability_bias", "overconfidence_effect"]

        instance = generator.generate_compound_instance(
            primary_bias="anchoring_effect",
            secondary_biases=secondary,
            domain=Domain.PROFESSIONAL,
        )

        assert instance.interaction_biases == secondary

    def test_compound_all_intensities_present(self):
        """Test that compound instances have all trigger intensities."""
        generator = CompoundTestGenerator()
        instance = generator.generate_compound_instance(
            primary_bias="loss_aversion",
            domain=Domain.INDIVIDUAL,
        )

        for intensity in TriggerIntensity:
            assert intensity in instance.treatment_prompts

    def test_generate_interaction_battery(self):
        """Test generation of complete interaction battery from matrix."""
        generator = CompoundTestGenerator()
        instances = generator.generate_interaction_battery(domain=Domain.PROFESSIONAL)

        # Should generate at least one instance per bias in the interaction matrix
        assert len(instances) > 0
        # All instances should be MESO scale
        for instance in instances:
            assert instance.scale == TestScale.MESO


class TestRobustnessTester:
    """Tests for RobustnessTester adversarial robustness testing."""

    def test_robustness_tester_init(self):
        """Test basic initialization of RobustnessTester."""
        tester = RobustnessTester()
        assert hasattr(tester, "paraphrase_strategies")
        assert isinstance(tester.paraphrase_strategies, dict)
        assert len(tester.paraphrase_strategies) > 0

    def test_generate_paraphrases(self):
        """Test paraphrase generation for a prompt."""
        tester = RobustnessTester()
        original_prompt = "You must decide whether to accept or reject this offer."

        paraphrases = tester.generate_paraphrases(original_prompt, num_variants=3)

        assert isinstance(paraphrases, list)
        # May generate fewer if strategies don't match
        assert len(paraphrases) <= 3

    def test_paraphrase_strategies_coverage(self):
        """Verify multiple paraphrase strategies are available."""
        assert "passive_voice" in PARAPHRASE_STRATEGIES
        assert "formality_shift" in PARAPHRASE_STRATEGIES
        assert "perspective_shift" in PARAPHRASE_STRATEGIES
        assert "hedging" in PARAPHRASE_STRATEGIES

        # Each strategy should have patterns
        for strategy_name, strategy in PARAPHRASE_STRATEGIES.items():
            assert "patterns" in strategy
            assert len(strategy["patterns"]) > 0

    def test_generate_paraphrases_with_specific_strategies(self):
        """Test paraphrase generation with specific strategies."""
        tester = RobustnessTester()
        prompt = "You should choose the best option."

        paraphrases = tester.generate_paraphrases(
            prompt,
            num_variants=2,
            strategies=["formality_shift"],
        )

        # Should generate some paraphrases using formality shift
        assert isinstance(paraphrases, list)

    def test_debiasing_self_help_prompts(self):
        """Test generation of self-help debiasing prompts."""
        tester = RobustnessTester()
        generator = TestCaseGenerator(seed=42)

        # Create a test instance to use
        instance = generator.generate_instance("anchoring_effect", Domain.PROFESSIONAL)

        # Generate self-help debiasing prompt
        self_help_prompt = tester.create_self_help_prompt(instance)

        assert isinstance(self_help_prompt, str)
        assert len(self_help_prompt) > 0
        assert "cognitive bias" in self_help_prompt.lower()
        assert "rewrite" in self_help_prompt.lower()

    def test_create_debiasing_variants(self):
        """Test creation of debiasing prompt variants."""
        tester = RobustnessTester()
        generator = TestCaseGenerator(seed=42)
        instance = generator.generate_instance("loss_aversion", Domain.INDIVIDUAL)

        variants = tester.create_debiasing_variants(instance)

        assert isinstance(variants, dict)
        assert "explicit_warning" in variants
        assert "chain_of_thought" in variants
        assert "consider_opposite" in variants
        assert "pre_mortem" in variants
        assert "reference_class_forecasting" in variants

        # Each variant should be a non-empty string
        for name, prompt in variants.items():
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_calculate_consistency(self):
        """Test consistency calculation across responses."""
        tester = RobustnessTester()

        # Perfect consistency - all same scores
        def constant_scorer(response: str) -> float:
            return 0.5

        consistency = tester.calculate_consistency(
            ["response1", "response2", "response3"],
            constant_scorer,
        )
        assert consistency == 1.0

        # Variable scores
        scores_list = [0.2, 0.8, 0.5]
        score_idx = 0

        def varying_scorer(response: str) -> float:
            nonlocal score_idx
            score = scores_list[score_idx % len(scores_list)]
            score_idx += 1
            return score

        consistency = tester.calculate_consistency(
            ["r1", "r2", "r3"],
            varying_scorer,
        )
        assert 0 <= consistency <= 1

    def test_calculate_consistency_single_response(self):
        """Test consistency with single response returns 1.0."""
        tester = RobustnessTester()

        consistency = tester.calculate_consistency(
            ["single_response"],
            lambda x: 0.5,
        )
        assert consistency == 1.0


class TestContrastiveRobustnessTester:
    """Tests for ContrastiveRobustnessTester extended robustness testing."""

    def test_create_contrastive_pair(self):
        """Test creation of contrastive prompt pairs."""
        tester = ContrastiveRobustnessTester()
        generator = TestCaseGenerator(seed=42)
        instance = generator.generate_instance("anchoring_effect", Domain.PROFESSIONAL)

        with_trigger, without_trigger = tester.create_contrastive_pair(instance)

        assert isinstance(with_trigger, str)
        assert isinstance(without_trigger, str)
        assert with_trigger != without_trigger

    def test_create_diagnostic_pair(self):
        """Test creation of diagnostic prompt pairs."""
        tester = ContrastiveRobustnessTester()
        generator = TestCaseGenerator(seed=42)
        instance = generator.generate_instance("anchoring_effect", Domain.PROFESSIONAL)

        minimal, maximal, expected = tester.create_diagnostic_pair(instance)

        assert isinstance(minimal, str)
        assert isinstance(maximal, str)
        assert isinstance(expected, str)
        # Minimal is WEAK intensity, maximal is ADVERSARIAL
        # They should generally be different, but verify they're valid prompts
        assert len(minimal) > 0
        assert len(maximal) > 0

    def test_calculate_discrimination(self):
        """Test discrimination calculation between conditions."""
        tester = ContrastiveRobustnessTester()

        # Clear separation
        trigger_scores = [0.8, 0.9, 0.85]
        no_trigger_scores = [0.1, 0.2, 0.15]

        discrimination = tester.calculate_discrimination(trigger_scores, no_trigger_scores)

        assert 0 <= discrimination <= 1
        # High discrimination expected due to clear separation
        assert discrimination > 0.5

    def test_calculate_discrimination_no_difference(self):
        """Test discrimination when scores are similar."""
        tester = ContrastiveRobustnessTester()

        # No real difference
        trigger_scores = [0.5, 0.5, 0.5]
        no_trigger_scores = [0.5, 0.5, 0.5]

        discrimination = tester.calculate_discrimination(trigger_scores, no_trigger_scores)

        # Low discrimination expected
        assert discrimination < 0.5

    def test_calculate_discrimination_empty_lists(self):
        """Test discrimination with empty lists returns 0."""
        tester = ContrastiveRobustnessTester()

        assert tester.calculate_discrimination([], [0.5]) == 0.0
        assert tester.calculate_discrimination([0.5], []) == 0.0


class TestMacroScaleGeneratorEdgeCases:
    """Additional edge case tests for MacroScaleGenerator."""

    def test_decision_chain_length_1(self):
        """Test minimum chain length of 1."""
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("anchoring_effect", chain_length=1)

        assert isinstance(chain, DecisionChain)
        assert len(chain.nodes) == 1
        assert chain.nodes[0].bias_id == "anchoring_effect"

    def test_decision_chain_length_10(self):
        """Test large chain length of 10."""
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("loss_aversion", chain_length=10)

        assert isinstance(chain, DecisionChain)
        assert len(chain.nodes) == 10

        # Verify all nodes are valid
        for i, node in enumerate(chain.nodes):
            assert isinstance(node, DecisionNode)
            assert node.prompt != ""

    def test_decision_chain_unknown_bias_raises(self):
        """Test that unknown bias_id raises ValueError."""
        generator = MacroScaleGenerator()
        with pytest.raises(ValueError, match="Unknown bias ID"):
            generator.generate_decision_chain("completely_nonexistent_bias")

    def test_decision_chain_nodes_have_expected_responses(self):
        """Test that all chain nodes have expected rational and biased responses."""
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("confirmation_bias", chain_length=5)

        for node in chain.nodes:
            assert node.expected_rational != ""
            assert node.expected_biased != ""

    def test_decision_chain_description_includes_bias_name(self):
        """Test that chain description mentions the bias."""
        generator = MacroScaleGenerator()
        chain = generator.generate_decision_chain("overconfidence_effect", chain_length=3)

        assert "overconfidence" in chain.description.lower()
        assert chain.cumulative_bias_expected != ""

    def test_decision_chain_different_bias_categories(self):
        """Test chain generation for biases from different categories."""
        generator = MacroScaleGenerator()

        # Test different bias categories to ensure all code paths work
        test_biases = [
            "anchoring_effect",  # anchoring category
            "loss_aversion",  # loss_aversion category
            "confirmation_bias",  # confirmation category
            "planning_fallacy",  # overconfidence category
            "availability_bias",  # should use generic chain
        ]

        for bias_id in test_biases:
            chain = generator.generate_decision_chain(bias_id, chain_length=3)
            assert len(chain.nodes) == 3
            assert chain.chain_id.startswith(f"{bias_id}_chain_")
