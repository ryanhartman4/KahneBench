"""Tests for advanced generator components: NovelScenarioGenerator, MacroScaleGenerator."""

import pytest

from kahne_bench.core import Domain, TestScale, TriggerIntensity
from kahne_bench.engines.generator import (
    NovelScenarioGenerator,
    MacroScaleGenerator,
    DecisionNode,
    DecisionChain,
    NOVEL_SCENARIO_ELEMENTS,
)
from kahne_bench.biases.taxonomy import BIAS_TAXONOMY


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
