"""Tests for the cognitive bias taxonomy."""


import re

import pytest
from kahne_bench.biases import (
    BIAS_TAXONOMY,
    get_bias_by_id,
    get_biases_by_category,
    get_all_bias_ids,
    BIAS_INTERACTION_MATRIX,
    BiasCategory,
)
from kahne_bench.biases.taxonomy import get_kt_core_biases, get_extended_biases
from kahne_bench.core import BiasDefinition


class TestBiasTaxonomy:
    """Tests for the bias taxonomy structure and content."""

    def test_taxonomy_has_exactly_69_biases(self):
        """Verify the taxonomy contains exactly 69 biases as specified in the framework documentation.

        The Kahne-Bench framework extends beyond the original 50-bias specification to include
        69 well-documented biases across 16 categories grounded in Kahneman-Tversky research.
        """
        assert len(BIAS_TAXONOMY) == 69, (
            f"Expected exactly 69 biases in taxonomy, found {len(BIAS_TAXONOMY)}"
        )

    def test_all_biases_have_required_fields(self):
        """Verify each bias has all required fields populated."""
        for bias_id, bias in BIAS_TAXONOMY.items():
            assert isinstance(bias, BiasDefinition)
            assert bias.id == bias_id
            assert len(bias.name) > 0
            assert isinstance(bias.category, BiasCategory)
            assert len(bias.description) > 0
            assert len(bias.theoretical_basis) > 0
            assert len(bias.system1_mechanism) > 0
            assert len(bias.system2_override) > 0
            assert len(bias.classic_paradigm) > 0
            assert len(bias.trigger_template) > 0

    def test_get_bias_by_id_returns_correct_bias(self):
        """Test retrieval of specific biases."""
        bias = get_bias_by_id("anchoring_effect")
        assert bias is not None
        assert bias.name == "Anchoring Effect"

    def test_get_bias_by_id_returns_none_for_unknown(self):
        """Test that unknown IDs return None."""
        assert get_bias_by_id("unknown_bias_xyz") is None

    def test_get_biases_by_category(self):
        """Test retrieval of biases by category."""
        for category in BiasCategory:
            biases = get_biases_by_category(category)
            for bias in biases:
                assert bias.category == category

    def test_all_categories_have_biases(self):
        """Verify all categories have at least one bias."""
        for category in BiasCategory:
            biases = get_biases_by_category(category)
            # Most categories should have biases, but some might be empty
            # At minimum, check the main categories
            if category in [
                BiasCategory.REPRESENTATIVENESS,
                BiasCategory.AVAILABILITY,
                BiasCategory.ANCHORING,
                BiasCategory.LOSS_AVERSION,
                BiasCategory.FRAMING,
            ]:
                assert len(biases) > 0, f"Category {category} has no biases"

    def test_get_all_bias_ids(self):
        """Test that we can get all bias IDs."""
        ids = get_all_bias_ids()
        assert len(ids) == len(BIAS_TAXONOMY)
        for bias_id in ids:
            assert bias_id in BIAS_TAXONOMY

    def test_key_biases_present(self):
        """Verify key biases from K&T research are included."""
        key_biases = [
            "anchoring_effect",
            "availability_bias",
            "base_rate_neglect",
            "conjunction_fallacy",
            "loss_aversion",
            "gain_loss_framing",
            "overconfidence_effect",
            "sunk_cost_fallacy",
            "status_quo_bias",
            "primacy_bias",
            "halo_effect",  # Mentioned in spec
            "group_attribution_bias",  # Mentioned in spec
            "neglect_of_probability",  # Mentioned in spec
        ]
        for bias_id in key_biases:
            assert bias_id in BIAS_TAXONOMY, f"Missing key bias: {bias_id}"

    def test_bias_category_distribution(self):
        """Verify each category has the expected number of biases per the framework specification.

        The distribution is documented in taxonomy.py header and must be maintained for
        consistent benchmark coverage across all cognitive bias categories.
        """
        # Expected counts from taxonomy.py docstring
        expected_distribution = {
            BiasCategory.REPRESENTATIVENESS: 8,
            BiasCategory.AVAILABILITY: 6,
            BiasCategory.ANCHORING: 5,
            BiasCategory.LOSS_AVERSION: 5,
            BiasCategory.FRAMING: 6,  # 7 in doc but mental_accounting makes 6 + 1 in FRAMING
            BiasCategory.PROBABILITY_DISTORTION: 7,  # includes affect_heuristic
            BiasCategory.OVERCONFIDENCE: 5,
            BiasCategory.CONFIRMATION: 3,
            BiasCategory.TEMPORAL_BIAS: 3,
            BiasCategory.EXTENSION_NEGLECT: 2,  # scope_insensitivity, identifiable_victim_effect
            BiasCategory.MEMORY_BIAS: 4,
            BiasCategory.ATTENTION_BIAS: 3,
            BiasCategory.ATTRIBUTION_BIAS: 3,
            BiasCategory.UNCERTAINTY_JUDGMENT: 3,
            BiasCategory.SOCIAL_BIAS: 5,  # group_attribution, halo, ingroup, false_consensus, outgroup_homogeneity
            BiasCategory.REFERENCE_DEPENDENCE: 1,  # reference_point_framing
        }

        # Check actual counts
        actual_distribution = {}
        for bias in BIAS_TAXONOMY.values():
            actual_distribution[bias.category] = actual_distribution.get(bias.category, 0) + 1

        for category, expected_count in expected_distribution.items():
            actual_count = actual_distribution.get(category, 0)
            assert actual_count == expected_count, (
                f"Category {category.name}: expected {expected_count} biases, found {actual_count}"
            )

    def test_all_bias_categories_are_used(self):
        """Verify all BiasCategory enum values have at least one bias assigned.

        This prevents orphaned categories that exist in the enum but have no
        corresponding biases in the taxonomy.
        """
        used_categories = {bias.category for bias in BIAS_TAXONOMY.values()}
        all_categories = set(BiasCategory)

        orphaned = all_categories - used_categories
        assert not orphaned, (
            f"The following BiasCategory enum values have no biases: "
            f"{[c.name for c in orphaned]}"
        )

    def test_bias_ids_follow_naming_convention(self):
        """Verify all bias IDs use snake_case naming convention.

        All bias IDs must consist only of lowercase letters and underscores
        (e.g., 'anchoring_effect', 'loss_aversion'). This ensures consistency
        across the codebase and compatibility with Python identifier conventions.
        """
        snake_case_pattern = re.compile(r"^[a-z][a-z_]*[a-z]$|^[a-z]$")

        violations = []
        for bias_id in BIAS_TAXONOMY.keys():
            if not snake_case_pattern.match(bias_id):
                violations.append(bias_id)

        assert not violations, (
            f"The following bias IDs do not follow snake_case convention: {violations}"
        )

    def test_trigger_templates_have_placeholders(self):
        """Verify all trigger templates contain variable placeholders for parameterization.

        Trigger templates must contain at least one {variable_name} placeholder
        to allow dynamic test case generation with scenario-specific values.
        Placeholders can contain lowercase letters, underscores, and digits
        (e.g., {anchor}, {n1}, {base_rate}).
        """
        placeholder_pattern = re.compile(r"\{[a-z][a-z0-9_]*\}")

        missing_placeholders = []
        for bias_id, bias in BIAS_TAXONOMY.items():
            if not placeholder_pattern.search(bias.trigger_template):
                missing_placeholders.append(bias_id)

        assert not missing_placeholders, (
            f"The following biases have trigger templates without placeholders: "
            f"{missing_placeholders}"
        )


class TestBiasInteractionMatrix:
    """Tests for the bias interaction matrix."""

    def test_interaction_matrix_not_empty(self):
        """Verify the interaction matrix has entries."""
        assert len(BIAS_INTERACTION_MATRIX) > 0

    def test_all_primary_biases_exist(self):
        """Verify all primary biases in matrix exist in taxonomy."""
        for primary_bias in BIAS_INTERACTION_MATRIX.keys():
            assert primary_bias in BIAS_TAXONOMY, f"Primary bias not in taxonomy: {primary_bias}"

    def test_all_secondary_biases_exist(self):
        """Verify all secondary biases in matrix exist in taxonomy."""
        for primary_bias, secondaries in BIAS_INTERACTION_MATRIX.items():
            for secondary_bias in secondaries:
                assert secondary_bias in BIAS_TAXONOMY, (
                    f"Secondary bias {secondary_bias} not in taxonomy "
                    f"(from primary {primary_bias})"
                )

    def test_each_primary_has_multiple_interactions(self):
        """Verify each primary bias has multiple interaction partners."""
        for primary_bias, secondaries in BIAS_INTERACTION_MATRIX.items():
            assert len(secondaries) >= 2, f"Primary bias {primary_bias} has too few interactions"

    def test_interaction_matrix_coverage_target(self):
        """Verify interaction matrix meets 60% coverage target."""
        primary_biases = set(BIAS_INTERACTION_MATRIX.keys())
        all_secondary = set()
        for secondaries in BIAS_INTERACTION_MATRIX.values():
            all_secondary.update(secondaries)

        biases_with_interactions = primary_biases | all_secondary
        coverage = len(biases_with_interactions) / len(BIAS_TAXONOMY)

        assert coverage >= 0.60, f"Interaction coverage {coverage:.1%} below 60% target"

    def test_interaction_matrix_reasonable_pair_counts(self):
        """Verify no bias has more than 15 interaction partners.

        This constraint ensures the interaction matrix remains theoretically
        meaningful rather than becoming a dense web where every bias interacts
        with every other bias. Having too many partners dilutes the theoretical
        significance of each interaction.
        """
        max_interactions = 15
        excessive_interactions = []

        for primary_bias, secondaries in BIAS_INTERACTION_MATRIX.items():
            if len(secondaries) > max_interactions:
                excessive_interactions.append(
                    f"{primary_bias}: {len(secondaries)} interactions"
                )

        assert not excessive_interactions, (
            f"The following biases exceed {max_interactions} interaction partners: "
            f"{excessive_interactions}"
        )


class TestKTCoreField:
    """Tests for the is_kt_core field on BiasDefinition."""

    def test_kt_core_field_exists(self):
        """Verify all biases have is_kt_core field."""
        for bias in BIAS_TAXONOMY.values():
            assert hasattr(bias, "is_kt_core")
            assert isinstance(bias.is_kt_core, bool)

    def test_kt_core_count_in_expected_range(self):
        """Verify K&T core count is in expected range."""
        kt_core = [b for b in BIAS_TAXONOMY.values() if b.is_kt_core]
        # Should be around 25 based on analysis
        assert len(kt_core) >= 20, f"Too few K&T core biases: {len(kt_core)}"
        assert len(kt_core) <= 35, f"Too many K&T core biases: {len(kt_core)}"

    def test_kt_core_biases_cite_kt(self):
        """Verify K&T core biases actually cite K&T in theoretical_basis."""
        kt_keywords = ["kahneman", "tversky", "k&t", "prospect theory"]
        for bias in BIAS_TAXONOMY.values():
            if bias.is_kt_core:
                basis_lower = bias.theoretical_basis.lower()
                has_kt = any(kw in basis_lower for kw in kt_keywords)
                assert has_kt, (
                    f"K&T core bias {bias.id} doesn't cite K&T: {bias.theoretical_basis}"
                )

    def test_helper_functions_exist(self):
        """Verify get_kt_core_biases and get_extended_biases work."""
        kt_core = get_kt_core_biases()
        extended = get_extended_biases()

        # Together should equal all biases
        assert len(kt_core) + len(extended) == len(BIAS_TAXONOMY)

        # All kt_core should have is_kt_core=True
        assert all(b.is_kt_core for b in kt_core)

        # All extended should have is_kt_core=False
        assert all(not b.is_kt_core for b in extended)

    def test_non_kt_core_biases_exist(self):
        """Verify the taxonomy contains biases not directly authored by Kahneman & Tversky.

        The framework includes biases documented by other researchers that are
        theoretically connected to dual-process theory but were not part of K&T's
        original research papers. This ensures we have both core and extended biases.
        """
        non_kt_core_biases = [b for b in BIAS_TAXONOMY.values() if not b.is_kt_core]

        assert len(non_kt_core_biases) > 0, (
            "Expected at least some biases with is_kt_core=False"
        )

        # Should have a reasonable number of extended biases (not just 1 or 2)
        assert len(non_kt_core_biases) >= 10, (
            f"Expected at least 10 non-K&T core biases, found {len(non_kt_core_biases)}"
        )

    def test_get_kt_core_biases_returns_subset(self):
        """Test that get_kt_core_biases() returns only biases with is_kt_core=True.

        The function should return a proper subset of the taxonomy containing
        only biases that were directly documented in Kahneman and/or Tversky's
        original research papers.
        """
        kt_core = get_kt_core_biases()

        # Should return a non-empty list
        assert len(kt_core) > 0, "get_kt_core_biases() returned empty list"

        # All returned biases must have is_kt_core=True
        for bias in kt_core:
            assert bias.is_kt_core, (
                f"Bias '{bias.id}' returned by get_kt_core_biases() "
                f"but has is_kt_core=False"
            )

        # Should be a proper subset (not all biases)
        assert len(kt_core) < len(BIAS_TAXONOMY), (
            "get_kt_core_biases() returned all biases, expected a subset"
        )

        # Verify the count matches filtering the taxonomy directly
        expected_kt_core = [b for b in BIAS_TAXONOMY.values() if b.is_kt_core]
        assert len(kt_core) == len(expected_kt_core), (
            f"get_kt_core_biases() returned {len(kt_core)} biases but "
            f"taxonomy has {len(expected_kt_core)} biases with is_kt_core=True"
        )
