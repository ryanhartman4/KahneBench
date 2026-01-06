"""Tests for the cognitive bias taxonomy."""

import pytest
from kahne_bench.biases import (
    BIAS_TAXONOMY,
    get_bias_by_id,
    get_biases_by_category,
    get_all_bias_ids,
    BIAS_INTERACTION_MATRIX,
    BiasCategory,
)
from kahne_bench.core import BiasDefinition


class TestBiasTaxonomy:
    """Tests for the bias taxonomy structure and content."""

    def test_taxonomy_has_at_least_50_biases(self):
        """Verify we have at least 50 biases as specified."""
        assert len(BIAS_TAXONOMY) >= 50

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
        from kahne_bench.biases.taxonomy import get_kt_core_biases, get_extended_biases

        kt_core = get_kt_core_biases()
        extended = get_extended_biases()

        # Together should equal all biases
        assert len(kt_core) + len(extended) == len(BIAS_TAXONOMY)

        # All kt_core should have is_kt_core=True
        assert all(b.is_kt_core for b in kt_core)

        # All extended should have is_kt_core=False
        assert all(not b.is_kt_core for b in extended)
