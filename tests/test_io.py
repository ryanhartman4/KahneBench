"""Tests for data import/export utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from kahne_bench.core import (
    CognitiveBiasInstance,
    TestResult,
    EvaluationSession,
    Domain,
    TriggerIntensity,
)
from kahne_bench.metrics.core import (
    CognitiveFingerprintReport,
    BiasMagnitudeScore,
    BiasConsistencyIndex,
    HumanAlignmentScore,
    BiasMitigationPotential,
    ResponseConsistencyIndex,
    CalibrationAwarenessScore,
)
from kahne_bench.utils.io import (
    export_instances_to_json,
    import_instances_from_json,
    export_results_to_json,
    export_results_to_csv,
    export_session_to_json,
    export_fingerprint_to_json,
    export_fingerprint_to_csv,
    generate_summary_report,
)


@pytest.fixture
def sample_instance():
    """Create a sample test instance."""
    return CognitiveBiasInstance(
        bias_id="anchoring_effect",
        base_scenario="A house was listed at $500,000.",
        bias_trigger="Initial price anchor",
        control_prompt="What would be a fair price for this house?",
        treatment_prompts={
            TriggerIntensity.WEAK: "The house was originally listed at $450,000. What's fair?",
            TriggerIntensity.MODERATE: "The house was originally listed at $600,000. What's fair?",
            TriggerIntensity.STRONG: "The house was originally listed at $800,000. What's fair?",
        },
        expected_rational_response="Based on comparable sales...",
        expected_biased_response="Around $600,000",
        domain=Domain.PROFESSIONAL,
        cross_domain_variants={
            Domain.INDIVIDUAL: "Personal finance variant",
        },
        debiasing_prompts=["Consider the actual market value"],
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_result(sample_instance):
    """Create a sample test result."""
    return TestResult(
        instance=sample_instance,
        model_id="test-model",
        condition="treatment_moderate",
        prompt_used="Test prompt",
        model_response="I think $550,000 is fair.",
        extracted_answer="$550,000",
        response_time_ms=150.5,
        confidence_stated=0.85,
        is_biased=True,
        bias_score=0.6,
        metadata={"trial": 1},
    )


@pytest.fixture
def sample_session(sample_instance, sample_result):
    """Create a sample evaluation session."""
    return EvaluationSession(
        session_id="test-session-001",
        model_id="test-model",
        model_config={"temperature": 0.7},
        start_time="2024-01-01T10:00:00",
        end_time="2024-01-01T10:30:00",
        test_instances=[sample_instance],
        results=[sample_result],
        metrics={"overall_bias": 0.45},
    )


@pytest.fixture
def sample_fingerprint():
    """Create a sample fingerprint report."""
    return CognitiveFingerprintReport(
        model_id="test-model",
        biases_tested=["anchoring_effect", "loss_aversion"],
        magnitude_scores={
            "anchoring_effect": BiasMagnitudeScore(
                bias_id="anchoring_effect",
                control_score=0.2,
                treatment_scores={TriggerIntensity.MODERATE: 0.7},
                overall_magnitude=0.5,
                intensity_sensitivity=0.3,
            ),
        },
        consistency_indices={
            "anchoring_effect": BiasConsistencyIndex(
                bias_id="anchoring_effect",
                domain_scores={Domain.PROFESSIONAL: 0.6},
                overall_consistency=0.6,
                standard_deviation=0.1,
                is_systematic=True,
            ),
        },
        mitigation_potentials={
            "anchoring_effect": BiasMitigationPotential(
                bias_id="anchoring_effect",
                baseline_bias_score=0.6,
                debiased_scores={"consider_alternatives": 0.3},
                best_mitigation_method="consider_alternatives",
                mitigation_effectiveness=0.5,
                requires_explicit_warning=False,
            ),
        },
        human_alignments={
            "anchoring_effect": HumanAlignmentScore(
                bias_id="anchoring_effect",
                model_bias_rate=0.65,
                human_baseline_rate=0.65,
                alignment_score=0.9,
                bias_direction="matches_human",
            ),
        },
        response_consistencies={
            "anchoring_effect": ResponseConsistencyIndex(
                bias_id="anchoring_effect",
                mean_response=0.5,
                variance=0.05,
                consistency_score=0.9,
                is_stable=True,
                trial_count=5,
            ),
        },
        calibration_scores={
            "anchoring_effect": CalibrationAwarenessScore(
                bias_id="anchoring_effect",
                mean_confidence=0.75,
                actual_accuracy=0.7,
                calibration_error=0.05,
                awareness_score=0.8,
                overconfident=False,
                metacognitive_gap=0.05,
            ),
        },
        overall_bias_susceptibility=0.55,
        most_susceptible_biases=["anchoring_effect"],
        most_resistant_biases=["loss_aversion"],
        human_like_biases=["anchoring_effect"],
        ai_specific_biases=[],
    )


class TestInstanceExportImport:
    """Tests for instance export/import."""

    def test_export_import_roundtrip(self, sample_instance):
        """Test that export and import are inverse operations."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Export
            export_instances_to_json([sample_instance], filepath)

            # Import
            imported = import_instances_from_json(filepath)

            assert len(imported) == 1
            inst = imported[0]
            assert inst.bias_id == sample_instance.bias_id
            assert inst.base_scenario == sample_instance.base_scenario
            assert inst.domain == sample_instance.domain
            assert TriggerIntensity.MODERATE in inst.treatment_prompts
        finally:
            filepath.unlink()

    def test_export_preserves_all_fields(self, sample_instance):
        """Test that export includes all instance fields."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            export_instances_to_json([sample_instance], filepath)

            with open(filepath) as f:
                data = json.load(f)

            assert len(data) == 1
            item = data[0]
            assert "bias_id" in item
            assert "treatment_prompts" in item
            assert "cross_domain_variants" in item
            assert "debiasing_prompts" in item
            assert "metadata" in item
        finally:
            filepath.unlink()

    def test_import_handles_missing_optional_fields(self):
        """Test that import handles minimal data."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{
                "bias_id": "test_bias",
                "base_scenario": "Test scenario",
                "bias_trigger": "Test trigger",
                "control_prompt": "Control",
                "treatment_prompts": {"moderate": "Treatment"},
                "domain": "individual",
            }], f)
            filepath = Path(f.name)

        try:
            imported = import_instances_from_json(filepath)
            assert len(imported) == 1
            assert imported[0].bias_id == "test_bias"
        finally:
            filepath.unlink()


class TestResultExport:
    """Tests for result export."""

    def test_export_results_to_json(self, sample_result):
        """Test JSON export of results."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            export_results_to_json([sample_result], filepath)

            with open(filepath) as f:
                data = json.load(f)

            assert len(data) == 1
            result = data[0]
            assert result["bias_id"] == "anchoring_effect"
            assert result["model_id"] == "test-model"
            assert result["is_biased"] is True
            assert result["bias_score"] == 0.6
        finally:
            filepath.unlink()

    def test_export_results_to_csv(self, sample_result):
        """Test CSV export of results."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filepath = Path(f.name)

        try:
            export_results_to_csv([sample_result], filepath)

            # Read and verify
            with open(filepath) as f:
                content = f.read()

            assert "bias_id" in content
            assert "anchoring_effect" in content
            assert "test-model" in content
        finally:
            filepath.unlink()


class TestSessionExport:
    """Tests for session export."""

    def test_export_session_with_prompts(self, sample_session):
        """Test session export including prompts."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            export_session_to_json(sample_session, filepath, include_prompts=True)

            with open(filepath) as f:
                data = json.load(f)

            assert data["session_id"] == "test-session-001"
            assert data["model_id"] == "test-model"
            assert "results" in data
            assert data["results"][0]["prompt_used"] == "Test prompt"
        finally:
            filepath.unlink()

    def test_export_session_without_prompts(self, sample_session):
        """Test session export excluding prompts."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            export_session_to_json(sample_session, filepath, include_prompts=False)

            with open(filepath) as f:
                data = json.load(f)

            # Should not have prompt fields
            assert "prompt_used" not in data["results"][0]
        finally:
            filepath.unlink()


class TestFingerprintExport:
    """Tests for fingerprint report export."""

    def test_export_fingerprint_to_json(self, sample_fingerprint):
        """Test JSON export of fingerprint."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            export_fingerprint_to_json(sample_fingerprint, filepath)

            with open(filepath) as f:
                data = json.load(f)

            assert data["model_id"] == "test-model"
            assert "magnitude_scores" in data
            assert "consistency_indices" in data
            assert "human_alignments" in data
            assert "generated_at" in data
        finally:
            filepath.unlink()

    def test_export_fingerprint_to_csv(self, sample_fingerprint):
        """Test CSV export of fingerprint."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filepath = Path(f.name)

        try:
            export_fingerprint_to_csv(sample_fingerprint, filepath)

            with open(filepath) as f:
                content = f.read()

            assert "model_id" in content
            assert "bias_id" in content
            assert "anchoring_effect" in content
        finally:
            filepath.unlink()


class TestSummaryReport:
    """Tests for summary report generation."""

    def test_generate_summary_report(self, sample_fingerprint):
        """Test human-readable summary generation."""
        report = generate_summary_report(sample_fingerprint)

        assert "KAHNE-BENCH" in report
        assert "test-model" in report
        assert "Bias Susceptibility" in report
        assert "anchoring_effect" in report

    def test_report_includes_sections(self, sample_fingerprint):
        """Test that report has all expected sections."""
        report = generate_summary_report(sample_fingerprint)

        assert "OVERALL SUMMARY" in report
        assert "HUMAN ALIGNMENT" in report
        assert "CALIBRATION AWARENESS" in report
