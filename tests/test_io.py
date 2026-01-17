"""Tests for data import/export utilities."""


import json
import tempfile
import warnings
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
                mean_bias_score=0.6,
                consistency_score=0.8,
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
                calibration_score=0.8,
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


class TestIOErrorHandling:
    """Tests for error handling in import/export operations."""

    def test_import_malformed_json(self):
        """Test import_instances_from_json with invalid JSON syntax raises JSONDecodeError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("{not valid json[")
            filepath = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                import_instances_from_json(filepath)
        finally:
            filepath.unlink()

    def test_import_missing_required_fields(self):
        """Test import with missing required fields (bias_id, control_prompt) raises KeyError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            # Missing bias_id and other required fields
            json.dump([{
                "base_scenario": "Test scenario",
                "domain": "individual",
            }], f)
            filepath = Path(f.name)

        try:
            with pytest.raises(KeyError):
                import_instances_from_json(filepath)
        finally:
            filepath.unlink()

    def test_import_invalid_domain_enum(self):
        """Test import with invalid Domain value emits warning and skips item."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{
                "bias_id": "test_bias",
                "base_scenario": "Test scenario",
                "bias_trigger": "Test trigger",
                "control_prompt": "Control prompt",
                "treatment_prompts": {"moderate": "Treatment"},
                "domain": "UNKNOWN_DOMAIN",  # Invalid domain
            }], f)
            filepath = Path(f.name)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                imported = import_instances_from_json(filepath)

                # Should skip the invalid item
                assert len(imported) == 0

                # Should have emitted a warning about invalid domain
                assert len(w) >= 1
                warning_messages = [str(warning.message) for warning in w]
                assert any("Domain" in msg or "UNKNOWN_DOMAIN" in msg for msg in warning_messages)
        finally:
            filepath.unlink()

    def test_import_invalid_trigger_intensity(self):
        """Test import with invalid TriggerIntensity value emits warning and skips intensity."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{
                "bias_id": "test_bias",
                "base_scenario": "Test scenario",
                "bias_trigger": "Test trigger",
                "control_prompt": "Control prompt",
                "treatment_prompts": {
                    "invalid_intensity": "Treatment with invalid intensity",
                    "moderate": "Valid treatment",
                },
                "domain": "individual",
            }], f)
            filepath = Path(f.name)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                imported = import_instances_from_json(filepath)

                # Should import the instance but skip the invalid intensity
                assert len(imported) == 1
                assert TriggerIntensity.MODERATE in imported[0].treatment_prompts
                # The invalid intensity should not be in the dict
                assert len(imported[0].treatment_prompts) == 1

                # Should have emitted a warning about invalid intensity
                warning_messages = [str(warning.message) for warning in w]
                assert any("TriggerIntensity" in msg or "invalid_intensity" in msg for msg in warning_messages)
        finally:
            filepath.unlink()

    def test_import_nonexistent_file(self):
        """Test import from file path that doesn't exist raises FileNotFoundError."""
        nonexistent_path = Path("/nonexistent/path/to/file.json")
        with pytest.raises(FileNotFoundError):
            import_instances_from_json(nonexistent_path)

    def test_export_results_with_none_values(self, sample_instance):
        """Test export when result fields contain None values."""
        result_with_nones = TestResult(
            instance=sample_instance,
            model_id="test-model",
            condition="control",
            prompt_used="Test prompt",
            model_response="Test response",
            extracted_answer="Answer",
            response_time_ms=100.0,
            confidence_stated=None,  # None value
            is_biased=None,  # None value
            bias_score=None,  # None value
            metadata={},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Should not raise an exception
            export_results_to_json([result_with_nones], filepath)

            # Verify the file was created and contains valid JSON
            with open(filepath) as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["confidence_stated"] is None
            assert data[0]["is_biased"] is None
            assert data[0]["bias_score"] is None
        finally:
            filepath.unlink()


class TestDataIntegrity:
    """Tests for data integrity during round-trip operations."""

    def test_instance_roundtrip_preserves_all_fields(self, sample_instance):
        """Test that export -> import roundtrip preserves all instance fields."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Export
            export_instances_to_json([sample_instance], filepath)

            # Import
            imported = import_instances_from_json(filepath)

            assert len(imported) == 1
            inst = imported[0]

            # Verify all fields are preserved
            assert inst.bias_id == sample_instance.bias_id
            assert inst.base_scenario == sample_instance.base_scenario
            assert inst.bias_trigger == sample_instance.bias_trigger
            assert inst.control_prompt == sample_instance.control_prompt
            assert inst.expected_rational_response == sample_instance.expected_rational_response
            assert inst.expected_biased_response == sample_instance.expected_biased_response
            assert inst.metadata == sample_instance.metadata
        finally:
            filepath.unlink()

    def test_instance_roundtrip_preserves_enums(self, sample_instance):
        """Test that Domain and TriggerIntensity enums survive roundtrip correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Export
            export_instances_to_json([sample_instance], filepath)

            # Import
            imported = import_instances_from_json(filepath)

            inst = imported[0]

            # Domain enum should be preserved
            assert inst.domain == sample_instance.domain
            assert isinstance(inst.domain, Domain)

            # TriggerIntensity enums in treatment_prompts should be preserved
            for intensity in sample_instance.treatment_prompts.keys():
                assert intensity in inst.treatment_prompts
                assert isinstance(intensity, TriggerIntensity)
                assert inst.treatment_prompts[intensity] == sample_instance.treatment_prompts[intensity]
        finally:
            filepath.unlink()

    def test_instance_roundtrip_with_debiasing(self):
        """Test roundtrip with debiasing_prompts included."""
        instance = CognitiveBiasInstance(
            bias_id="test_bias_with_debiasing",
            base_scenario="Test scenario",
            bias_trigger="Test trigger",
            control_prompt="Control prompt",
            treatment_prompts={
                TriggerIntensity.MODERATE: "Moderate treatment",
            },
            expected_rational_response="Rational response",
            expected_biased_response="Biased response",
            domain=Domain.INDIVIDUAL,
            debiasing_prompts=[
                "Consider the base rate probability",
                "Think about alternative explanations",
                "What would a statistician say?",
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Export
            export_instances_to_json([instance], filepath)

            # Import
            imported = import_instances_from_json(filepath)

            inst = imported[0]

            # Verify debiasing prompts are preserved
            assert inst.debiasing_prompts == instance.debiasing_prompts
            assert len(inst.debiasing_prompts) == 3
            assert "Consider the base rate probability" in inst.debiasing_prompts
        finally:
            filepath.unlink()

    def test_instance_roundtrip_with_context_variants(self):
        """Test roundtrip with cross_domain_variants included."""
        instance = CognitiveBiasInstance(
            bias_id="test_bias_with_variants",
            base_scenario="Test scenario",
            bias_trigger="Test trigger",
            control_prompt="Control prompt",
            treatment_prompts={
                TriggerIntensity.MODERATE: "Moderate treatment",
            },
            expected_rational_response="Rational response",
            expected_biased_response="Biased response",
            domain=Domain.PROFESSIONAL,
            cross_domain_variants={
                Domain.INDIVIDUAL: "Individual variant for personal finance",
                Domain.SOCIAL: "Social variant for negotiation context",
                Domain.RISK: "Risk variant for policy decisions",
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Export
            export_instances_to_json([instance], filepath)

            # Import
            imported = import_instances_from_json(filepath)

            inst = imported[0]

            # Verify cross-domain variants are preserved
            assert len(inst.cross_domain_variants) == 3
            assert Domain.INDIVIDUAL in inst.cross_domain_variants
            assert Domain.SOCIAL in inst.cross_domain_variants
            assert Domain.RISK in inst.cross_domain_variants
            assert inst.cross_domain_variants[Domain.INDIVIDUAL] == "Individual variant for personal finance"
        finally:
            filepath.unlink()


class TestFingerprintExportExtended:
    """Extended tests for fingerprint export edge cases."""

    def test_fingerprint_csv_has_expected_columns(self, sample_fingerprint):
        """Verify CSV structure has all expected metric columns."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filepath = Path(f.name)

        try:
            export_fingerprint_to_csv(sample_fingerprint, filepath)

            with open(filepath) as f:
                content = f.read()
                header_line = content.split('\n')[0]

            # Verify all expected columns are present
            expected_columns = [
                "model_id",
                "bias_id",
                "magnitude_overall",
                "magnitude_sensitivity",
                "mean_bias_score",
                "consistency_score",
                "is_systematic",
                "mitigation_effectiveness",
                "best_mitigation_method",
                "human_alignment",
                "bias_direction",
                "response_consistency",
                "is_stable",
                "calibration_score",
                "is_overconfident",
            ]

            for column in expected_columns:
                assert column in header_line, f"Expected column '{column}' not found in CSV header"
        finally:
            filepath.unlink()

    def test_fingerprint_with_empty_metrics(self):
        """Test edge case: fingerprint with empty metric dictionaries."""
        empty_fingerprint = CognitiveFingerprintReport(
            model_id="empty-metrics-model",
            biases_tested=["test_bias_1", "test_bias_2"],
            magnitude_scores={},  # Empty
            consistency_indices={},  # Empty
            mitigation_potentials={},  # Empty
            human_alignments={},  # Empty
            response_consistencies={},  # Empty
            calibration_scores={},  # Empty
            overall_bias_susceptibility=0.0,
            most_susceptible_biases=[],
            most_resistant_biases=[],
            human_like_biases=[],
            ai_specific_biases=[],
        )

        # Test JSON export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            # JSON export should work without errors
            export_fingerprint_to_json(empty_fingerprint, json_path)

            with open(json_path) as f:
                data = json.load(f)

            assert data["model_id"] == "empty-metrics-model"
            assert data["magnitude_scores"] == {}
            assert data["consistency_indices"] == {}

            # CSV export should work without errors
            export_fingerprint_to_csv(empty_fingerprint, csv_path)

            with open(csv_path) as f:
                content = f.read()

            # Should have header and two data rows (for two biases_tested)
            lines = [line for line in content.strip().split('\n') if line]
            assert len(lines) == 3  # header + 2 bias rows
        finally:
            json_path.unlink()
            csv_path.unlink()

    def test_summary_report_with_minimal_data(self):
        """Test generate_summary_report with minimal fingerprint data."""
        minimal_fingerprint = CognitiveFingerprintReport(
            model_id="minimal-model",
            biases_tested=[],  # No biases tested
            magnitude_scores={},
            consistency_indices={},
            mitigation_potentials={},
            human_alignments={},
            response_consistencies={},
            calibration_scores={},
            overall_bias_susceptibility=0.0,
            most_susceptible_biases=[],
            most_resistant_biases=[],
            human_like_biases=[],
            ai_specific_biases=[],
        )

        # Should not raise an exception
        report = generate_summary_report(minimal_fingerprint)

        # Verify basic structure is present
        assert "KAHNE-BENCH" in report
        assert "minimal-model" in report
        assert "OVERALL SUMMARY" in report
        assert "Biases Tested: 0" in report
        assert "Bias Susceptibility: 0.00%" in report


class TestSessionExportExtended:
    """Extended tests for session export edge cases."""

    def test_session_export_includes_metadata(self, sample_session):
        """Verify model_config, start_time, end_time are included in export."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            export_session_to_json(sample_session, filepath, include_prompts=True)

            with open(filepath) as f:
                data = json.load(f)

            # Verify metadata fields are present
            assert "model_config" in data
            assert data["model_config"] == {"temperature": 0.7}

            assert "start_time" in data
            assert data["start_time"] == "2024-01-01T10:00:00"

            assert "end_time" in data
            assert data["end_time"] == "2024-01-01T10:30:00"

            # Verify session-level metadata
            assert "session_id" in data
            assert "model_id" in data
            assert "num_instances" in data
            assert "num_results" in data
            assert "metrics" in data
        finally:
            filepath.unlink()

    def test_session_export_with_empty_results(self, sample_instance):
        """Test edge case: session with no results."""
        empty_session = EvaluationSession(
            session_id="empty-session-001",
            model_id="test-model",
            model_config={"temperature": 0.5},
            start_time="2024-01-01T09:00:00",
            end_time="2024-01-01T09:00:01",
            test_instances=[sample_instance],
            results=[],  # No results
            metrics={},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Should not raise an exception
            export_session_to_json(empty_session, filepath, include_prompts=True)

            with open(filepath) as f:
                data = json.load(f)

            assert data["session_id"] == "empty-session-001"
            assert data["num_instances"] == 1
            assert data["num_results"] == 0
            assert data["results"] == []
            assert data["metrics"] == {}
        finally:
            filepath.unlink()
