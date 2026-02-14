"""Tests for CLI commands: tier enforcement, intensity policy, canonical export, provenance."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from kahne_bench.cli import main
from kahne_bench.core import CognitiveBiasInstance, Domain, TriggerIntensity
from kahne_bench.engines.generator import get_tier_biases
from kahne_bench.utils.io import export_instances_to_json, import_instances_from_json

# Common args for evaluate that avoid needing real API keys
MOCK_EVAL_ARGS = ["--judge-provider", "mock"]


@pytest.fixture
def runner():
    return CliRunner()


def _make_instances(bias_ids: list[str]) -> list[CognitiveBiasInstance]:
    """Create minimal valid instances for the given bias IDs."""
    instances = []
    for bias_id in bias_ids:
        instances.append(CognitiveBiasInstance(
            bias_id=bias_id,
            base_scenario="Test scenario",
            bias_trigger="Test trigger",
            control_prompt="Control prompt for testing",
            treatment_prompts={
                TriggerIntensity.WEAK: "Weak treatment",
                TriggerIntensity.MODERATE: "Moderate treatment",
                TriggerIntensity.STRONG: "Strong treatment",
            },
            expected_rational_response="Rational response",
            expected_biased_response="Biased response",
            domain=Domain.PROFESSIONAL,
            debiasing_prompts=["Consider alternatives"],
        ))
    return instances


def _write_instances_file(bias_ids: list[str]) -> Path:
    """Create a temp JSON file with instances for given bias IDs."""
    instances = _make_instances(bias_ids)
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    filepath = Path(f.name)
    f.close()
    export_instances_to_json(instances, filepath)
    return filepath


class TestTierEnforcement:
    """PP-002: Tier consistency enforcement tests."""

    def test_mismatched_tier_causes_nonzero_exit(self, runner):
        """Input with wrong biases for tier should fail with non-zero exit."""
        filepath = _write_instances_file(["some_fake_bias", "another_fake_bias"])
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code != 0
            assert "Tier mismatch" in result.output
        finally:
            filepath.unlink()

    def test_mismatched_tier_shows_missing_biases(self, runner):
        """Error message should list missing biases."""
        filepath = _write_instances_file(["anchoring_effect", "loss_aversion"])
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code != 0
            assert "Missing biases" in result.output
        finally:
            filepath.unlink()

    def test_mismatched_tier_shows_extra_biases(self, runner):
        """Error message should list extra biases not in tier."""
        core_biases = get_tier_biases("core")
        extra_biases = core_biases + ["totally_unknown_bias"]
        filepath = _write_instances_file(extra_biases)
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code != 0
            assert "Extra biases" in result.output
        finally:
            filepath.unlink()

    def test_allow_tier_mismatch_flag_proceeds(self, runner):
        """--allow-tier-mismatch should allow running despite mismatch."""
        filepath = _write_instances_file(["anchoring_effect"])
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        fppath = Path(fpfile.name)
        outfile.close()
        fpfile.close()
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                "--allow-tier-mismatch",
                "-o", str(outpath),
                "-f", str(fppath),
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"
            assert "Proceeding with --allow-tier-mismatch" in result.output
        finally:
            filepath.unlink()
            outpath.unlink(missing_ok=True)
            fppath.unlink(missing_ok=True)

    def test_matching_tier_succeeds(self, runner):
        """Input that matches core tier exactly should succeed."""
        core_biases = get_tier_biases("core")
        filepath = _write_instances_file(core_biases)
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        fppath = Path(fpfile.name)
        outfile.close()
        fpfile.close()
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                "-o", str(outpath),
                "-f", str(fppath),
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"
            assert "Tier mismatch" not in result.output
        finally:
            filepath.unlink()
            outpath.unlink(missing_ok=True)
            fppath.unlink(missing_ok=True)


class TestIntensityPolicy:
    """PP-009: 3 vs 4 intensity policy tests."""

    def test_default_uses_three_intensities(self, runner):
        """Default evaluate should report 3 intensities (no ADVERSARIAL)."""
        core_biases = get_tier_biases("core")
        filepath = _write_instances_file(core_biases)
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        fppath = Path(fpfile.name)
        outfile.close()
        fpfile.close()
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                "-o", str(outpath),
                "-f", str(fppath),
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"
            assert "weak, moderate, strong" in result.output
            # Verify ADVERSARIAL is NOT on the Intensities line
            for line in result.output.split("\n"):
                if "Intensities:" in line:
                    assert "adversarial" not in line.lower()
                    break
        finally:
            filepath.unlink()
            outpath.unlink(missing_ok=True)
            fppath.unlink(missing_ok=True)

    def test_include_adversarial_adds_fourth(self, runner):
        """--include-adversarial should report 4 intensities."""
        core_biases = get_tier_biases("core")
        filepath = _write_instances_file(core_biases)
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        fppath = Path(fpfile.name)
        outfile.close()
        fpfile.close()
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                "--include-adversarial",
                "-o", str(outpath),
                "-f", str(fppath),
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"
            assert "adversarial" in result.output.lower()
        finally:
            filepath.unlink()
            outpath.unlink(missing_ok=True)
            fppath.unlink(missing_ok=True)

    def test_intensity_recorded_in_output_metadata(self, runner):
        """Intensity list should be persisted in output metadata."""
        core_biases = get_tier_biases("core")
        filepath = _write_instances_file(core_biases)
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        fppath = Path(fpfile.name)
        outfile.close()
        fpfile.close()
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                "-o", str(outpath),
                "-f", str(fppath),
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"

            with open(fppath) as f:
                fp_data = json.load(f)
            assert "run_metadata" in fp_data
            assert fp_data["run_metadata"]["intensities"] == ["weak", "moderate", "strong"]
        finally:
            filepath.unlink()
            outpath.unlink(missing_ok=True)
            fppath.unlink(missing_ok=True)


class TestCanonicalExport:
    """PP-010: Generate command uses canonical IO serializer."""

    def test_generate_includes_all_canonical_fields(self, runner):
        """generate command output should include cross_domain_variants and interaction_biases."""
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        outfile.close()
        try:
            result = runner.invoke(main, [
                "generate",
                "--bias", "anchoring_effect",
                "--domain", "professional",
                "--instances", "1",
                "--output", str(outpath),
                "--seed", "42",
            ])
            assert result.exit_code == 0

            with open(outpath) as f:
                data = json.load(f)

            assert len(data) >= 1
            item = data[0]
            # Canonical fields that generator.export_to_json was missing
            assert "cross_domain_variants" in item
            assert "interaction_biases" in item
            assert "debiasing_prompts" in item
            assert "metadata" in item
            assert "bias_id" in item
            assert "base_scenario" in item
        finally:
            outpath.unlink(missing_ok=True)

    def test_generate_roundtrip(self, runner):
        """generate -> export -> import -> verify no schema loss."""
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        outfile.close()
        try:
            result = runner.invoke(main, [
                "generate",
                "--bias", "anchoring_effect",
                "--domain", "professional",
                "--instances", "1",
                "--output", str(outpath),
                "--seed", "42",
            ])
            assert result.exit_code == 0

            # Round-trip: import the generated output
            imported = import_instances_from_json(outpath)
            assert len(imported) >= 1
            inst = imported[0]
            assert inst.bias_id == "anchoring_effect"
            assert inst.domain == Domain.PROFESSIONAL
            assert len(inst.treatment_prompts) > 0
        finally:
            outpath.unlink(missing_ok=True)


class TestRunProvenance:
    """PP-004: Run provenance metadata in outputs."""

    REQUIRED_METADATA_KEYS = [
        "provider", "model", "judge_provider", "judge_model",
        "temperature", "max_tokens", "num_trials", "max_concurrent_requests",
        "rate_limit_retries", "rate_limit_retry_delay_s",
        "intensities", "include_control", "include_debiasing",
        "tier", "input_file", "bias_manifest", "bias_manifest_hash", "instance_count_by_bias",
        "git_commit", "timestamp", "python_version", "kahne_bench_version",
    ]

    def test_results_json_contains_provenance(self, runner):
        """results.json should contain run_metadata with all required keys."""
        core_biases = get_tier_biases("core")
        filepath = _write_instances_file(core_biases)
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        fppath = Path(fpfile.name)
        outfile.close()
        fpfile.close()
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                "-o", str(outpath),
                "-f", str(fppath),
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"

            with open(outpath) as f:
                data = json.load(f)

            assert "run_metadata" in data
            meta = data["run_metadata"]
            for key in self.REQUIRED_METADATA_KEYS:
                assert key in meta, f"Missing metadata key: {key}"

            # Verify specific values
            assert meta["provider"] == "mock"
            assert meta["tier"] == "core"
            assert meta["num_trials"] == 3
            assert meta["rate_limit_retries"] == 1
            assert meta["rate_limit_retry_delay_s"] == 5.0
            assert meta["input_file"] == str(filepath)
            assert isinstance(meta["bias_manifest"], list)
            assert len(meta["bias_manifest"]) == len(core_biases)
            assert isinstance(meta["timestamp"], str)
        finally:
            filepath.unlink()
            outpath.unlink(missing_ok=True)
            fppath.unlink(missing_ok=True)

    def test_fingerprint_json_contains_provenance(self, runner):
        """fingerprint.json should contain run_metadata with all required keys."""
        core_biases = get_tier_biases("core")
        filepath = _write_instances_file(core_biases)
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        fppath = Path(fpfile.name)
        outfile.close()
        fpfile.close()
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                "-o", str(outpath),
                "-f", str(fppath),
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"

            with open(fppath) as f:
                data = json.load(f)

            assert "run_metadata" in data
            meta = data["run_metadata"]
            for key in self.REQUIRED_METADATA_KEYS:
                assert key in meta, f"Missing metadata key: {key}"
        finally:
            filepath.unlink()
            outpath.unlink(missing_ok=True)
            fppath.unlink(missing_ok=True)

    def test_provenance_includes_correct_instance_counts(self, runner):
        """instance_count_by_bias should accurately reflect input instances."""
        core_biases = get_tier_biases("core")
        filepath = _write_instances_file(core_biases)
        fpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fppath = Path(fpfile.name)
        fpfile.close()
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        outfile.close()
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                "-o", str(outpath),
                "-f", str(fppath),
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"

            with open(fppath) as f:
                data = json.load(f)

            counts = data["run_metadata"]["instance_count_by_bias"]
            # Each bias should have exactly 1 instance (from _make_instances)
            for bias_id in core_biases:
                assert counts.get(bias_id) == 1, f"Expected 1 for {bias_id}"
        finally:
            filepath.unlink()
            outpath.unlink(missing_ok=True)
            fppath.unlink(missing_ok=True)

    def test_cli_passes_trial_settings_to_metric_calculator(self, runner, monkeypatch):
        """CLI evaluate should pass trials/temperature into calculate_all_metrics."""
        from kahne_bench.metrics import MetricCalculator

        captured = {}
        original = MetricCalculator.calculate_all_metrics

        def _spy(self, model_id, results, temperature=0.0, num_trials=3):
            captured["temperature"] = temperature
            captured["num_trials"] = num_trials
            return original(self, model_id, results, temperature=temperature, num_trials=num_trials)

        monkeypatch.setattr(MetricCalculator, "calculate_all_metrics", _spy)

        filepath = _write_instances_file(["anchoring_effect"])
        outfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        fpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        outpath = Path(outfile.name)
        fppath = Path(fpfile.name)
        outfile.close()
        fpfile.close()
        try:
            result = runner.invoke(main, [
                "evaluate",
                "-i", str(filepath),
                "-p", "mock",
                "-t", "core",
                "--allow-tier-mismatch",
                "--trials", "5",
                "-o", str(outpath),
                "-f", str(fppath),
                *MOCK_EVAL_ARGS,
            ])
            assert result.exit_code == 0, f"Output: {result.output}"
            assert captured["temperature"] == 0.0
            assert captured["num_trials"] == 5
        finally:
            filepath.unlink()
            outpath.unlink(missing_ok=True)
            fppath.unlink(missing_ok=True)


class TestExportBackwardsCompat:
    """Ensure io.py export functions remain backwards-compatible."""

    def test_export_results_without_metadata_returns_list(self):
        """export_results_to_json without metadata should produce a plain list."""
        from kahne_bench.utils.io import export_results_to_json
        from kahne_bench.core import TestResult

        inst = CognitiveBiasInstance(
            bias_id="test",
            base_scenario="s",
            bias_trigger="t",
            control_prompt="c",
            treatment_prompts={TriggerIntensity.MODERATE: "m"},
            expected_rational_response="r",
            expected_biased_response="b",
            domain=Domain.INDIVIDUAL,
        )
        result = TestResult(
            instance=inst,
            model_id="m",
            condition="control",
            prompt_used="p",
            model_response="resp",
            extracted_answer="a",
            response_time_ms=10.0,
            confidence_stated=None,
            is_biased=False,
            bias_score=0.0,
            metadata={},
        )

        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        filepath = Path(f.name)
        f.close()
        try:
            export_results_to_json([result], filepath)
            with open(filepath) as fh:
                data = json.load(fh)
            # Without metadata, result is a plain list (backwards compat)
            assert isinstance(data, list)
        finally:
            filepath.unlink()
