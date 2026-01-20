"""
Data import/export utilities for Kahne-Bench.

Supports JSON and CSV formats for test instances, results, and metrics.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import pandas as pd

from kahne_bench.core import (
    CognitiveBiasInstance,
    Domain,
    EvaluationSession,
    TestResult,
    TestScale,
    TriggerIntensity,
)
from kahne_bench.metrics.core import CognitiveFingerprintReport

# Type variable for enum parsing
_E = TypeVar("_E")


def _safe_parse_enum(value: str, enum_class: type[_E], context: str = "") -> _E | None:
    """
    Safely parse an enum value, returning None and warning on invalid values.

    Args:
        value: String value to parse
        enum_class: Enum class to parse into
        context: Additional context for warning message

    Returns:
        Parsed enum value or None if invalid
    """
    try:
        return enum_class(value)
    except ValueError:
        ctx = f" in {context}" if context else ""
        warnings.warn(
            f"Unknown {enum_class.__name__} '{value}'{ctx}, skipping",
            stacklevel=3
        )
        return None


def export_instances_to_json(
    instances: list[CognitiveBiasInstance],
    filepath: str | Path,
    indent: int = 2,
) -> None:
    """
    Export test instances to JSON file.

    Args:
        instances: List of test instances to export
        filepath: Output file path
        indent: JSON indentation level
    """
    data = []
    for inst in instances:
        data.append({
            "bias_id": inst.bias_id,
            "base_scenario": inst.base_scenario,
            "bias_trigger": inst.bias_trigger,
            "control_prompt": inst.control_prompt,
            "treatment_prompts": {
                k.value: v for k, v in inst.treatment_prompts.items()
            },
            "expected_rational_response": inst.expected_rational_response,
            "expected_biased_response": inst.expected_biased_response,
            "domain": inst.domain.value,
            "scale": inst.scale.value,
            "cross_domain_variants": {
                k.value: v for k, v in inst.cross_domain_variants.items()
            },
            "debiasing_prompts": inst.debiasing_prompts,
            "interaction_biases": inst.interaction_biases,
            "metadata": inst.metadata,
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def import_instances_from_json(
    filepath: str | Path,
) -> list[CognitiveBiasInstance]:
    """
    Import test instances from JSON file.

    Args:
        filepath: Input file path

    Returns:
        List of CognitiveBiasInstance objects
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = []
    for idx, item in enumerate(data):
        # Convert string keys back to enums with validation
        treatment_prompts = {}
        for k, v in item.get("treatment_prompts", {}).items():
            intensity = _safe_parse_enum(k, TriggerIntensity, f"treatment_prompts (item {idx})")
            if intensity is not None:
                treatment_prompts[intensity] = v

        cross_domain_variants = {}
        for k, v in item.get("cross_domain_variants", {}).items():
            domain = _safe_parse_enum(k, Domain, f"cross_domain_variants (item {idx})")
            if domain is not None:
                cross_domain_variants[domain] = v

        # Parse required domain with fallback
        domain_str = item.get("domain", "individual")
        domain = _safe_parse_enum(domain_str, Domain, f"domain (item {idx})")
        if domain is None:
            warnings.warn(f"Skipping item {idx} due to invalid domain '{domain_str}'")
            continue

        # Parse optional scale with default
        scale_str = item.get("scale", "micro")
        scale = _safe_parse_enum(scale_str, TestScale, f"scale (item {idx})")
        if scale is None:
            scale = TestScale.MICRO  # Default fallback

        instance = CognitiveBiasInstance(
            bias_id=item["bias_id"],
            base_scenario=item["base_scenario"],
            bias_trigger=item["bias_trigger"],
            control_prompt=item["control_prompt"],
            treatment_prompts=treatment_prompts,
            expected_rational_response=item.get("expected_rational_response", ""),
            expected_biased_response=item.get("expected_biased_response", ""),
            domain=domain,
            scale=scale,
            cross_domain_variants=cross_domain_variants,
            debiasing_prompts=item.get("debiasing_prompts", []),
            interaction_biases=item.get("interaction_biases", []),
            metadata=item.get("metadata", {}),
        )
        instances.append(instance)

    return instances


def export_results_to_json(
    results: list[TestResult],
    filepath: str | Path,
    indent: int = 2,
) -> None:
    """
    Export test results to JSON file.

    Args:
        results: List of test results to export
        filepath: Output file path
        indent: JSON indentation level
    """
    data = []
    for result in results:
        data.append({
            "bias_id": result.instance.bias_id,
            "model_id": result.model_id,
            "condition": result.condition,
            "prompt_used": result.prompt_used,
            "model_response": result.model_response,
            "extracted_answer": result.extracted_answer,
            "response_time_ms": result.response_time_ms,
            "confidence_stated": result.confidence_stated,
            "is_biased": result.is_biased,
            "bias_score": result.bias_score,
            "domain": result.instance.domain.value,
            "metadata": result.metadata,
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def export_results_to_csv(
    results: list[TestResult],
    filepath: str | Path,
) -> None:
    """
    Export test results to CSV file.

    Args:
        results: List of test results to export
        filepath: Output file path
    """
    rows = []
    for result in results:
        rows.append({
            "bias_id": result.instance.bias_id,
            "model_id": result.model_id,
            "condition": result.condition,
            "extracted_answer": result.extracted_answer,
            "response_time_ms": result.response_time_ms,
            "confidence_stated": result.confidence_stated,
            "is_biased": result.is_biased,
            "bias_score": result.bias_score,
            "domain": result.instance.domain.value,
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)


def export_session_to_json(
    session: EvaluationSession,
    filepath: str | Path,
    include_prompts: bool = True,
    indent: int = 2,
) -> None:
    """
    Export a complete evaluation session to JSON.

    Args:
        session: The evaluation session to export
        filepath: Output file path
        include_prompts: Whether to include full prompt texts
        indent: JSON indentation level
    """
    results_data = []
    for result in session.results:
        result_dict = {
            "bias_id": result.instance.bias_id,
            "model_id": result.model_id,
            "condition": result.condition,
            "extracted_answer": result.extracted_answer,
            "response_time_ms": result.response_time_ms,
            "confidence_stated": result.confidence_stated,
            "is_biased": result.is_biased,
            "bias_score": result.bias_score,
            "domain": result.instance.domain.value,
            "metadata": result.metadata,
        }
        if include_prompts:
            result_dict["prompt_used"] = result.prompt_used
            result_dict["model_response"] = result.model_response
        results_data.append(result_dict)

    data = {
        "session_id": session.session_id,
        "model_id": session.model_id,
        "model_config": session.model_config,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "num_instances": len(session.test_instances),
        "num_results": len(session.results),
        "metrics": session.metrics,
        "results": results_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def export_fingerprint_to_json(
    report: CognitiveFingerprintReport,
    filepath: str | Path,
    indent: int = 2,
) -> None:
    """
    Export a cognitive fingerprint report to JSON.

    Args:
        report: The fingerprint report to export
        filepath: Output file path
        indent: JSON indentation level
    """
    data = {
        "model_id": report.model_id,
        "biases_tested": report.biases_tested,
        "summary": {
            "overall_bias_susceptibility": report.overall_bias_susceptibility,
            "most_susceptible_biases": report.most_susceptible_biases,
            "most_resistant_biases": report.most_resistant_biases,
            "human_like_biases": report.human_like_biases,
            "ai_specific_biases": report.ai_specific_biases,
        },
        "unknown_rates_by_bias": report.unknown_rates_by_bias,
        "magnitude_scores": {
            bias_id: {
                "control_score": bms.control_score,
                "treatment_scores": {k.value: v for k, v in bms.treatment_scores.items()},
                "overall_magnitude": bms.overall_magnitude,
                "intensity_sensitivity": bms.intensity_sensitivity,
                "unknown_rate": bms.unknown_rate,
            }
            for bias_id, bms in report.magnitude_scores.items()
        },
        "consistency_indices": {
            bias_id: {
                "domain_scores": {k.value: v for k, v in bci.domain_scores.items()},
                "mean_bias_score": bci.mean_bias_score,
                "consistency_score": bci.consistency_score,
                "standard_deviation": bci.standard_deviation,
                "is_systematic": bci.is_systematic,
                "unknown_rate": bci.unknown_rate,
            }
            for bias_id, bci in report.consistency_indices.items()
        },
        "mitigation_potentials": {
            bias_id: {
                "baseline_bias_score": bmp.baseline_bias_score,
                "debiased_scores": bmp.debiased_scores,
                "best_mitigation_method": bmp.best_mitigation_method,
                "mitigation_effectiveness": bmp.mitigation_effectiveness,
                "requires_explicit_warning": bmp.requires_explicit_warning,
                "unknown_rate": bmp.unknown_rate,
            }
            for bias_id, bmp in report.mitigation_potentials.items()
        },
        "human_alignments": {
            bias_id: {
                "model_bias_rate": has.model_bias_rate,
                "human_baseline_rate": has.human_baseline_rate,
                "alignment_score": has.alignment_score,
                "bias_direction": has.bias_direction,
                "unknown_rate": has.unknown_rate,
            }
            for bias_id, has in report.human_alignments.items()
        },
        "response_consistencies": {
            bias_id: {
                "mean_response": rci.mean_response,
                "variance": rci.variance,
                "consistency_score": rci.consistency_score,
                "is_stable": rci.is_stable,
                "trial_count": rci.trial_count,
                "unknown_rate": rci.unknown_rate,
            }
            for bias_id, rci in report.response_consistencies.items()
        },
        "calibration_scores": {
            bias_id: {
                "mean_confidence": cas.mean_confidence,
                "actual_accuracy": cas.actual_accuracy,
                "calibration_error": cas.calibration_error,
                "calibration_score": cas.calibration_score,
                "overconfident": cas.overconfident,
                "metacognitive_gap": cas.metacognitive_gap,
                "unknown_rate": cas.unknown_rate,
            }
            for bias_id, cas in report.calibration_scores.items()
        },
        "generated_at": datetime.now().isoformat(),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def export_fingerprint_to_csv(
    report: CognitiveFingerprintReport,
    filepath: str | Path,
) -> None:
    """
    Export a cognitive fingerprint summary to CSV.

    Args:
        report: The fingerprint report to export
        filepath: Output file path
    """
    rows = []
    for bias_id in report.biases_tested:
        row = {
            "model_id": report.model_id,
            "bias_id": bias_id,
        }

        # Magnitude
        if bias_id in report.magnitude_scores:
            bms = report.magnitude_scores[bias_id]
            row["magnitude_overall"] = bms.overall_magnitude
            row["magnitude_sensitivity"] = bms.intensity_sensitivity

        # Consistency
        if bias_id in report.consistency_indices:
            bci = report.consistency_indices[bias_id]
            row["mean_bias_score"] = bci.mean_bias_score
            row["consistency_score"] = bci.consistency_score
            row["is_systematic"] = bci.is_systematic

        # Mitigation
        if bias_id in report.mitigation_potentials:
            bmp = report.mitigation_potentials[bias_id]
            row["mitigation_effectiveness"] = bmp.mitigation_effectiveness
            row["best_mitigation_method"] = bmp.best_mitigation_method

        # Human alignment
        if bias_id in report.human_alignments:
            has = report.human_alignments[bias_id]
            row["human_alignment"] = has.alignment_score
            row["bias_direction"] = has.bias_direction

        # Response consistency
        if bias_id in report.response_consistencies:
            rci = report.response_consistencies[bias_id]
            row["response_consistency"] = rci.consistency_score
            row["is_stable"] = rci.is_stable

        # Calibration
        if bias_id in report.calibration_scores:
            cas = report.calibration_scores[bias_id]
            row["calibration_score"] = cas.calibration_score
            row["is_overconfident"] = cas.overconfident

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)


def generate_summary_report(
    report: CognitiveFingerprintReport,
) -> str:
    """
    Generate a human-readable summary report.

    Args:
        report: The fingerprint report

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 60,
        "KAHNE-BENCH COGNITIVE FINGERPRINT REPORT",
        "=" * 60,
        "",
        f"Model: {report.model_id}",
        f"Biases Tested: {len(report.biases_tested)}",
        "",
        "-" * 40,
        "OVERALL SUMMARY",
        "-" * 40,
        f"Bias Susceptibility: {report.overall_bias_susceptibility:.2%}",
        "",
        "Most Susceptible Biases:",
    ]

    for bias_id in report.most_susceptible_biases[:5]:
        if bias_id in report.magnitude_scores:
            mag = report.magnitude_scores[bias_id].overall_magnitude
            lines.append(f"  - {bias_id}: {mag:.3f}")

    lines.extend([
        "",
        "Most Resistant Biases:",
    ])

    for bias_id in report.most_resistant_biases[:5]:
        if bias_id in report.magnitude_scores:
            mag = report.magnitude_scores[bias_id].overall_magnitude
            lines.append(f"  - {bias_id}: {mag:.3f}")

    lines.extend([
        "",
        "-" * 40,
        "HUMAN ALIGNMENT",
        "-" * 40,
    ])

    if report.human_like_biases:
        lines.append("Human-Like Biases (high alignment):")
        for bias_id in report.human_like_biases[:5]:
            if bias_id in report.human_alignments:
                score = report.human_alignments[bias_id].alignment_score
                lines.append(f"  - {bias_id}: {score:.3f}")

    if report.ai_specific_biases:
        lines.append("\nAI-Specific Biases (deviates from human patterns):")
        for bias_id in report.ai_specific_biases[:5]:
            if bias_id in report.human_alignments:
                has = report.human_alignments[bias_id]
                lines.append(f"  - {bias_id}: {has.bias_direction}")

    lines.extend([
        "",
        "-" * 40,
        "CALIBRATION AWARENESS",
        "-" * 40,
    ])

    overconfident_biases = [
        bias_id for bias_id, cas in report.calibration_scores.items()
        if cas.overconfident
    ]

    if overconfident_biases:
        lines.append("Biases where model is overconfident:")
        for bias_id in overconfident_biases[:5]:
            cas = report.calibration_scores[bias_id]
            lines.append(f"  - {bias_id}: gap = {cas.metacognitive_gap:.2f}")
    else:
        lines.append("No significant overconfidence detected.")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
