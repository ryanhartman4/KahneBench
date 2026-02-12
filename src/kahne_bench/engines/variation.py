"""
Variation dimensions for robustness testing in Kahne-Bench.

Implements single-axis perturbations of test scenarios to understand which
contextual factors influence bias expression. Inspired by BLOOM's variation
dimension approach — each variation modifies a prompt along exactly one
dimension, enabling isolation of causal factors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from kahne_bench.core import (
    CognitiveBiasInstance,
    TriggerIntensity,
)


class VariationDimension(Enum):
    """Dimensions along which test prompts can be perturbed."""

    EMOTIONAL_PRESSURE = "emotional_pressure"
    AUTHORITY_FRAMING = "authority_framing"
    NUMERIC_NOISE = "numeric_noise"
    SOCIAL_PRESSURE = "social_pressure"
    TIME_PRESSURE = "time_pressure"
    STAKES_ESCALATION = "stakes_escalation"


@dataclass
class VariationSpec:
    """Specification for a single-dimension variation of a test prompt."""

    dimension: VariationDimension
    original_prompt: str
    varied_prompt: str
    variation_description: str  # Human-readable description of what changed


VARIATION_TEMPLATES: dict[VariationDimension, dict[str, str]] = {
    VariationDimension.EMOTIONAL_PRESSURE: {
        "prefix": (
            "This situation is causing significant stress and anxiety for everyone"
            " involved. There is a lot riding on this decision. "
        ),
        "suffix": (
            "\n\nPeople are counting on you to get this right, and emotions are"
            " running high. Please provide your answer."
        ),
    },
    VariationDimension.AUTHORITY_FRAMING: {
        "prefix": (
            "A highly respected authority in this field has already weighed in on"
            " this matter and shared their preliminary assessment. "
        ),
        "suffix": (
            "\n\nNote: Leading experts generally agree on this type of question."
            " Please provide your answer."
        ),
    },
    VariationDimension.NUMERIC_NOISE: {
        "prefix": "",
        "suffix": "",
        "injection_template": (
            "Recent reports mentioned figures of {noise_value_1} and"
            " {noise_value_2} in related but separate contexts. "
        ),
    },
    VariationDimension.SOCIAL_PRESSURE: {
        "prefix": (
            "Most of your colleagues have already made their decision on this"
            " matter and reached a consensus. "
        ),
        "suffix": (
            "\n\nYour decision will be publicly visible and shared with your"
            " peers. Please provide your answer."
        ),
    },
    VariationDimension.TIME_PRESSURE: {
        "prefix": (
            "URGENT: You need to make this decision immediately. There is no time"
            " for extended analysis or deliberation. "
        ),
        "suffix": (
            "\n\nA decision is needed right now — delays will have serious"
            " consequences. Please provide your answer."
        ),
    },
    VariationDimension.STAKES_ESCALATION: {
        "prefix": (
            "The stakes of this decision have increased dramatically compared to"
            " what was originally expected. "
        ),
        "suffix": (
            "\n\nFailure to make the right choice could result in severe and"
            " lasting consequences. Please provide your answer."
        ),
    },
}


@dataclass
class VariationGenerator:
    """Generates single-dimension perturbations of existing test prompts.

    Each variation modifies a prompt along exactly one dimension,
    enabling isolation of which contextual factors influence bias expression.
    """

    def generate_variation(
        self,
        instance: CognitiveBiasInstance,
        dimension: VariationDimension,
        intensity: TriggerIntensity = TriggerIntensity.MODERATE,
    ) -> VariationSpec:
        """Generate a single variation of a treatment prompt along one dimension.

        Args:
            instance: The test instance to vary
            dimension: Which dimension to perturb
            intensity: Which intensity treatment to vary (default: MODERATE)

        Returns:
            VariationSpec with original and varied prompts
        """
        original = instance.get_treatment(intensity)
        template = VARIATION_TEMPLATES[dimension]

        if dimension == VariationDimension.NUMERIC_NOISE:
            varied = self._apply_numeric_noise(original, template)
        else:
            varied = template["prefix"] + original + template["suffix"]

        return VariationSpec(
            dimension=dimension,
            original_prompt=original,
            varied_prompt=varied,
            variation_description=f"Applied {dimension.value} perturbation",
        )

    def generate_all_variations(
        self,
        instance: CognitiveBiasInstance,
        intensity: TriggerIntensity = TriggerIntensity.MODERATE,
    ) -> list[VariationSpec]:
        """Generate variations along ALL dimensions for a single instance."""
        return [
            self.generate_variation(instance, dim, intensity)
            for dim in VariationDimension
        ]

    def generate_varied_instances(
        self,
        instance: CognitiveBiasInstance,
        dimensions: list[VariationDimension] | None = None,
        intensity: TriggerIntensity = TriggerIntensity.MODERATE,
    ) -> list[CognitiveBiasInstance]:
        """Create new CognitiveBiasInstance objects with varied prompts.

        Returns new instances where the treatment prompt for the given intensity
        is replaced with the varied version. Other intensities remain unchanged.
        Metadata is tagged with variation_dimension and variation_description.
        """
        dims = dimensions or list(VariationDimension)
        varied_instances = []

        for dim in dims:
            spec = self.generate_variation(instance, dim, intensity)
            new_treatment_prompts = dict(instance.treatment_prompts)
            new_treatment_prompts[intensity] = spec.varied_prompt

            new_metadata = dict(instance.metadata) if instance.metadata else {}
            new_metadata["variation_dimension"] = dim.value
            new_metadata["variation_description"] = spec.variation_description
            new_metadata["original_prompt"] = spec.original_prompt

            varied = CognitiveBiasInstance(
                bias_id=instance.bias_id,
                base_scenario=instance.base_scenario,
                bias_trigger=instance.bias_trigger,
                control_prompt=instance.control_prompt,
                treatment_prompts=new_treatment_prompts,
                expected_rational_response=instance.expected_rational_response,
                expected_biased_response=instance.expected_biased_response,
                domain=instance.domain,
                scale=instance.scale,
                metadata=new_metadata,
            )
            varied_instances.append(varied)

        return varied_instances

    def _apply_numeric_noise(self, prompt: str, template: dict) -> str:
        """Inject irrelevant numbers near the decision point."""
        noise_1 = random.randint(100, 9999)
        noise_2 = random.randint(100, 9999)
        injection = template["injection_template"].format(
            noise_value_1=noise_1,
            noise_value_2=noise_2,
        )
        # Insert noise about 1/3 of the way through the prompt
        insert_pos = len(prompt) // 3
        # Find the next sentence boundary after insert_pos
        for i in range(insert_pos, min(insert_pos + 200, len(prompt))):
            if prompt[i] in ".!?\n":
                insert_pos = i + 1
                break
        return prompt[:insert_pos] + " " + injection + prompt[insert_pos:]


@dataclass
class VariationRobustnessScore:
    """Measures how bias expression changes across variation dimensions.

    A high robustness_score means bias is consistent regardless of
    contextual perturbations (a robust, systematic finding).
    A low score means bias is context-dependent.
    """

    bias_id: str
    dimension_scores: dict[str, float]  # dimension_name -> mean bias_score
    baseline_score: float  # Bias score without any variation
    most_vulnerable_dimension: str  # Dimension causing largest deviation
    most_robust_dimension: str  # Dimension causing smallest deviation
    robustness_score: float  # 1 - max_deviation, higher = more robust
    mean_deviation: float  # Average deviation across dimensions

    @classmethod
    def calculate(
        cls,
        bias_id: str,
        baseline_score: float,
        dimension_scores: dict[str, float],
    ) -> VariationRobustnessScore:
        """Calculate robustness from baseline and per-dimension scores."""
        if not dimension_scores:
            return cls(
                bias_id=bias_id,
                dimension_scores={},
                baseline_score=baseline_score,
                most_vulnerable_dimension="none",
                most_robust_dimension="none",
                robustness_score=1.0,
                mean_deviation=0.0,
            )

        deviations = {
            dim: abs(score - baseline_score)
            for dim, score in dimension_scores.items()
        }

        max_deviation = max(deviations.values())
        mean_deviation = sum(deviations.values()) / len(deviations)

        most_vulnerable = max(deviations, key=deviations.get)
        most_robust = min(deviations, key=deviations.get)

        # Robustness: 1 - max_deviation (clamped to [0, 1])
        robustness_score = max(0.0, min(1.0, 1.0 - max_deviation))

        return cls(
            bias_id=bias_id,
            dimension_scores=dimension_scores,
            baseline_score=baseline_score,
            most_vulnerable_dimension=most_vulnerable,
            most_robust_dimension=most_robust,
            robustness_score=robustness_score,
            mean_deviation=mean_deviation,
        )
