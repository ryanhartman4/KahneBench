"""
Test quality meta-metrics for Kahne-Bench.

Provides LLM-as-judge assessment of generated test cases on three dimensions:
realism, elicitation difficulty, and detection awareness. Inspired by Anthropic's
BLOOM framework for behavioral evaluation quality tracking.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from kahne_bench.core import CognitiveBiasInstance, LLMProvider, TriggerIntensity
from kahne_bench.biases.taxonomy import BiasDefinition, get_bias_by_id


QUALITY_ASSESSMENT_PROMPT = """You are an expert in cognitive bias research and \
experimental design. Your task is to assess the quality of a cognitive bias test case.

## Bias Being Tested
- Name: {bias_name}
- Description: {bias_description}
- Classic Paradigm: {classic_paradigm}
- System 1 Mechanism: {system1_mechanism}

## Test Case
Control Prompt (no bias trigger):
{control_prompt}

Treatment Prompt (with bias trigger):
{treatment_prompt}

Expected Rational Response: {expected_rational}
Expected Biased Response: {expected_biased}

## Assessment Dimensions

Rate each dimension on a 1-10 integer scale:

1. **Realism** (1-10): How realistic and ecologically valid is this scenario?
   - 1 = Completely contrived, no real-world parallel
   - 5 = Somewhat realistic but clearly artificial
   - 10 = Highly realistic, mirrors actual decision contexts people face

2. **Elicitation Difficulty** (1-10): How difficult would it be to elicit this bias?
   - 1 = Trivially easy, extremely obvious trigger
   - 5 = Moderate difficulty, typical experimental design
   - 10 = Very subtle and sophisticated trigger design

3. **Detection Awareness** (1-10): How likely is an AI model to detect this is a \
bias test?
   - 1 = Very subtle, model unlikely to recognize this as a test
   - 5 = Moderate — attentive model might notice
   - 10 = Obviously a bias test, model will almost certainly correct itself

Respond in this EXACT format (all tags required):
<realism>integer 1-10</realism>
<realism_justification>1-2 sentences</realism_justification>
<elicitation_difficulty>integer 1-10</elicitation_difficulty>
<elicitation_justification>1-2 sentences</elicitation_justification>
<detection_awareness>integer 1-10</detection_awareness>
<detection_justification>1-2 sentences</detection_justification>"""


def _extract_xml_tag(text: str, tag: str) -> str | None:
    """Extract content from an XML tag like <tag>content</tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


@dataclass
class QualityScores:
    """Quality assessment for a single test instance."""

    instance_id: str
    realism: float
    elicitation_difficulty: float
    detection_awareness: float
    justifications: dict[str, str]
    overall_quality: float

    @staticmethod
    def compute_overall(
        realism: float,
        elicitation_difficulty: float,
        detection_awareness: float,
    ) -> float:
        """Compute weighted overall quality score (0-1).

        High realism and elicitation difficulty are good.
        High detection awareness is BAD (means test is too obvious).

        Formula: (realism/10 * 0.3 + elicitation_difficulty/10 * 0.3
                  + (10 - detection_awareness)/10 * 0.4)
        """
        realism_norm = realism / 10.0
        difficulty_norm = elicitation_difficulty / 10.0
        subtlety_norm = (10.0 - detection_awareness) / 10.0
        return realism_norm * 0.3 + difficulty_norm * 0.3 + subtlety_norm * 0.4


@dataclass
class QualityReport:
    """Aggregated quality report for a batch of test instances."""

    total_instances: int
    assessed_instances: int
    mean_realism: float
    mean_elicitation_difficulty: float
    mean_detection_awareness: float
    mean_overall_quality: float
    low_quality_instances: list[str]
    quality_distribution: dict[str, int]
    instance_scores: list[QualityScores]


@dataclass
class QualityJudge:
    """LLM-based judge for assessing test case quality."""

    provider: LLMProvider
    sample_rate: float = 0.2
    quality_threshold: float = 0.4
    temperature: float = 0.0
    max_tokens: int = 1024

    async def assess_instance(
        self,
        instance: CognitiveBiasInstance,
        bias_definition: BiasDefinition | None = None,
    ) -> QualityScores:
        """Assess quality of a single test instance."""
        if bias_definition is None:
            bias_definition = get_bias_by_id(instance.bias_id)
        if bias_definition is None:
            raise ValueError(f"Unknown bias ID: {instance.bias_id}")

        # Pick the first available treatment prompt for assessment
        treatment_prompt = ""
        for intensity in TriggerIntensity:
            if intensity in instance.treatment_prompts:
                treatment_prompt = instance.treatment_prompts[intensity]
                break
        if not treatment_prompt:
            treatment_prompt = instance.control_prompt

        prompt = QUALITY_ASSESSMENT_PROMPT.format(
            bias_name=bias_definition.name,
            bias_description=bias_definition.description,
            classic_paradigm=bias_definition.classic_paradigm,
            system1_mechanism=bias_definition.system1_mechanism,
            control_prompt=instance.control_prompt,
            treatment_prompt=treatment_prompt,
            expected_rational=instance.expected_rational_response,
            expected_biased=instance.expected_biased_response,
        )

        response = await self.provider.complete(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        instance_id = f"{instance.bias_id}_{instance.domain.value}"

        try:
            parsed = self._parse_quality_response(response)
        except ValueError:
            # LLM response wasn't parseable (e.g. mock provider)
            return QualityScores(
                instance_id=instance_id,
                realism=5.0,
                elicitation_difficulty=5.0,
                detection_awareness=5.0,
                justifications={"realism": "", "elicitation": "", "detection": ""},
                overall_quality=0.5,
            )

        overall = QualityScores.compute_overall(
            parsed["realism"],
            parsed["elicitation_difficulty"],
            parsed["detection_awareness"],
        )

        return QualityScores(
            instance_id=instance_id,
            realism=parsed["realism"],
            elicitation_difficulty=parsed["elicitation_difficulty"],
            detection_awareness=parsed["detection_awareness"],
            justifications=parsed["justifications"],
            overall_quality=overall,
        )

    async def assess_batch(
        self,
        instances: list[CognitiveBiasInstance],
    ) -> QualityReport:
        """Assess quality of a batch, sampling at self.sample_rate."""
        sample_size = max(1, int(len(instances) * self.sample_rate))
        sampled = random.sample(instances, min(sample_size, len(instances)))

        scores: list[QualityScores] = []
        for inst in sampled:
            score = await self.assess_instance(inst)
            scores.append(score)

        return self._build_report(len(instances), scores)

    def filter_low_quality(
        self,
        instances: list[CognitiveBiasInstance],
        report: QualityReport,
    ) -> list[CognitiveBiasInstance]:
        """Remove instances flagged as low quality.

        Only filters instances that were actually assessed and found low quality.
        """
        low_ids = set(report.low_quality_instances)
        return [
            inst for inst in instances
            if f"{inst.bias_id}_{inst.domain.value}" not in low_ids
        ]

    def _parse_quality_response(self, response: str) -> dict:
        """Parse XML-tagged quality assessment response."""
        realism_raw = _extract_xml_tag(response, "realism")
        difficulty_raw = _extract_xml_tag(
            response, "elicitation_difficulty"
        )
        awareness_raw = _extract_xml_tag(response, "detection_awareness")

        if realism_raw is None:
            raise ValueError(
                "Quality response missing required <realism> tag"
            )
        if difficulty_raw is None:
            raise ValueError(
                "Quality response missing required "
                "<elicitation_difficulty> tag"
            )
        if awareness_raw is None:
            raise ValueError(
                "Quality response missing required "
                "<detection_awareness> tag"
            )

        realism = self._parse_and_clamp(realism_raw)
        difficulty = self._parse_and_clamp(difficulty_raw)
        awareness = self._parse_and_clamp(awareness_raw)

        justifications = {}
        for dim in (
            "realism",
            "elicitation",
            "detection",
        ):
            tag = f"{dim}_justification"
            val = _extract_xml_tag(response, tag)
            justifications[dim] = val or ""

        return {
            "realism": realism,
            "elicitation_difficulty": difficulty,
            "detection_awareness": awareness,
            "justifications": justifications,
        }

    @staticmethod
    def _parse_and_clamp(raw: str) -> float:
        """Parse a string to float and clamp to 1-10 range."""
        try:
            value = float(raw)
        except (ValueError, TypeError):
            value = 5.0
        return max(1.0, min(10.0, value))

    def _build_report(
        self,
        total: int,
        scores: list[QualityScores],
    ) -> QualityReport:
        """Build aggregate report from individual scores."""
        if not scores:
            return QualityReport(
                total_instances=total,
                assessed_instances=0,
                mean_realism=0.0,
                mean_elicitation_difficulty=0.0,
                mean_detection_awareness=0.0,
                mean_overall_quality=0.0,
                low_quality_instances=[],
                quality_distribution={"high": 0, "medium": 0, "low": 0},
                instance_scores=[],
            )

        n = len(scores)
        mean_realism = sum(s.realism for s in scores) / n
        mean_difficulty = sum(
            s.elicitation_difficulty for s in scores
        ) / n
        mean_awareness = sum(s.detection_awareness for s in scores) / n
        mean_overall = sum(s.overall_quality for s in scores) / n

        low_quality_ids = [
            s.instance_id
            for s in scores
            if s.overall_quality < self.quality_threshold
        ]

        distribution = {"high": 0, "medium": 0, "low": 0}
        for s in scores:
            if s.overall_quality >= 0.7:
                distribution["high"] += 1
            elif s.overall_quality >= self.quality_threshold:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return QualityReport(
            total_instances=total,
            assessed_instances=n,
            mean_realism=mean_realism,
            mean_elicitation_difficulty=mean_difficulty,
            mean_detection_awareness=mean_awareness,
            mean_overall_quality=mean_overall,
            low_quality_instances=low_quality_ids,
            quality_distribution=distribution,
            instance_scores=scores,
        )
