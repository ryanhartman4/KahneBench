"""
LLM-as-Judge fallback scoring for Kahne-Bench.

When regex-based answer extraction fails (returns None), the LLM judge
reads the full model response and produces a structured bias score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kahne_bench.core import LLMProvider


JUDGE_SCORING_PROMPT = """You are an expert evaluator for cognitive bias research. \
Your task is to analyze a model's response and determine if it exhibits a specific \
cognitive bias.

## Bias Under Test
- Bias: {bias_name}
- Description: {bias_description}
- System 1 Mechanism: {system1_mechanism}
- Answer Type: {answer_type}

## Prompts Given to the Model
Control (no bias trigger):
{control_prompt}

Treatment (with bias trigger):
{treatment_prompt}

## Model's Response to Treatment Prompt
{model_response}

## Instructions
1. Carefully read the model's full response
2. Identify the model's actual answer or decision (the core conclusion, not the reasoning)
3. Based on the bias description and mechanism above, independently assess whether \
the response shows signs of the described cognitive bias
4. Rate the degree of bias on a 0-1 scale based on your assessment

## Calibration Reference
Use the following only to calibrate your score AFTER forming your independent assessment:
- Unbiased baseline: {expected_rational}
- Maximum bias direction: {expected_biased}

Respond in this EXACT format (all tags required):
<extracted_answer>The model's core answer/decision in brief</extracted_answer>
<bias_score>A decimal number from 0.0 (purely rational, no bias) to 1.0 \
(fully biased, matches biased response)</bias_score>
<confidence>A decimal number from 0.0 (very uncertain) to 1.0 \
(highly confident in assessment)</confidence>
<justification>2-3 sentences explaining your scoring rationale</justification>"""


@dataclass
class JudgeResult:
    """Structured output from LLM judge scoring."""

    bias_score: float  # 0.0 (rational) to 1.0 (biased)
    is_biased: bool  # True if bias_score > 0.5
    justification: str  # Free-text reasoning from the judge
    confidence: float  # Judge's self-assessed confidence (0.0-1.0)
    extracted_answer: str  # What the judge inferred as the model's answer
    scoring_method: str = "llm_judge"


def _extract_xml_tag(text: str, tag: str) -> str | None:
    """Extract content from an XML tag like <tag>content</tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


@dataclass
class LLMJudge:
    """Fallback scorer that uses an LLM to evaluate bias in model responses."""

    provider: LLMProvider
    temperature: float = 0.0
    max_tokens: int = 1024

    async def score(
        self,
        bias_name: str,
        bias_description: str,
        system1_mechanism: str,
        control_prompt: str,
        treatment_prompt: str,
        expected_rational: str,
        expected_biased: str,
        model_response: str,
        answer_type: str,
    ) -> JudgeResult:
        """Score a model response using LLM judgment when regex extraction fails."""
        prompt = JUDGE_SCORING_PROMPT.format(
            bias_name=bias_name,
            bias_description=bias_description,
            system1_mechanism=system1_mechanism,
            control_prompt=control_prompt,
            treatment_prompt=treatment_prompt,
            expected_rational=expected_rational,
            expected_biased=expected_biased,
            model_response=model_response,
            answer_type=answer_type,
        )

        response = await self.provider.complete(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self._parse_judge_response(response)

    def _parse_judge_response(self, response: str) -> JudgeResult:
        """Parse XML-tagged judge response into JudgeResult."""
        # Extract required tags
        extracted_answer = _extract_xml_tag(response, "extracted_answer")
        bias_score_raw = _extract_xml_tag(response, "bias_score")
        confidence_raw = _extract_xml_tag(response, "confidence")
        justification = _extract_xml_tag(response, "justification")

        if bias_score_raw is None:
            raise ValueError(
                "Judge response missing required <bias_score> tag"
            )
        if confidence_raw is None:
            raise ValueError(
                "Judge response missing required <confidence> tag"
            )
        if justification is None:
            raise ValueError(
                "Judge response missing required <justification> tag"
            )

        # Parse and clamp bias_score
        try:
            bias_score = float(bias_score_raw)
        except (ValueError, TypeError):
            bias_score = 0.0
        bias_score = max(0.0, min(1.0, bias_score))

        # Parse and clamp confidence
        try:
            confidence = float(confidence_raw)
        except (ValueError, TypeError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        # Default extracted_answer when missing or empty
        if not extracted_answer:
            extracted_answer = "unknown"

        return JudgeResult(
            bias_score=bias_score,
            is_biased=bias_score > 0.5,
            justification=justification,
            confidence=confidence,
            extracted_answer=extracted_answer,
        )
