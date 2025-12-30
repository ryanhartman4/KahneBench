"""
Evaluation engine for Kahne-Bench.

Executes bias tests on target LLMs, extracts answers, and prepares
results for metric calculation.
"""

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol

from kahne_bench.core import (
    CognitiveBiasInstance,
    EvaluationSession,
    TestResult,
    TestScale,
    TriggerIntensity,
    TemporalCondition,
)


class LLMProvider(Protocol):
    """Protocol for LLM API providers."""

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for the given prompt."""
        ...


@dataclass
class OpenAIProvider:
    """OpenAI API provider implementation."""

    client: Any  # openai.AsyncOpenAI
    model: str = "gpt-4o"

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


@dataclass
class AnthropicProvider:
    """Anthropic API provider implementation."""

    client: Any  # anthropic.AsyncAnthropic
    model: str = "claude-sonnet-4-20250514"

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


@dataclass
class EvaluationConfig:
    """Configuration for bias evaluation runs."""

    # Response settings
    max_tokens: int = 1024
    temperature: float = 0.0  # Deterministic for reproducibility

    # Trial settings
    num_trials: int = 3  # Multiple trials for Response Consistency Index
    trial_delay_ms: int = 100  # Delay between trials

    # Testing scope
    intensities: list[TriggerIntensity] = field(
        default_factory=lambda: list(TriggerIntensity)
    )
    include_control: bool = True
    include_debiasing: bool = True

    # Temporal conditions for dynamic testing
    temporal_conditions: list[TemporalCondition] = field(
        default_factory=lambda: [TemporalCondition.IMMEDIATE]
    )

    # Rate limiting
    requests_per_minute: int = 60


class AnswerExtractor:
    """
    Extracts structured answers from free-form LLM responses.

    Uses pattern matching and optional LLM-based extraction to
    identify the final decision/answer from verbose responses.
    """

    # Common answer patterns
    OPTION_PATTERNS = [
        r"(?:I (?:would )?(?:choose|select|prefer|recommend)(?:ing)?)\s*(?:option\s*)?([A-D])",
        r"(?:my (?:choice|selection|preference|answer) is)\s*(?:option\s*)?([A-D])",
        r"(?:option\s*)?([A-D])\s*(?:is (?:the )?(?:best|better|correct|right))",
        r"(?:^|\n)\s*([A-D])\s*[:\.\)]",
        r"(?:answer|choice|selection):\s*([A-D])",
    ]

    NUMERIC_PATTERNS = [
        r"(?:estimate|answer|value|result)[:\s]+\$?([\d,]+(?:\.\d+)?)",
        r"(?:approximately|about|around)\s+\$?([\d,]+(?:\.\d+)?)",
        r"\$?([\d,]+(?:\.\d+)?)\s*(?:dollars|percent|%|years|people)",
    ]

    YES_NO_PATTERNS = [
        r"(?:I (?:would )?(?:recommend|suggest|advise))\s*(yes|no|accepting|rejecting)",
        r"(?:my (?:answer|recommendation) is)\s*(yes|no)",
        r"(?:^|\n)\s*(yes|no)\s*[,\.\:]",
    ]

    def __init__(self, llm_extractor: LLMProvider | None = None):
        """
        Initialize the answer extractor.

        Args:
            llm_extractor: Optional LLM for complex answer extraction
        """
        self.llm_extractor = llm_extractor

    def extract(self, response: str, expected_type: str = "option") -> str:
        """
        Extract the final answer from an LLM response.

        Args:
            response: Full text response from LLM
            expected_type: Type of answer expected ("option", "numeric", "yes_no", "text")

        Returns:
            Extracted answer string
        """
        response_lower = response.lower()

        if expected_type == "option":
            patterns = self.OPTION_PATTERNS
        elif expected_type == "numeric":
            patterns = self.NUMERIC_PATTERNS
        elif expected_type == "yes_no":
            patterns = self.YES_NO_PATTERNS
        else:
            return self._extract_text_answer(response)

        for pattern in patterns:
            match = re.search(pattern, response_lower, re.IGNORECASE)
            if match:
                return match.group(1).upper() if expected_type == "option" else match.group(1)

        # Fallback: look for the last mentioned option/value
        return self._fallback_extraction(response, expected_type)

    def _extract_text_answer(self, response: str) -> str:
        """Extract a text-based answer (last sentence or explicit answer)."""
        # Look for explicit answer markers
        answer_markers = ["therefore", "in conclusion", "my answer is", "final answer"]
        for marker in answer_markers:
            idx = response.lower().rfind(marker)
            if idx != -1:
                return response[idx:].split(".")[0].strip()

        # Return last sentence as fallback
        sentences = response.split(".")
        return sentences[-2].strip() if len(sentences) > 1 else response.strip()

    def _fallback_extraction(self, response: str, expected_type: str) -> str:
        """Fallback extraction when patterns don't match."""
        if expected_type == "option":
            # Find last mentioned option letter
            options = re.findall(r"\b([A-D])\b", response)
            return options[-1] if options else "UNKNOWN"

        elif expected_type == "numeric":
            # Find last number in response
            numbers = re.findall(r"[\d,]+(?:\.\d+)?", response)
            return numbers[-1].replace(",", "") if numbers else "UNKNOWN"

        elif expected_type == "yes_no":
            # Check for presence of yes/no keywords
            if "yes" in response.lower() or "accept" in response.lower():
                return "yes"
            elif "no" in response.lower() or "reject" in response.lower():
                return "no"
            return "UNKNOWN"

        return "UNKNOWN"

    def extract_confidence(self, response: str) -> float | None:
        """Extract stated confidence level from response."""
        patterns = [
            r"(\d{1,3})\s*%?\s*(?:confident|confidence|certain|sure)",
            r"(?:confidence|certainty)[:\s]+(\d{1,3})\s*%?",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                conf = float(match.group(1))
                return conf / 100 if conf > 1 else conf

        return None


class BiasEvaluator:
    """
    Main evaluation engine for running cognitive bias tests.

    Coordinates test execution, answer extraction, and result collection
    across multiple test instances and trial configurations.
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: EvaluationConfig | None = None,
        answer_extractor: AnswerExtractor | None = None,
    ):
        """
        Initialize the evaluator.

        Args:
            provider: LLM provider for test execution
            config: Evaluation configuration
            answer_extractor: Custom answer extractor (defaults to built-in)
        """
        self.provider = provider
        self.config = config or EvaluationConfig()
        self.extractor = answer_extractor or AnswerExtractor()

    async def evaluate_instance(
        self,
        instance: CognitiveBiasInstance,
        model_id: str,
    ) -> list[TestResult]:
        """
        Evaluate a single test instance across all configured conditions.

        Args:
            instance: The cognitive bias test instance
            model_id: Identifier for the model being tested

        Returns:
            List of TestResult objects for each condition tested
        """
        results = []

        # Test control condition
        if self.config.include_control:
            control_results = await self._run_trials(
                instance=instance,
                prompt=instance.control_prompt,
                condition="control",
                model_id=model_id,
            )
            results.extend(control_results)

        # Test treatment conditions at each intensity
        for intensity in self.config.intensities:
            treatment_prompt = instance.get_treatment(intensity)
            treatment_results = await self._run_trials(
                instance=instance,
                prompt=treatment_prompt,
                condition=f"treatment_{intensity.value}",
                model_id=model_id,
            )
            results.extend(treatment_results)

        # Test with debiasing prompts
        if self.config.include_debiasing and instance.has_debiasing():
            for i, debiasing_prompt in enumerate(instance.debiasing_prompts):
                debiasing_results = await self._run_trials(
                    instance=instance,
                    prompt=debiasing_prompt,
                    condition=f"debiasing_{i}",
                    model_id=model_id,
                )
                results.extend(debiasing_results)

        return results

    async def _run_trials(
        self,
        instance: CognitiveBiasInstance,
        prompt: str,
        condition: str,
        model_id: str,
    ) -> list[TestResult]:
        """Run multiple trials for a single condition."""
        results = []

        for trial in range(self.config.num_trials):
            start_time = time.time()

            try:
                response = await self.provider.complete(
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                response = f"ERROR: {str(e)}"

            elapsed_ms = (time.time() - start_time) * 1000

            # Extract answer and confidence
            extracted = self.extractor.extract(response, self._infer_answer_type(instance))
            confidence = self.extractor.extract_confidence(response)

            result = TestResult(
                instance=instance,
                model_id=model_id,
                condition=condition,
                prompt_used=prompt,
                model_response=response,
                extracted_answer=extracted,
                response_time_ms=elapsed_ms,
                confidence_stated=confidence,
                metadata={"trial": trial},
            )
            results.append(result)

            # Rate limiting delay
            if trial < self.config.num_trials - 1:
                await asyncio.sleep(self.config.trial_delay_ms / 1000)

        return results

    def _infer_answer_type(self, instance: CognitiveBiasInstance) -> str:
        """Infer the expected answer type from the instance."""
        prompt_lower = instance.control_prompt.lower()

        if any(opt in prompt_lower for opt in ["option a", "option b", "program a", "program b"]):
            return "option"
        elif any(word in prompt_lower for word in ["estimate", "how much", "how many", "probability"]):
            return "numeric"
        elif any(word in prompt_lower for word in ["accept", "reject", "continue", "should you"]):
            return "yes_no"
        else:
            return "text"

    async def evaluate_batch(
        self,
        instances: list[CognitiveBiasInstance],
        model_id: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EvaluationSession:
        """
        Evaluate a batch of test instances.

        Args:
            instances: List of test instances to evaluate
            model_id: Identifier for the model being tested
            progress_callback: Optional callback for progress updates

        Returns:
            EvaluationSession with all results
        """
        session = EvaluationSession(
            session_id=str(uuid.uuid4()),
            model_id=model_id,
            model_config={
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "num_trials": self.config.num_trials,
            },
            test_instances=instances,
            start_time=datetime.now().isoformat(),
        )

        for i, instance in enumerate(instances):
            results = await self.evaluate_instance(instance, model_id)
            session.results.extend(results)

            if progress_callback:
                progress_callback(i + 1, len(instances))

            # Rate limiting between instances
            if i < len(instances) - 1:
                await asyncio.sleep(60 / self.config.requests_per_minute)

        session.end_time = datetime.now().isoformat()
        return session

    def score_response(
        self,
        result: TestResult,
        rational_answer: str,
        biased_answer: str,
    ) -> tuple[bool, float]:
        """
        Score a single response for bias presence.

        Args:
            result: The test result to score
            rational_answer: The expected rational response
            biased_answer: The expected biased response

        Returns:
            Tuple of (is_biased, bias_score)
        """
        extracted = result.extracted_answer.lower()
        rational = rational_answer.lower()
        biased = biased_answer.lower()

        # Simple matching for option/yes_no answers
        if extracted == rational:
            return False, 0.0
        elif extracted == biased:
            return True, 1.0
        else:
            # Partial scoring for numeric answers
            try:
                extracted_num = float(extracted.replace(",", ""))
                rational_num = float(rational.replace(",", ""))
                biased_num = float(biased.replace(",", ""))

                # Calculate position between rational and biased
                if abs(biased_num - rational_num) > 0:
                    bias_score = abs(extracted_num - rational_num) / abs(biased_num - rational_num)
                    bias_score = min(max(bias_score, 0.0), 1.0)
                    return bias_score > 0.5, bias_score
            except (ValueError, ZeroDivisionError):
                pass

            return True, 0.5  # Unknown, default to partial bias


class TemporalEvaluator(BiasEvaluator):
    """
    Extended evaluator for temporal dynamics testing.

    Tests how biases evolve across sequential prompts and
    how models respond to corrective feedback.
    """

    async def evaluate_persistent(
        self,
        instance: CognitiveBiasInstance,
        model_id: str,
        num_rounds: int = 5,
    ) -> list[TestResult]:
        """
        Test bias persistence across sequential related prompts.

        Args:
            instance: The test instance
            model_id: Model identifier
            num_rounds: Number of sequential prompts

        Returns:
            Results showing bias evolution over rounds
        """
        results = []
        context = ""

        for round_num in range(num_rounds):
            # Build on previous context
            if round_num == 0:
                prompt = instance.get_treatment(TriggerIntensity.MODERATE)
            else:
                prompt = f"""
Continuing from our previous discussion:
{context}

Now consider a similar but distinct situation:
{instance.get_treatment(TriggerIntensity.MODERATE)}
"""

            start_time = time.time()
            response = await self.provider.complete(prompt, self.config.max_tokens)
            elapsed_ms = (time.time() - start_time) * 1000

            context = response[:500]  # Use response as context for next round

            result = TestResult(
                instance=instance,
                model_id=model_id,
                condition=f"persistent_round_{round_num}",
                prompt_used=prompt,
                model_response=response,
                extracted_answer=self.extractor.extract(response, self._infer_answer_type(instance)),
                response_time_ms=elapsed_ms,
                metadata={"round": round_num, "temporal_condition": "persistent"},
            )
            results.append(result)

        return results

    async def evaluate_adaptive(
        self,
        instance: CognitiveBiasInstance,
        model_id: str,
    ) -> list[TestResult]:
        """
        Test model's ability to adapt after corrective feedback.

        Args:
            instance: The test instance
            model_id: Model identifier

        Returns:
            Results showing pre and post-feedback behavior
        """
        results = []

        # Initial biased prompt
        initial_prompt = instance.get_treatment(TriggerIntensity.STRONG)
        start_time = time.time()
        initial_response = await self.provider.complete(initial_prompt, self.config.max_tokens)
        elapsed_ms = (time.time() - start_time) * 1000

        results.append(TestResult(
            instance=instance,
            model_id=model_id,
            condition="adaptive_pre_feedback",
            prompt_used=initial_prompt,
            model_response=initial_response,
            extracted_answer=self.extractor.extract(initial_response, self._infer_answer_type(instance)),
            response_time_ms=elapsed_ms,
            metadata={"temporal_condition": "adaptive", "phase": "pre_feedback"},
        ))

        # Corrective feedback
        feedback_prompt = f"""
Your previous response showed signs of {instance.bias_id.replace('_', ' ')}.

{instance.control_prompt}

Please reconsider, being careful to avoid this cognitive bias.
"""

        start_time = time.time()
        feedback_response = await self.provider.complete(feedback_prompt, self.config.max_tokens)
        elapsed_ms = (time.time() - start_time) * 1000

        results.append(TestResult(
            instance=instance,
            model_id=model_id,
            condition="adaptive_post_feedback",
            prompt_used=feedback_prompt,
            model_response=feedback_response,
            extracted_answer=self.extractor.extract(feedback_response, self._infer_answer_type(instance)),
            response_time_ms=elapsed_ms,
            metadata={"temporal_condition": "adaptive", "phase": "post_feedback"},
        ))

        return results
