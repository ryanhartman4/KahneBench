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
class XAIProvider:
    """xAI Grok API provider implementation.

    Uses the xai-sdk package. The SDK is synchronous, so we wrap calls
    with asyncio.to_thread() to maintain async compatibility.
    """

    client: Any  # xai_sdk.Client
    model: str = "grok-4-1-fast-reasoning"

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        def _sync_complete() -> str:
            from xai_sdk.chat import user, system

            chat = self.client.chat.create(model=self.model)
            chat.append(system("You are a helpful assistant."))
            chat.append(user(prompt))
            response = chat.sample()
            return response.content

        return await asyncio.to_thread(_sync_complete)


@dataclass
class GeminiProvider:
    """Google Gemini API provider implementation.

    Uses the google-genai package. The SDK is synchronous, so we wrap calls
    with asyncio.to_thread() to maintain async compatibility.
    """

    client: Any  # google.genai.Client
    model: str = "gemini-3-pro-preview"

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        def _sync_complete() -> str:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text

        return await asyncio.to_thread(_sync_complete)


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

    def extract(self, response: str, expected_type: str = "option") -> str | None:
        """
        Extract the final answer from an LLM response.

        Args:
            response: Full text response from LLM
            expected_type: Type of answer expected ("option", "numeric", "yes_no", "text")

        Returns:
            Extracted answer string, or None if extraction failed
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

    def _extract_text_answer(self, response: str) -> str | None:
        """Extract a text-based answer (last sentence or explicit answer)."""
        # Look for explicit answer markers
        answer_markers = ["therefore", "in conclusion", "my answer is", "final answer"]
        for marker in answer_markers:
            idx = response.lower().rfind(marker)
            if idx != -1:
                return response[idx:].split(".")[0].strip()

        # Return last sentence as fallback
        sentences = response.split(".")
        if len(sentences) > 1 and sentences[-2].strip():
            return sentences[-2].strip()
        elif response.strip():
            return response.strip()
        return None

    def _fallback_extraction(self, response: str, expected_type: str) -> str | None:
        """Fallback extraction when patterns don't match."""
        if expected_type == "option":
            # Find last mentioned option letter
            options = re.findall(r"\b([A-D])\b", response)
            return options[-1] if options else None

        elif expected_type == "numeric":
            # Try to find numbers near answer keywords first
            answer_context = re.search(
                r"(?:estimate|answer|value|result|approximately|about|around)[:\s]+\$?([\d,]+(?:\.\d+)?)",
                response,
                re.IGNORECASE,
            )
            if answer_context:
                return answer_context.group(1).replace(",", "")

            # Find numbers with units (excluding confidence percentages)
            numbers_with_units = re.findall(
                r"\$?([\d,]+(?:\.\d+)?)\s*(?:dollars?|people|years|months|days|units?)\b",
                response,
                re.IGNORECASE,
            )
            if numbers_with_units:
                return numbers_with_units[-1].replace(",", "")

            # Exclude numbers that are part of confidence statements
            response_no_confidence = re.sub(
                r"\d{1,3}\s*%?\s*(?:confident|confidence|certain|sure)",
                "",
                response,
                flags=re.IGNORECASE,
            )
            numbers = re.findall(r"[\d,]+(?:\.\d+)?", response_no_confidence)
            return numbers[-1].replace(",", "") if numbers else None

        elif expected_type == "yes_no":
            # Check for presence of yes/no keywords
            if "yes" in response.lower() or "accept" in response.lower():
                return "yes"
            elif "no" in response.lower() or "reject" in response.lower():
                return "no"
            return None

        return None

    def extract_confidence(self, response: str) -> float | None:
        """Extract stated confidence level from response.

        Returns:
            Confidence value clamped to [0.0, 1.0], or None if not found.
        """
        patterns = [
            r"(\d{1,3})\s*%?\s*(?:confident|confidence|certain|sure)",
            r"(?:confidence|certainty)[:\s]+(\d{1,3})\s*%?",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                conf = float(match.group(1))
                normalized = conf / 100 if conf > 1 else conf
                return max(0.0, min(1.0, normalized))  # Clamp to [0, 1]

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

            # Score the response for bias (only if we have expected answers)
            if (instance.expected_rational_response and
                instance.expected_biased_response and
                not instance.expected_rational_response.startswith("[") and
                not instance.expected_biased_response.startswith("[")):
                is_biased, bias_score = self.score_response(
                    result,
                    instance.expected_rational_response,
                    instance.expected_biased_response,
                )
                result.is_biased = is_biased
                result.bias_score = bias_score

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
    ) -> tuple[bool | None, float | None]:
        """
        Score a single response for bias presence.

        Args:
            result: The test result to score
            rational_answer: The expected rational response
            biased_answer: The expected biased response

        Returns:
            Tuple of (is_biased, bias_score) where:
            - is_biased: True if biased, False if rational, None if unknown
            - bias_score: 0.0 for rational, 1.0 for biased, 0-1 for partial, None if unknown
        """
        # Handle placeholder expected answers at entry point
        if rational_answer.startswith("[") or biased_answer.startswith("["):
            return None, None

        # Handle extraction failures
        if result.extracted_answer is None:
            return None, None

        extracted = result.extracted_answer.lower().strip()
        rational = rational_answer.lower().strip()
        biased = biased_answer.lower().strip()

        # Numeric tolerance (1% relative epsilon)
        EPSILON = 0.01

        # Try numeric comparison FIRST (handles "100" vs "100.0" cases)
        try:
            extracted_num = float(extracted.replace(",", "").replace("$", ""))
            rational_num = float(rational.replace(",", "").replace("$", ""))
            biased_num = float(biased.replace(",", "").replace("$", ""))

            # Check if within epsilon of rational answer
            if abs(extracted_num - rational_num) <= EPSILON * max(abs(rational_num), 1.0):
                return False, 0.0

            # Check if within epsilon of biased answer
            if abs(extracted_num - biased_num) <= EPSILON * max(abs(biased_num), 1.0):
                return True, 1.0

            # Calculate position between rational and biased
            if abs(biased_num - rational_num) > EPSILON:
                bias_score = abs(extracted_num - rational_num) / abs(biased_num - rational_num)
                bias_score = max(0.0, min(1.0, bias_score))
                return bias_score > 0.5, bias_score

            # Rational and biased are essentially equal - can't determine bias
            return None, None

        except (ValueError, TypeError):
            pass

        # Fall back to exact string matching for non-numeric answers
        if extracted == rational:
            return False, 0.0
        elif extracted == biased:
            return True, 1.0

        # Unknown answer - neither matches
        return None, None


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


class ContextSensitivityEvaluator(BiasEvaluator):
    """
    Extended evaluator for context sensitivity testing.

    Implements Section 4.2 of the Kahne-Bench specification:
    Tests how expertise level, formality, and stakes affect bias expression.
    """

    async def evaluate_context_sensitivity(
        self,
        instance: CognitiveBiasInstance,
        model_id: str,
        intensity: TriggerIntensity = TriggerIntensity.MODERATE,
    ) -> list[TestResult]:
        """
        Evaluate bias expression across different context configurations.

        Tests all combinations of expertise, formality, and stakes to
        measure how context affects bias susceptibility.

        Args:
            instance: The test instance
            model_id: Model identifier
            intensity: Trigger intensity to use

        Returns:
            Results across all context configurations
        """
        from kahne_bench.core import (
            ExpertiseLevel,
            Formality,
            Stakes,
            ContextSensitivityConfig,
        )

        results = []

        # Test key context combinations
        context_configs = [
            # Low context pressure
            ContextSensitivityConfig(
                expertise_level=ExpertiseLevel.NOVICE,
                formality=Formality.CASUAL,
                stakes=Stakes.LOW,
            ),
            # Medium context pressure (baseline)
            ContextSensitivityConfig(
                expertise_level=ExpertiseLevel.INTERMEDIATE,
                formality=Formality.PROFESSIONAL,
                stakes=Stakes.MODERATE,
            ),
            # High context pressure
            ContextSensitivityConfig(
                expertise_level=ExpertiseLevel.EXPERT,
                formality=Formality.FORMAL,
                stakes=Stakes.HIGH,
            ),
            # Maximum pressure (authority + critical stakes)
            ContextSensitivityConfig(
                expertise_level=ExpertiseLevel.AUTHORITY,
                formality=Formality.FORMAL,
                stakes=Stakes.CRITICAL,
            ),
            # Expert with low stakes (test expertise effect)
            ContextSensitivityConfig(
                expertise_level=ExpertiseLevel.EXPERT,
                formality=Formality.CASUAL,
                stakes=Stakes.LOW,
            ),
            # Novice with high stakes (test vulnerability)
            ContextSensitivityConfig(
                expertise_level=ExpertiseLevel.NOVICE,
                formality=Formality.FORMAL,
                stakes=Stakes.HIGH,
            ),
        ]

        for config in context_configs:
            prompt = instance.apply_context_sensitivity(
                instance.get_treatment(intensity),
                config,
            )

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

            extracted = self.extractor.extract(
                response, self._infer_answer_type(instance)
            )
            confidence = self.extractor.extract_confidence(response)

            result = TestResult(
                instance=instance,
                model_id=model_id,
                condition=f"context_{config.expertise_level.value}_{config.stakes.value}",
                prompt_used=prompt,
                model_response=response,
                extracted_answer=extracted,
                response_time_ms=elapsed_ms,
                confidence_stated=confidence,
                metadata={
                    "context_sensitivity": True,
                    "expertise_level": config.expertise_level.value,
                    "formality": config.formality.value,
                    "stakes": config.stakes.value,
                },
            )

            # Score the response
            if (instance.expected_rational_response and
                instance.expected_biased_response and
                not instance.expected_rational_response.startswith("[") and
                not instance.expected_biased_response.startswith("[")):
                is_biased, bias_score = self.score_response(
                    result,
                    instance.expected_rational_response,
                    instance.expected_biased_response,
                )
                result.is_biased = is_biased
                result.bias_score = bias_score

            results.append(result)

        return results

    async def evaluate_expertise_gradient(
        self,
        instance: CognitiveBiasInstance,
        model_id: str,
        intensity: TriggerIntensity = TriggerIntensity.MODERATE,
    ) -> list[TestResult]:
        """
        Evaluate bias expression specifically across expertise levels.

        Tests whether expert framing reduces bias susceptibility.

        Args:
            instance: The test instance
            model_id: Model identifier
            intensity: Trigger intensity to use

        Returns:
            Results for each expertise level
        """
        from kahne_bench.core import ExpertiseLevel, Formality, Stakes

        results = []

        for expertise in ExpertiseLevel:
            prompt = instance.get_context_variant(
                intensity=intensity,
                expertise=expertise,
                formality=Formality.PROFESSIONAL,
                stakes=Stakes.MODERATE,
            )

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

            result = TestResult(
                instance=instance,
                model_id=model_id,
                condition=f"expertise_{expertise.value}",
                prompt_used=prompt,
                model_response=response,
                extracted_answer=self.extractor.extract(
                    response, self._infer_answer_type(instance)
                ),
                response_time_ms=elapsed_ms,
                metadata={
                    "context_sensitivity": True,
                    "expertise_level": expertise.value,
                    "test_type": "expertise_gradient",
                },
            )

            # Score the response
            if (instance.expected_rational_response and
                instance.expected_biased_response and
                not instance.expected_rational_response.startswith("[") and
                not instance.expected_biased_response.startswith("[")):
                is_biased, bias_score = self.score_response(
                    result,
                    instance.expected_rational_response,
                    instance.expected_biased_response,
                )
                result.is_biased = is_biased
                result.bias_score = bias_score

            results.append(result)

        return results

    async def evaluate_stakes_gradient(
        self,
        instance: CognitiveBiasInstance,
        model_id: str,
        intensity: TriggerIntensity = TriggerIntensity.MODERATE,
    ) -> list[TestResult]:
        """
        Evaluate bias expression specifically across stakes levels.

        Tests whether high-stakes framing increases or decreases bias.

        Args:
            instance: The test instance
            model_id: Model identifier
            intensity: Trigger intensity to use

        Returns:
            Results for each stakes level
        """
        from kahne_bench.core import ExpertiseLevel, Formality, Stakes

        results = []

        for stakes in Stakes:
            prompt = instance.get_context_variant(
                intensity=intensity,
                expertise=ExpertiseLevel.INTERMEDIATE,
                formality=Formality.PROFESSIONAL,
                stakes=stakes,
            )

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

            result = TestResult(
                instance=instance,
                model_id=model_id,
                condition=f"stakes_{stakes.value}",
                prompt_used=prompt,
                model_response=response,
                extracted_answer=self.extractor.extract(
                    response, self._infer_answer_type(instance)
                ),
                response_time_ms=elapsed_ms,
                metadata={
                    "context_sensitivity": True,
                    "stakes": stakes.value,
                    "test_type": "stakes_gradient",
                },
            )

            # Score the response
            if (instance.expected_rational_response and
                instance.expected_biased_response and
                not instance.expected_rational_response.startswith("[") and
                not instance.expected_biased_response.startswith("[")):
                is_biased, bias_score = self.score_response(
                    result,
                    instance.expected_rational_response,
                    instance.expected_biased_response,
                )
                result.is_biased = is_biased
                result.bias_score = bias_score

            results.append(result)

        return results
