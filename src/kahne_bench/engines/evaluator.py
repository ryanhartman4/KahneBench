"""
Evaluation engine for Kahne-Bench.

Executes bias tests on target LLMs, extracts answers, and prepares
results for metric calculation.
"""

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

from kahne_bench.core import (
    CognitiveBiasInstance,
    EvaluationSession,
    TestResult,
    TriggerIntensity,
    TemporalCondition,
)
from kahne_bench.engines.judge import LLMJudge, JudgeResult
from kahne_bench.biases.taxonomy import get_bias_by_id


# Answer normalization mappings for common synonyms
# Maps canonical form -> list of acceptable variations
ANSWER_SYNONYMS: dict[str, list[str]] = {
    # Decision/action answers
    "accept": ["accept", "yes", "take", "agree", "proceed", "go ahead"],
    "reject": ["reject", "no", "decline", "refuse", "pass", "avoid"],
    "continue": ["continue", "proceed", "keep going", "persist", "stay"],
    "abandon": ["abandon", "stop", "quit", "cancel", "discontinue"],
    "change": ["change", "switch", "new", "different"],
    "keep": ["keep", "maintain", "stay", "current", "status quo"],
    # Option answers
    "a": ["a", "option a", "choice a", "program a", "plan a"],
    "b": ["b", "option b", "choice b", "program b", "plan b"],
    "both": ["both", "equal", "same", "either", "no difference", "indifferent"],
    # Comparison answers
    "small": ["small", "smaller", "the small", "small hospital"],
    "large": ["large", "larger", "the large", "large hospital"],
    "average": ["average", "similar", "same as average", "base rate", "mean"],
    "lower": ["lower", "less", "below average", "less than average", "decreased"],
    "higher": ["higher", "more", "above average", "greater than average", "increased"],
    "extreme": ["extreme", "continuation", "persist", "same direction"],
    # Certainty answers
    "uncertain": ["uncertain", "unsure", "unclear", "unknown", "unpredictable"],
    "predictable": ["predictable", "obvious", "foreseeable", "expected", "inevitable"],
    "confident": ["confident", "certain", "sure", "high confidence"],
    # Evidence/information answers
    "confirming": ["confirming", "supporting", "favorable", "positive evidence"],
    "statistical": ["statistical", "base rate", "data-driven", "objective"],
    "emotional": ["emotional", "affect", "feeling", "gut"],
    "salient": ["salient", "memorable", "vivid", "dramatic"],
    # Evaluation answers
    "compare": ["compare", "evaluate", "assess", "analyze", "consider both"],
    "evaluate": ["evaluate", "assess", "consider", "analyze", "judge on merits"],
    "default": ["default", "unchanged", "original"],
    "adopt": ["adopt", "follow", "join", "go with"],
    # Attribution answers
    "situational": ["situational", "external", "circumstances", "context"],
    "dispositional": ["dispositional", "internal", "personality", "character"],
    # Timing answers
    "now": ["now", "immediate", "today", "present"],
    "later": ["later", "delayed", "future", "wait"],
    "historical": ["historical", "long-term", "past average"],
    "recent": ["recent", "latest", "current trend"],
    # Memory answers
    "original": ["original", "actual", "true", "real"],
    "reconstructed": ["reconstructed", "false", "suggested", "modified"],
    "correct": ["correct", "right", "accurate"],
    "wrong": ["wrong", "incorrect", "mistaken"],
    # Group answers
    "individual": ["individual", "person", "specific", "single"],
    "group": ["group", "collective", "all", "everyone"],
    "ingroup": ["ingroup", "team", "us", "own group"],
    "equal": ["equal", "fair", "unbiased", "merit-based"],
    # Variety answers
    "varied": ["varied", "diverse", "heterogeneous", "different"],
    "homogeneous": ["homogeneous", "uniform", "same", "similar"],
    # Update answers
    "update": ["update", "revise", "change belief", "modify"],
    "maintain": ["maintain", "persist", "keep belief", "unchanged"],
    # Correlation answers
    "none": ["none", "no correlation", "unrelated", "independent"],
    "correlated": ["correlated", "related", "connected", "associated"],
    # Proportion answers
    "proportional": ["proportional", "scaled", "relative to size"],
    "similar": ["similar", "same amount", "flat", "regardless of scale"],
    # Other
    "multiple": ["multiple", "several", "many factors", "all factors"],
    "single": ["single", "one", "focal", "main factor"],
    "first": ["first", "initial", "primary", "earliest"],
    "all": ["all", "everything", "complete", "comprehensive"],
    "attended": ["attended", "noticed", "salient", "focused on"],
    "objective": ["objective", "unbiased", "neutral", "as is"],
    "fungible": ["fungible", "interchangeable", "same money"],
    "separate": ["separate", "different", "mental account"],
    "noticed": ["noticed", "saw", "detected", "observed"],
    "missed": ["missed", "didn't see", "overlooked", "blind to"],
    "peak": ["peak", "highest", "best moment", "peak-end"],
    "typical": ["typical", "representative", "prototype", "stereotypical"],
    "own": ["own", "my side", "personal view"],
    "asymmetric": ["asymmetric", "different attribution", "self-serving"],
    "ambiguous": ["ambiguous", "unknown probability", "uncertain odds"],
    "known": ["known", "certain probability", "clear odds"],
}


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer to its canonical form for comparison.

    Args:
        answer: The raw answer string (will be lowercased and stripped)

    Returns:
        Canonical form if a match is found, otherwise the original lowercased answer
    """
    answer_lower = answer.lower().strip()

    # Check each canonical form - require word boundary or exact match
    for canonical, variations in ANSWER_SYNONYMS.items():
        for variation in variations:
            # Exact match
            if answer_lower == variation:
                return canonical
            # Check if answer contains the variation as a complete word
            # (e.g., "i accept" contains "accept" but not just "a" or "e")
            if len(variation) >= 3 and variation in answer_lower:
                # Ensure it's a word boundary match (not substring of another word)
                if re.search(rf'\b{re.escape(variation)}\b', answer_lower):
                    return canonical

    return answer_lower


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
        # Newer models (gpt-5.x, o3, o1) use max_completion_tokens instead of max_tokens
        uses_completion_tokens = any(
            self.model.startswith(prefix)
            for prefix in ("gpt-5", "o3", "o1", "chatgpt-")
        )

        if uses_completion_tokens:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
        else:
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
        # Handle empty content array (e.g., from content filtering)
        if not response.content:
            return ""
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
            # Handle None content from xAI
            return response.content or ""

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
            # Handle None text (e.g., from safety filtering)
            return response.text or ""

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

    # Rate limiting and concurrency
    requests_per_minute: int = 60  # Legacy, kept for compatibility
    max_concurrent_requests: int = 50  # Concurrent API calls via semaphore

    def __post_init__(self):
        """Validate configuration values."""
        if self.requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")


# Pre-compiled regex patterns for answer extraction (module level for performance)
_OPTION_PATTERNS = [
    re.compile(r"(?:I (?:would )?(?:choose|select|prefer|recommend)(?:ing)?)\s*(?:option\s*)?([A-D])", re.IGNORECASE),
    re.compile(r"(?:my (?:choice|selection|preference|answer) is)\s*(?:option\s*)?([A-D])", re.IGNORECASE),
    re.compile(r"(?:option\s*)?([A-D])\s*(?:is (?:the )?(?:best|better|correct|right))", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*([A-D])\s*[:\.\)]"),
    re.compile(r"(?:answer|choice|selection):\s*([A-D])", re.IGNORECASE),
]

_NUMERIC_PATTERNS = [
    re.compile(r"(?:estimate|answer|value|result)[:\s]+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"(?:approximately|about|around)\s+\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\$?([\d,]+(?:\.\d+)?)\s*(?:dollars|percent|%|years|people)", re.IGNORECASE),
]

_YES_NO_PATTERNS = [
    re.compile(r"(?:I (?:would )?(?:recommend|suggest|advise))\s*(yes|no|accepting|rejecting)", re.IGNORECASE),
    re.compile(r"(?:my (?:answer|recommendation) is)\s*(yes|no)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(yes|no)\s*[,\.\:]", re.IGNORECASE),
]

# Priority patterns for explicit "Answer:" lines - checked FIRST before other patterns
# These ensure that when a model follows the prompt format (e.g., "Answer: [your answer]"),
# we always capture their intended answer regardless of verbose reasoning.
_ANSWER_LINE_OPTION = re.compile(
    r"(?:^|\n)\s*\*{0,2}Answer\*{0,2}\s*[:]\s*\*{0,2}(?:option\s*)?([A-D])\b",
    re.IGNORECASE | re.MULTILINE,
)

_ANSWER_LINE_NUMERIC = re.compile(
    r"(?:^|\n)\s*\*{0,2}Answer\*{0,2}\s*[:]\s*\*{0,2}(?:approximately\s+|about\s+|around\s+)?\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE | re.MULTILINE,
)

# Regex matching a negative-context phrase DIRECTLY before a number.
# Catches anchor references like "based on 6500" but NOT conversational
# usage like "Based on my analysis, the answer is 75000".
_NEGATIVE_CONTEXT_RE = re.compile(
    r"(?:"
    r"based\s+on"
    r"|anchored\s+(?:to|by|at)(?:\s+(?:the\s+)?(?:value|number|figure)(?:\s+of))?"
    r"|influenced\s+by(?:\s+(?:the\s+)?(?:initial\s+)?(?:value|number|figure)(?:\s+of))?"
    r"|start(?:ing)?\s+from"
    r"|(?:the\s+)?anchor(?:\s+(?:was|is|of))?"
    r"|reference\s+(?:value\s+)?(?:of|was|is)"
    r"|initial\s+(?:value|estimate|number|figure)(?:\s+(?:of|was|is))?"
    r"|(?:given|provided)(?:\s+(?:the\s+)?(?:value|number|figure))?(?:\s+(?:of|was|is))?"
    r"|regardless\s+of"
    r"|ignoring(?:\s+the)?"
    r"|disregarding"
    r")\s+\$?"
    r"([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


class AnswerExtractor:
    """
    Extracts structured answers from free-form LLM responses.

    Uses pattern matching and optional LLM-based extraction to
    identify the final decision/answer from verbose responses.
    """

    # Use pre-compiled module-level patterns for performance
    OPTION_PATTERNS = _OPTION_PATTERNS
    NUMERIC_PATTERNS = _NUMERIC_PATTERNS
    YES_NO_PATTERNS = _YES_NO_PATTERNS

    def __init__(self, llm_extractor: LLMProvider | None = None):
        """
        Initialize the answer extractor.

        Args:
            llm_extractor: Optional LLM for complex answer extraction
        """
        self.llm_extractor = llm_extractor

    @staticmethod
    def _has_negative_context(text: str, match_start: int) -> bool:
        """Check if a number at match_start is in a negative/rejection context.

        Uses regex to match anchor reference patterns where the negative phrase
        is directly adjacent to the number (e.g., "based on 6500"), not
        conversational usage (e.g., "Based on my analysis, ... 75000").
        """
        for m in _NEGATIVE_CONTEXT_RE.finditer(text):
            if m.start(1) == match_start:
                return True
        return False

    def extract(self, response: str, expected_type: str = "option") -> str | None:
        """
        Extract the final answer from an LLM response.

        Priority: Explicit "Answer:" lines are always preferred when present.
        This ensures models following prompt format have their intended answer captured.

        Args:
            response: Full text response from LLM
            expected_type: Type of answer expected ("option", "numeric", "yes_no", "text")

        Returns:
            Extracted answer string, or None if extraction failed
        """
        if expected_type == "confidence":
            confidence = self.extract_confidence(response)
            if confidence is None:
                return None
            percentage = round(confidence * 100, 2)
            if percentage.is_integer():
                return str(int(percentage))
            return str(percentage)

        # PRIORITY CHECK: Look for explicit "Answer:" line first
        # This takes precedence over all other patterns to ensure we capture
        # the model's intended answer when it follows the requested format.
        if expected_type == "option":
            priority_match = _ANSWER_LINE_OPTION.search(response)
            if priority_match:
                return priority_match.group(1).upper()
        elif expected_type == "numeric":
            # Strip confidence statements before checking Answer: line
            response_no_confidence = re.sub(
                r"(?:confidence|confident|certain|sure)[:\s]+\d{1,3}\s*%?",
                "",
                response,
                flags=re.IGNORECASE,
            )
            priority_match = _ANSWER_LINE_NUMERIC.search(response_no_confidence)
            if priority_match:
                return priority_match.group(1).replace(",", "")

        # Standard pattern matching (fallback when no explicit Answer: line)
        response_lower = response.lower()

        if expected_type == "option":
            patterns = self.OPTION_PATTERNS
        elif expected_type == "numeric":
            # Strip confidence statements before pattern matching to avoid
            # extracting confidence values (e.g., "30%" from "Confidence: 30%")
            response_for_patterns = re.sub(
                r"(?:confidence|confident|certain|sure)[:\s]+\d{1,3}\s*%?",
                "",
                response_lower,
                flags=re.IGNORECASE,
            )
            response_for_patterns = re.sub(
                r"\d{1,3}\s*%?\s*(?:confident|confidence|certain|sure)",
                "",
                response_for_patterns,
                flags=re.IGNORECASE,
            )
            patterns = self.NUMERIC_PATTERNS
        elif expected_type == "yes_no":
            patterns = self.YES_NO_PATTERNS
        else:
            return self._extract_text_answer(response)

        # Use confidence-stripped response for numeric, original for others
        search_text = response_for_patterns if expected_type == "numeric" else response_lower

        for pattern in patterns:
            # Patterns are pre-compiled with re.IGNORECASE
            match = pattern.search(search_text)
            if match:
                if expected_type == "option":
                    return match.group(1).upper()
                if expected_type == "yes_no":
                    value = match.group(1).lower()
                    if value in ("accepting", "rejecting"):
                        return "yes" if value == "accepting" else "no"
                    return value
                # For numeric, skip numbers in negative/anchor context
                if self._has_negative_context(search_text, match.start(1)):
                    continue
                return match.group(1)

        # Fallback: look for the last mentioned option/value
        return self._fallback_extraction(response, expected_type)

    def _extract_text_answer(self, response: str) -> str | None:
        """Extract a text-based answer (last sentence or explicit answer)."""
        # Look for explicit answer markers
        answer_markers = ["therefore", "in conclusion", "my answer is", "final answer"]
        for marker in answer_markers:
            idx = response.lower().rfind(marker)
            if idx != -1:
                # Use sentence-aware splitting (don't split on decimal points)
                remainder = response[idx:]
                sentences = self._split_into_sentences(remainder)
                return sentences[0].strip() if sentences else remainder.strip()

        # Return last sentence as fallback using sentence-aware splitting
        sentences = self._split_into_sentences(response)
        if len(sentences) > 1 and sentences[-1].strip():
            return sentences[-1].strip()
        elif response.strip():
            return response.strip()
        return None

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences, preserving decimal numbers.

        Splits on periods followed by whitespace and uppercase letter,
        or periods at end of text. Does NOT split on decimal points like 0.30.
        """
        # Pattern: period followed by space and capital letter, or period at end
        # Negative lookbehind for digits to avoid splitting "0.30" or "P(X) = 0.5"
        pattern = r'(?<!\d)\.(?=\s+[A-Z])|(?<!\d)\.$'
        sentences = re.split(pattern, text)
        # Filter out empty strings and strip whitespace
        return [s.strip() for s in sentences if s and s.strip()]

    def _fallback_extraction(self, response: str, expected_type: str) -> str | None:
        """Fallback extraction when patterns don't match."""
        if expected_type == "option":
            # Find last mentioned option letter
            options = re.findall(r"\b([A-D])\b", response)
            return options[-1] if options else None

        elif expected_type == "numeric":
            # Try to find numbers near answer keywords (handles "answer is X", "answer: X", "answer X")
            answer_context = re.search(
                r"(?:estimate|answer|value|result|approximately|about|around|probability|chance|likelihood|odds)(?:\s+is\s+|\s*:\s*|\s+)\$?([\d,]+(?:\.\d+)?)",
                response,
                re.IGNORECASE,
            )
            if answer_context and not self._has_negative_context(response, answer_context.start(1)):
                return answer_context.group(1).replace(",", "")

            # Find numbers with units (excluding confidence percentages),
            # filtering out anchor/rejection context
            valid_unit_numbers = []
            for match in re.finditer(
                r"\$?([\d,]+(?:\.\d+)?)\s*(?:dollars?|people|years|months|days|units?)\b",
                response,
                re.IGNORECASE,
            ):
                if not self._has_negative_context(response, match.start(1)):
                    valid_unit_numbers.append(match.group(1))
            if valid_unit_numbers:
                return valid_unit_numbers[-1].replace(",", "")

            # Exclude numbers that are part of confidence statements
            response_no_confidence = re.sub(
                r"\d{1,3}\s*%?\s*(?:confident|confidence|certain|sure)",
                "",
                response,
                flags=re.IGNORECASE,
            )
            # Find all numbers, filtering out those in anchor/rejection context
            for match in re.finditer(r"[\d,]+(?:\.\d+)?", response_no_confidence):
                if not self._has_negative_context(response_no_confidence, match.start()):
                    return match.group().replace(",", "")
            return None

        elif expected_type == "yes_no":
            response_lower = response.lower()

            # Check for negation patterns first (higher priority)
            negation_patterns = [
                r"\bnot\s+accept",
                r"\bwould\s+not\b",
                r"\bdon'?t\s+accept",
                r"\bshould\s+not\b",
                r"\bdo\s+not\b",
                r"\bcannot\s+recommend",
                r"\bwouldn'?t\s+recommend",
                r"\bwouldn'?t\s+accept",
                r"\brefuse\b",
                r"\bdecline\b",
                r"\breject\b",
            ]

            for pattern in negation_patterns:
                if re.search(pattern, response_lower):
                    return "no"

            # Check for affirmative patterns
            affirmative_patterns = [
                r"\byes\b",
                r"\baccept\b",
                r"\bwould\s+recommend\b",
                r"\bshould\s+proceed\b",
                r"\bagree\b",
            ]

            for pattern in affirmative_patterns:
                if re.search(pattern, response_lower):
                    return "yes"

            # Fallback: check for standalone "no"
            if re.search(r"\bno\b", response_lower):
                return "no"

            return None

        return None

    def extract_confidence(self, response: str) -> float | None:
        """Extract stated confidence level from response.

        Priority order:
        1. Explicit 'Confidence:' line markers (highest priority)
        2. Inline confidence statements with keywords

        Supports both percentage (0-100) and fractional (0-1) formats.

        Returns:
            Confidence value clamped to [0.0, 1.0], or None if not found.
        """
        # Priority 1: Explicit "Confidence:" line markers (most reliable)
        # Matches: "Confidence: 85%", "Confidence: 85", "Confidence: 0.85"
        explicit_patterns = [
            # "Confidence: 0.85" or "Confidence: .85" (decimal fraction) - check first
            r"(?:^|\n)\s*confidence\s*:\s*(0?\.\d+)",
            # "Confidence: 85%" or "Confidence: 85" (integer percentage)
            # Negative lookahead (?!\.) prevents matching "0" in "0.85"
            r"(?:^|\n)\s*confidence\s*:\s*(\d{1,3})(?!\.)\s*%?",
        ]

        for pattern in explicit_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return self._normalize_confidence(match.group(1))

        # Priority 2: Inline confidence statements (fallback)
        # Matches: "85% confident", "I am 70% certain", "confidence is 80"
        inline_patterns = [
            # "85% confident" or "85 percent confident"
            r"(\d{1,3})\s*%?\s*(?:percent\s+)?(?:confident|certain|sure)",
            # "confidence is 70" or "certainty: 80%"
            r"(?:confidence|certainty)\s*(?:is|of|:)\s*(\d{1,3})\s*%?",
            # "0.85 confident" (fractional inline)
            r"(0?\.\d+)\s*(?:confident|certain|sure)",
        ]

        for pattern in inline_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return self._normalize_confidence(match.group(1))

        return None

    def _normalize_confidence(self, value_str: str) -> float:
        """Normalize confidence value to [0.0, 1.0] range.

        Args:
            value_str: String representation of confidence (e.g., "85", "0.85", ".75")

        Returns:
            Confidence value clamped to [0.0, 1.0]
        """
        value = float(value_str)

        # Values > 1 are treated as percentages (divide by 100)
        # Values <= 1 are treated as fractions (keep as-is)
        if value > 1:
            normalized = value / 100
        else:
            normalized = value

        return max(0.0, min(1.0, normalized))


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
        judge: LLMJudge | None = None,
    ):
        """
        Initialize the evaluator.

        Args:
            provider: LLM provider for test execution
            config: Evaluation configuration
            answer_extractor: Custom answer extractor (defaults to built-in)
            judge: Optional LLM judge for fallback scoring when regex fails
        """
        self.provider = provider
        self.config = config or EvaluationConfig()
        self.extractor = answer_extractor or AnswerExtractor()
        self.judge = judge
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    def _resolve_biased_answer(
        self,
        instance: CognitiveBiasInstance,
        condition: str,
    ) -> str:
        """Resolve the correct expected biased answer for frame-dependent biases.

        For gain_loss_framing, the biased answer depends on which frame was used:
        - Gain frame (WEAK/MODERATE): biased = gain_frame_biased (typically "A")
        - Loss frame (STRONG/ADVERSARIAL): biased = loss_frame_biased (typically "B")

        For non-framing biases, returns instance.expected_biased_response unchanged.
        """
        biased = instance.expected_biased_response
        if not (instance.metadata and "frame_map" in instance.metadata):
            return biased

        frame_map = instance.metadata["frame_map"]
        # Match intensity key in condition (e.g., "treatment_strong" contains "strong")
        matched = False
        for intensity_key, frame in frame_map.items():
            if intensity_key in condition:
                matched = True
                if frame == "loss":
                    biased = instance.metadata.get("loss_frame_biased", biased)
                elif frame == "gain":
                    biased = instance.metadata.get("gain_frame_biased", biased)
                break

        if not matched:
            # No intensity token found in condition. This happens with context
            # evaluator conditions (e.g., "context_novice_low", "expertise_expert",
            # "stakes_high") and control conditions. Default to gain-frame biased
            # answer, since context evaluators use moderate intensity (gain frame)
            # by default.
            biased = instance.metadata.get("gain_frame_biased", biased)

        return biased

    def _resolve_rational_answer(
        self,
        instance: CognitiveBiasInstance,
        condition: str,
    ) -> str:
        """Resolve the correct expected rational answer for frame-dependent biases.

        For gain_loss_framing, the rational answer depends on which frame was used:
        - Gain frame (WEAK/MODERATE): rational = gain_frame_rational (typically "B")
        - Loss frame (STRONG/ADVERSARIAL): rational = loss_frame_rational (typically "A")

        For non-framing biases, returns instance.expected_rational_response unchanged.
        """
        rational = instance.expected_rational_response
        if not (instance.metadata and "frame_map" in instance.metadata):
            return rational

        frame_map = instance.metadata["frame_map"]
        matched = False
        for intensity_key, frame in frame_map.items():
            if intensity_key in condition:
                matched = True
                if frame == "loss":
                    rational = instance.metadata.get("loss_frame_rational", rational)
                elif frame == "gain":
                    rational = instance.metadata.get("gain_frame_rational", rational)
                break

        if not matched:
            # No intensity token found in condition (context evaluators, control, etc.).
            # Default to gain-frame rational answer, matching the assumption in
            # _resolve_biased_answer.
            rational = instance.metadata.get("gain_frame_rational", rational)

        return rational

    async def _call_provider(self, prompt: str) -> str:
        """Make API call with semaphore-based concurrency limiting."""
        async with self._semaphore:
            return await self.provider.complete(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

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
        """Run multiple trials for a single condition concurrently."""

        async def run_single_trial(trial_num: int) -> TestResult:
            """Execute a single trial with timing and scoring."""
            start_time = time.time()

            try:
                response = await self._call_provider(prompt)
            except Exception as e:
                response = f"ERROR: {str(e)}"

            elapsed_ms = (time.time() - start_time) * 1000

            # Extract answer and confidence (skip for error responses)
            answer_type = self._infer_answer_type(instance)
            if response.startswith("ERROR:"):
                extracted = None
                confidence = None
            else:
                extracted = self.extractor.extract(response, answer_type)
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
                metadata={"trial": trial_num, "answer_type": answer_type},
            )

            # Score the response for bias (only if we have expected answers)
            if (instance.expected_rational_response and
                instance.expected_biased_response and
                not instance.expected_rational_response.startswith("[") and
                not instance.expected_biased_response.startswith("[")):
                rational_answer = self._resolve_rational_answer(instance, condition)
                biased_answer = self._resolve_biased_answer(instance, condition)
                is_biased, bias_score = self.score_response(
                    result,
                    rational_answer,
                    biased_answer,
                )
                result.is_biased = is_biased
                result.bias_score = bias_score
                if result.is_biased is not None:
                    result.metadata["scoring_method"] = "regex"

            # LLM judge fallback when regex extraction fails
            if result.is_biased is None and self.judge is not None:
                try:
                    # Extract base bias id (may be prefixed like "abcd1234_anchoring_effect")
                    bias_id = instance.bias_id
                    bias_def = get_bias_by_id(bias_id)
                    if bias_def is None and "_" in bias_id:
                        # Try stripping a UUID/hash prefix
                        base_id = bias_id.split("_", 1)[-1]
                        bias_def = get_bias_by_id(base_id)
                    if bias_def is not None:
                        judge_result = await self.judge.score(
                            bias_name=bias_def.name,
                            bias_description=bias_def.description,
                            system1_mechanism=bias_def.system1_mechanism,
                            control_prompt=instance.control_prompt,
                            treatment_prompt=prompt,
                            expected_rational=self._resolve_rational_answer(instance, condition),
                            expected_biased=self._resolve_biased_answer(instance, condition),
                            model_response=response,
                            answer_type=self._infer_answer_type(instance),
                        )
                        result.is_biased = judge_result.is_biased
                        result.bias_score = judge_result.bias_score
                        result.extracted_answer = judge_result.extracted_answer
                        result.metadata["scoring_method"] = "llm_judge"
                        result.metadata["judge_confidence"] = judge_result.confidence
                        result.metadata["judge_justification"] = judge_result.justification
                except Exception as e:
                    logger.warning("LLM judge failed for %s: %s", instance.bias_id, e)

            return result

        # Run all trials concurrently (semaphore limits actual concurrency)
        tasks = [run_single_trial(t) for t in range(self.config.num_trials)]
        results = await asyncio.gather(*tasks)
        return list(results)

    def _infer_answer_type(self, instance: CognitiveBiasInstance) -> str:
        """Infer the expected answer type from the instance.

        First checks metadata for explicit answer_type, then falls back to prompt inference.
        """
        # Check metadata first for explicit answer_type
        if instance.metadata and "answer_type" in instance.metadata:
            answer_type = instance.metadata["answer_type"]
            # Handle categorical as non-evaluable
            if answer_type == "categorical":
                return "text"  # Will extract but scoring handles placeholder answers
            if answer_type == "choice":
                return "option"
            return answer_type

        # Fall back to prompt-based inference
        prompt_lower = instance.control_prompt.lower()

        if any(opt in prompt_lower for opt in ["option a", "option b", "program a", "program b"]):
            return "option"
        elif any(word in prompt_lower for word in ["estimate", "how much", "how many", "probability", "confidence:"]):
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

        # Track progress with thread-safe counter
        completed_count = 0
        progress_lock = asyncio.Lock()

        async def eval_with_progress(instance: CognitiveBiasInstance) -> list[TestResult]:
            nonlocal completed_count
            results = await self.evaluate_instance(instance, model_id)
            async with progress_lock:
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, len(instances))
            return results

        # Run all instances concurrently (semaphore limits actual API concurrency)
        tasks = [eval_with_progress(inst) for inst in instances]
        all_results = await asyncio.gather(*tasks)

        # Flatten results
        for results in all_results:
            session.results.extend(results)

        # Compute per-bias unknown rates for downstream reporting
        unknown_counts: dict[str, dict[str, int]] = {}
        for result in session.results:
            bias_id = result.instance.bias_id
            if bias_id not in unknown_counts:
                unknown_counts[bias_id] = {"total": 0, "unknown": 0}
            unknown_counts[bias_id]["total"] += 1
            if result.is_biased is None:
                unknown_counts[bias_id]["unknown"] += 1
        unknown_rates = {
            bid: {
                "total": c["total"],
                "unknown": c["unknown"],
                "rate": c["unknown"] / c["total"] if c["total"] > 0 else 0.0,
            }
            for bid, c in unknown_counts.items()
        }
        session.metrics["unknown_rates_by_bias"] = unknown_rates

        session.end_time = datetime.now().isoformat()
        return session

    @staticmethod
    def _is_descriptive_answer(answer: str) -> bool:
        """Check if an expected answer is a descriptive prose string.

        Descriptive answers (e.g., "based on statistical data rather than
        memorable examples") cannot be scored via regex matching and require
        LLM judge fallback for proper evaluation.

        Returns True for answers that are too long/complex for regex matching.
        """
        # Short answers (<=5 words) are concrete enough for regex matching
        if len(answer.split()) <= 5:
            return False
        # Try to parse as a number — numeric answers are concrete
        try:
            float(answer.replace(",", "").replace("$", ""))
            return False
        except (ValueError, TypeError):
            pass
        # Single-word canonical answers (e.g., "accept", "reject", "a", "b") are concrete
        if answer.strip().lower() in ANSWER_SYNONYMS:
            return False
        # Multi-word answers longer than 5 words are likely descriptive
        return True

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

        # Detect descriptive expected answers that can't be matched via regex.
        # These are long prose descriptions like "based on statistical data rather
        # than memorable examples" — they require LLM judge for proper scoring.
        if self._is_descriptive_answer(rational_answer) or self._is_descriptive_answer(biased_answer):
            logger.debug(
                "Descriptive expected answer for %s requires LLM judge: rational=%r, biased=%r",
                result.instance.bias_id, rational_answer[:50], biased_answer[:50],
            )
            result.metadata["requires_llm_judge"] = True
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

        # Fall back to normalized string matching for non-numeric answers
        extracted_norm = normalize_answer(extracted)
        rational_norm = normalize_answer(rational)
        biased_norm = normalize_answer(biased)

        if extracted_norm == rational_norm:
            return False, 0.0
        elif extracted_norm == biased_norm:
            return True, 1.0

        # Also try exact matching as fallback
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

            # Score the response for bias
            if (instance.expected_rational_response and
                instance.expected_biased_response and
                not instance.expected_rational_response.startswith("[") and
                not instance.expected_biased_response.startswith("[")):
                condition = f"persistent_round_{round_num}"
                rational_answer = self._resolve_rational_answer(instance, condition)
                biased_answer = self._resolve_biased_answer(instance, condition)
                is_biased, bias_score = self.score_response(
                    result,
                    rational_answer,
                    biased_answer,
                )
                result.is_biased = is_biased
                result.bias_score = bias_score

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

        pre_result = TestResult(
            instance=instance,
            model_id=model_id,
            condition="adaptive_pre_feedback",
            prompt_used=initial_prompt,
            model_response=initial_response,
            extracted_answer=self.extractor.extract(initial_response, self._infer_answer_type(instance)),
            response_time_ms=elapsed_ms,
            metadata={"temporal_condition": "adaptive", "phase": "pre_feedback"},
        )

        # Score the pre-feedback response (uses STRONG intensity → loss frame if applicable)
        if (instance.expected_rational_response and
            instance.expected_biased_response and
            not instance.expected_rational_response.startswith("[") and
            not instance.expected_biased_response.startswith("[")):
            pre_condition = "adaptive_pre_feedback_strong"
            rational_answer = self._resolve_rational_answer(instance, pre_condition)
            biased_answer = self._resolve_biased_answer(instance, pre_condition)
            is_biased, bias_score = self.score_response(
                pre_result,
                rational_answer,
                biased_answer,
            )
            pre_result.is_biased = is_biased
            pre_result.bias_score = bias_score

        results.append(pre_result)

        # Corrective feedback
        feedback_prompt = f"""
Your previous response showed signs of {instance.bias_id.replace('_', ' ')}.

{instance.control_prompt}

Please reconsider, being careful to avoid this cognitive bias.
"""

        start_time = time.time()
        feedback_response = await self.provider.complete(feedback_prompt, self.config.max_tokens)
        elapsed_ms = (time.time() - start_time) * 1000

        post_result = TestResult(
            instance=instance,
            model_id=model_id,
            condition="adaptive_post_feedback",
            prompt_used=feedback_prompt,
            model_response=feedback_response,
            extracted_answer=self.extractor.extract(feedback_response, self._infer_answer_type(instance)),
            response_time_ms=elapsed_ms,
            metadata={"temporal_condition": "adaptive", "phase": "post_feedback"},
        )

        # Score the post-feedback response (uses control prompt, default frame)
        if (instance.expected_rational_response and
            instance.expected_biased_response and
            not instance.expected_rational_response.startswith("[") and
            not instance.expected_biased_response.startswith("[")):
            post_condition = "adaptive_post_feedback"
            rational_answer = self._resolve_rational_answer(instance, post_condition)
            biased_answer = self._resolve_biased_answer(instance, post_condition)
            is_biased, bias_score = self.score_response(
                post_result,
                rational_answer,
                biased_answer,
            )
            post_result.is_biased = is_biased
            post_result.bias_score = bias_score

        results.append(post_result)

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
                rational_answer = self._resolve_rational_answer(
                    instance, result.condition
                )
                biased_answer = self._resolve_biased_answer(
                    instance, result.condition
                )
                is_biased, bias_score = self.score_response(
                    result,
                    rational_answer,
                    biased_answer,
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
                rational_answer = self._resolve_rational_answer(
                    instance, result.condition
                )
                biased_answer = self._resolve_biased_answer(
                    instance, result.condition
                )
                is_biased, bias_score = self.score_response(
                    result,
                    rational_answer,
                    biased_answer,
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
                rational_answer = self._resolve_rational_answer(
                    instance, result.condition
                )
                biased_answer = self._resolve_biased_answer(
                    instance, result.condition
                )
                is_biased, bias_score = self.score_response(
                    result,
                    rational_answer,
                    biased_answer,
                )
                result.is_biased = is_biased
                result.bias_score = bias_score

            results.append(result)

        return results
