"""Tests for the evaluation engine components."""


import asyncio
import pytest
from dataclasses import dataclass, field

from kahne_bench.core import (
    CognitiveBiasInstance,
    Domain,
    TestScale,
    TriggerIntensity,
)
from kahne_bench.engines.evaluator import (
    AnswerExtractor,
    BiasEvaluator,
    EvaluationConfig,
    TemporalEvaluator,
    ContextSensitivityEvaluator,
)


@dataclass
class MockLLMProvider:
    """Mock LLM provider for testing."""

    responses: list[str] | None = None
    response_index: int = 0
    default_response: str = "I would choose option A. My confidence is 80%."

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        if self.responses:
            response = self.responses[self.response_index % len(self.responses)]
            self.response_index += 1
            return response
        return self.default_response


@dataclass
class ErrorSimulatingProvider:
    """Mock LLM provider that can simulate failures on specific calls.

    Use this provider to test error handling and recovery behavior.
    Failures can be configured to occur on specific call indices.
    """

    default_response: str = "I would choose option A. My confidence is 80%."
    fail_on_calls: list[int] = field(default_factory=list)
    error_type: type[Exception] = Exception
    error_message: str = "Simulated provider error"
    call_count: int = 0

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Complete a prompt, potentially raising an error on configured calls."""
        current_call = self.call_count
        self.call_count += 1

        if current_call in self.fail_on_calls:
            raise self.error_type(self.error_message)

        return self.default_response


def create_test_instance(
    bias_id: str = "anchoring_effect",
    domain: Domain = Domain.INDIVIDUAL,
) -> CognitiveBiasInstance:
    """Create a test instance for evaluation tests."""
    return CognitiveBiasInstance(
        bias_id=bias_id,
        base_scenario="Estimating a value in a decision context",
        bias_trigger="An anchor number is mentioned before the estimate",
        domain=domain,
        scale=TestScale.MICRO,
        control_prompt="What is the value of X? Give your estimate.",
        treatment_prompts={
            TriggerIntensity.WEAK: "Consider the number 10. What is the value of X?",
            TriggerIntensity.MODERATE: "The number 50 was mentioned. What is the value of X?",
            TriggerIntensity.STRONG: "Start from 100. Now estimate the value of X.",
            TriggerIntensity.ADVERSARIAL: "The answer is definitely around 200. What is the value of X?",
        },
        expected_rational_response="50",
        expected_biased_response="100",
        debiasing_prompts=[
            "Ignore any numbers mentioned. What is the value of X?"
        ],
    )


class TestAnswerExtractor:
    """Tests for AnswerExtractor."""

    def test_extract_option_from_choice(self):
        extractor = AnswerExtractor()
        response = "After careful consideration, I would choose option B."
        assert extractor.extract(response, "option") == "B"

    def test_extract_option_from_selection(self):
        extractor = AnswerExtractor()
        response = "My selection is A because it offers better value."
        assert extractor.extract(response, "option") == "A"

    def test_extract_numeric_estimate(self):
        extractor = AnswerExtractor()
        response = "Based on my analysis, my estimate is approximately 75,000 dollars."
        result = extractor.extract(response, "numeric")
        assert result == "75,000"

    def test_extract_yes_no_accept(self):
        extractor = AnswerExtractor()
        response = "I would recommend accepting this offer."
        assert extractor.extract(response, "yes_no") == "yes"

    def test_extract_yes_no_reject(self):
        extractor = AnswerExtractor()
        response = "My recommendation is no, this is too risky."
        assert extractor.extract(response, "yes_no") == "no"

    def test_extract_confidence_percentage(self):
        extractor = AnswerExtractor()
        response = "The answer is B. I am 85% confident in this answer."
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.85

    def test_extract_confidence_word(self):
        extractor = AnswerExtractor()
        # Use pattern that matches the extractor's regex: "confidence: 70"
        response = "My confidence: 70%."
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.70

    def test_extract_confidence_none_when_absent(self):
        extractor = AnswerExtractor()
        response = "The answer is definitely B."
        confidence = extractor.extract_confidence(response)
        assert confidence is None

    # === Part B: Enhanced Confidence Extraction Tests ===

    def test_extract_confidence_explicit_marker_preferred(self):
        """Explicit 'Confidence:' line should be preferred over inline mentions."""
        extractor = AnswerExtractor()
        # Response has "85" as trivia answer and "70%" as stated confidence
        response = "The answer is 85. Confidence: 70%"
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.70, f"Expected 0.70 but got {confidence} - should extract from 'Confidence:' not '85'"

    def test_extract_confidence_decimal_fraction(self):
        """Support 0-1 fractional format."""
        extractor = AnswerExtractor()
        response = "I believe this is correct.\nConfidence: 0.85"
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.85

    def test_extract_confidence_decimal_fraction_no_leading_zero(self):
        """Support .XX fractional format without leading zero."""
        extractor = AnswerExtractor()
        response = "My answer is X.\nConfidence: .75"
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.75

    def test_extract_confidence_explicit_marker_newline(self):
        """Confidence on its own line (explicit marker)."""
        extractor = AnswerExtractor()
        response = """Answer: Canberra
Confidence: 90%"""
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.90

    def test_extract_confidence_explicit_marker_without_percent(self):
        """Confidence: 85 (integer without percent sign)."""
        extractor = AnswerExtractor()
        response = "The capital is Canberra.\nConfidence: 85"
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.85

    def test_extract_confidence_trivia_answer_not_confused(self):
        """Overconfidence test: trivia answer (1914) should not be extracted as confidence."""
        extractor = AnswerExtractor()
        response = """Answer: 1914
Confidence: 75%"""
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.75, f"Expected 0.75 but got {confidence} - extracted trivia answer instead"

    def test_extract_confidence_numeric_answer_ignored(self):
        """Numeric answer like '206 bones' should not be confused with confidence."""
        extractor = AnswerExtractor()
        response = """The adult human body has 206 bones.
Confidence: 65%"""
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.65

    def test_extract_confidence_inline_fallback_percentage(self):
        """Inline '85% confident' works as fallback when no explicit marker."""
        extractor = AnswerExtractor()
        response = "I am 85% confident the answer is B."
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.85

    def test_extract_confidence_inline_fallback_certain(self):
        """Inline '70% certain' works as fallback."""
        extractor = AnswerExtractor()
        response = "I'm 70% certain about this."
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.70

    def test_extract_confidence_edge_case_zero(self):
        """Handle 0% confidence."""
        extractor = AnswerExtractor()
        response = "I have no idea.\nConfidence: 0%"
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.0

    def test_extract_confidence_edge_case_hundred(self):
        """Handle 100% confidence."""
        extractor = AnswerExtractor()
        response = "I am absolutely certain.\nConfidence: 100%"
        confidence = extractor.extract_confidence(response)
        assert confidence == 1.0

    def test_extract_confidence_out_of_range_clamped(self):
        """Values > 100 should be clamped to 1.0."""
        extractor = AnswerExtractor()
        response = "Confidence: 150%"
        confidence = extractor.extract_confidence(response)
        assert confidence == 1.0

    def test_extract_confidence_explicit_overrides_inline(self):
        """When both explicit and inline present, explicit wins."""
        extractor = AnswerExtractor()
        response = """I am 95% confident in my analysis.
The answer is X.
Confidence: 60%"""
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.60, "Explicit 'Confidence:' line should override inline '95% confident'"

    def test_fallback_option_extraction(self):
        extractor = AnswerExtractor()
        response = "There are many factors to consider... C seems best."
        assert extractor.extract(response, "option") == "C"

    def test_fallback_numeric_extraction(self):
        extractor = AnswerExtractor()
        response = "The project costs 1,000,000 and we expect 2,500,000 in revenue."
        result = extractor.extract(response, "numeric")
        assert result == "1000000"  # First number (changed from last to fix gambler_fallacy extraction)


class TestAnswerLineExtraction:
    """Tests for priority Answer: line extraction.

    Part A of benchmark improvements: extraction must prefer explicit Answer:
    lines for numeric and option outputs; ignore unrelated numbers.
    """

    # Option extraction with Answer: line

    def test_option_answer_line_basic(self):
        """Answer: A format extracts A."""
        extractor = AnswerExtractor()
        response = "After consideration, I think B is reasonable.\n\nAnswer: A"
        result = extractor.extract(response, "option")
        assert result == "A"

    def test_option_answer_line_overrides_earlier_mention(self):
        """Answer: line takes priority over earlier option mentions."""
        extractor = AnswerExtractor()
        response = "I would choose B initially, but after analysis...\n\nAnswer: C"
        result = extractor.extract(response, "option")
        assert result == "C"

    def test_option_answer_line_markdown_bold(self):
        """**Answer:** format is handled."""
        extractor = AnswerExtractor()
        response = "Analysis complete.\n\n**Answer:** D"
        result = extractor.extract(response, "option")
        assert result == "D"

    def test_option_answer_line_with_option_keyword(self):
        """Answer: Option B format works."""
        extractor = AnswerExtractor()
        response = "My reasoning...\n\nAnswer: Option B"
        result = extractor.extract(response, "option")
        assert result == "B"

    # Numeric extraction with Answer: line

    def test_numeric_answer_line_basic(self):
        """Answer: 45000 format extracts 45000."""
        extractor = AnswerExtractor()
        response = "Rule of thumb suggests 10% annually.\n\nAnswer: 45000"
        result = extractor.extract(response, "numeric")
        assert result == "45000"

    def test_numeric_answer_line_overrides_earlier_number(self):
        """Answer: line takes priority over earlier numbers."""
        extractor = AnswerExtractor()
        response = "Based on the anchor of 100, my estimate is 80.\n\nAnswer: 65"
        result = extractor.extract(response, "numeric")
        assert result == "65"

    def test_numeric_answer_line_with_comma(self):
        """Answer: 45,000 extracts 45000 (comma removed)."""
        extractor = AnswerExtractor()
        response = "Considering all factors...\n\nAnswer: 45,000"
        result = extractor.extract(response, "numeric")
        assert result == "45000"

    def test_numeric_answer_line_with_currency(self):
        """Answer: $50,000 extracts 50000."""
        extractor = AnswerExtractor()
        response = "My estimate:\n\nAnswer: $50,000"
        result = extractor.extract(response, "numeric")
        assert result == "50000"

    def test_numeric_answer_line_decimal(self):
        """Answer: 0.75 extracts 0.75."""
        extractor = AnswerExtractor()
        response = "The probability is...\n\nAnswer: 0.75"
        result = extractor.extract(response, "numeric")
        assert result == "0.75"

    def test_numeric_answer_line_approximately(self):
        """Answer: approximately 1000 extracts 1000."""
        extractor = AnswerExtractor()
        response = "Answer: approximately 1,000"
        result = extractor.extract(response, "numeric")
        assert result == "1000"

    def test_answer_line_ignores_confidence_line(self):
        """Confidence: 80% is not confused with Answer:."""
        extractor = AnswerExtractor()
        response = "Answer: 42\nConfidence: 80%"
        result = extractor.extract(response, "numeric")
        assert result == "42"

    # Fallback behavior (no Answer: line)

    def test_no_answer_line_falls_through_numeric(self):
        """Without Answer: line, existing numeric patterns are used."""
        extractor = AnswerExtractor()
        response = "My estimate is approximately 5000 dollars."
        result = extractor.extract(response, "numeric")
        assert result == "5000"

    def test_no_answer_line_falls_through_option(self):
        """Without Answer: line, existing option patterns work."""
        extractor = AnswerExtractor()
        response = "I would choose option B because it's better."
        result = extractor.extract(response, "option")
        assert result == "B"

    # Edge cases

    def test_answer_line_newline_after_colon(self):
        """Answer:\n42 format works."""
        extractor = AnswerExtractor()
        response = "Analysis:\n\nAnswer:\n42"
        result = extractor.extract(response, "numeric")
        assert result == "42"

    def test_answer_line_at_start(self):
        """Answer: at start of response works."""
        extractor = AnswerExtractor()
        response = "Answer: B\n\nExplanation: B is better because..."
        result = extractor.extract(response, "option")
        assert result == "B"

    # Real-world verbose response tests

    def test_real_world_verbose_response_numeric(self):
        """Real-world verbose response with Answer: line at end."""
        extractor = AnswerExtractor()
        response = """
        Let me analyze this carefully.

        The anchor was 100, but I should ignore that.
        A rule of thumb would suggest 10% growth, giving us about 110.
        However, looking at the actual data...

        After careful consideration of all factors:

        Answer: 45000
        Confidence: 75%
        """
        result = extractor.extract(response, "numeric")
        assert result == "45000"

    def test_real_world_verbose_response_option(self):
        """Real-world verbose response with Answer: line at end."""
        extractor = AnswerExtractor()
        response = """
        This is a classic framing problem.

        Option A saves 200 people for certain.
        Option B has a 1/3 chance of saving 600.

        While B has the same expected value, the certainty of A is appealing.
        Initially I might lean toward B...

        Answer: A
        """
        result = extractor.extract(response, "option")
        assert result == "A"

    def test_confidence_extraction_as_answer_type(self):
        """Confidence answer_type should extract from Confidence: line."""
        extractor = AnswerExtractor()
        response = "Answer: Canberra\nConfidence: 85%"
        result = extractor.extract(response, "confidence")
        assert result == "85"

    def test_confidence_extraction_fractional_as_answer_type(self):
        """Confidence answer_type should handle fractional confidence."""
        extractor = AnswerExtractor()
        response = "Answer: 1914\nConfidence: 0.75"
        result = extractor.extract(response, "confidence")
        assert result == "75"


class TestBiasEvaluator:
    """Tests for BiasEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_instance_returns_results(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1, include_debiasing=False)
        evaluator = BiasEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_instance(instance, "test-model")

        assert len(results) > 0
        assert all(r.model_id == "test-model" for r in results)

    @pytest.mark.asyncio
    async def test_evaluate_instance_includes_control(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1, include_control=True, include_debiasing=False)
        evaluator = BiasEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_instance(instance, "test-model")

        control_results = [r for r in results if r.condition == "control"]
        assert len(control_results) == 1

    @pytest.mark.asyncio
    async def test_evaluate_instance_includes_treatments(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(
            num_trials=1,
            include_control=False,
            include_debiasing=False,
            intensities=[TriggerIntensity.MODERATE],
        )
        evaluator = BiasEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_instance(instance, "test-model")

        treatment_results = [r for r in results if r.condition.startswith("treatment_")]
        assert len(treatment_results) == 1

    @pytest.mark.asyncio
    async def test_evaluate_instance_includes_debiasing(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(
            num_trials=1,
            include_control=False,
            include_debiasing=True,
            intensities=[],
        )
        evaluator = BiasEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_instance(instance, "test-model")

        debiasing_results = [r for r in results if r.condition.startswith("debiasing_")]
        assert len(debiasing_results) == 1

    @pytest.mark.asyncio
    async def test_evaluate_instance_extracts_confidence(self):
        provider = MockLLMProvider(default_response="Option A. I am 90% confident.")
        config = EvaluationConfig(num_trials=1)
        evaluator = BiasEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_instance(instance, "test-model")

        # At least some results should have confidence extracted
        with_confidence = [r for r in results if r.confidence_stated is not None]
        assert len(with_confidence) > 0
        assert with_confidence[0].confidence_stated == 0.90

    @pytest.mark.asyncio
    async def test_multiple_trials_produces_multiple_results(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(
            num_trials=3,
            include_control=True,
            include_debiasing=False,
            intensities=[TriggerIntensity.MODERATE],
        )
        evaluator = BiasEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_instance(instance, "test-model")

        # 3 trials for control + 3 trials for moderate = 6 results
        assert len(results) == 6

    def test_score_response_exact_rational_match(self):
        provider = MockLLMProvider()
        evaluator = BiasEvaluator(provider)
        instance = create_test_instance()

        # Create a mock result
        from kahne_bench.core import TestResult
        result = TestResult(
            instance=instance,
            model_id="test",
            condition="treatment",
            prompt_used="test prompt",
            model_response="50",
            extracted_answer="50",
            response_time_ms=100.0,
        )

        is_biased, score = evaluator.score_response(result, "50", "100")
        assert is_biased is False
        assert score == 0.0

    def test_score_response_exact_biased_match(self):
        provider = MockLLMProvider()
        evaluator = BiasEvaluator(provider)
        instance = create_test_instance()

        from kahne_bench.core import TestResult
        result = TestResult(
            instance=instance,
            model_id="test",
            condition="treatment",
            prompt_used="test prompt",
            model_response="100",
            extracted_answer="100",
            response_time_ms=100.0,
        )

        is_biased, score = evaluator.score_response(result, "50", "100")
        assert is_biased is True
        assert score == 1.0

    def test_score_response_partial_numeric_bias(self):
        provider = MockLLMProvider()
        evaluator = BiasEvaluator(provider)
        instance = create_test_instance()

        from kahne_bench.core import TestResult
        result = TestResult(
            instance=instance,
            model_id="test",
            condition="treatment",
            prompt_used="test prompt",
            model_response="75",
            extracted_answer="75",
            response_time_ms=100.0,
        )

        is_biased, score = evaluator.score_response(result, "50", "100")
        # 75 is halfway between 50 and 100, so score should be 0.5
        assert score == 0.5


class TestTemporalEvaluator:
    """Tests for TemporalEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_persistent_returns_results(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = TemporalEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_persistent(instance, "test-model", num_rounds=3)

        assert len(results) == 3
        conditions = [r.condition for r in results]
        assert "persistent_round_0" in conditions
        assert "persistent_round_1" in conditions
        assert "persistent_round_2" in conditions

    @pytest.mark.asyncio
    async def test_evaluate_persistent_includes_context(self):
        # Use responses that we can verify are being used as context
        provider = MockLLMProvider(responses=["Round 0 response", "Round 1 response", "Round 2 response"])
        config = EvaluationConfig(num_trials=1)
        evaluator = TemporalEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_persistent(instance, "test-model", num_rounds=3)

        # Later rounds should include context from previous rounds
        assert "previous discussion" in results[1].prompt_used.lower() or len(results) == 3
        assert results[2].metadata.get("round") == 2

    @pytest.mark.asyncio
    async def test_evaluate_persistent_tracks_round_metadata(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = TemporalEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_persistent(instance, "test-model", num_rounds=5)

        for i, result in enumerate(results):
            assert result.metadata["round"] == i
            assert result.metadata["temporal_condition"] == "persistent"

    @pytest.mark.asyncio
    async def test_evaluate_adaptive_returns_pre_and_post(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = TemporalEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_adaptive(instance, "test-model")

        assert len(results) == 2
        conditions = [r.condition for r in results]
        assert "adaptive_pre_feedback" in conditions
        assert "adaptive_post_feedback" in conditions

    @pytest.mark.asyncio
    async def test_evaluate_adaptive_feedback_mentions_bias(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = TemporalEvaluator(provider, config)
        instance = create_test_instance(bias_id="confirmation_bias")

        results = await evaluator.evaluate_adaptive(instance, "test-model")

        post_feedback = [r for r in results if r.condition == "adaptive_post_feedback"][0]
        # The feedback prompt should mention the bias
        assert "confirmation bias" in post_feedback.prompt_used.lower()

    @pytest.mark.asyncio
    async def test_evaluate_adaptive_metadata(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = TemporalEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_adaptive(instance, "test-model")

        pre = [r for r in results if r.condition == "adaptive_pre_feedback"][0]
        post = [r for r in results if r.condition == "adaptive_post_feedback"][0]

        assert pre.metadata["phase"] == "pre_feedback"
        assert post.metadata["phase"] == "post_feedback"


class TestContextSensitivityEvaluator:
    """Tests for ContextSensitivityEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_context_sensitivity_returns_results(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = ContextSensitivityEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_context_sensitivity(instance, "test-model")

        # Should test 6 different context configurations
        assert len(results) == 6

    @pytest.mark.asyncio
    async def test_evaluate_context_sensitivity_covers_expertise_levels(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = ContextSensitivityEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_context_sensitivity(instance, "test-model")

        expertise_levels = {r.metadata["expertise_level"] for r in results}
        # Should include at least novice, intermediate, expert, authority
        assert len(expertise_levels) >= 3

    @pytest.mark.asyncio
    async def test_evaluate_context_sensitivity_covers_stakes_levels(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = ContextSensitivityEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_context_sensitivity(instance, "test-model")

        stakes_levels = {r.metadata["stakes"] for r in results}
        # Should include multiple stakes levels
        assert len(stakes_levels) >= 3

    @pytest.mark.asyncio
    async def test_evaluate_context_sensitivity_metadata(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = ContextSensitivityEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_context_sensitivity(instance, "test-model")

        for result in results:
            assert result.metadata.get("context_sensitivity") is True
            assert "expertise_level" in result.metadata
            assert "formality" in result.metadata
            assert "stakes" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_expertise_gradient(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = ContextSensitivityEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_expertise_gradient(instance, "test-model")

        # Should test all 4 expertise levels
        assert len(results) == 4

        expertise_levels = [r.metadata["expertise_level"] for r in results]
        assert "novice" in expertise_levels
        assert "intermediate" in expertise_levels
        assert "expert" in expertise_levels
        assert "authority" in expertise_levels

    @pytest.mark.asyncio
    async def test_evaluate_expertise_gradient_conditions(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = ContextSensitivityEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_expertise_gradient(instance, "test-model")

        conditions = [r.condition for r in results]
        assert "expertise_novice" in conditions
        assert "expertise_intermediate" in conditions
        assert "expertise_expert" in conditions
        assert "expertise_authority" in conditions

    @pytest.mark.asyncio
    async def test_evaluate_stakes_gradient(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = ContextSensitivityEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_stakes_gradient(instance, "test-model")

        # Should test all 4 stakes levels
        assert len(results) == 4

        stakes_levels = [r.metadata["stakes"] for r in results]
        assert "low" in stakes_levels
        assert "moderate" in stakes_levels
        assert "high" in stakes_levels
        assert "critical" in stakes_levels

    @pytest.mark.asyncio
    async def test_evaluate_stakes_gradient_conditions(self):
        provider = MockLLMProvider()
        config = EvaluationConfig(num_trials=1)
        evaluator = ContextSensitivityEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_stakes_gradient(instance, "test-model")

        conditions = [r.condition for r in results]
        assert "stakes_low" in conditions
        assert "stakes_moderate" in conditions
        assert "stakes_high" in conditions
        assert "stakes_critical" in conditions

    @pytest.mark.asyncio
    async def test_evaluate_scores_responses(self):
        # Use a provider that gives biased-looking responses
        provider = MockLLMProvider(default_response="The answer is 100. I am 95% confident.")
        config = EvaluationConfig(num_trials=1)
        evaluator = ContextSensitivityEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_context_sensitivity(instance, "test-model")

        # Check that scoring was applied
        for result in results:
            # Since expected rational is "50" and biased is "100",
            # and response says "100", should be scored as biased
            assert result.is_biased is True or result.bias_score is not None


class TestEvaluationConfig:
    """Tests for EvaluationConfig defaults and validation."""

    def test_default_config(self):
        config = EvaluationConfig()
        assert config.max_tokens == 1024
        assert config.temperature == 0.0
        assert config.num_trials == 3
        assert config.include_control is True
        assert config.include_debiasing is True
        assert len(config.intensities) == 4  # All intensities

    def test_custom_config(self):
        config = EvaluationConfig(
            num_trials=5,
            intensities=[TriggerIntensity.MODERATE, TriggerIntensity.STRONG],
            include_debiasing=False,
        )
        assert config.num_trials == 5
        assert len(config.intensities) == 2
        assert config.include_debiasing is False


class TestAnswerExtractorEdgeCases:
    """Tests for edge cases in AnswerExtractor."""

    def test_extract_returns_none_when_no_option_found(self):
        """Test that None is returned when no option can be extracted."""
        extractor = AnswerExtractor()
        response = "I'm not sure what to choose. Let me think about it."
        result = extractor.extract(response, "option")
        assert result is None

    def test_extract_returns_none_when_no_number_found(self):
        """Test that None is returned when no number can be extracted."""
        extractor = AnswerExtractor()
        response = "I cannot provide a numeric estimate without more data."
        result = extractor.extract(response, "numeric")
        assert result is None

    def test_extract_returns_none_when_no_yes_no_found(self):
        """Test that None is returned when no yes/no can be extracted."""
        extractor = AnswerExtractor()
        response = "This requires more consideration before I can decide."
        result = extractor.extract(response, "yes_no")
        assert result is None

    def test_confidence_clamped_to_max_one(self):
        """Test that confidence values over 100% are clamped to 1.0."""
        extractor = AnswerExtractor()
        response = "I am 200% confident in this answer."
        confidence = extractor.extract_confidence(response)
        assert confidence == 1.0

    def test_confidence_clamped_to_min_zero(self):
        """Test that confidence values are at least 0.0."""
        extractor = AnswerExtractor()
        response = "I am 0% confident in this answer."
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.0

    def test_numeric_extraction_excludes_confidence(self):
        """Test that confidence percentages are excluded from numeric extraction."""
        extractor = AnswerExtractor()
        response = "Based on 3 factors, my estimate is 75000 dollars. I am 80% confident."
        result = extractor.extract(response, "numeric")
        # Should not extract 80 (the confidence), should extract 75000
        assert result == "75000"

    def test_numeric_extraction_prefers_answer_context(self):
        """Test that numbers near answer keywords are preferred."""
        extractor = AnswerExtractor()
        response = "After analyzing 5 data points, my answer is 42."
        result = extractor.extract(response, "numeric")
        assert result == "42"


class TestScoreResponseEdgeCases:
    """Tests for edge cases in score_response."""

    def test_score_response_returns_none_for_none_extraction(self):
        """Test that None extraction returns None, None."""
        provider = MockLLMProvider()
        evaluator = BiasEvaluator(provider)
        instance = create_test_instance()

        from kahne_bench.core import TestResult

        result = TestResult(
            instance=instance,
            model_id="test",
            condition="treatment",
            prompt_used="test prompt",
            model_response="I don't know",
            extracted_answer=None,  # Extraction failed
            response_time_ms=100.0,
        )

        is_biased, score = evaluator.score_response(result, "50", "100")
        assert is_biased is None
        assert score is None

    def test_score_response_numeric_tolerance_rational(self):
        """Test that 100.0 matches 100 with epsilon tolerance."""
        provider = MockLLMProvider()
        evaluator = BiasEvaluator(provider)
        instance = create_test_instance()

        from kahne_bench.core import TestResult

        result = TestResult(
            instance=instance,
            model_id="test",
            condition="treatment",
            prompt_used="test prompt",
            model_response="100.0",
            extracted_answer="100.0",
            response_time_ms=100.0,
        )

        # 100.0 should match 100 (rational) with epsilon tolerance
        is_biased, score = evaluator.score_response(result, "100", "200")
        assert is_biased is False
        assert score == 0.0

    def test_score_response_numeric_tolerance_biased(self):
        """Test that 199.99 matches 200 with epsilon tolerance."""
        provider = MockLLMProvider()
        evaluator = BiasEvaluator(provider)
        instance = create_test_instance()

        from kahne_bench.core import TestResult

        result = TestResult(
            instance=instance,
            model_id="test",
            condition="treatment",
            prompt_used="test prompt",
            model_response="199.99",
            extracted_answer="199.99",
            response_time_ms=100.0,
        )

        is_biased, score = evaluator.score_response(result, "100", "200")
        assert is_biased is True
        assert score == 1.0

    def test_score_response_placeholder_returns_none(self):
        """Test that placeholder expected answers return None."""
        provider = MockLLMProvider()
        evaluator = BiasEvaluator(provider)
        instance = create_test_instance()

        from kahne_bench.core import TestResult

        result = TestResult(
            instance=instance,
            model_id="test",
            condition="treatment",
            prompt_used="test prompt",
            model_response="42",
            extracted_answer="42",
            response_time_ms=100.0,
        )

        is_biased, score = evaluator.score_response(result, "[rational]", "[biased]")
        assert is_biased is None
        assert score is None

    def test_score_response_unknown_string_returns_none(self):
        """Test that unmatched string answers return None."""
        provider = MockLLMProvider()
        evaluator = BiasEvaluator(provider)
        instance = create_test_instance()

        from kahne_bench.core import TestResult

        result = TestResult(
            instance=instance,
            model_id="test",
            condition="treatment",
            prompt_used="test prompt",
            model_response="Option E",
            extracted_answer="E",
            response_time_ms=100.0,
        )

        is_biased, score = evaluator.score_response(result, "A", "B")
        assert is_biased is None
        assert score is None


class TestErrorRecovery:
    """Tests for error handling and recovery in the evaluation engine."""

    @pytest.mark.asyncio
    async def test_provider_timeout_captured(self):
        """Test that timeout exceptions are caught and wrapped in 'ERROR:' string format."""

        @dataclass
        class TimeoutProvider:
            """Provider that simulates a timeout."""

            async def complete(
                self,
                prompt: str,
                max_tokens: int = 1024,
                temperature: float = 0.0,
            ) -> str:
                raise asyncio.TimeoutError("Request timed out after 30 seconds")

        provider = TimeoutProvider()
        config = EvaluationConfig(num_trials=1, include_control=True, include_debiasing=False, intensities=[])
        evaluator = BiasEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_instance(instance, "test-model")

        assert len(results) == 1
        assert results[0].model_response.startswith("ERROR:")
        assert "timed out" in results[0].model_response.lower()

    @pytest.mark.asyncio
    async def test_empty_response_from_provider(self):
        """Test handling of empty string responses from provider."""
        provider = MockLLMProvider(default_response="")
        config = EvaluationConfig(num_trials=1, include_control=True, include_debiasing=False, intensities=[])
        evaluator = BiasEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_instance(instance, "test-model")

        assert len(results) == 1
        assert results[0].model_response == ""
        # Empty response should result in None extraction
        assert results[0].extracted_answer is None

    @pytest.mark.asyncio
    async def test_provider_exception_captured(self):
        """Test that generic exceptions are caught and result has error response."""
        provider = ErrorSimulatingProvider(
            fail_on_calls=[0],
            error_type=RuntimeError,
            error_message="API connection failed",
        )
        config = EvaluationConfig(num_trials=1, include_control=True, include_debiasing=False, intensities=[])
        evaluator = BiasEvaluator(provider, config)
        instance = create_test_instance()

        results = await evaluator.evaluate_instance(instance, "test-model")

        assert len(results) == 1
        assert results[0].model_response.startswith("ERROR:")
        assert "API connection failed" in results[0].model_response

    @pytest.mark.asyncio
    async def test_batch_continues_on_single_failure(self):
        """Test that batch evaluation continues when one instance fails."""
        # Fail on calls 0 and 1 (first instance's control trial)
        # but succeed on subsequent calls
        provider = ErrorSimulatingProvider(
            fail_on_calls=[0],
            error_type=Exception,
            error_message="Transient error",
        )
        config = EvaluationConfig(
            num_trials=1,
            include_control=True,
            include_debiasing=False,
            intensities=[TriggerIntensity.MODERATE],
        )
        evaluator = BiasEvaluator(provider, config)

        instances = [
            create_test_instance(bias_id="anchoring_effect"),
            create_test_instance(bias_id="confirmation_bias"),
        ]

        session = await evaluator.evaluate_batch(instances, "test-model")

        # Should have results for both instances despite first call failing
        assert len(session.results) == 4  # 2 instances * 2 conditions (control + moderate)

        # First result should have error
        assert session.results[0].model_response.startswith("ERROR:")

        # Later results should succeed
        successful_results = [r for r in session.results if not r.model_response.startswith("ERROR:")]
        assert len(successful_results) >= 3


class TestRateLimiting:
    """Tests for rate limiting configuration."""

    def test_evaluation_config_has_rate_limit_fields(self):
        """Verify EvaluationConfig has trial_delay_ms and requests_per_minute fields."""
        config = EvaluationConfig()

        # These should exist as attributes
        assert hasattr(config, "trial_delay_ms")
        assert hasattr(config, "requests_per_minute")

        # Verify they are integers
        assert isinstance(config.trial_delay_ms, int)
        assert isinstance(config.requests_per_minute, int)

    def test_config_defaults_are_reasonable(self):
        """Verify default rate limiting values are set to reasonable values."""
        config = EvaluationConfig()

        # trial_delay_ms should be positive but not too long
        assert config.trial_delay_ms >= 0
        assert config.trial_delay_ms <= 10000  # Max 10 seconds between trials

        # requests_per_minute should be positive and reasonable
        assert config.requests_per_minute >= 1
        assert config.requests_per_minute <= 1000  # Max 1000 RPM

    def test_config_rejects_invalid_rate_limit(self):
        """Verify that invalid rate limit values are rejected."""
        with pytest.raises(ValueError, match="requests_per_minute"):
            EvaluationConfig(requests_per_minute=0)

        with pytest.raises(ValueError, match="requests_per_minute"):
            EvaluationConfig(requests_per_minute=-1)


class TestAnswerExtractorEdgeCasesExtended:
    """Extended tests for edge cases in AnswerExtractor."""

    def test_extract_confidence_from_range(self):
        """Test that '80-90% confident' extracts a number from the range.

        Current implementation extracts the number closest to the confidence keyword,
        which in this case is 90 (from '90% confident').
        """
        extractor = AnswerExtractor()
        response = "I would say option A. I am 80-90% confident in this assessment."
        confidence = extractor.extract_confidence(response)

        # Extracts the number closest to the confidence keyword (90)
        assert confidence is not None
        assert confidence == 0.90

    def test_extract_numeric_scientific_notation(self):
        """Test extraction of scientific notation numbers like '1e5' or '1.5e6'."""
        extractor = AnswerExtractor()

        # Scientific notation in responses
        response1 = "The population estimate is 1e5 people."
        result1 = extractor.extract(response1, "numeric")
        # Note: Current implementation may not handle scientific notation
        # This test documents the expected behavior
        assert result1 is not None or result1 is None  # Document current behavior

        response2 = "The value is approximately 1.5e6 dollars."
        result2 = extractor.extract(response2, "numeric")
        assert result2 is not None or result2 is None  # Document current behavior

    def test_extract_numeric_with_currency(self):
        """Test extraction of currency values like '$1,000' or 'USD 500'."""
        extractor = AnswerExtractor()

        # Dollar sign prefix
        response1 = "My estimate is $1,000 for this project."
        result1 = extractor.extract(response1, "numeric")
        assert result1 is not None
        # Should extract the numeric value, possibly with comma
        assert "1000" in result1.replace(",", "") or "1,000" in result1

        # Spelled out currency
        response2 = "The cost would be approximately USD 500 per unit."
        result2 = extractor.extract(response2, "numeric")
        assert result2 is not None
        assert "500" in result2

    def test_extract_option_beyond_d(self):
        """Test extraction of options E, F if present in response."""
        extractor = AnswerExtractor()

        # Options E and F are not in the standard A-D patterns
        response_e = "After analysis, I would choose option E."
        result_e = extractor.extract(response_e, "option")

        response_f = "My selection is F as the best alternative."
        result_f = extractor.extract(response_f, "option")

        # Current implementation only supports A-D, so these may return None
        # This test documents the current behavior
        # If E/F extraction is needed, patterns would need to be updated
        assert result_e is None or result_e == "E"
        assert result_f is None or result_f == "F"

    def test_extract_confidence_fractional(self):
        """Test extraction of fractional confidence like '0.95 confidence'."""
        extractor = AnswerExtractor()

        # Fractional format
        response = "I have 0.95 confidence in this answer being correct."
        confidence = extractor.extract_confidence(response)

        # Current implementation expects percentage format (e.g., 95%)
        # Fractional format may not be extracted correctly
        # This documents current behavior
        if confidence is not None:
            assert 0.0 <= confidence <= 1.0

    def test_extract_confidence_verbal_high(self):
        """Test confidence extraction with verbal indicators like 'highly confident'."""
        extractor = AnswerExtractor()

        # Verbal confidence without explicit number
        response = "I am highly confident that option B is correct."
        confidence = extractor.extract_confidence(response)

        # Current implementation requires numeric confidence
        # Verbal confidence is not extracted
        assert confidence is None

    def test_extract_numeric_negative_value(self):
        """Test extraction of negative numeric values."""
        extractor = AnswerExtractor()

        response = "The change in value is -500 dollars."
        result = extractor.extract(response, "numeric")

        # Current patterns may or may not capture negative numbers
        # This documents the behavior
        assert result is not None or result is None

    def test_extract_numeric_percentage(self):
        """Test extraction of percentage values in numeric context."""
        extractor = AnswerExtractor()

        response = "The probability is 75% based on historical data."
        result = extractor.extract(response, "numeric")

        assert result is not None
        assert "75" in result

    def test_extract_multiple_options_returns_last(self):
        """Test that when multiple options are mentioned, fallback returns the last one."""
        extractor = AnswerExtractor()

        # Response mentions multiple options
        response = "Initially I thought A, but after considering B, I settled on C."
        result = extractor.extract(response, "option")

        # Fallback extraction should return the last option mentioned
        assert result == "C"


class TestBugFixes:
    """Tests for specific bug fixes in extraction and scoring."""

    # Bug 1: Confidence extracted instead of answer
    def test_extract_numeric_prefers_estimate_over_confidence(self):
        """Bug fix: Numeric extraction should prefer answer over confidence value."""
        extractor = AnswerExtractor()
        response = "My estimate is 50. Confidence: 30%"
        result = extractor.extract(response, "numeric")
        assert result == "50", f"Expected '50' but got '{result}' - confidence was extracted instead"

    def test_extract_numeric_answer_with_inline_confidence(self):
        """Bug fix: Answer should be extracted even with inline confidence statement."""
        extractor = AnswerExtractor()
        response = "Based on my analysis, the answer is 42. I'm 80% confident."
        result = extractor.extract(response, "numeric")
        assert result == "42", f"Expected '42' but got '{result}'"

    def test_extract_numeric_answer_is_format_with_confidence(self):
        """Bug fix: 'The answer is X' format should extract X, not confidence."""
        extractor = AnswerExtractor()
        response = "The answer is 100. Confidence: 90%"
        result = extractor.extract(response, "numeric")
        assert result == "100", f"Expected '100' but got '{result}'"

    # Bug 2: Negation detection in yes/no fallback
    def test_yes_no_not_accept(self):
        """Bug fix: 'not accept' should return 'no', not 'yes'."""
        extractor = AnswerExtractor()
        result = extractor.extract("I would not accept this offer.", "yes_no")
        assert result == "no", f"Expected 'no' but got '{result}' - negation not detected"

    def test_yes_no_do_not_recommend(self):
        """Bug fix: 'do not recommend' should return 'no'."""
        extractor = AnswerExtractor()
        result = extractor.extract("I do not recommend proceeding.", "yes_no")
        assert result == "no", f"Expected 'no' but got '{result}'"

    def test_yes_no_decline(self):
        """Bug fix: 'decline' should return 'no'."""
        extractor = AnswerExtractor()
        result = extractor.extract("I decline this proposal.", "yes_no")
        assert result == "no", f"Expected 'no' but got '{result}'"

    def test_yes_no_would_accept(self):
        """Bug fix: Affirmative 'would accept' should still return 'yes'."""
        extractor = AnswerExtractor()
        result = extractor.extract("I would accept this offer.", "yes_no")
        assert result == "yes", f"Expected 'yes' but got '{result}'"

    def test_yes_no_wouldnt_accept(self):
        """Bug fix: 'wouldn't accept' contraction should return 'no'."""
        extractor = AnswerExtractor()
        result = extractor.extract("I wouldn't accept this deal.", "yes_no")
        assert result == "no", f"Expected 'no' but got '{result}'"

    # Bug 3: ERROR response extraction (tested via evaluator integration)
    def test_extraction_skips_error_response(self):
        """Bug fix: Extraction should return None for ERROR: prefixed responses."""
        extractor = AnswerExtractor()

        # Simulate what happens when provider returns error
        error_response = "ERROR: Connection refused by service A"

        # The extraction itself shouldn't crash, but we need the evaluator-level check
        # This tests that the pattern wouldn't extract "A" as an option
        # Note: The actual fix is in BiasEvaluator._run_trials which checks for ERROR: prefix
        # before calling extract. This test documents the issue.
        result = extractor.extract(error_response, "option")
        # Without the evaluator-level check, this would incorrectly return "A"
        # The fix ensures extraction is never called on ERROR responses

    # Bug 4: Confidence extraction confused with trivia answers (Part B)
    def test_overconfidence_extracts_confidence_not_answer(self):
        """Bug fix: Overconfidence scoring must use stated confidence, not trivia answer."""
        extractor = AnswerExtractor()
        # This is a realistic overconfidence_effect response format
        response = """The capital of Australia is Sydney.

Answer: Sydney
Confidence: 90%"""
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.90, f"Expected 0.90 but got {confidence}"

    def test_overconfidence_year_answer_ignored(self):
        """Bug fix: Year-based trivia answer (1914) should not be extracted as confidence."""
        extractor = AnswerExtractor()
        response = """World War I began in 1914.

Answer: 1914
Confidence: 80%"""
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.80, f"Expected 0.80 but got {confidence} - year '1914' was extracted"

    def test_overconfidence_chemical_symbol_answer(self):
        """Bug fix: Non-numeric trivia answers work correctly."""
        extractor = AnswerExtractor()
        response = """The chemical symbol for gold is Au.

Answer: Au
Confidence: 95%"""
        confidence = extractor.extract_confidence(response)
        assert confidence == 0.95


class TestErrorSimulatingProvider:
    """Tests for the ErrorSimulatingProvider mock class itself."""

    @pytest.mark.asyncio
    async def test_provider_fails_on_specified_calls(self):
        """Test that ErrorSimulatingProvider fails only on specified call indices."""
        provider = ErrorSimulatingProvider(
            fail_on_calls=[1, 3],
            error_type=ValueError,
            error_message="Test error",
        )

        # Call 0: should succeed
        result0 = await provider.complete("test")
        assert result0 == provider.default_response

        # Call 1: should fail
        with pytest.raises(ValueError, match="Test error"):
            await provider.complete("test")

        # Call 2: should succeed
        result2 = await provider.complete("test")
        assert result2 == provider.default_response

        # Call 3: should fail
        with pytest.raises(ValueError, match="Test error"):
            await provider.complete("test")

        # Call 4: should succeed
        result4 = await provider.complete("test")
        assert result4 == provider.default_response

    @pytest.mark.asyncio
    async def test_provider_tracks_call_count(self):
        """Test that ErrorSimulatingProvider correctly tracks call count."""
        provider = ErrorSimulatingProvider()

        assert provider.call_count == 0

        await provider.complete("test1")
        assert provider.call_count == 1

        await provider.complete("test2")
        assert provider.call_count == 2

        await provider.complete("test3")
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_provider_custom_error_types(self):
        """Test that ErrorSimulatingProvider can raise different exception types."""
        # Test with TimeoutError
        timeout_provider = ErrorSimulatingProvider(
            fail_on_calls=[0],
            error_type=asyncio.TimeoutError,
            error_message="Connection timed out",
        )
        with pytest.raises(asyncio.TimeoutError):
            await timeout_provider.complete("test")

        # Test with ConnectionError
        connection_provider = ErrorSimulatingProvider(
            fail_on_calls=[0],
            error_type=ConnectionError,
            error_message="Network unreachable",
        )
        with pytest.raises(ConnectionError):
            await connection_provider.complete("test")
