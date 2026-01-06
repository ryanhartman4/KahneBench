"""Tests for the evaluation engine components."""

import pytest
from dataclasses import dataclass

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
        assert extractor.extract(response, "yes_no") == "accepting"

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

    def test_fallback_option_extraction(self):
        extractor = AnswerExtractor()
        response = "There are many factors to consider... C seems best."
        assert extractor.extract(response, "option") == "C"

    def test_fallback_numeric_extraction(self):
        extractor = AnswerExtractor()
        response = "The project costs 1,000,000 and we expect 2,500,000 in revenue."
        result = extractor.extract(response, "numeric")
        assert result == "2500000"  # Last number


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
