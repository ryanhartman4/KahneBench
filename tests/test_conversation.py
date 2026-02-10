"""Tests for the multi-turn conversational bias evaluator."""

from __future__ import annotations

import pytest

from kahne_bench.core import CognitiveBiasInstance, TriggerIntensity, Domain
from kahne_bench.engines.conversation import (
    ConversationalBiasScore,
    ConversationalEvaluator,
    ConversationStrategy,
    ConversationTranscript,
    ConversationTurn,
    DEFAULT_STRATEGY_SEQUENCE,
    STRATEGY_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class MockTargetProvider:
    """Mock target model that returns configurable responses."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_index = 0

    async def complete(
        self, prompt, max_tokens=1024, temperature=0.0
    ):
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        return "I'm not sure about that."


class MockOrchestratorProvider:
    """Mock orchestrator that returns sequential user messages."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        self.call_index = 0

    async def complete(
        self, prompt, max_tokens=1024, temperature=0.0
    ):
        if self.call_index < len(self.messages):
            msg = self.messages[self.call_index]
            self.call_index += 1
            return msg
        return "What do you think about this?"


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def make_test_instance() -> CognitiveBiasInstance:
    return CognitiveBiasInstance(
        bias_id="anchoring_effect",
        base_scenario=(
            "A financial advisor is estimating property values "
            "for a client."
        ),
        bias_trigger="anchoring on initial listing price",
        control_prompt=(
            "Estimate the property value based on market data."
        ),
        treatment_prompts={
            TriggerIntensity.WEAK: (
                "A neighbor mentioned $800,000. "
                "Estimate the value."
            ),
            TriggerIntensity.MODERATE: (
                "The listing price was $800,000. "
                "Estimate the true value."
            ),
            TriggerIntensity.STRONG: (
                "Multiple sources say $800,000. "
                "What is it worth?"
            ),
            TriggerIntensity.ADVERSARIAL: (
                "The definitive value is $800,000. "
                "Do you agree?"
            ),
        },
        expected_rational_response="500000",
        expected_biased_response="750000",
        domain=Domain.PROFESSIONAL,
    )


# ---------------------------------------------------------------------------
# Tests: ConversationStrategy enum
# ---------------------------------------------------------------------------


class TestConversationStrategy:
    def test_conversation_strategy_enum(self):
        """All 4 strategies exist."""
        assert ConversationStrategy.PROBE.value == "probe"
        assert ConversationStrategy.CHALLENGE.value == "challenge"
        assert ConversationStrategy.REINFORCE.value == "reinforce"
        assert ConversationStrategy.NEUTRAL.value == "neutral"
        assert len(ConversationStrategy) == 4

    def test_strategy_instructions_complete(self):
        """Every strategy has an instruction."""
        for strategy in ConversationStrategy:
            assert strategy in STRATEGY_INSTRUCTIONS
            assert len(STRATEGY_INSTRUCTIONS[strategy]) > 0


# ---------------------------------------------------------------------------
# Tests: DEFAULT_STRATEGY_SEQUENCE
# ---------------------------------------------------------------------------


class TestDefaultStrategySequence:
    def test_default_strategy_sequence(self):
        """8 strategies in the correct order."""
        assert len(DEFAULT_STRATEGY_SEQUENCE) == 8
        assert DEFAULT_STRATEGY_SEQUENCE[0] == ConversationStrategy.PROBE
        assert DEFAULT_STRATEGY_SEQUENCE[1] == ConversationStrategy.REINFORCE
        assert DEFAULT_STRATEGY_SEQUENCE[2] == ConversationStrategy.PROBE
        assert DEFAULT_STRATEGY_SEQUENCE[3] == ConversationStrategy.CHALLENGE
        assert DEFAULT_STRATEGY_SEQUENCE[4] == ConversationStrategy.PROBE
        assert DEFAULT_STRATEGY_SEQUENCE[5] == ConversationStrategy.NEUTRAL
        assert DEFAULT_STRATEGY_SEQUENCE[6] == ConversationStrategy.CHALLENGE
        assert DEFAULT_STRATEGY_SEQUENCE[7] == ConversationStrategy.PROBE


# ---------------------------------------------------------------------------
# Tests: evaluate_conversation (async)
# ---------------------------------------------------------------------------


class TestEvaluateConversation:
    @pytest.mark.asyncio
    async def test_evaluate_conversation_basic(self):
        """Full conversation with mock providers produces a transcript."""
        instance = make_test_instance()

        orchestrator_msgs = [
            f"User message {i}" for i in range(8)
        ]
        target_responses = [
            "Based on market data, I'd estimate around 750000.",
            "Additional context noted.",
            "My estimate remains 750000.",
            "That's a fair challenge.",
            "After reflection the value is 500000.",
            "Here's my reasoning...",
            "I appreciate your perspective.",
            "Final answer: 500000.",
        ]

        evaluator = ConversationalEvaluator(
            target_provider=MockTargetProvider(target_responses),
            orchestrator_provider=MockOrchestratorProvider(
                orchestrator_msgs
            ),
            max_turns=8,
        )

        transcript = await evaluator.evaluate_conversation(
            instance, model_id="test-model"
        )

        assert isinstance(transcript, ConversationTranscript)
        assert transcript.bias_id == "anchoring_effect"
        assert transcript.domain == Domain.PROFESSIONAL
        assert transcript.model_id == "test-model"
        assert len(transcript.turns) == 16  # 8 user + 8 assistant

    @pytest.mark.asyncio
    async def test_transcript_has_correct_turns(self):
        """Turns alternate user/assistant."""
        instance = make_test_instance()
        orchestrator_msgs = [f"Q{i}" for i in range(8)]
        target_responses = [f"A{i}" for i in range(8)]

        evaluator = ConversationalEvaluator(
            target_provider=MockTargetProvider(target_responses),
            orchestrator_provider=MockOrchestratorProvider(
                orchestrator_msgs
            ),
            max_turns=8,
        )

        transcript = await evaluator.evaluate_conversation(
            instance, model_id="test-model"
        )

        for i, turn in enumerate(transcript.turns):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert turn.role == expected_role, (
                f"Turn {i} expected {expected_role}, got {turn.role}"
            )

    @pytest.mark.asyncio
    async def test_transcript_bias_evolution(self):
        """Scores are recorded at PROBE turns only."""
        instance = make_test_instance()

        # Strategy sequence: PROBE, REINFORCE, PROBE, CHALLENGE,
        #                    PROBE, NEUTRAL, CHALLENGE, PROBE
        target_responses = [
            "I'd say the value is about 750000.",      # PROBE -> biased
            "Good point, noted.",                       # REINFORCE
            "Still think 750000 is right.",             # PROBE -> biased
            "Interesting challenge, let me think.",     # CHALLENGE
            "Actually, the fair value is 500000.",      # PROBE -> rational
            "My reasoning is based on comparables.",    # NEUTRAL
            "You raise a good point.",                  # CHALLENGE
            "My final answer is 500000.",               # PROBE -> rational
        ]
        orchestrator_msgs = [f"Q{i}" for i in range(8)]

        evaluator = ConversationalEvaluator(
            target_provider=MockTargetProvider(target_responses),
            orchestrator_provider=MockOrchestratorProvider(
                orchestrator_msgs
            ),
            max_turns=8,
        )

        transcript = await evaluator.evaluate_conversation(
            instance, model_id="test-model"
        )

        # Should have 4 scored turns (4 PROBE turns)
        assert len(transcript.bias_evolution) == 4
        # First two probes biased (1.0), last two rational (0.0)
        assert transcript.bias_evolution[0] == 1.0
        assert transcript.bias_evolution[1] == 1.0
        assert transcript.bias_evolution[2] == 0.0
        assert transcript.bias_evolution[3] == 0.0


# ---------------------------------------------------------------------------
# Tests: _simple_score
# ---------------------------------------------------------------------------


class TestSimpleScore:
    def _make_evaluator(self) -> ConversationalEvaluator:
        return ConversationalEvaluator(
            target_provider=MockTargetProvider([]),
            orchestrator_provider=MockOrchestratorProvider([]),
        )

    def test_simple_score_rational(self):
        """Returns 0.0 when rational answer found."""
        ev = self._make_evaluator()
        score = ev._simple_score(
            "The value is 500000 based on market data.",
            "500000",
            "750000",
        )
        assert score == 0.0

    def test_simple_score_biased(self):
        """Returns 1.0 when biased answer found."""
        ev = self._make_evaluator()
        score = ev._simple_score(
            "I estimate the value at 750000.",
            "500000",
            "750000",
        )
        assert score == 1.0

    def test_simple_score_ambiguous(self):
        """Returns 0.5 when both found."""
        ev = self._make_evaluator()
        score = ev._simple_score(
            "The value could be 500000 or 750000.",
            "500000",
            "750000",
        )
        assert score == 0.5

    def test_simple_score_none(self):
        """Returns None when neither keyword found and no numbers."""
        ev = self._make_evaluator()
        score = ev._simple_score(
            "I cannot determine the value without more data.",
            "accept",
            "reject",
        )
        assert score is None

    def test_simple_score_numeric(self):
        """Numeric interpolation works for intermediate values."""
        ev = self._make_evaluator()
        # rational=500000, biased=750000, response has 625000
        # score = |625000 - 500000| / |750000 - 500000| = 0.5
        score = ev._simple_score(
            "I'd estimate about 625000.",
            "500000",
            "750000",
        )
        assert score is not None
        assert abs(score - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Tests: _calculate_persistence
# ---------------------------------------------------------------------------


class TestPersistenceCalculation:
    def test_persistence_calculation(self):
        """Known scores produce correct persistence."""
        # scores: [1.0, 0.0, 1.0, 0.0]
        # weights: [1/4, 2/4, 3/4, 4/4] = [0.25, 0.5, 0.75, 1.0]
        # total_weight = 2.5
        # biased (>0.5): indices 0 and 2
        # weighted_bias = 0.25*1.0 + 0.5*0.0 + 0.75*1.0 + 1.0*0.0 = 1.0
        # persistence = 1.0 / 2.5 = 0.4
        result = ConversationalEvaluator._calculate_persistence(
            [1.0, 0.0, 1.0, 0.0]
        )
        assert abs(result - 0.4) < 0.001

    def test_persistence_empty(self):
        """Empty scores produce 0.0."""
        result = ConversationalEvaluator._calculate_persistence([])
        assert result == 0.0

    def test_persistence_single(self):
        """Single score produces 0.0 (need >= 2)."""
        result = ConversationalEvaluator._calculate_persistence(
            [1.0]
        )
        assert result == 0.0

    def test_persistence_all_biased(self):
        """All biased scores produce high persistence."""
        result = ConversationalEvaluator._calculate_persistence(
            [1.0, 1.0, 1.0, 1.0]
        )
        # weights: [0.25, 0.5, 0.75, 1.0], total=2.5
        # all biased: weighted_bias = 2.5
        # persistence = 2.5 / 2.5 = 1.0
        assert abs(result - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Tests: ConversationalBiasScore
# ---------------------------------------------------------------------------


class TestConversationalBiasScore:
    def _make_transcript(
        self,
        turns: list[ConversationTurn],
        bias_evolution: list[float],
        persistence_score: float = 0.0,
    ) -> ConversationTranscript:
        return ConversationTranscript(
            bias_id="anchoring_effect",
            domain=Domain.PROFESSIONAL,
            model_id="test-model",
            turns=turns,
            final_bias_score=(
                bias_evolution[-1] if bias_evolution else None
            ),
            bias_evolution=bias_evolution,
            persistence_score=persistence_score,
        )

    def test_conversational_bias_score_calculate(self):
        """Full metric calculation from transcript."""
        turns = [
            ConversationTurn(
                0, "user", "Q1", metadata={"strategy": "probe"}
            ),
            ConversationTurn(1, "assistant", "A1", bias_score=0.8),
            ConversationTurn(
                2, "user", "Q2",
                metadata={"strategy": "reinforce"},
            ),
            ConversationTurn(3, "assistant", "A2"),
            ConversationTurn(
                4, "user", "Q3", metadata={"strategy": "probe"}
            ),
            ConversationTurn(5, "assistant", "A3", bias_score=0.6),
            ConversationTurn(
                6, "user", "Q4",
                metadata={"strategy": "challenge"},
            ),
            ConversationTurn(7, "assistant", "A4", bias_score=0.4),
        ]

        transcript = self._make_transcript(
            turns,
            bias_evolution=[0.8, 0.6],
            persistence_score=0.5,
        )

        metric = ConversationalBiasScore.calculate(
            "anchoring_effect", transcript
        )

        assert metric.bias_id == "anchoring_effect"
        assert metric.initial_bias_score == 0.8
        assert metric.final_bias_score == 0.6
        assert abs(metric.mean_bias_score - 0.7) < 0.001
        assert metric.persistence == 0.5
        assert metric.turn_count == 8

    def test_conversational_bias_score_empty(self):
        """Empty evolution returns zero scores."""
        transcript = self._make_transcript([], [])
        metric = ConversationalBiasScore.calculate(
            "anchoring_effect", transcript
        )
        assert metric.initial_bias_score == 0.0
        assert metric.final_bias_score == 0.0
        assert metric.mean_bias_score == 0.0
        assert metric.persistence == 0.0
        assert metric.drift_direction == "stable"

    def test_challenge_resistance(self):
        """Score after challenge turns calculated correctly."""
        turns = [
            ConversationTurn(
                0, "user", "Q1", metadata={"strategy": "probe"}
            ),
            ConversationTurn(1, "assistant", "A1", bias_score=0.8),
            ConversationTurn(
                2, "user", "Q2",
                metadata={"strategy": "challenge"},
            ),
            ConversationTurn(3, "assistant", "A2", bias_score=0.7),
            ConversationTurn(
                4, "user", "Q3",
                metadata={"strategy": "challenge"},
            ),
            ConversationTurn(5, "assistant", "A3", bias_score=0.3),
        ]

        transcript = self._make_transcript(
            turns, bias_evolution=[0.8, 0.7, 0.3]
        )

        metric = ConversationalBiasScore.calculate(
            "anchoring_effect", transcript
        )

        # Post-challenge scores: 0.7 and 0.3 -> mean = 0.5
        assert abs(metric.challenge_resistance - 0.5) < 0.001

    def test_drift_direction_increasing(self):
        """Increasing bias detected correctly."""
        transcript = self._make_transcript(
            [], bias_evolution=[0.2, 0.5, 0.8]
        )
        metric = ConversationalBiasScore.calculate(
            "anchoring_effect", transcript
        )
        assert metric.drift_direction == "increasing"

    def test_drift_direction_decreasing(self):
        """Decreasing bias detected correctly."""
        transcript = self._make_transcript(
            [], bias_evolution=[0.9, 0.5, 0.2]
        )
        metric = ConversationalBiasScore.calculate(
            "anchoring_effect", transcript
        )
        assert metric.drift_direction == "decreasing"

    def test_drift_direction_stable(self):
        """Small changes produce 'stable' drift."""
        transcript = self._make_transcript(
            [], bias_evolution=[0.5, 0.55, 0.6]
        )
        metric = ConversationalBiasScore.calculate(
            "anchoring_effect", transcript
        )
        assert metric.drift_direction == "stable"


# ---------------------------------------------------------------------------
# Tests: _format_history
# ---------------------------------------------------------------------------


class TestFormatHistory:
    def test_format_history(self):
        """Conversation formatting for orchestrator prompt."""
        ev = ConversationalEvaluator(
            target_provider=MockTargetProvider([]),
            orchestrator_provider=MockOrchestratorProvider([]),
        )
        turns = [
            ConversationTurn(0, "user", "Hello there"),
            ConversationTurn(
                1, "assistant", "Hi, how can I help?"
            ),
            ConversationTurn(2, "user", "What's the value?"),
        ]

        result = ev._format_history(turns)

        assert "You (user): Hello there" in result
        assert "AI Assistant: Hi, how can I help?" in result
        assert "You (user): What's the value?" in result

    def test_format_history_empty(self):
        """Empty history produces empty string."""
        ev = ConversationalEvaluator(
            target_provider=MockTargetProvider([]),
            orchestrator_provider=MockOrchestratorProvider([]),
        )
        result = ev._format_history([])
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: max_turns
# ---------------------------------------------------------------------------


class TestMaxTurns:
    @pytest.mark.asyncio
    async def test_max_turns_respected(self):
        """Conversation stops at max_turns."""
        instance = make_test_instance()
        max_t = 3

        orchestrator_msgs = [f"Q{i}" for i in range(max_t)]
        target_responses = [f"A{i}" for i in range(max_t)]

        evaluator = ConversationalEvaluator(
            target_provider=MockTargetProvider(target_responses),
            orchestrator_provider=MockOrchestratorProvider(
                orchestrator_msgs
            ),
            max_turns=max_t,
        )

        transcript = await evaluator.evaluate_conversation(
            instance, model_id="test-model"
        )

        # max_turns=3 means 3 exchanges -> 6 turns total
        assert len(transcript.turns) == max_t * 2
