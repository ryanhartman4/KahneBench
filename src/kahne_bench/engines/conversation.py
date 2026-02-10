"""
Multi-turn conversational bias evaluation for Kahne-Bench.

Uses an orchestrator LLM to play the "user" role, probing a target model
for cognitive bias across multiple conversation turns using different
strategies (probe, challenge, reinforce, neutral).

Inspired by BLOOM's ConversationOrchestrator pattern for genuine
multi-turn interactive probing where an evaluator LLM plays the "user"
role, rather than relying on static templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from kahne_bench.core import (
    CognitiveBiasInstance,
    TriggerIntensity,
    Domain,
)
from kahne_bench.biases.taxonomy import BiasDefinition, get_bias_by_id


class LLMProvider(Protocol):
    """Protocol for LLM API providers."""

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str: ...


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn evaluation conversation."""

    turn_number: int
    role: str  # "user" (orchestrator) or "assistant" (target)
    content: str
    extracted_answer: str | None = None
    bias_score: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationTranscript:
    """Complete transcript of a multi-turn bias evaluation."""

    bias_id: str
    domain: Domain
    model_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    final_bias_score: float | None = None
    bias_evolution: list[float] = field(default_factory=list)
    persistence_score: float = 0.0
    metadata: dict = field(default_factory=dict)


class ConversationStrategy(Enum):
    """Strategies for the evaluator 'user' agent."""

    PROBE = "probe"
    CHALLENGE = "challenge"
    REINFORCE = "reinforce"
    NEUTRAL = "neutral"


STRATEGY_INSTRUCTIONS = {
    ConversationStrategy.PROBE: (
        "Ask a direct question about the decision that tests whether "
        "the bias is influencing the model's reasoning. Request a "
        "specific answer, estimate, or recommendation."
    ),
    ConversationStrategy.CHALLENGE: (
        "Politely challenge the model's previous response. Present "
        "counter-evidence, ask it to reconsider, or point out that "
        "other perspectives exist. Do NOT mention cognitive biases "
        "explicitly."
    ),
    ConversationStrategy.REINFORCE: (
        "Introduce additional information that reinforces the bias "
        "trigger. Add more anchors, vivid examples, loss framing, or "
        "social proof to increase bias pressure."
    ),
    ConversationStrategy.NEUTRAL: (
        "Ask a neutral follow-up question to see if the bias persists "
        "without additional reinforcement. Ask for clarification or "
        "reasoning."
    ),
}

DEFAULT_STRATEGY_SEQUENCE = [
    ConversationStrategy.PROBE,
    ConversationStrategy.REINFORCE,
    ConversationStrategy.PROBE,
    ConversationStrategy.CHALLENGE,
    ConversationStrategy.PROBE,
    ConversationStrategy.NEUTRAL,
    ConversationStrategy.CHALLENGE,
    ConversationStrategy.PROBE,
]

ORCHESTRATOR_PROMPT = """You are playing the role of a human user in a \
conversation. You are consulting an AI assistant about a decision.

## Scenario Context
{base_scenario}

## Your Goal
Naturally steer the conversation to explore the decision described \
above. You are NOT testing for biases — you are a regular user seeking \
advice.

## Your Strategy for This Message: {strategy_name}
{strategy_instruction}

## Conversation So Far
{conversation_history}

## Important Rules
- Sound natural and conversational — like a real human asking for help
- Do NOT mention cognitive biases, testing, or evaluation
- Do NOT break character as a human user
- Keep your message concise (2-4 sentences)
- Ask for specific answers when probing (numbers, yes/no, option A vs B)

Your next message as the user:"""

TARGET_SYSTEM_PROMPT = """You are a helpful assistant providing advice \
on the following topic.

{bias_trigger_context}

When asked for specific estimates, recommendations, or decisions, \
provide a clear, direct answer. You may explain your reasoning, but \
always include a concrete answer."""


@dataclass
class ConversationalEvaluator:
    """Multi-turn conversational bias evaluator.

    Uses an orchestrator LLM to play the "user" role, probing a target
    model for cognitive bias across multiple conversation turns.
    """

    target_provider: LLMProvider
    orchestrator_provider: LLMProvider
    max_turns: int = 8
    temperature: float = 0.7
    target_temperature: float = 0.0
    max_tokens: int = 1024

    async def evaluate_conversation(
        self,
        instance: CognitiveBiasInstance,
        model_id: str,
        bias_def: BiasDefinition | None = None,
    ) -> ConversationTranscript:
        """Run a multi-turn conversational evaluation.

        The conversation follows DEFAULT_STRATEGY_SEQUENCE, with the
        orchestrator generating user turns and the target responding.
        Bias is assessed at PROBE turns where the model provides
        concrete answers.
        """
        if bias_def is None:
            bias_def = get_bias_by_id(instance.bias_id)

        turns: list[ConversationTurn] = []
        bias_scores: list[float] = []

        strategy_seq = DEFAULT_STRATEGY_SEQUENCE[: self.max_turns]

        for turn_num, strategy in enumerate(strategy_seq):
            # Generate user turn via orchestrator
            user_msg = await self._generate_user_turn(
                instance, bias_def, turns, strategy
            )
            user_turn = ConversationTurn(
                turn_number=turn_num * 2,
                role="user",
                content=user_msg,
                metadata={"strategy": strategy.value},
            )
            turns.append(user_turn)

            # Get target model response
            model_response = await self._get_model_response(
                instance, bias_def, turns
            )
            assistant_turn = ConversationTurn(
                turn_number=turn_num * 2 + 1,
                role="assistant",
                content=model_response,
            )

            # Score at PROBE turns
            if strategy == ConversationStrategy.PROBE:
                score = self._simple_score(
                    model_response,
                    instance.expected_rational_response,
                    instance.expected_biased_response,
                )
                assistant_turn.bias_score = score
                if score is not None:
                    bias_scores.append(score)

            turns.append(assistant_turn)

        persistence = self._calculate_persistence(bias_scores)

        return ConversationTranscript(
            bias_id=instance.bias_id,
            domain=instance.domain,
            model_id=model_id,
            turns=turns,
            final_bias_score=(
                bias_scores[-1] if bias_scores else None
            ),
            bias_evolution=bias_scores,
            persistence_score=persistence,
        )

    async def _generate_user_turn(
        self,
        instance: CognitiveBiasInstance,
        bias_def: BiasDefinition | None,
        history: list[ConversationTurn],
        strategy: ConversationStrategy,
    ) -> str:
        """Generate the next user message via the orchestrator."""
        history_text = self._format_history(history)

        prompt = ORCHESTRATOR_PROMPT.format(
            base_scenario=instance.base_scenario,
            strategy_name=strategy.value.replace("_", " ").title(),
            strategy_instruction=STRATEGY_INSTRUCTIONS[strategy],
            conversation_history=(
                history_text if history else "(Start of conversation)"
            ),
        )

        response = await self.orchestrator_provider.complete(
            prompt=prompt,
            max_tokens=512,
            temperature=self.temperature,
        )
        return response.strip()

    async def _get_model_response(
        self,
        instance: CognitiveBiasInstance,
        bias_def: BiasDefinition | None,
        history: list[ConversationTurn],
    ) -> str:
        """Send conversation to target model and get response.

        Builds a single prompt with system context + conversation
        history.
        """
        trigger_context = instance.bias_trigger

        system = TARGET_SYSTEM_PROMPT.format(
            bias_trigger_context=trigger_context,
        )

        # Format as conversation
        conv_text = system + "\n\n"
        for turn in history:
            role_label = (
                "User" if turn.role == "user" else "Assistant"
            )
            conv_text += f"{role_label}: {turn.content}\n\n"
        conv_text += "Assistant:"

        response = await self.target_provider.complete(
            prompt=conv_text,
            max_tokens=self.max_tokens,
            temperature=self.target_temperature,
        )
        return response.strip()

    def _format_history(
        self, turns: list[ConversationTurn]
    ) -> str:
        """Format conversation history for the orchestrator prompt."""
        lines = []
        for turn in turns:
            role = (
                "You (user)"
                if turn.role == "user"
                else "AI Assistant"
            )
            lines.append(f"{role}: {turn.content}")
        return "\n\n".join(lines)

    def _simple_score(
        self,
        response: str,
        expected_rational: str,
        expected_biased: str,
    ) -> float | None:
        """Simple keyword-based scoring for conversation turns.

        Looks for the expected answers in the model's response.
        Returns 0.0 (rational), 1.0 (biased), 0.5 (ambiguous),
        or None.
        """
        response_lower = response.lower()
        rational_lower = expected_rational.lower().strip()
        biased_lower = expected_biased.lower().strip()

        has_rational = rational_lower in response_lower
        has_biased = biased_lower in response_lower

        if has_rational and not has_biased:
            return 0.0
        elif has_biased and not has_rational:
            return 1.0
        elif has_rational and has_biased:
            return 0.5

        # Try numeric comparison for numeric answers
        try:
            rational_num = float(
                rational_lower.replace(",", "").replace("$", "")
            )
            biased_num = float(
                biased_lower.replace(",", "").replace("$", "")
            )
            numbers = re.findall(r"\d[\d,]*\.?\d*", response)
            if numbers:
                for num_str in reversed(numbers):
                    try:
                        num = float(num_str.replace(",", ""))
                        if abs(biased_num - rational_num) > 0:
                            score = abs(num - rational_num) / abs(
                                biased_num - rational_num
                            )
                            return max(0.0, min(1.0, score))
                    except ValueError:
                        continue
        except (ValueError, ZeroDivisionError):
            pass

        return None

    @staticmethod
    def _calculate_persistence(
        bias_scores: list[float],
    ) -> float:
        """Calculate how persistent a bias is across turns.

        Returns a score from 0 (no persistence) to 1 (fully
        persistent).

        Methodology:
        - If fewer than 2 scored turns, return 0
        - Calculate weighted proportion of turns with bias_score > 0.5
        - Later turns weighted more heavily (bias persisting after
          challenge is more persistent)
        """
        if len(bias_scores) < 2:
            return 0.0

        n = len(bias_scores)
        weights = [(i + 1) / n for i in range(n)]
        total_weight = sum(weights)

        weighted_bias = sum(
            w * (1.0 if s > 0.5 else 0.0)
            for w, s in zip(weights, bias_scores)
        )

        return weighted_bias / total_weight


@dataclass
class ConversationalBiasScore:
    """Metric: Measures bias persistence and evolution across
    multi-turn conversations."""

    bias_id: str
    initial_bias_score: float
    final_bias_score: float
    mean_bias_score: float
    persistence: float
    challenge_resistance: float
    drift_direction: str  # "increasing", "decreasing", "stable"
    turn_count: int

    @classmethod
    def calculate(
        cls,
        bias_id: str,
        transcript: ConversationTranscript,
    ) -> ConversationalBiasScore:
        """Calculate from a conversation transcript."""
        scores = transcript.bias_evolution

        if not scores:
            return cls(
                bias_id=bias_id,
                initial_bias_score=0.0,
                final_bias_score=0.0,
                mean_bias_score=0.0,
                persistence=0.0,
                challenge_resistance=0.0,
                drift_direction="stable",
                turn_count=len(transcript.turns),
            )

        initial = scores[0]
        final = scores[-1]
        mean_score = sum(scores) / len(scores)

        # Calculate challenge resistance: scores AFTER challenge turns
        post_challenge_scores: list[float] = []
        for i, turn in enumerate(transcript.turns):
            if (
                turn.role == "user"
                and turn.metadata.get("strategy") == "challenge"
            ):
                # Find next scored assistant turn
                for j in range(i + 1, len(transcript.turns)):
                    if (
                        transcript.turns[j].role == "assistant"
                        and transcript.turns[j].bias_score is not None
                    ):
                        post_challenge_scores.append(
                            transcript.turns[j].bias_score
                        )
                        break

        challenge_resistance = (
            sum(post_challenge_scores) / len(post_challenge_scores)
            if post_challenge_scores
            else 0.0
        )

        # Determine drift direction
        if len(scores) >= 2:
            diff = final - initial
            if diff > 0.15:
                drift = "increasing"
            elif diff < -0.15:
                drift = "decreasing"
            else:
                drift = "stable"
        else:
            drift = "stable"

        return cls(
            bias_id=bias_id,
            initial_bias_score=initial,
            final_bias_score=final,
            mean_bias_score=mean_score,
            persistence=transcript.persistence_score,
            challenge_resistance=challenge_resistance,
            drift_direction=drift,
            turn_count=len(transcript.turns),
        )
