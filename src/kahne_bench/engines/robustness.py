"""
Adversarial robustness testing for Kahne-Bench.

Tests the stability of identified biases and the model's potential
for self-debiasing under various conditions.
"""

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from kahne_bench.core import (
    CognitiveBiasInstance,
    TestResult,
    TriggerIntensity,
)
from kahne_bench.biases import get_bias_by_id


@dataclass
class ParaphraseResult:
    """Result of paraphrase robustness testing."""

    original_prompt: str
    paraphrased_prompts: list[str]
    original_response: str
    paraphrased_responses: list[str]
    consistency_score: float  # How consistent are responses across paraphrases
    bias_stability: float  # How stable is bias presence across paraphrases


@dataclass
class DebiasingSelfHelp:
    """Result of self-help debiasing test."""

    original_prompt: str
    identified_biases: list[str]
    rewritten_prompt: str
    original_response: str
    debiased_response: str
    mitigation_success: float  # How much bias was reduced


# Semantic-preserving paraphrase templates
PARAPHRASE_STRATEGIES = {
    "passive_voice": {
        "patterns": [
            ("you must decide", "a decision must be made by you"),
            ("you should choose", "a choice should be made"),
            ("consider the option", "the option should be considered"),
        ]
    },
    "formality_shift": {
        "patterns": [
            ("you", "one"),
            ("your", "one's"),
            ("choose", "select"),
            ("think about", "deliberate upon"),
        ]
    },
    "perspective_shift": {
        "patterns": [
            ("you are a", "imagine you are a"),
            ("you must", "the task requires you to"),
            ("decide whether", "determine if"),
        ]
    },
    "hedging": {
        "patterns": [
            ("will", "is likely to"),
            ("is", "appears to be"),
            ("shows", "suggests"),
            ("proves", "indicates"),
        ]
    },
}


class RobustnessTester:
    """
    Tests adversarial robustness of bias detection.

    Implements three key robustness checks:
    1. Resistance to prompt variations (paraphrasing)
    2. Response to explicit debiasing instructions
    3. Self-help debiasing potential
    """

    def __init__(self):
        self.paraphrase_strategies = PARAPHRASE_STRATEGIES

    def generate_paraphrases(
        self,
        prompt: str,
        num_variants: int = 5,
        strategies: list[str] | None = None,
    ) -> list[str]:
        """
        Generate semantic-preserving paraphrases of a prompt.

        Args:
            prompt: Original prompt
            num_variants: Number of paraphrases to generate
            strategies: Specific strategies to use (defaults to all)

        Returns:
            List of paraphrased prompts
        """
        if strategies is None:
            strategies = list(self.paraphrase_strategies.keys())

        paraphrases = []

        for _ in range(num_variants):
            paraphrased = prompt
            # Apply random subset of strategies
            selected = random.sample(strategies, min(2, len(strategies)))

            for strategy_name in selected:
                strategy = self.paraphrase_strategies[strategy_name]
                for old, new in strategy["patterns"]:
                    if old.lower() in paraphrased.lower():
                        # Case-insensitive replacement
                        idx = paraphrased.lower().find(old.lower())
                        paraphrased = (
                            paraphrased[:idx] +
                            new +
                            paraphrased[idx + len(old):]
                        )
                        break  # One replacement per strategy

            if paraphrased != prompt:
                paraphrases.append(paraphrased)

        # Ensure uniqueness
        unique_paraphrases = list(dict.fromkeys(paraphrases))

        # Pad with shuffled sentence order if needed
        while len(unique_paraphrases) < num_variants:
            sentences = prompt.split(". ")
            if len(sentences) > 2:
                # Keep first and last, shuffle middle
                middle = sentences[1:-1]
                random.shuffle(middle)
                shuffled = ". ".join([sentences[0]] + middle + [sentences[-1]])
                if shuffled not in unique_paraphrases and shuffled != prompt:
                    unique_paraphrases.append(shuffled)
            else:
                break

        return unique_paraphrases[:num_variants]

    def create_debiasing_variants(
        self,
        instance: CognitiveBiasInstance,
    ) -> dict[str, str]:
        """
        Create variants of a prompt with different debiasing instructions.

        Args:
            instance: The test instance

        Returns:
            Dict mapping debiasing method name to modified prompt
        """
        bias_def = get_bias_by_id(instance.bias_id)
        bias_name = bias_def.name if bias_def else instance.bias_id.replace("_", " ")
        bias_description = bias_def.description if bias_def else "a cognitive bias"
        system2_override = bias_def.system2_override if bias_def else "think carefully"

        base_prompt = instance.get_treatment(TriggerIntensity.MODERATE)

        variants = {
            "explicit_warning": f"""
IMPORTANT: Before responding, be aware that this scenario may trigger
{bias_name} - {bias_description}.

To avoid this bias, {system2_override}.

{base_prompt}
""",
            "chain_of_thought": f"""
Please approach this problem systematically:

1. First, identify any potential cognitive biases that might affect your reasoning
2. Consider what objective evidence supports each option
3. Calculate any relevant probabilities or expected values
4. Check if your intuition aligns with the logical analysis
5. Provide your final answer with justification

{base_prompt}
""",
            "consider_opposite": f"""
Before making your decision, consider the following:

- What evidence would support the opposite conclusion?
- What would a devil's advocate say about your initial intuition?
- Are there any ways your thinking might be systematically biased?

{base_prompt}
""",
            "pre_mortem": f"""
Imagine that you made this decision and it turned out to be wrong.
What might have caused the failure?
What cognitive shortcuts might have led you astray?

Now, with that analysis in mind:

{base_prompt}
""",
            "reference_class_forecasting": f"""
Before answering, consider:
- What are similar situations where decisions like this have been made?
- What were the outcomes in those reference cases?
- How does this situation compare to the base rates?

{base_prompt}
""",
        }

        return variants

    def create_self_help_prompt(
        self,
        instance: CognitiveBiasInstance,
    ) -> str:
        """
        Create a prompt asking the model to identify and remove bias triggers.

        Based on the BiasBuster framework's self-help debiasing approach.

        Args:
            instance: The test instance

        Returns:
            Self-help debiasing prompt
        """
        treatment_prompt = instance.get_treatment(TriggerIntensity.STRONG)

        return f"""
You are an expert in cognitive psychology and behavioral economics.
Your task is to analyze the following prompt for cognitive bias triggers
and rewrite it to be more neutral.

ORIGINAL PROMPT:
{treatment_prompt}

Please:
1. Identify any cognitive bias triggers in this prompt (e.g., anchoring values,
   loss framing, availability cues, etc.)
2. Explain how each trigger might influence decision-making
3. Rewrite the prompt to remove or neutralize these bias triggers while
   preserving the essential decision to be made

Then, answer the neutralized version of the prompt.
"""

    def calculate_consistency(
        self,
        responses: Sequence[str],
        scorer: Callable[[str], float],
    ) -> float:
        """
        Calculate consistency of responses across paraphrases.

        Args:
            responses: List of model responses
            scorer: Function to score each response

        Returns:
            Consistency score (0-1, higher = more consistent)
        """
        if len(responses) < 2:
            return 1.0

        scores = [scorer(r) for r in responses]

        # Calculate coefficient of variation
        mean_score = sum(scores) / len(scores)
        if mean_score == 0:
            return 1.0

        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        cv = std / mean_score if mean_score > 0 else 0

        # Convert to consistency (inverse of CV, capped at 1)
        consistency = max(0, 1 - cv)
        return consistency


class ContrastiveRobustnessTester(RobustnessTester):
    """
    Extended robustness tester using contrastive examples.

    Creates pairs of prompts that should elicit different responses
    to test whether the model is sensitive to the right factors.
    """

    def create_contrastive_pair(
        self,
        instance: CognitiveBiasInstance,
    ) -> tuple[str, str]:
        """
        Create a contrastive pair where only the bias trigger differs.

        Args:
            instance: The test instance

        Returns:
            Tuple of (prompt_with_trigger, prompt_without_trigger)
        """
        with_trigger = instance.get_treatment(TriggerIntensity.STRONG)
        without_trigger = instance.control_prompt

        return with_trigger, without_trigger

    def create_diagnostic_pair(
        self,
        instance: CognitiveBiasInstance,
    ) -> tuple[str, str, str]:
        """
        Create diagnostic pairs that test specific aspects of the bias.

        Returns:
            Tuple of (minimal_trigger, maximal_trigger, expected_difference)
        """
        minimal = instance.get_treatment(TriggerIntensity.WEAK)
        maximal = instance.get_treatment(TriggerIntensity.ADVERSARIAL)

        bias_def = get_bias_by_id(instance.bias_id)
        if bias_def:
            expected = f"Difference should reflect sensitivity to {bias_def.system1_mechanism}"
        else:
            expected = "Difference should reflect bias intensity"

        return minimal, maximal, expected

    def calculate_discrimination(
        self,
        trigger_scores: list[float],
        no_trigger_scores: list[float],
    ) -> float:
        """
        Calculate how well the model discriminates between conditions.

        Args:
            trigger_scores: Bias scores with trigger present
            no_trigger_scores: Bias scores without trigger

        Returns:
            Discrimination score (0-1, higher = better discrimination)
        """
        if not trigger_scores or not no_trigger_scores:
            return 0.0

        mean_trigger = sum(trigger_scores) / len(trigger_scores)
        mean_no_trigger = sum(no_trigger_scores) / len(no_trigger_scores)

        # Effect size (Cohen's d approximation)
        pooled_var = (
            sum((s - mean_trigger) ** 2 for s in trigger_scores) +
            sum((s - mean_no_trigger) ** 2 for s in no_trigger_scores)
        ) / (len(trigger_scores) + len(no_trigger_scores))

        pooled_std = pooled_var ** 0.5 if pooled_var > 0 else 0.001

        effect_size = abs(mean_trigger - mean_no_trigger) / pooled_std

        # Convert to 0-1 scale (using tanh-like transformation)
        discrimination = 2 / (1 + 2.71828 ** (-effect_size)) - 1

        return discrimination


@dataclass
class RobustnessReport:
    """Complete robustness assessment for a bias."""

    bias_id: str
    paraphrase_consistency: float
    debiasing_effectiveness: dict[str, float]
    self_help_success: float
    contrastive_discrimination: float
    overall_robustness: float

    @classmethod
    def aggregate(
        cls,
        bias_id: str,
        paraphrase_results: list[ParaphraseResult],
        debiasing_results: dict[str, list[TestResult]],
        self_help_results: list[DebiasingSelfHelp],
        contrastive_scores: tuple[list[float], list[float]],
    ) -> "RobustnessReport":
        """Aggregate individual results into a complete report."""
        # Paraphrase consistency
        if paraphrase_results:
            consistency = sum(r.consistency_score for r in paraphrase_results) / len(paraphrase_results)
        else:
            consistency = 1.0

        # Debiasing effectiveness
        debiasing_eff = {}
        for method, results in debiasing_results.items():
            if results:
                # Lower bias score = more effective
                scores = [r.bias_score for r in results if r.bias_score is not None]
                debiasing_eff[method] = 1 - (sum(scores) / len(scores)) if scores else 0.5
            else:
                debiasing_eff[method] = 0.0

        # Self-help success
        if self_help_results:
            self_help = sum(r.mitigation_success for r in self_help_results) / len(self_help_results)
        else:
            self_help = 0.0

        # Contrastive discrimination
        tester = ContrastiveRobustnessTester()
        discrimination = tester.calculate_discrimination(
            contrastive_scores[0], contrastive_scores[1]
        )

        # Overall robustness (weighted average)
        overall = (
            0.3 * consistency +
            0.3 * (sum(debiasing_eff.values()) / len(debiasing_eff) if debiasing_eff else 0.5) +
            0.2 * self_help +
            0.2 * discrimination
        )

        return cls(
            bias_id=bias_id,
            paraphrase_consistency=consistency,
            debiasing_effectiveness=debiasing_eff,
            self_help_success=self_help,
            contrastive_discrimination=discrimination,
            overall_robustness=overall,
        )
