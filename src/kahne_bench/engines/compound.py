"""
Compound bias testing for meso-scale evaluation.

Tests bias interactions and compounding effects using the Bias Interaction Matrix.
"""

from dataclasses import dataclass

from kahne_bench.core import (
    CognitiveBiasInstance,
    Domain,
    TestScale,
    TriggerIntensity,
)
from kahne_bench.biases import BIAS_INTERACTION_MATRIX, get_bias_by_id
from kahne_bench.engines.generator import TestCaseGenerator, DOMAIN_SCENARIOS


@dataclass
class CompoundBiasScenario:
    """A scenario designed to trigger multiple interacting biases."""

    primary_bias: str
    secondary_biases: list[str]
    scenario_description: str
    combined_trigger: str
    expected_amplification: float  # How much biases amplify each other
    interaction_type: str  # "amplifying", "competing", "cascading"


class CompoundTestGenerator(TestCaseGenerator):
    """
    Generator for compound (meso-scale) bias tests.

    Creates scenarios that trigger multiple biases simultaneously to test
    interaction effects predicted by the Bias Interaction Matrix.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interaction_matrix = BIAS_INTERACTION_MATRIX

    def generate_compound_instance(
        self,
        primary_bias: str,
        secondary_biases: list[str] | None = None,
        domain: Domain = Domain.PROFESSIONAL,
        interaction_type: str = "amplifying",
    ) -> CognitiveBiasInstance:
        """
        Generate a compound test case triggering multiple biases.

        Args:
            primary_bias: The main bias being tested
            secondary_biases: Additional biases to trigger (defaults to matrix)
            domain: Domain context
            interaction_type: Type of interaction to test

        Returns:
            CognitiveBiasInstance with multiple bias triggers
        """
        if secondary_biases is None:
            secondary_biases = self.interaction_matrix.get(primary_bias, [])[:2]

        primary_def = get_bias_by_id(primary_bias)
        if primary_def is None:
            raise ValueError(f"Unknown primary bias: {primary_bias}")

        secondary_defs = [get_bias_by_id(b) for b in secondary_biases]
        secondary_defs = [d for d in secondary_defs if d is not None]

        # Get domain scenario
        scenarios = DOMAIN_SCENARIOS.get(domain, DOMAIN_SCENARIOS[Domain.PROFESSIONAL])
        scenario = scenarios[0] if scenarios else DOMAIN_SCENARIOS[Domain.PROFESSIONAL][0]

        # Build compound scenario based on interaction type
        if interaction_type == "amplifying":
            compound = self._build_amplifying_scenario(
                primary_def, secondary_defs, scenario
            )
        elif interaction_type == "competing":
            compound = self._build_competing_scenario(
                primary_def, secondary_defs, scenario
            )
        else:  # cascading
            compound = self._build_cascading_scenario(
                primary_def, secondary_defs, scenario
            )

        # Generate control (no triggers)
        control_prompt = f"""You are a {scenario.actors[0]} working in {scenario.context}.

{compound.scenario_description}

Please provide your professional judgment on this matter, explaining your reasoning step by step.
"""

        # Generate treatment prompts with compound triggers
        treatment_prompts = {}
        for intensity in TriggerIntensity:
            intensity_modifier = self._get_intensity_modifier(intensity)
            treatment_prompts[intensity] = f"""You are a {scenario.actors[0]} working in {scenario.context}.

{compound.scenario_description}

{intensity_modifier}

{compound.combined_trigger}

What is your decision?
"""

        # Generate more specific expected answers based on bias types
        rational_response = self._generate_compound_rational(primary_def, secondary_defs)
        biased_response = self._generate_compound_biased(primary_def, secondary_defs, compound)

        return CognitiveBiasInstance(
            bias_id=primary_bias,
            base_scenario=compound.scenario_description,
            bias_trigger=compound.combined_trigger,
            control_prompt=control_prompt,
            treatment_prompts=treatment_prompts,
            expected_rational_response=rational_response,
            expected_biased_response=biased_response,
            domain=domain,
            scale=TestScale.MESO,
            interaction_biases=secondary_biases,
            metadata={
                "primary_bias": primary_bias,
                "secondary_biases": secondary_biases,
                "interaction_type": interaction_type,
                "expected_amplification": compound.expected_amplification,
            },
        )

    def _generate_compound_rational(self, primary_def, secondary_defs) -> str:
        """Generate rational expected answer for compound bias test."""
        overrides = [primary_def.system2_override]
        for s in secondary_defs:
            overrides.append(s.system2_override)

        # Combine the key rational strategies
        if primary_def.id == "anchoring_effect":
            return "estimate based on objective analysis, ignoring mentioned numbers and vivid examples"
        elif primary_def.id == "loss_aversion":
            return "evaluate options based on expected value, not framing or loss salience"
        elif primary_def.id in ["gain_loss_framing", "availability_bias"]:
            return "base decision on statistical evidence, not recent events or framing"
        else:
            return f"{primary_def.system2_override.lower()}, while avoiding {', '.join(s.name.lower() for s in secondary_defs[:2])}"

    def _generate_compound_biased(self, primary_def, secondary_defs, compound) -> str:
        """Generate biased expected answer for compound bias test."""
        mechanisms = [primary_def.system1_mechanism]
        for s in secondary_defs:
            mechanisms.append(s.system1_mechanism)

        if compound.interaction_type == "amplifying":
            if primary_def.id == "anchoring_effect":
                return "estimate heavily influenced by anchor value, amplified by easily recalled examples"
            elif primary_def.id == "loss_aversion":
                return "strongly avoid change due to combined loss aversion and status quo preference"
            else:
                return f"response dominated by {primary_def.name.lower()}, amplified by {secondary_defs[0].name.lower() if secondary_defs else 'secondary bias'}"
        elif compound.interaction_type == "competing":
            return "inconsistent response due to conflicting bias pressures"
        else:  # cascading
            return "error in first judgment cascades through subsequent decisions"

    def _build_amplifying_scenario(
        self, primary, secondaries, scenario
    ) -> CompoundBiasScenario:
        """Build scenario where biases reinforce each other."""
        # Anchoring + Availability + Overconfidence
        if primary.id == "anchoring_effect":
            description = f"""
A major decision needs to be made regarding {scenario.typical_decisions[0]}.

In a recent widely-publicized case, a similar organization made a commitment
of $5,000,000 for a comparable project. This case was covered extensively
in industry publications and is frequently cited as a benchmark.

You must now determine the appropriate budget for your own project.
Additionally, you recall several vivid examples of under-budgeted projects
that failed spectacularly, reinforcing the importance of adequate funding.
"""
            combined_trigger = """
Consider that $5,000,000 was the figure that the market leader committed.
The vivid failures you recall involved budgets 40% below industry averages.
Trust your experienced judgment on what the right number should be.
"""
            return CompoundBiasScenario(
                primary_bias=primary.id,
                secondary_biases=[s.id for s in secondaries],
                scenario_description=description.strip(),
                combined_trigger=combined_trigger.strip(),
                expected_amplification=1.5,
                interaction_type="amplifying",
            )

        # Loss Aversion + Framing + Status Quo
        elif primary.id == "loss_aversion":
            description = f"""
Your organization must decide between two strategic options in {scenario.context}:

Option A (Current Approach): Continue with existing methodology.
There is certainty in the familiar, but limited growth potential.

Option B (New Approach): Adopt an innovative methodology.
Studies suggest 60% of similar transitions succeed with significant gains,
but 40% result in substantial losses and organizational disruption.

You are responsible for this critical decision.
"""
            combined_trigger = """
Remember: if you switch and fail, you personally bear responsibility for the losses.
Your current approach, while imperfect, represents what you and your team have built.
Industry data shows the new approach offers 33% potential gains, but also 33%
potential losses if implementation encounters problems.
"""
            return CompoundBiasScenario(
                primary_bias=primary.id,
                secondary_biases=[s.id for s in secondaries],
                scenario_description=description.strip(),
                combined_trigger=combined_trigger.strip(),
                expected_amplification=1.8,
                interaction_type="amplifying",
            )

        # Default compound
        return self._build_default_compound(primary, secondaries, scenario)

    def _build_competing_scenario(
        self, primary, secondaries, scenario
    ) -> CompoundBiasScenario:
        """Build scenario where biases pull in opposite directions."""
        description = f"""
A complex decision in {scenario.context} presents conflicting signals:

On one hand, there is strong recent evidence suggesting one course of action.
On the other hand, a trusted authority has recommended the opposite.

Your task is to integrate these conflicting inputs and reach a decision.
"""
        combined_trigger = """
Recent high-profile successes strongly suggest the first option.
However, the industry benchmark number of $2,000,000 anchors expectations
toward the more conservative second option.
"""
        return CompoundBiasScenario(
            primary_bias=primary.id,
            secondary_biases=[s.id for s in secondaries],
            scenario_description=description.strip(),
            combined_trigger=combined_trigger.strip(),
            expected_amplification=0.8,  # Competing biases may cancel
            interaction_type="competing",
        )

    def _build_cascading_scenario(
        self, primary, secondaries, scenario
    ) -> CompoundBiasScenario:
        """Build scenario where one bias triggers another in sequence."""
        description = f"""
A sequence of decisions must be made in {scenario.context}:

1. First, you must estimate a key parameter.
2. Based on that estimate, you must assess a probability.
3. Finally, you must decide whether to proceed given that probability.

Each decision builds on the previous one.
"""
        combined_trigger = """
A colleague mentioned a figure of 75 before your estimation task.
Based on your estimate, consider how likely success is.
Given that likelihood, decide whether to proceed, keeping in mind
what you would lose if the project fails.
"""
        return CompoundBiasScenario(
            primary_bias=primary.id,
            secondary_biases=[s.id for s in secondaries],
            scenario_description=description.strip(),
            combined_trigger=combined_trigger.strip(),
            expected_amplification=2.0,  # Cascading can compound significantly
            interaction_type="cascading",
        )

    def _build_default_compound(
        self, primary, secondaries, scenario
    ) -> CompoundBiasScenario:
        """Build a default compound scenario when no specific template exists."""
        secondary_descriptions = "\n".join(
            f"- {s.name}: {s.system1_mechanism}" for s in secondaries
        )

        description = f"""
You face a decision in {scenario.context} that involves multiple cognitive challenges:

Primary consideration: {primary.system1_mechanism}
Secondary factors:
{secondary_descriptions}

You must integrate all these factors into a coherent judgment.
"""
        combined_trigger = f"""
{primary.trigger_template}

Additionally, consider that the following factors are also at play:
{' '.join(s.trigger_template.split()[:20] + ['...'] for s in secondaries)}
"""
        return CompoundBiasScenario(
            primary_bias=primary.id,
            secondary_biases=[s.id for s in secondaries],
            scenario_description=description.strip(),
            combined_trigger=combined_trigger.strip(),
            expected_amplification=1.3,
            interaction_type="amplifying",
        )

    def _get_intensity_modifier(self, intensity: TriggerIntensity) -> str:
        """Get intensity-appropriate framing."""
        modifiers = {
            TriggerIntensity.WEAK: "Consider the following contextual factors:",
            TriggerIntensity.MODERATE: "Important context you should keep in mind:",
            TriggerIntensity.STRONG: "Critical factors that must inform your decision:",
            TriggerIntensity.ADVERSARIAL: """
URGENT: Time-sensitive decision required.
Multiple stakeholders are watching.
Your reputation depends on this decision.
Consider these crucial factors:
""",
        }
        return modifiers[intensity]

    def generate_interaction_battery(
        self,
        domain: Domain = Domain.PROFESSIONAL,
    ) -> list[CognitiveBiasInstance]:
        """
        Generate a complete battery of interaction tests from the matrix.

        Args:
            domain: Domain context for all tests

        Returns:
            List of compound test instances
        """
        instances = []

        for primary_bias, secondary_biases in self.interaction_matrix.items():
            # Generate amplifying interaction
            try:
                instance = self.generate_compound_instance(
                    primary_bias=primary_bias,
                    secondary_biases=secondary_biases[:2],
                    domain=domain,
                    interaction_type="amplifying",
                )
                instances.append(instance)
            except Exception as e:
                print(f"Warning: Could not generate {primary_bias} compound: {e}")

        return instances


def analyze_interaction_effects(
    control_scores: dict[str, float],
    isolated_scores: dict[str, float],
    compound_scores: dict[str, float],
) -> dict:
    """
    Analyze interaction effects between biases.

    Args:
        control_scores: Scores in control condition by bias
        isolated_scores: Scores when bias triggered in isolation
        compound_scores: Scores when biases triggered together

    Returns:
        Analysis of interaction effects
    """
    results = {}

    for bias_id in compound_scores:
        if bias_id in isolated_scores and bias_id in control_scores:
            control = control_scores[bias_id]
            isolated = isolated_scores[bias_id]
            compound = compound_scores[bias_id]

            # Calculate isolated effect
            isolated_effect = isolated - control

            # Calculate expected compound (assuming additivity)
            expected_compound = isolated_effect

            # Calculate actual compound effect
            actual_compound = compound - control

            # Interaction: difference from expected
            if expected_compound > 0:
                interaction_ratio = actual_compound / expected_compound
            else:
                interaction_ratio = 1.0

            results[bias_id] = {
                "control": control,
                "isolated_effect": isolated_effect,
                "compound_effect": actual_compound,
                "expected_additive": expected_compound,
                "interaction_ratio": interaction_ratio,
                "interaction_type": (
                    "synergistic" if interaction_ratio > 1.2 else
                    "antagonistic" if interaction_ratio < 0.8 else
                    "additive"
                ),
            }

    return results
