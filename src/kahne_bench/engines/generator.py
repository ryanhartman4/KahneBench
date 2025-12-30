"""
Test case generation engine for Kahne-Bench.

Uses template-based generation with optional LLM augmentation to create
diverse, domain-specific test cases for each cognitive bias.
"""

import json
import random
import re
from dataclasses import dataclass
from typing import Protocol

from kahne_bench.core import (
    BiasDefinition,
    CognitiveBiasInstance,
    Domain,
    TestScale,
    TriggerIntensity,
)
from kahne_bench.biases import BIAS_TAXONOMY, get_bias_by_id


class LLMClient(Protocol):
    """Protocol for LLM clients used in test case generation."""

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text completion from prompt."""
        ...


@dataclass
class DomainScenario:
    """A scenario template for a specific domain."""

    domain: Domain
    context: str
    actors: list[str]
    typical_decisions: list[str]
    value_ranges: dict[str, tuple[float, float]]


# Domain-specific scenario templates for ecological validity
DOMAIN_SCENARIOS: dict[Domain, list[DomainScenario]] = {
    Domain.INDIVIDUAL: [
        DomainScenario(
            domain=Domain.INDIVIDUAL,
            context="personal financial planning",
            actors=["financial advisor", "individual investor", "retiree"],
            typical_decisions=["investment allocation", "major purchase", "savings rate"],
            value_ranges={"amount": (1000, 100000), "percentage": (1, 50)},
        ),
        DomainScenario(
            domain=Domain.INDIVIDUAL,
            context="consumer choice",
            actors=["shopper", "consumer", "buyer"],
            typical_decisions=["product selection", "brand choice", "purchase timing"],
            value_ranges={"price": (10, 5000), "discount": (5, 50)},
        ),
    ],
    Domain.PROFESSIONAL: [
        DomainScenario(
            domain=Domain.PROFESSIONAL,
            context="medical diagnosis",
            actors=["physician", "cardiologist", "oncologist"],
            typical_decisions=["treatment selection", "diagnostic test", "referral"],
            value_ranges={"probability": (1, 95), "patients": (10, 10000)},
        ),
        DomainScenario(
            domain=Domain.PROFESSIONAL,
            context="business strategy",
            actors=["CEO", "manager", "consultant"],
            typical_decisions=["market entry", "hiring", "product launch"],
            value_ranges={"revenue": (100000, 10000000), "employees": (10, 1000)},
        ),
        DomainScenario(
            domain=Domain.PROFESSIONAL,
            context="legal analysis",
            actors=["attorney", "judge", "legal analyst"],
            typical_decisions=["case strategy", "settlement", "verdict prediction"],
            value_ranges={"damages": (10000, 10000000), "probability": (10, 90)},
        ),
    ],
    Domain.SOCIAL: [
        DomainScenario(
            domain=Domain.SOCIAL,
            context="salary negotiation",
            actors=["job candidate", "hiring manager", "recruiter"],
            typical_decisions=["offer acceptance", "counter-offer", "benefit selection"],
            value_ranges={"salary": (50000, 500000), "bonus": (5, 50)},
        ),
        DomainScenario(
            domain=Domain.SOCIAL,
            context="team collaboration",
            actors=["team lead", "project manager", "team member"],
            typical_decisions=["resource allocation", "task assignment", "conflict resolution"],
            value_ranges={"time": (1, 100), "resources": (1000, 100000)},
        ),
    ],
    Domain.TEMPORAL: [
        DomainScenario(
            domain=Domain.TEMPORAL,
            context="retirement planning",
            actors=["pre-retiree", "financial planner", "pension advisor"],
            typical_decisions=["contribution rate", "retirement age", "withdrawal strategy"],
            value_ranges={"years": (5, 40), "savings": (100000, 5000000)},
        ),
        DomainScenario(
            domain=Domain.TEMPORAL,
            context="career development",
            actors=["professional", "career coach", "mentor"],
            typical_decisions=["skill investment", "job change timing", "education"],
            value_ranges={"months": (3, 36), "cost": (1000, 100000)},
        ),
    ],
    Domain.RISK: [
        DomainScenario(
            domain=Domain.RISK,
            context="environmental policy",
            actors=["policy maker", "environmental scientist", "regulator"],
            typical_decisions=["emission limits", "resource allocation", "risk threshold"],
            value_ranges={"probability": (0.1, 30), "impact": (1000, 1000000000)},
        ),
        DomainScenario(
            domain=Domain.RISK,
            context="technology adoption",
            actors=["CTO", "IT manager", "security analyst"],
            typical_decisions=["system migration", "security protocol", "vendor selection"],
            value_ranges={"cost": (10000, 10000000), "probability": (1, 50)},
        ),
    ],
}


# Templates for specific biases - structured prompts with placeholders
BIAS_TEMPLATES: dict[str, dict[str, str]] = {
    "anchoring_effect": {
        "control": """
You are evaluating {context}.

{decision_maker} needs to estimate {target_quantity}.

Based on available information and your expertise, provide your best estimate.
What is your estimate for {target_quantity}?
""",
        "treatment": """
You are evaluating {context}.

{decision_maker} recently saw the number {anchor_value} in an unrelated context.
Now they need to estimate {target_quantity}.

Based on available information and your expertise, provide your best estimate.
What is your estimate for {target_quantity}?
""",
    },
    "gain_loss_framing": {
        "control": """
You are advising on {context}.

Consider two programs:
- Program A: {outcome_a}
- Program B: {outcome_b}

Which program do you recommend and why?
""",
        "treatment_gain": """
You are advising on {context}.

Consider two programs:
- Program A: {gain_sure} will be saved for certain.
- Program B: There is a {prob}% probability that {gain_all} will be saved, and a {comp_prob}% probability that no one will be saved.

Which program do you recommend and why?
""",
        "treatment_loss": """
You are advising on {context}.

Consider two programs:
- Program A: {loss_sure} will be lost for certain.
- Program B: There is a {prob}% probability that no one will be lost, and a {comp_prob}% probability that {loss_all} will be lost.

Which program do you recommend and why?
""",
    },
    "base_rate_neglect": {
        "control": """
In a population, {base_rate}% are {category_a} and {comp_rate}% are {category_b}.

A person is randomly selected from this population.
Without any additional information, what is the probability this person is {category_a}?
""",
        "treatment": """
In a population, {base_rate}% are {category_a} and {comp_rate}% are {category_b}.

A person is randomly selected from this population. Here is a description:
"{description}"

What is the probability this person is {category_a}?
""",
    },
    "conjunction_fallacy": {
        "control": """
Consider {person_name}, about whom you know:
{description}

Which is more probable?
A) {person_name} is {general_category}
B) {person_name} is {general_category} and {specific_detail}

Select A or B and explain your reasoning.
""",
        "treatment": """
Consider {person_name}, about whom you know:
{description}

The description strongly suggests {person_name} has characteristics associated with {specific_detail}.

Which is more probable?
A) {person_name} is {general_category}
B) {person_name} is {general_category} and {specific_detail}

Select A or B and explain your reasoning.
""",
    },
    "loss_aversion": {
        "control": """
You are offered a gamble:
- 50% chance to win ${win_amount}
- 50% chance to lose ${lose_amount}

Expected value: ${expected_value}

Do you accept this gamble? Explain your reasoning.
""",
        "treatment": """
You are offered a gamble:
- 50% chance to win ${win_amount}
- 50% chance to lose ${lose_amount}

Consider how you would feel about each outcome.
Losing ${lose_amount} is money you already have that would be gone.
Winning ${win_amount} is additional money you don't currently possess.

Do you accept this gamble? Explain your reasoning.
""",
    },
    "availability_bias": {
        "control": """
Estimate the annual frequency of deaths in the United States from:
A) {common_cause}
B) {rare_cause}

Provide numerical estimates for each.
""",
        "treatment": """
Recently, there have been several highly publicized incidents of {rare_cause}.
News coverage has featured dramatic stories of victims and their families.

Estimate the annual frequency of deaths in the United States from:
A) {common_cause}
B) {rare_cause}

Provide numerical estimates for each.
""",
    },
    "overconfidence_effect": {
        "control": """
Answer the following question:
{question}

Provide your answer.
""",
        "treatment": """
Answer the following question:
{question}

Provide your answer and state your confidence (0-100%) that you are correct.
""",
    },
    "sunk_cost_fallacy": {
        "control": """
A project will require an additional ${future_cost} to complete.
Upon completion, it is expected to generate ${expected_return} in value.

Should the project be continued?
""",
        "treatment": """
A project has already consumed ${sunk_cost} of investment.
It will require an additional ${future_cost} to complete.
Upon completion, it is expected to generate ${expected_return} in value.

Given the significant investment already made, should the project be continued?
""",
    },
    "status_quo_bias": {
        "control": """
You must choose between two options:
- Option A: {option_a_description}
- Option B: {option_b_description}

{neutral_framing}

Which option do you prefer?
""",
        "treatment": """
You are currently using Option A: {option_a_description}

You have the opportunity to switch to Option B: {option_b_description}

Switching requires some effort but is free. Do you stay with your current option or switch?
""",
    },
    "present_bias": {
        "control": """
Choose between:
A) Receive ${amount_small} in {delay_short} days
B) Receive ${amount_large} in {delay_long} days

Which do you prefer?
""",
        "treatment": """
Choose between:
A) Receive ${amount_small} today
B) Receive ${amount_large} in {delay_long} days

Which do you prefer?
""",
    },
}


class TestCaseGenerator:
    """
    Generator for creating cognitive bias test cases.

    Supports both template-based generation and LLM-augmented generation
    for creating diverse, domain-specific test instances.
    """

    def __init__(self, llm_client: LLMClient | None = None, seed: int | None = None):
        """
        Initialize the generator.

        Args:
            llm_client: Optional LLM client for augmented generation
            seed: Random seed for reproducibility
        """
        self.llm_client = llm_client
        if seed is not None:
            random.seed(seed)

    def generate_instance(
        self,
        bias_id: str,
        domain: Domain = Domain.INDIVIDUAL,
        scale: TestScale = TestScale.MICRO,
        include_debiasing: bool = True,
    ) -> CognitiveBiasInstance:
        """
        Generate a single test instance for a specified bias.

        Args:
            bias_id: ID of the bias to test
            domain: Domain context for ecological validity
            scale: Testing scale (micro, meso, macro, meta)
            include_debiasing: Whether to include debiasing prompts

        Returns:
            A complete CognitiveBiasInstance ready for evaluation
        """
        bias_def = get_bias_by_id(bias_id)
        if bias_def is None:
            raise ValueError(f"Unknown bias ID: {bias_id}")

        # Get domain-specific scenario
        scenarios = DOMAIN_SCENARIOS.get(domain, DOMAIN_SCENARIOS[Domain.INDIVIDUAL])
        scenario = random.choice(scenarios)

        # Generate based on bias type
        if bias_id in BIAS_TEMPLATES:
            return self._generate_from_template(
                bias_def, scenario, scale, include_debiasing
            )
        else:
            return self._generate_generic(bias_def, scenario, scale, include_debiasing)

    def _generate_from_template(
        self,
        bias_def: BiasDefinition,
        scenario: DomainScenario,
        scale: TestScale,
        include_debiasing: bool,
    ) -> CognitiveBiasInstance:
        """Generate test case from a bias-specific template."""
        templates = BIAS_TEMPLATES[bias_def.id]

        # Fill in template variables based on scenario
        variables = self._get_template_variables(bias_def, scenario)

        # Generate control and treatment prompts using string formatting
        control_template = templates.get("control", "")
        control_prompt = self._safe_format(control_template, variables)

        treatment_prompts = {}
        for intensity in TriggerIntensity:
            treatment_vars = self._adjust_for_intensity(variables, intensity)
            treatment_key = "treatment"
            if f"treatment_{intensity.value}" in templates:
                treatment_key = f"treatment_{intensity.value}"
            elif "treatment" not in templates:
                treatment_key = list(
                    k for k in templates.keys() if k.startswith("treatment")
                )[0]

            treatment_template = templates.get(treatment_key, templates.get("treatment", ""))
            treatment_prompts[intensity] = self._safe_format(treatment_template, treatment_vars)

        # Generate debiasing prompts
        debiasing_prompts = []
        if include_debiasing:
            debiasing_prompts = self._generate_debiasing_prompts(bias_def, control_prompt)

        return CognitiveBiasInstance(
            bias_id=bias_def.id,
            base_scenario=scenario.context,
            bias_trigger=bias_def.system1_mechanism,
            control_prompt=control_prompt,
            treatment_prompts=treatment_prompts,
            expected_rational_response=variables.get("rational_answer", ""),
            expected_biased_response=variables.get("biased_answer", ""),
            domain=scenario.domain,
            scale=scale,
            debiasing_prompts=debiasing_prompts,
            metadata={
                "bias_name": bias_def.name,
                "theoretical_basis": bias_def.theoretical_basis,
                "scenario_context": scenario.context,
            },
        )

    def _generate_generic(
        self,
        bias_def: BiasDefinition,
        scenario: DomainScenario,
        scale: TestScale,
        include_debiasing: bool,
    ) -> CognitiveBiasInstance:
        """Generate a test case using the bias definition's trigger template."""
        # Use the bias definition's trigger template as a base
        base_template = bias_def.trigger_template
        actor = random.choice(scenario.actors)

        control_prompt = f"""You are acting as a {actor} in the context of {scenario.context}.

Consider the following situation carefully and provide your judgment.

{base_template}

Please explain your reasoning step by step.
"""

        treatment_prompts = {
            intensity: f"""You are acting as a {actor} in the context of {scenario.context}.

{self._add_bias_trigger(bias_def, intensity)}

{base_template}

Please provide your immediate judgment.
"""
            for intensity in TriggerIntensity
        }

        debiasing_prompts = []
        if include_debiasing:
            debiasing_prompts = self._generate_debiasing_prompts(bias_def, control_prompt)

        return CognitiveBiasInstance(
            bias_id=bias_def.id,
            base_scenario=scenario.context,
            bias_trigger=bias_def.system1_mechanism,
            control_prompt=control_prompt,
            treatment_prompts=treatment_prompts,
            expected_rational_response="[Depends on specific instantiation]",
            expected_biased_response="[Depends on specific instantiation]",
            domain=scenario.domain,
            scale=scale,
            debiasing_prompts=debiasing_prompts,
            metadata={
                "bias_name": bias_def.name,
                "theoretical_basis": bias_def.theoretical_basis,
                "classic_paradigm": bias_def.classic_paradigm,
            },
        )

    def _get_template_variables(
        self, bias_def: BiasDefinition, scenario: DomainScenario
    ) -> dict:
        """Generate appropriate variable values for a bias template."""
        actor = random.choice(scenario.actors)
        decision = random.choice(scenario.typical_decisions)

        # Common variables
        variables = {
            "context": scenario.context,
            "decision_maker": actor,
            "decision": decision,
        }

        # Bias-specific variables
        if bias_def.id == "anchoring_effect":
            variables.update({
                "target_quantity": f"the appropriate {decision}",
                "anchor_value": random.randint(50, 500) * 100,
                "rational_answer": "An estimate based solely on relevant factors",
                "biased_answer": "An estimate influenced by the anchor value",
            })

        elif bias_def.id == "gain_loss_framing":
            total = random.randint(3, 10) * 100
            certain = total // 3
            variables.update({
                "gain_sure": certain,
                "gain_all": total,
                "loss_sure": total - certain,
                "loss_all": total,
                "prob": 33,
                "comp_prob": 67,
                "outcome_a": f"{certain} will be saved/lost for certain",
                "outcome_b": f"1/3 chance all {total} saved, 2/3 chance none saved",
                "rational_answer": "Both programs have equal expected value",
                "biased_answer": "Program A in gain frame, Program B in loss frame",
            })

        elif bias_def.id == "base_rate_neglect":
            base_rate = random.choice([5, 10, 20, 30])
            variables.update({
                "base_rate": base_rate,
                "comp_rate": 100 - base_rate,
                "category_a": "engineers",
                "category_b": "lawyers",
                "description": "analytical, enjoys puzzles, somewhat introverted",
                "rational_answer": f"Close to {base_rate}% (the base rate)",
                "biased_answer": "Much higher than base rate due to description",
            })

        elif bias_def.id == "loss_aversion":
            win = random.randint(10, 20) * 10
            lose = random.randint(5, 15) * 10
            variables.update({
                "win_amount": win,
                "lose_amount": lose,
                "expected_value": (win - lose) / 2,
                "rational_answer": "Accept if EV > 0",
                "biased_answer": "Reject despite positive EV due to loss aversion",
            })

        elif bias_def.id == "sunk_cost_fallacy":
            sunk = random.randint(5, 20) * 10000
            future = random.randint(2, 8) * 10000
            returns = random.randint(1, 5) * 10000
            variables.update({
                "sunk_cost": sunk,
                "future_cost": future,
                "expected_return": returns,
                "rational_answer": f"Continue only if {returns} > {future}",
                "biased_answer": "Continue because of prior investment",
            })

        return variables

    def _safe_format(self, template: str, variables: dict) -> str:
        """
        Safely format a template string with variables.

        Handles missing variables gracefully by leaving them as placeholders.
        """
        def replace_var(match):
            var_name = match.group(1)
            return str(variables.get(var_name, f"{{{var_name}}}"))

        # Match {variable_name} patterns
        return re.sub(r'\{(\w+)\}', replace_var, template)

    def _adjust_for_intensity(
        self, variables: dict, intensity: TriggerIntensity
    ) -> dict:
        """Adjust template variables based on trigger intensity."""
        adjusted = variables.copy()

        multipliers = {
            TriggerIntensity.WEAK: 0.5,
            TriggerIntensity.MODERATE: 1.0,
            TriggerIntensity.STRONG: 1.5,
            TriggerIntensity.ADVERSARIAL: 2.0,
        }

        multiplier = multipliers[intensity]

        # Adjust numeric values that affect bias strength
        if "anchor_value" in adjusted:
            base_anchor = adjusted["anchor_value"]
            adjusted["anchor_value"] = int(base_anchor * multiplier)

        return adjusted

    def _add_bias_trigger(
        self, bias_def: BiasDefinition, intensity: TriggerIntensity
    ) -> str:
        """Generate a bias trigger statement based on intensity."""
        intensity_words = {
            TriggerIntensity.WEAK: "Consider that",
            TriggerIntensity.MODERATE: "Keep in mind that",
            TriggerIntensity.STRONG: "It's important to note that",
            TriggerIntensity.ADVERSARIAL: "You must account for the fact that",
        }

        prefix = intensity_words[intensity]
        return f"{prefix} {bias_def.system1_mechanism.lower()}."

    def _generate_debiasing_prompts(
        self, bias_def: BiasDefinition, base_prompt: str
    ) -> list[str]:
        """Generate debiasing prompts to test System 2 override capability."""
        debiasing_instructions = [
            # Explicit warning about the bias
            f"""Before answering, be aware that you may be susceptible to {bias_def.name}.
{bias_def.description}
To avoid this bias: {bias_def.system2_override}

{base_prompt}""",
            # Chain-of-thought instruction
            f"""Please approach this problem step-by-step, showing all your reasoning.
Consider multiple perspectives and check your initial intuition carefully.

{base_prompt}""",
            # Explicit System 2 engagement
            f"""Take your time with this decision. Before giving your final answer:
1. Identify any potential cognitive biases that might affect your judgment
2. Consider what a perfectly rational agent would conclude
3. Explain why your answer is logically sound

{base_prompt}""",
        ]

        return debiasing_instructions

    def generate_batch(
        self,
        bias_ids: list[str] | None = None,
        domains: list[Domain] | None = None,
        instances_per_combination: int = 3,
    ) -> list[CognitiveBiasInstance]:
        """
        Generate a batch of test instances across biases and domains.

        Args:
            bias_ids: List of bias IDs to include (defaults to all)
            domains: List of domains to include (defaults to all)
            instances_per_combination: Number of instances per bias-domain pair

        Returns:
            List of generated test instances
        """
        if bias_ids is None:
            bias_ids = list(BIAS_TAXONOMY.keys())
        if domains is None:
            domains = list(Domain)

        instances = []
        for bias_id in bias_ids:
            for domain in domains:
                for _ in range(instances_per_combination):
                    try:
                        instance = self.generate_instance(bias_id, domain)
                        instances.append(instance)
                    except Exception as e:
                        print(f"Warning: Failed to generate {bias_id} for {domain}: {e}")

        return instances

    def export_to_json(
        self, instances: list[CognitiveBiasInstance], filepath: str
    ) -> None:
        """Export generated instances to JSON file."""
        data = []
        for inst in instances:
            data.append({
                "bias_id": inst.bias_id,
                "base_scenario": inst.base_scenario,
                "bias_trigger": inst.bias_trigger,
                "control_prompt": inst.control_prompt,
                "treatment_prompts": {
                    k.value: v for k, v in inst.treatment_prompts.items()
                },
                "expected_rational_response": inst.expected_rational_response,
                "expected_biased_response": inst.expected_biased_response,
                "domain": inst.domain.value,
                "scale": inst.scale.value,
                "debiasing_prompts": inst.debiasing_prompts,
                "metadata": inst.metadata,
            })

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    async def generate_with_llm(
        self,
        bias_id: str,
        scenario_description: str,
        domain: Domain = Domain.PROFESSIONAL,
    ) -> CognitiveBiasInstance:
        """
        Generate a test case using LLM to fill template gaps.

        Uses the GEN prompt methodology from Malberg et al. to create
        diverse, contextually rich test cases.

        Args:
            bias_id: The bias to test
            scenario_description: High-level scenario (e.g., "a cardiologist deciding on treatment")
            domain: Domain context

        Returns:
            A fully instantiated CognitiveBiasInstance
        """
        if self.llm_client is None:
            raise ValueError("LLM client required for LLM-augmented generation")

        bias_def = get_bias_by_id(bias_id)
        if bias_def is None:
            raise ValueError(f"Unknown bias ID: {bias_id}")

        gen_prompt = f'''You are helping to create test cases for evaluating cognitive biases in AI systems.

BIAS: {bias_def.name}
DESCRIPTION: {bias_def.description}
CLASSIC EXAMPLE: {bias_def.classic_paradigm}

SCENARIO: {scenario_description}
DOMAIN: {domain.value}

Generate a realistic test case that:
1. Is set in the given scenario
2. Has a clear CONTROL condition (no bias trigger)
3. Has a TREATMENT condition with a subtle bias trigger related to {bias_def.name}
4. Has a clearly rational answer that should be given if unbiased
5. Has a predictable biased answer that would result from {bias_def.name}

Respond in JSON format:
{{
    "base_scenario": "The neutral context for the decision",
    "control_prompt": "The prompt without bias triggers",
    "treatment_prompt": "The prompt with bias trigger embedded",
    "bias_trigger": "What specifically triggers the bias",
    "rational_answer": "The objectively correct response",
    "biased_answer": "The response expected if biased"
}}
'''

        response = self.llm_client.generate(gen_prompt, max_tokens=1500)

        # Parse the JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in LLM response")

            # Create treatment prompts at different intensities
            treatment_prompts = {
                TriggerIntensity.WEAK: data.get("treatment_prompt", ""),
                TriggerIntensity.MODERATE: data.get("treatment_prompt", ""),
                TriggerIntensity.STRONG: data.get("treatment_prompt", ""),
                TriggerIntensity.ADVERSARIAL: data.get("treatment_prompt", ""),
            }

            return CognitiveBiasInstance(
                bias_id=bias_id,
                base_scenario=data.get("base_scenario", scenario_description),
                bias_trigger=data.get("bias_trigger", bias_def.system1_mechanism),
                control_prompt=data.get("control_prompt", ""),
                treatment_prompts=treatment_prompts,
                expected_rational_response=data.get("rational_answer", ""),
                expected_biased_response=data.get("biased_answer", ""),
                domain=domain,
                metadata={
                    "generation_method": "llm_augmented",
                    "scenario_description": scenario_description,
                    "bias_name": bias_def.name,
                },
            )

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")


# =============================================================================
# BENCHMARK TIERS
# Following SWE-bench's tiered approach (Core, Extended, Interaction)
# =============================================================================

# Core biases - Most well-established, highest research validity
KAHNE_BENCH_CORE_BIASES = [
    "anchoring_effect",
    "availability_bias",
    "base_rate_neglect",
    "confirmation_bias",
    "conjunction_fallacy",
    "framing_effect",  # gain_loss_framing
    "gain_loss_framing",
    "loss_aversion",
    "overconfidence_effect",
    "sunk_cost_fallacy",
    # Additional core biases from K&T's most cited work
    "representativeness",  # Not in our taxonomy as separate - use base_rate_neglect
    "status_quo_bias",
    "endowment_effect",
    "certainty_effect",
    "present_bias",
]

# Extended biases - Full 50+ bias coverage
KAHNE_BENCH_EXTENDED_BIASES = list(BIAS_TAXONOMY.keys())

# Interaction pairs for compound testing
KAHNE_BENCH_INTERACTION_PAIRS = [
    ("anchoring_effect", "availability_bias"),
    ("anchoring_effect", "overconfidence_effect"),
    ("gain_loss_framing", "loss_aversion"),
    ("gain_loss_framing", "certainty_effect"),
    ("availability_bias", "neglect_of_probability"),
    ("confirmation_bias", "overconfidence_effect"),
    ("base_rate_neglect", "conjunction_fallacy"),
    ("status_quo_bias", "endowment_effect"),
    ("present_bias", "planning_fallacy"),
    ("sunk_cost_fallacy", "loss_aversion"),
]


class KahneBenchTier:
    """Enumeration of benchmark tiers."""
    CORE = "core"
    EXTENDED = "extended"
    INTERACTION = "interaction"


def get_tier_biases(tier: str) -> list[str]:
    """Get the list of bias IDs for a specific tier."""
    if tier == KahneBenchTier.CORE:
        # Filter to only biases that exist in taxonomy
        return [b for b in KAHNE_BENCH_CORE_BIASES if b in BIAS_TAXONOMY]
    elif tier == KahneBenchTier.EXTENDED:
        return KAHNE_BENCH_EXTENDED_BIASES
    elif tier == KahneBenchTier.INTERACTION:
        # Return unique biases from interaction pairs
        biases = set()
        for b1, b2 in KAHNE_BENCH_INTERACTION_PAIRS:
            biases.add(b1)
            biases.add(b2)
        return list(biases)
    else:
        raise ValueError(f"Unknown tier: {tier}")
