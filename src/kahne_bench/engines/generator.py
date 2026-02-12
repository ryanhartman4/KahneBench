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
# NOTE: typical_decisions should be numeric/estimable for anchoring bias tests
# Categorical decisions (test selection, protocol choice) are handled separately
DOMAIN_SCENARIOS: dict[Domain, list[DomainScenario]] = {
    Domain.INDIVIDUAL: [
        DomainScenario(
            domain=Domain.INDIVIDUAL,
            context="personal financial planning",
            actors=["financial advisor", "individual investor", "retiree"],
            # Numeric decisions suitable for anchoring tests
            typical_decisions=["portfolio value estimate", "emergency fund amount", "monthly budget"],
            value_ranges={"amount": (5000, 50000), "percentage": (10, 25)},
        ),
        DomainScenario(
            domain=Domain.INDIVIDUAL,
            context="consumer choice",
            actors=["shopper", "consumer", "buyer"],
            typical_decisions=["fair price estimate", "budget amount", "discount threshold"],
            value_ranges={"price": (50, 500), "discount": (10, 40)},
        ),
    ],
    Domain.PROFESSIONAL: [
        DomainScenario(
            domain=Domain.PROFESSIONAL,
            context="medical diagnosis",
            actors=["physician", "cardiologist", "oncologist"],
            # Numeric decisions - probability estimates suitable for anchoring
            typical_decisions=["disease probability estimate", "treatment success rate", "patient count estimate"],
            value_ranges={"probability": (10, 60), "patients": (50, 500), "percentage": (20, 70)},
        ),
        DomainScenario(
            domain=Domain.PROFESSIONAL,
            context="business strategy",
            actors=["CEO", "manager", "consultant"],
            typical_decisions=["revenue forecast", "headcount estimate", "market share projection"],
            value_ranges={"revenue": (100000, 1000000), "employees": (10, 200), "percentage": (5, 40)},
        ),
        DomainScenario(
            domain=Domain.PROFESSIONAL,
            context="legal analysis",
            actors=["attorney", "judge", "legal analyst"],
            typical_decisions=["damages estimate", "settlement value", "case success probability"],
            value_ranges={"damages": (50000, 500000), "probability": (20, 80), "percentage": (20, 80)},
        ),
    ],
    Domain.SOCIAL: [
        DomainScenario(
            domain=Domain.SOCIAL,
            context="salary negotiation",
            actors=["job candidate", "hiring manager", "recruiter"],
            typical_decisions=["salary estimate", "bonus percentage", "total compensation value"],
            value_ranges={"salary": (60000, 150000), "bonus": (5, 25), "percentage": (5, 25)},
        ),
        DomainScenario(
            domain=Domain.SOCIAL,
            context="team collaboration",
            actors=["team lead", "project manager", "team member"],
            typical_decisions=["project timeline estimate", "resource budget", "team size recommendation"],
            value_ranges={"time": (2, 24), "resources": (10000, 100000), "count": (3, 15)},
        ),
    ],
    Domain.TEMPORAL: [
        DomainScenario(
            domain=Domain.TEMPORAL,
            context="retirement planning",
            actors=["pre-retiree", "financial planner", "pension advisor"],
            typical_decisions=["retirement savings target", "annual contribution amount", "withdrawal amount"],
            value_ranges={"years": (5, 30), "savings": (100000, 1000000), "percentage": (3, 15)},
        ),
        DomainScenario(
            domain=Domain.TEMPORAL,
            context="career development",
            actors=["professional", "career coach", "mentor"],
            typical_decisions=["training budget", "expected salary increase", "time to promotion estimate"],
            value_ranges={"months": (6, 36), "cost": (2000, 20000), "percentage": (5, 30)},
        ),
    ],
    Domain.RISK: [
        DomainScenario(
            domain=Domain.RISK,
            context="environmental policy",
            actors=["policy maker", "environmental scientist", "regulator"],
            typical_decisions=["emission reduction target", "compliance cost estimate", "risk probability"],
            value_ranges={"probability": (5, 30), "impact": (100000, 10000000), "percentage": (10, 50)},
        ),
        DomainScenario(
            domain=Domain.RISK,
            context="technology adoption",
            actors=["CTO", "IT manager", "security analyst"],
            typical_decisions=["implementation cost estimate", "downtime estimate", "failure rate estimate"],
            value_ranges={"cost": (50000, 500000), "probability": (1, 20), "percentage": (1, 20)},
        ),
    ],
}


# Risk-free rate assumption for present bias calculations
ANNUAL_RFR = 0.05  # 5% annual risk-free rate

# Templates for specific biases - structured prompts with placeholders
BIAS_TEMPLATES: dict[str, dict[str, str]] = {
    "anchoring_effect": {
        "control": """
You are evaluating {context}.

{decision_maker} needs to estimate {target_quantity}.

{range_hint}

Based on available information and your expertise, provide your best estimate.
Provide your answer as a single number only.

Answer:
""",
        "treatment": """
You are evaluating {context}.

{decision_maker} needs to estimate {target_quantity}.

For context, a recent {anchor_source} reported a figure of {anchor_value} for a related metric. This background information may or may not be directly applicable.

{range_hint}

Based on your own independent analysis, provide your best estimate.
Provide your answer as a single number only.

Answer:
""",
    },
    "gain_loss_framing": {
        "control": """
You are advising on {context}.

A situation affects {total_affected} people. Consider two programs:
- Program A: A fixed outcome affecting {fixed_outcome} people (EV = {ev_a} people)
- Program B: A {prob}% probability of affecting all {total_affected} people, and a {comp_prob}% probability of affecting no one (EV = {ev_b} people)

A rational decision-maker should select the option with the higher expected value.

Which program do you recommend? Reply with A or B.
""",
        "treatment_gain": """
You are advising on {context}.

A disease threatens {total_affected} lives. Consider two programs:
- Program A: {gain_sure} people will be saved for certain.
- Program B: There is a {prob}% probability that all {gain_all} will be saved, and a {comp_prob}% probability that no one will be saved.

Which program do you recommend? Reply with A or B.
""",
        "treatment_loss": """
You are advising on {context}.

A disease threatens {total_affected} lives. Consider two programs:
- Program A: {loss_sure} people will die for certain.
- Program B: There is a {prob}% probability that no one will die, and a {comp_prob}% probability that all {loss_all} will die.

Which program do you recommend? Reply with A or B.
""",
    },
    "base_rate_neglect": {
        "control": """
In a population, {base_rate}% are engineers and {comp_rate}% are lawyers.

A person is randomly selected from this population.
Without any additional information, which is more likely?

A) This person is {category_a}
B) This person is {category_b}

Reply with just A or B.
""",
        "treatment": """
In a population, {base_rate}% are engineers and {comp_rate}% are lawyers.

A person is randomly selected from this population. Here is a description:
"{description}"

Which is more likely?

A) This person is {category_a}
B) This person is {category_b}

Reply with just A or B.
""",
    },
    "conjunction_fallacy": {
        "control": """
{person_name} works at a large organization.

Which of the following is more likely?

A) {person_name} {general_category}
B) {person_name} {general_category} and {specific_detail}

Respond with your choice in this format:
Answer: [A or B]
""",
        "treatment": """
{person_name} is described by colleagues as: {description}

Which of the following is more likely?

A) {person_name} {general_category}
B) {person_name} {general_category} and {specific_detail}

Respond with your choice in this format:
Answer: [A or B]
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
Estimate the annual number of deaths in the United States from {rare_cause}.
For context, {common_cause} is a common cause of death.

Provide a single numeric estimate only.

Answer:
""",
        "treatment": """
Recently, there have been several highly publicized incidents of {rare_cause}.
News coverage has featured dramatic stories of victims and their families.

Estimate the annual number of deaths in the United States from {rare_cause}.
For context, {common_cause} is a common cause of death.

Provide a single numeric estimate only.

Answer:
""",
    },
    "overconfidence_effect": {
        "control": """
Answer the following question:
{question}

After answering, rate your confidence (0-100%) that your answer is correct.

Format your response as:
Answer: [your answer]
Confidence: [0-100]%
""",
        "treatment": """
You have strong expertise in this area.

Answer the following question:
{question}

After answering, rate your confidence (0-100%) that your answer is correct.

Format your response as:
Answer: [your answer]
Confidence: [0-100]%
""",
    },
    "sunk_cost_fallacy": {
        "control": """
{scenario_description}

Looking at the projected costs and outcomes going forward:

Option A: {option_a_label} — Additional investment needed: ${continue_cost}. Projected {time_horizon} value: ${continue_value}.
Option B: {option_b_label} — Investment needed: ${switch_cost}. Projected {time_horizon} value: ${switch_value}.

Based on the forward-looking financials, which option do you recommend?

Respond with your choice in this format:
Answer: [A or B]
""",
        "treatment": """
{scenario_description}

{sunk_cost_mention}

Looking at the projected costs and outcomes going forward:

Option A: {option_a_label} — Additional investment needed: ${continue_cost}. Projected {time_horizon} value: ${continue_value}.
Option B: {option_b_label} — Investment needed: ${switch_cost}. Projected {time_horizon} value: ${switch_value}.

Which option do you recommend?

Respond with your choice in this format:
Answer: [A or B]
""",
    },
    "status_quo_bias": {
        "control": """
You are evaluating options for your {context_label}. Consider these two:

Option A: {option_a_description}
Option B: {option_b_description}

Based purely on the features and value, which option do you recommend?

Respond with your choice in this format:
Answer: [A or B]
""",
        "treatment": """
{incumbent_framing}

You now have an opportunity to evaluate an alternative.

Option A: {option_a_description}
Option B: {option_b_description}

Which option do you recommend?

Respond with your choice in this format:
Answer: [A or B]
""",
    },
    "present_bias": {
        "control": """
{scenario_context}

You have two options:
A) {option_a_description}
B) {option_b_description}

For reference, the current annual risk-free rate is 5%. Over {months} months, this corresponds to approximately {rfr_pct}% growth. Use this as a discount rate to compare the present value of each option.

Which option do you choose? Reply with just A or B.
""",
        "treatment": """
{scenario_context}

You have two options:
A) {option_a_description}
B) {option_b_description}

Which option do you choose? Reply with just A or B.
""",
    },
    "gambler_fallacy": {
        "control": """
{scenario_context_control}

{neutral_question}

Please explain your reasoning and provide your recommendation.
""",
        "treatment": """
{scenario_context_treatment}

{question}

Please explain your reasoning and provide your recommendation.
""",
    },
    "endowment_effect": {
        "control": """
A {item} has a market value of ${market_value}.

A buyer offers exactly ${market_value} for this {item}.

Is this a fair offer?

Option A: Yes, this is a fair offer at market value
Option B: No, the item is worth more than market value

Which option do you choose?

Respond with your choice in this format:
Answer: [A or B]
""",
        "treatment": """
You own a {item} that you've had for a while. It has become part of your possessions.

The market value is ${market_value}. A buyer offers exactly ${market_value} for your {item}.

Should you accept this offer?

Option A: Yes, accept the fair market value offer
Option B: No, I would need more than the listed market price to part with it

Which option do you choose?

Respond with your choice in this format:
Answer: [A or B]
""",
    },
    "confirmation_bias": {
        "control": """
You are investigating whether {hypothesis}.

Which of the following investigation plans would you follow?

Option A: Review the cases and data that are consistent with the initial findings
Option B: Prioritize examining cases where the findings did not hold
Option C: Conduct a comprehensive review of all relevant data regardless of direction

Which option do you choose?

Respond with your choice in this format:
Answer: [A, B, or C]
""",
        "treatment": """
You believe that {hypothesis}.

You have gathered some initial evidence that supports your belief.

Which of the following next steps would you take?

Option A: Review additional cases and data that are consistent with your initial findings
Option B: Prioritize examining cases where the findings did not hold
Option C: Conduct a comprehensive review of all relevant data regardless of direction

Which option do you choose?

Respond with your choice in this format:
Answer: [A, B, or C]
""",
    },
    "hindsight_bias": {
        "control": """
A company is about to launch a new product. Industry analysts are divided - some predict success, others predict failure.

How predictable is the outcome?

Option A: The outcome is highly unpredictable - could go either way
Option B: The outcome is fairly predictable based on available information

Which option do you choose?

Respond with your choice in this format:
Answer: [A or B]
""",
        "treatment": """
A company launched a new product last year. It {outcome}.

Before the launch, industry analysts were divided - some predicted success, others predicted failure.

Looking back, how predictable was this outcome BEFORE it happened?

Option A: It was genuinely unpredictable - the divided analysts show real uncertainty existed
Option B: It was fairly predictable - the signs were there all along

Which option do you choose?

Respond with your choice in this format:
Answer: [A or B]
""",
    },
    "neglect_of_probability": {
        "control": """
Consider two options:
A) A {prob_a}% chance to {outcome_a}
B) A {prob_b}% chance to {outcome_b}

Calculate the expected value of each option and choose the better one.
""",
        "treatment": """
Imagine you might {negative_outcome}! This terrible event could happen!

Consider two options:
A) A {prob_a}% chance to {outcome_a}
B) A {prob_b}% chance to {outcome_b}

Which option do you choose?
""",
    },
    "certainty_effect": {
        "control": """
Choose between:
A) {prob_certain}% chance to win ${amount_certain}
B) {prob_gamble}% chance to win ${amount_gamble}

Expected values: A = ${ev_a}, B = ${ev_b}

Which do you prefer?
""",
        "treatment": """
Choose between:
A) Win ${amount_certain} for CERTAIN (guaranteed, 100% sure, no risk)
B) {prob_gamble}% chance to win ${amount_gamble}, otherwise win nothing

Which do you prefer?
""",
    },
    "planning_fallacy": {
        "control": """
A new {project_type} project needs to be scheduled.

Based on {reference_data} similar projects, what is a realistic timeline?
""",
        "treatment": """
You are planning your own {project_type} project. You've thought carefully about each step and feel optimistic.

How long will YOUR project take to complete?
""",
    },
    "insensitivity_to_sample_size": {
        "control": """
Hospital A delivers about {large_sample} babies per day.
Hospital B delivers about {small_sample} babies per day.

On average, about 50% of babies are boys, though the percentage varies day-to-day.

Which hospital is more likely to have days where more than 60% of babies born are boys?
""",
        "treatment": """
Two hospitals track births:
- Large Hospital: {large_sample} births per day
- Small Hospital: {small_sample} births per day

About 50% of babies are typically boys.

On a given day, which hospital is more likely to record that more than 60% of babies born are boys?

Consider that both hospitals are reputable and have good data collection.
""",
    },
    # Note: "regression_to_mean" renamed to "regression_neglect" to match taxonomy
    "illusory_correlation": {
        "control": """
You are analyzing data about {variable_a} and {variable_b}.

The statistical analysis shows {correlation_value}. What is the relationship between these variables?

Option A: There is a meaningful correlation between {variable_a} and {variable_b}
Option B: There is no meaningful correlation between {variable_a} and {variable_b}

Which option is supported by the data?
""",
        "treatment": """
You have heard many anecdotes about {variable_a} being linked to {variable_b}.

These memorable stories are quite vivid and frequently discussed.

What is the relationship between {variable_a} and {variable_b}?

Option A: There is a meaningful correlation between {variable_a} and {variable_b}
Option B: There is no meaningful correlation between {variable_a} and {variable_b}

Which option do you believe?
""",
    },
    "peak_end_rule": {
        "control": """
Compare two experiences:
Experience A: {duration_a} minutes at {intensity_a} intensity
Experience B: {duration_b} minutes at {intensity_b} intensity

Which experience was objectively more unpleasant overall (total discomfort)?
""",
        "treatment": """
Recall two experiences you had:
Experience A: {duration_a} minutes, mostly unpleasant, ending moderately
Experience B: {duration_b} minutes, varying intensity, ending with relief

Which experience would you be more willing to repeat?
""",
    },
    "scope_insensitivity": {
        "control": """
Calculate the proportional value of saving:
A) {small_count} birds
B) {medium_count} birds
C) {large_count} birds

If saving {small_count} birds is worth $X, what should saving the other amounts be worth?
""",
        "treatment": """
An environmental charity is asking for donations to save endangered birds.

How much would you donate to save:
A) {small_count} birds from an oil spill?
B) {medium_count} birds from an oil spill?
C) {large_count} birds from an oil spill?
""",
    },
    "identifiable_victim_effect": {
        "control": """
An intervention can help address a problem affecting {statistical_count} people.

What priority should this intervention receive? How much funding is appropriate?
""",
        "treatment": """
Meet {victim_name}, a {victim_age}-year-old who {victim_story}.

{victim_name} is one of {statistical_count} people affected by this problem.

What priority should helping {victim_name} and others like them receive?
""",
    },
    "affect_heuristic": {
        "control": """
Evaluate {technology_or_activity} based on:
- Documented risks: {risk_data}
- Documented benefits: {benefit_data}

What is your overall assessment?
""",
        "treatment": """
{technology_or_activity} makes people feel {emotion} when they think about it.

What are the risks and benefits of {technology_or_activity}?
""",
    },
    "distinction_bias": {
        "control": """
Rate the quality of Option A on its own merits.
Option A: {option_a_details}

Then separately rate Option B.
Option B: {option_b_details}
""",
        "treatment": """
Compare these two options side by side:
Option A: {option_a_details}
Option B: {option_b_details}

Which is better, and by how much?
""",
    },
    "denomination_effect": {
        "control": """
You have ${total_amount} for discretionary spending.

How would you allocate this amount?
""",
        "treatment": """
You have ${total_amount} for discretionary spending.

You received this as {denomination_form} (a single ${total_amount} bill / many small bills).

How would you allocate this amount?
""",
    },
    "decoy_effect": {
        "control": """
Choose between two options:
A) {option_a}: {option_a_attr1}, {option_a_attr2}
B) {option_b}: {option_b_attr1}, {option_b_attr2}

Which do you prefer?
""",
        "treatment": """
Choose between three options:
A) {option_a}: {option_a_attr1}, {option_a_attr2}
B) {option_b}: {option_b_attr1}, {option_b_attr2}
C) {decoy}: {decoy_attr1}, {decoy_attr2}

Which do you prefer?
""",
    },
    "illusion_of_control": {
        "control": """
You are participating in a lottery with {probability}% chance of winning.

What is your probability of winning?
""",
        "treatment": """
You are participating in a lottery with {probability}% chance of winning.

You get to choose your own numbers and you've developed a personal system.

What is your probability of winning?
""",
    },
    "optimism_bias": {
        "control": """
What is the average probability of {event} happening to a person in your demographic?
""",
        "treatment": """
What is the probability of {event} happening to YOU personally, compared to the average person?
""",
    },
    "attribute_substitution": {
        "control": """
Answer this question: {hard_question}

Take your time to analyze all relevant factors.
""",
        "treatment": """
Answer this question: {hard_question}

As you consider this, your immediate feeling is {gut_feeling}.
""",
    },
    "default_effect": {
        "control": """
You must choose between:
- Option A: {option_a_description}
- Option B: {option_b_description}

Neither option is pre-selected. Which do you choose?
""",
        "treatment": """
Your current setting is Option A: {option_a_description}

You can switch to Option B: {option_b_description} by opting out.

Do you keep your current setting or switch?
""",
    },
    "zero_risk_bias": {
        "control": """
Choose between two safety interventions:
A) Reduce risk from {initial_risk_a}% to {final_risk_a}% (reduction of {reduction_a} percentage points)
B) Reduce risk from {initial_risk_b}% to {final_risk_b}% (reduction of {reduction_b} percentage points)

Which intervention saves more lives in expectation?
""",
        "treatment": """
Choose between two safety interventions:
A) COMPLETELY ELIMINATE the {small_risk}% risk (reduce to 0%)
B) Reduce the {large_risk}% risk by {large_reduction} percentage points

Which intervention do you prefer?
""",
    },
    "outcome_bias": {
        "control": """
Evaluate this decision based on the information available at the time:
Decision: {decision_description}
Information available: {info_at_time}

Was this a good decision?
""",
        "treatment": """
Evaluate this decision:
Decision: {decision_description}
Outcome: {actual_outcome}

Was this a good decision?
""",
    },
    "bandwagon_effect": {
        "control": """
Consider adopting {innovation_or_practice}.

Based on the objective evidence, should you adopt it?
""",
        "treatment": """
{percentage}% of your peers have already adopted {innovation_or_practice}.

Should you adopt it too?
""",
    },
    "halo_effect": {
        "control": """
Evaluate this person's competence in {skill_area}:
Information: {relevant_skill_info}

Rate their competence in {skill_area}.
""",
        "treatment": """
Evaluate this person's competence in {skill_area}:
Background: They are {positive_trait} and {other_positive_trait}.
Information: {relevant_skill_info}

Rate their competence in {skill_area}.
""",
    },
    # =========================================================================
    # REPRESENTATIVENESS HEURISTIC - Additional Templates
    # =========================================================================
    "hot_hand_fallacy": {
        "control": """
A basketball player has a career free-throw percentage of {baseline_rate}%.

They are about to take a free throw.

What is the probability they will make this shot?
""",
        "treatment": """
A basketball player has a career free-throw percentage of {baseline_rate}%.

They have just made {streak_length} consecutive free throws in a row - they're "on fire"!

They are about to take another free throw.

What is the probability they will make this shot?
""",
    },
    "regression_neglect": {
        "control": """
{person_name} scored {extreme_score} on a test, which is {direction} the average of {average_score}.

Predict {person_name}'s score on the next similar test.
""",
        "treatment": """
{person_name} scored {extreme_score} on a test, which was {direction} the average of {average_score}.

After their exceptional performance, the instructor decided to {intervention} them.
On the next test, their score was closer to average.

Did the instructor's {intervention} cause this change in performance?
""",
    },
    "stereotype_bias": {
        "control": """
In a group of 100 people, {engineer_rate} are engineers and {lawyer_rate} are lawyers.

A person is randomly selected from this group.

What is the probability this person is an engineer?
""",
        "treatment": """
In a group of 100 people, {engineer_rate} are engineers and {lawyer_rate} are lawyers.

A person is randomly selected from this group. Here is their description:
"{description}"

This description was written by a psychologist based on projective tests.

What is the probability this person is an engineer?
""",
    },
    "prototype_heuristic": {
        "control": """
Consider these two creatures:
A) A robin
B) A penguin

Both are members of the category "bird."

What percentage of all birds do you think share characteristics with each?
""",
        "treatment": """
Consider {instance_name}, who has the following characteristics:
{characteristics}

How typical is {instance_name} of the category "{category}"?

Based on this typicality, estimate the probability that {instance_name} truly belongs to "{category}".
""",
    },
    # =========================================================================
    # AVAILABILITY HEURISTIC - Additional Templates
    # =========================================================================
    "recency_bias": {
        "control": """
Based on historical data over the past 20 years, estimate the annual probability of:

A) {event_type_a}
B) {event_type_b}

Provide your probability estimates.
""",
        "treatment": """
In the past month, there have been several news reports about {event_type_b}.

Based on your assessment, estimate the annual probability of:

A) {event_type_a}
B) {event_type_b}

Provide your probability estimates.
""",
    },
    "salience_bias": {
        "control": """
Estimate the annual number of deaths in the United States from:

A) Heart disease
B) Homicide

Provide numerical estimates.
""",
        "treatment": """
You recently watched a documentary featuring dramatic footage of violent crimes, interviews with grieving families, and statistics presented in alarming graphics.

Estimate the annual number of deaths in the United States from:

A) Heart disease
B) Homicide

Provide numerical estimates.
""",
    },
    "simulation_heuristic": {
        "control": """
{person_name} was in a car accident on their way to the airport.

They missed their flight by {time_missed}.

How upset do you think {person_name} is about missing the flight? Rate from 1-10.
""",
        "treatment": """
{person_name} was in a car accident on their way to the airport.

If only they had left {counterfactual_time} earlier, they would have made the flight.
As it happened, they missed their flight by just {time_missed}.

How upset do you think {person_name} is about missing the flight? Rate from 1-10.
""",
    },
    "primacy_bias": {
        "control": """
Evaluate the following job candidates for a management position:

Candidate A: {candidate_a_description}
Candidate B: {candidate_b_description}
Candidate C: {candidate_c_description}
Candidate D: {candidate_d_description}

Note: Candidates are listed in random order.

Rank the candidates from most to least qualified.
""",
        "treatment": """
Evaluate the following job candidates for a management position:

Candidate A: {candidate_a_description}
Candidate B: {candidate_b_description}
Candidate C: {candidate_c_description}
Candidate D: {candidate_d_description}

Consider that Candidate A was reviewed first and made a strong initial impression.

Rank the candidates from most to least qualified.
""",
    },
    # =========================================================================
    # ANCHORING - Additional Templates
    # =========================================================================
    "insufficient_adjustment": {
        "control": """
Estimate the population of {city_name}.

Provide your best estimate.
""",
        "treatment": """
Start by considering whether the population of {city_name} is more or less than {anchor_value}.

Now, estimate the actual population of {city_name}.

Have you adjusted sufficiently from your starting point?
""",
    },
    "focalism": {
        "control": """
Consider all the factors that might affect your happiness over the next year, including:
- Work and career
- Relationships and family
- Health and fitness
- Hobbies and leisure
- Financial situation
- Living environment

How happy do you expect to be overall next year? Rate from 1-10.
""",
        "treatment": """
Imagine you receive {focal_event} next year.

How happy do you expect to be overall next year? Rate from 1-10.
""",
    },
    "first_offer_anchoring": {
        "control": """
You are selling your used car. Based on market research:
- Similar cars sell for ${market_low} to ${market_high}
- Average sale price is ${market_average}
- Your car is in {condition} condition

What price would you accept for your car?
""",
        "treatment": """
You are selling your used car. A potential buyer makes an opening offer of ${first_offer}.

Based on market research:
- Similar cars sell for ${market_low} to ${market_high}
- Average sale price is ${market_average}
- Your car is in {condition} condition

What price would you accept for your car?
""",
    },
    "numeric_priming": {
        "control": """
Estimate the price of a typical bottle of wine at a mid-range restaurant.

Provide your estimate in dollars.
""",
        "treatment": """
Please write down the last two digits of your phone number: {phone_digits}

Now, estimate the price of a typical bottle of wine at a mid-range restaurant.

Provide your estimate in dollars.
""",
    },
    # =========================================================================
    # LOSS AVERSION - Additional Templates
    # =========================================================================
    "disposition_effect": {
        "control": """
You need to sell one stock from your portfolio to raise cash.

Stock A: Current value ${value_a}, no particular history
Stock B: Current value ${value_b}, no particular history

Both stocks have similar future prospects.

Which stock do you sell?
""",
        "treatment": """
You need to sell one stock from your portfolio to raise cash.

Stock A: You bought at ${purchase_a}, now worth ${value_a} (up {gain_pct}%)
Stock B: You bought at ${purchase_b}, now worth ${value_b} (down {loss_pct}%)

Both stocks have similar future prospects.

Which stock do you sell?
""",
    },
    # =========================================================================
    # FRAMING EFFECTS - Additional Templates
    # =========================================================================
    "attribute_framing": {
        "control": """
Evaluate this ground beef product:

Nutritional information: Contains {fat_percentage}% fat, {lean_percentage}% lean meat.

How would you rate the quality of this beef? (1-10)
""",
        "treatment_positive": """
Evaluate this ground beef product:

Label: "{lean_percentage}% LEAN"

How would you rate the quality of this beef? (1-10)
""",
        "treatment_negative": """
Evaluate this ground beef product:

Label: "{fat_percentage}% FAT"

How would you rate the quality of this beef? (1-10)
""",
    },
    "reference_point_framing": {
        "control": """
Your current salary is $X.

You receive a new job offer with salary ${new_salary}.

Evaluate this offer.
""",
        "treatment_gain": """
Your current salary is ${current_salary}.

You receive a new job offer with salary ${new_salary}.

This represents a gain of ${gain_amount} from your current position.

Evaluate this offer.
""",
        "treatment_loss": """
You were previously earning ${previous_salary}.

You now have an offer of ${new_salary}.

This represents a loss of ${loss_amount} from your previous position.

Evaluate this offer.
""",
    },
    "risk_framing": {
        "control": """
A medical treatment has the following outcomes:
- {outcome_rate}% of patients {outcome_description}

Would you recommend this treatment?
""",
        "treatment_frequency": """
A medical treatment has the following outcomes:
- {numerator} out of {denominator} patients {outcome_description}

Would you recommend this treatment?
""",
        "treatment_percentage": """
A medical treatment has the following outcomes:
- {outcome_rate}% of patients {outcome_description}

Note: This is equivalent to {numerator} out of {denominator} patients.

Would you recommend this treatment?
""",
    },
    "temporal_framing": {
        "control": """
A subscription service costs ${annual_cost} per year (equivalent to ${daily_cost} per day).

How would you evaluate this pricing?

Option A: This seems like a good value
Option B: This seems expensive
Option C: Need more information about the service to judge

Which option best describes your view?
""",
        "treatment": """
A subscription service costs just ${daily_cost} per day!

That's less than the price of a cup of coffee!

(Note: This works out to ${annual_cost} per year)

How would you evaluate this pricing?

Option A: This seems like a good value - less than a coffee per day is reasonable
Option B: This seems expensive - ${annual_cost}/year is a significant amount
Option C: Need more information about the service to judge

Which option best describes your view?
""",
    },
    "mental_accounting": {
        "control": """
You have ${total_amount} available.

Would you spend ${purchase_amount} on {purchase_item}?
""",
        "treatment": """
You have ${total_amount} available.

This money came from {money_source}.

Would you spend ${purchase_amount} on {purchase_item}?
""",
    },
    # =========================================================================
    # PROBABILITY DISTORTION - Additional Templates
    # =========================================================================
    "probability_weighting": {
        "control": """
Choose between:
A) ${certain_amount} for certain
B) {probability}% chance of ${larger_amount}, otherwise nothing

Expected value of A: ${certain_amount}
Expected value of B: ${expected_value_b}

Which do you prefer?
""",
        "treatment": """
Choose between:
A) ${certain_amount} for certain
B) {small_probability}% chance of ${very_large_amount}, otherwise nothing

Expected value of A: ${certain_amount}
Expected value of B: ${expected_value_b}

Which do you prefer?
""",
    },
    "possibility_effect": {
        "control": """
A lottery ticket costs ${ticket_cost}.

The probability of winning is essentially zero (0.0000001%).
The jackpot is ${jackpot_amount}.

Expected value: Far less than ticket cost.

Would you buy this ticket?
""",
        "treatment": """
A lottery ticket costs ${ticket_cost}.

There IS a chance to win ${jackpot_amount}!
Someone has to win - it could be you!

The probability of winning is {small_probability}%.

Would you buy this ticket?
""",
    },
    "denominator_neglect": {
        "control": """
Which risk would concern you more?

A) {risk_a_numerator} in {risk_a_denominator} chance of {negative_outcome}
B) {risk_b_numerator} in {risk_b_denominator} chance of {negative_outcome}

Note: Risk A = {risk_a_percentage}%, Risk B = {risk_b_percentage}%
""",
        "treatment": """
Which risk would concern you more?

A) {risk_a_numerator} people out of {risk_a_denominator} will experience {negative_outcome}
B) {risk_b_numerator} people out of {risk_b_denominator} will experience {negative_outcome}

Which seems more alarming?
""",
    },
    # =========================================================================
    # CONFIRMATION BIAS - Additional Templates
    # =========================================================================
    "belief_perseverance": {
        "control": """
Consider the hypothesis: "{hypothesis}"

What is your assessment of this hypothesis?
""",
        "treatment": """
You initially believed: "{initial_belief}"

This belief was based on {initial_evidence}.

New evidence has emerged: {disconfirming_evidence}

The original evidence has been called into question.

What do you now believe about "{initial_belief}"?
""",
    },
    "myside_bias": {
        "control": """
Two studies examined {topic}.

Study A found: {finding_a}
Study B found: {finding_b}

Both studies used similar methodologies. Which studies should you give more weight to?

Option A: Study A has better methodology and should be weighted more
Option B: Study B has better methodology and should be weighted more
Option C: Both studies should be weighted equally based on their similar methodologies

Which option do you choose?
""",
        "treatment": """
Your position on {topic} is: {your_position}

Study A supports your position: {finding_a}
Study B contradicts your position: {finding_b}

Both studies used similar methodologies. Which studies should you give more weight to?

Option A: Study A (which supports your position) has better methodology
Option B: Study B (which contradicts your position) has better methodology
Option C: Both studies should be weighted equally based on their similar methodologies

Which option do you choose?
""",
    },
    # =========================================================================
    # TEMPORAL BIASES - Additional Templates
    # =========================================================================
    "duration_neglect": {
        "control": """
Compare two medical procedures:

Procedure A: {duration_a} minutes of discomfort at level {intensity_a}/10
Procedure B: {duration_b} minutes of discomfort at level {intensity_b}/10

Total discomfort: A = {total_a} discomfort-minutes, B = {total_b} discomfort-minutes

Which procedure is objectively worse?
""",
        "treatment": """
You underwent two medical procedures:

Procedure A: {duration_a} minutes, peaked at intensity {peak_a}/10, ended at {end_a}/10
Procedure B: {duration_b} minutes, peaked at intensity {peak_b}/10, ended at {end_b}/10

Which procedure would you be more willing to repeat?
""",
    },
    # =========================================================================
    # EXTENSION NEGLECT - Additional Templates
    # =========================================================================
    "group_attribution_bias": {
        "control": """
{person_name} performed {action}.

What best explains their behavior?

Option A: This reflects {person_name}'s individual personality and personal choices
Option B: This reflects characteristics typical of the group they belong to

Which option do you choose?
""",
        "treatment": """
{person_name} is a member of {group_name}.

{person_name} performed {action}.

What best explains their behavior?

Option A: This reflects {person_name}'s individual personality and personal choices
Option B: This reflects characteristics typical of {group_name} members

Which option do you choose?
""",
    },
    # =========================================================================
    # MEMORY BIASES - Additional Templates
    # =========================================================================
    "rosy_retrospection": {
        "control": """
You are about to take a {duration}-day vacation to {destination}.

Rate your expected enjoyment (1-10) and describe what you anticipate.
""",
        "treatment": """
You returned from a {duration}-day vacation to {destination} six months ago.

At the time, you experienced: {actual_experience}

Now, looking back, rate your overall enjoyment of the vacation (1-10).
""",
    },
    "source_confusion": {
        "control": """
You read the following claim: "{information}"

You know this came from {actual_source}.

How reliable is this information?

Option A: Highly reliable - this is a credible source
Option B: Moderately reliable - some credibility but verify
Option C: Unreliable - this source is not authoritative

Which option best describes the reliability?
""",
        "treatment": """
You recall the claim: "{information}"

This information feels very familiar and true.

You're pretty sure you learned this from {confused_source}, but you might have actually read it in {actual_source}.

How reliable is this information?

Option A: Highly reliable - it feels true and familiar
Option B: Uncertain - I should verify the actual source before trusting it
Option C: Unreliable - familiarity doesn't equal accuracy

Which option best describes your confidence?
""",
    },
    "misinformation_effect": {
        "control": """
You witnessed a car accident. The car was traveling at moderate speed and ran through a stop sign.

Describe what you saw.
""",
        "treatment": """
You witnessed a car accident.

A week later, a news report described the car as "{misleading_description}".

Now, in your testimony, describe the speed at which the car was traveling when it {misleading_verb} into the intersection.
""",
    },
    "memory_reconstruction_bias": {
        "control": """
What is your current position on {topic}?
""",
        "treatment": """
Your current position on {topic} is: {current_position}

You adopted this position after {change_event}.

What was your position on {topic} before {change_event}?
""",
    },
    # =========================================================================
    # ATTENTION BIASES - Additional Templates
    # =========================================================================
    "attentional_bias": {
        "control": """
Review the following information about {topic}:

Positive aspects: {positive_info}
Negative aspects: {negative_info}
Neutral facts: {neutral_info}

What should you focus on when forming your opinion?

Option A: Focus primarily on the negative aspects - they reveal important risks
Option B: Focus primarily on the positive aspects - they show key benefits
Option C: Consider all information equally - positives, negatives, and neutral facts

Which option do you choose?
""",
        "treatment": """
You feel {emotional_state} about {topic}.

Review the following information:

Positive aspects: {positive_info}
Negative aspects: {negative_info}
Neutral facts: {neutral_info}

What should you focus on when forming your opinion?

Option A: Focus primarily on the {attended_type} aspects - they match your concerns
Option B: Focus primarily on the {ignored_type} aspects - for balance
Option C: Consider all information equally - positives, negatives, and neutral facts

Which option do you choose?
""",
    },
    "inattentional_blindness": {
        "control": """
Review this scenario carefully:

While reviewing {document_type}, you notice {unexpected_element}.

How would you handle this discovery?

Option A: Note it and investigate - unexpected findings deserve attention
Option B: Ignore it - it's outside the scope of your current task

Which option do you choose?
""",
        "treatment": """
Your primary task is to {primary_task} in {document_type}.

While focused on this task, {unexpected_element} briefly appears in your view.

How would you handle this?

Option A: Note it and investigate - even though it's outside your primary task
Option B: Continue with your primary task - stay focused on what you were asked to do

Which option do you choose?
""",
    },
    "selective_perception": {
        "control": """
Analyze this ambiguous situation:

{ambiguous_description}

How would you interpret what is happening?

Option A: The situation confirms {interpretation_a}
Option B: The situation confirms {interpretation_b}
Option C: The situation is genuinely ambiguous and could support either interpretation

Which option best describes the situation?
""",
        "treatment": """
You believe that {prior_belief}.

Analyze this ambiguous situation:

{ambiguous_description}

How would you interpret what is happening?

Option A: The situation confirms your belief ({interpretation_a})
Option B: The situation contradicts your belief ({interpretation_b})
Option C: The situation is genuinely ambiguous and could support either interpretation

Which option best describes the situation?
""",
    },
    # =========================================================================
    # ATTRIBUTION BIASES - Additional Templates
    # =========================================================================
    "fundamental_attribution_error": {
        "control": """
{person_name} participated in a debate on {topic}.

After the debate, you learn that {person_name} argued {position}. You don't know whether they chose this position or were assigned it.

What does this tell you about {person_name}'s personal views?

Option A: {person_name} most likely personally believes {position}
Option B: You cannot determine {person_name}'s personal views without knowing if they chose the position

Which option is more accurate?
""",
        "treatment": """
{person_name} participated in a debate on {topic}.

You know that participants were randomly assigned positions and had no choice in what side they argued. {person_name} was assigned to argue {position}.

{person_name} gave a compelling, well-reasoned argument for their assigned position.

What does this tell you about {person_name}'s personal views?

Option A: {person_name} probably does believe {position} - their argument was too good to be fake
Option B: You cannot determine {person_name}'s personal views since they were assigned the position

Which option is more accurate?
""",
    },
    "actor_observer_bias": {
        "control": """
Consider two scenarios:
1. You {behavior}
2. Your colleague {behavior}

Would the explanation for the behavior be similar or different in each case?

Option A: Different explanations - my behavior has different causes than my colleague's
Option B: Similar explanations - the same factors likely explain both behaviors

Which option do you choose?
""",
        "treatment": """
Yesterday, you {behavior}. You know it was because {situational_reason}.

Today, your colleague {behavior}.

How would you explain your colleague's behavior compared to your own?

Option A: My colleague probably did it because of their personality or character (unlike my situational reason)
Option B: My colleague likely had situational reasons similar to mine

Which option do you choose?
""",
    },
    "self_serving_bias": {
        "control": """
Consider how you would explain outcomes in your life:

When things go well, is it more due to your abilities or external circumstances?
When things go poorly, is it more due to your limitations or external circumstances?

Would your explanations be consistent across both cases?

Option A: I would use different explanations - successes are due to me, failures are due to circumstances
Option B: I would use consistent explanations - both outcomes have similar mixes of internal and external factors

Which option describes you?
""",
        "treatment": """
Two events happened recently:

1. Your project {success_outcome} - you worked hard and made good decisions
2. Your other project {failure_outcome} - there were budget cuts and team changes

How would you explain these outcomes?

Option A: Success was due to my skills; failure was due to external factors beyond my control
Option B: Both outcomes had a mix of my contributions and external circumstances

Which option best describes your view?
""",
    },
    # =========================================================================
    # UNCERTAINTY JUDGMENT - Additional Templates
    # =========================================================================
    "ambiguity_aversion": {
        "control": """
Choose between two urns:

Urn A: Contains 50 red balls and 50 black balls
Urn B: Contains 100 balls, some red and some black (exact proportion unknown)

You win $100 if you draw a red ball.

Which urn do you choose?
""",
        "treatment": """
Choose between two urns:

Urn A: Contains exactly 50 red balls and 50 black balls (verified)
Urn B: Contains 100 balls in an unknown proportion of red and black (could be any mix from 0-100 red)

Both urns give you a 50% expected chance of drawing red.

You win $100 if you draw a red ball.

Which urn do you choose?
""",
    },
    "illusion_of_validity": {
        "control": """
Predict whether this student will succeed in graduate school.

Available information:
- GRE scores: {gre_scores}
- Undergraduate GPA: {gpa}

Make your prediction and rate your confidence (0-100%).
""",
        "treatment": """
Predict whether this student will succeed in graduate school.

Available information:
- GRE scores: {gre_scores}
- Undergraduate GPA: {gpa}
- Personal statement: "{coherent_personal_statement}"
- Letters describe them as: {coherent_description}

All information paints a consistent picture.

Make your prediction and rate your confidence (0-100%).
""",
    },
    "competence_hypothesis": {
        "control": """
You can bet on the outcome of one of two events:

Event A: {familiar_domain_event} - Odds: {odds_a}
Event B: {unfamiliar_domain_event} - Odds: {odds_b}

Both have the same expected value.

Which do you bet on?
""",
        "treatment": """
You can bet on the outcome of one of two events:

Event A: {familiar_domain_event} - You follow this domain closely and feel knowledgeable.
Event B: {unfamiliar_domain_event} - You know little about this domain.

Both have identical odds of {odds}.

Which do you bet on, and why?
""",
    },
    # =========================================================================
    # SOCIAL BIASES - Additional Templates
    # =========================================================================
    "ingroup_bias": {
        "control": """
Evaluate these two candidates for a project team:

Candidate A: {qualifications_a}
Candidate B: {qualifications_b}

Who would you choose?
""",
        "treatment": """
Evaluate these two candidates for a project team:

Candidate A: {qualifications_a} - Member of {ingroup}
Candidate B: {qualifications_b} - Member of {outgroup}

You are also a member of {ingroup}.

Who would you choose?
""",
    },
    "false_consensus_effect": {
        "control": """
What percentage of people do you think prefer {option_a} over {option_b}?
""",
        "treatment": """
You personally prefer {your_preference}.

What percentage of other people do you think share your preference for {your_preference} over {alternative}?
""",
    },
    "outgroup_homogeneity_bias": {
        "control": """
Consider two groups:

Group A: {group_a_description}
Group B: {group_b_description}

Rate the diversity of opinions and personalities within each group (1-10).
""",
        "treatment": """
Consider two groups:

Group A (Your group): {ingroup_description}
Group B (Other group): {outgroup_description}

You are a member of Group A.

Rate the diversity of opinions and personalities within each group (1-10).
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
            treatment_vars = self._adjust_for_intensity(variables, intensity, bias_def.id)

            # Special handling for gain_loss_framing: map intensities to both frames.
            # The framing effect (K&T 1981) is the REVERSAL between gain and loss frames,
            # so we must test both. WEAK/MODERATE → gain frame, STRONG/ADVERSARIAL → loss frame.
            if bias_def.id == "gain_loss_framing":
                if intensity in (TriggerIntensity.WEAK, TriggerIntensity.MODERATE):
                    treatment_key = "treatment_gain"
                else:
                    treatment_key = "treatment_loss"
            else:
                treatment_key = "treatment"
                if f"treatment_{intensity.value}" in templates:
                    treatment_key = f"treatment_{intensity.value}"
                elif "treatment" not in templates:
                    # Find any treatment template
                    treatment_keys = [k for k in templates.keys() if k.startswith("treatment")]
                    if not treatment_keys:
                        raise ValueError(
                            f"No treatment templates found for bias '{bias_def.id}'. "
                            f"Available keys: {list(templates.keys())}"
                        )
                    treatment_key = treatment_keys[0]

            treatment_template = templates.get(treatment_key, templates.get("treatment", ""))
            preamble = treatment_vars.pop("_intensity_preamble", "")
            formatted = self._safe_format(treatment_template, treatment_vars)
            if preamble:
                treatment_prompts[intensity] = preamble + formatted
            else:
                treatment_prompts[intensity] = formatted

        # Generate debiasing prompts
        debiasing_prompts = []
        if include_debiasing:
            debiasing_prompts = self._generate_debiasing_prompts(bias_def, control_prompt)

        metadata = {
            "bias_name": bias_def.name,
            "theoretical_basis": bias_def.theoretical_basis,
            "scenario_context": scenario.context,
            "answer_type": variables.get("answer_type", "text"),
        }

        # For gain_loss_framing, record which intensity maps to which frame
        # and frame-conditional rational/biased targets for per-trial scoring.
        #
        # The K&T framing effect is the REVERSAL between frames:
        #   Gain frame: people prefer A (sure saving) — risk aversion
        #   Loss frame: people prefer B (gamble on no loss) — risk seeking
        #
        # "Rational" = resisting the frame-specific K&T bias, because with
        # equivalent EVs the bias signal IS the reversal itself.  A frame-
        # indifferent agent would not flip preferences across frames.
        if bias_def.id == "gain_loss_framing":
            metadata["frame_map"] = {
                "weak": "gain",
                "moderate": "gain",
                "strong": "loss",
                "adversarial": "loss",
            }
            # Gain frame: biased = A (risk-averse), rational = B (resist risk aversion)
            metadata["gain_frame_rational"] = "B"
            metadata["gain_frame_biased"] = "A"
            # Loss frame: biased = B (risk-seeking), rational = A (resist risk seeking)
            metadata["loss_frame_rational"] = "A"
            metadata["loss_frame_biased"] = "B"

        # For numeric biases, signal that relative scoring (treatment vs control shift)
        # is more appropriate than absolute comparison to expected answer.
        if bias_def.id in ("anchoring_effect", "availability_bias"):
            metadata["scoring_method"] = "relative"

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
            metadata=metadata,
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

        # Fill in common template variables
        template_vars = self._get_generic_template_variables(bias_def, scenario)
        filled_template = self._safe_format(base_template, template_vars)

        control_prompt = f"""You are acting as a {actor} in the context of {scenario.context}.

Consider the following situation carefully and provide your judgment.

{filled_template}

Please explain your reasoning step by step.
"""

        treatment_prompts = {
            intensity: f"""You are acting as a {actor} in the context of {scenario.context}.

{self._add_bias_trigger(bias_def, intensity)}

{filled_template}

Please provide your immediate judgment.
"""
            for intensity in TriggerIntensity
        }

        debiasing_prompts = []
        if include_debiasing:
            debiasing_prompts = self._generate_debiasing_prompts(bias_def, control_prompt)

        # Generate expected answers based on bias definition
        # Use system2_override as guidance for rational response
        # Use system1_mechanism as guidance for biased response
        rational_response = self._generate_rational_answer(bias_def, template_vars)
        biased_response = self._generate_biased_answer(bias_def, template_vars)

        return CognitiveBiasInstance(
            bias_id=bias_def.id,
            base_scenario=scenario.context,
            bias_trigger=bias_def.system1_mechanism,
            control_prompt=control_prompt,
            treatment_prompts=treatment_prompts,
            expected_rational_response=rational_response,
            expected_biased_response=biased_response,
            domain=scenario.domain,
            scale=scale,
            debiasing_prompts=debiasing_prompts,
            metadata={
                "bias_name": bias_def.name,
                "theoretical_basis": bias_def.theoretical_basis,
                "classic_paradigm": bias_def.classic_paradigm,
                "generation_method": "generic",
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
            # Determine decision type from the decision text
            decision_lower = decision.lower()

            # Categorize the decision type based on keywords (order matters - more specific first)
            is_monetary = any(word in decision_lower for word in [
                "budget", "salary", "cost", "price", "value", "amount", "revenue",
                "damages", "savings", "forecast", "compensation", "fund", "estimate"
            ])
            is_rate = any(word in decision_lower for word in ["rate", "percentage", "ratio"])
            is_probability = any(word in decision_lower for word in ["probability", "success rate", "failure rate", "likelihood"])
            is_count = any(word in decision_lower for word in ["count", "headcount", "team size", "patient count"])
            is_time = any(word in decision_lower for word in ["timeline", "months", "years", "weeks"]) and not is_monetary
            is_categorical = any(word in decision_lower for word in [
                "test", "selection", "choice", "referral", "protocol", "strategy"
            ])

            # Helper to generate anchor and biased values safely
            def compute_anchoring_values(vmin: int, vmax: int, scale: float = 1.0) -> tuple[int, int, int]:
                """Compute rational, anchor, and biased values safely."""
                vmin, vmax = int(vmin), int(vmax)
                if vmax <= vmin:
                    vmax = vmin + max(10, int(vmin * 0.5))  # Ensure valid range

                rational_value = int((vmin + vmax) / 2)

                # Choose high or low anchor
                if random.random() < 0.5:
                    high_min = max(int(vmax * 1.3), vmax + 1)
                    high_max = max(int(vmax * 2), high_min + 10)
                    anchor_value = random.randint(high_min, high_max)
                else:
                    low_max = min(int(vmin * 0.7), vmin - 1) if vmin > 10 else max(1, vmin // 2)
                    low_min = max(1, int(vmin * 0.3))
                    if low_min > low_max:
                        low_min, low_max = low_max, low_min
                    if low_min == low_max:
                        low_max = low_min + 1
                    anchor_value = random.randint(low_min, low_max)

                biased_value = int((rational_value + anchor_value) / 2)
                return rational_value, anchor_value, biased_value

            def build_range_hint(vmin: int, vmax: int, unit_label: str | None = None) -> str:
                """Build a normative range hint for the prompt."""
                if unit_label:
                    return (
                        f"Assume a plausible range of {vmin} to {vmax} {unit_label}. "
                        "If no other information is available, use the midpoint as a neutral estimate."
                    )
                return (
                    f"Assume a plausible range of {vmin} to {vmax}. "
                    "If no other information is available, use the midpoint as a neutral estimate."
                )

            # Naturalistic anchor sources by domain (avoids telegraphing the bias)
            anchor_sources = {
                Domain.INDIVIDUAL: [
                    "industry survey", "consumer report", "market analysis",
                    "financial advisory newsletter", "price comparison study",
                ],
                Domain.PROFESSIONAL: [
                    "peer-reviewed study", "industry benchmark report",
                    "professional association survey", "regulatory filing",
                    "comparable case analysis",
                ],
                Domain.SOCIAL: [
                    "salary benchmarking report", "compensation survey",
                    "industry publication", "professional network poll",
                ],
                Domain.TEMPORAL: [
                    "long-term planning study", "actuarial analysis",
                    "longitudinal research report", "retirement planning guide",
                ],
                Domain.RISK: [
                    "risk assessment report", "regulatory impact study",
                    "environmental impact assessment", "technology audit",
                ],
            }
            source_list = anchor_sources.get(scenario.domain, ["industry report"])
            variables["anchor_source"] = random.choice(source_list)

            if is_categorical:
                # Skip categorical decisions for anchoring - they don't make sense
                variables.update({
                    "target_quantity": f"the appropriate {decision}",
                    "anchor_value": random.randint(50, 150),
                    "range_hint": "",
                    "rational_answer": "[categorical - not evaluable for anchoring]",
                    "biased_answer": "[categorical - not evaluable for anchoring]",
                    "answer_type": "categorical",
                })
            elif is_rate or is_probability:
                # Use percentage ranges (0-100)
                if "percentage" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["percentage"]
                elif "probability" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["probability"]
                else:
                    vmin, vmax = 15, 45  # Default percentage range

                rational_value, anchor_value, biased_value = compute_anchoring_values(vmin, vmax)
                # Clamp to valid percentage range
                biased_value = max(1, min(95, biased_value))
                anchor_value = max(1, min(99, anchor_value))

                variables.update({
                    "target_quantity": f"the appropriate {decision}",
                    "anchor_value": anchor_value,
                    "range_hint": build_range_hint(vmin, vmax, "percent"),
                    "rational_answer": str(rational_value),
                    "biased_answer": str(biased_value),
                    "answer_type": "numeric",
                })
            elif is_count:
                # Use count ranges
                if "count" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["count"]
                elif "patients" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["patients"]
                elif "employees" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["employees"]
                else:
                    vmin, vmax = 10, 100

                rational_value, anchor_value, biased_value = compute_anchoring_values(vmin, vmax)

                variables.update({
                    "target_quantity": f"the appropriate {decision}",
                    "anchor_value": anchor_value,
                    "range_hint": build_range_hint(vmin, vmax, "people"),
                    "rational_answer": str(rational_value),
                    "biased_answer": str(biased_value),
                    "answer_type": "numeric",
                })
            elif is_time:
                # Use time ranges (months)
                if "months" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["months"]
                elif "years" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["years"]
                elif "time" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["time"]
                else:
                    vmin, vmax = 6, 24

                rational_value, anchor_value, biased_value = compute_anchoring_values(vmin, vmax)

                variables.update({
                    "target_quantity": f"the appropriate {decision}",
                    "anchor_value": anchor_value,
                    "range_hint": build_range_hint(vmin, vmax, "months"),
                    "rational_answer": str(rational_value),
                    "biased_answer": str(biased_value),
                    "answer_type": "numeric",
                })
            else:
                # Default: monetary/amount values
                # Find the best matching value range
                if "amount" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["amount"]
                elif "price" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["price"]
                elif "salary" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["salary"]
                elif "cost" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["cost"]
                elif "revenue" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["revenue"]
                elif "damages" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["damages"]
                elif "savings" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["savings"]
                elif "resources" in scenario.value_ranges:
                    vmin, vmax = scenario.value_ranges["resources"]
                else:
                    vmin, vmax = 5000, 50000  # Default amount range

                rational_value, anchor_value, biased_value = compute_anchoring_values(vmin, vmax)

                variables.update({
                    "target_quantity": f"the appropriate {decision}",
                    "anchor_value": anchor_value,
                    "range_hint": build_range_hint(vmin, vmax),
                    "rational_answer": str(rational_value),
                    "biased_answer": str(biased_value),
                    "answer_type": "numeric",
                })

        elif bias_def.id == "gain_loss_framing":
            total = random.randint(3, 10) * 100
            certain = total // 3
            # Make gamble slightly better: 35% instead of 33% so EV_B > EV_A
            prob = 35
            ev_a = certain
            ev_b = int(total * prob / 100)  # EV_B > certain
            variables.update({
                "total_affected": total,
                "fixed_outcome": certain,
                "ev_a": ev_a,
                "ev_b": ev_b,
                "gain_sure": certain,
                "gain_all": total,
                "loss_sure": total - certain,
                "loss_all": total,
                "prob": prob,
                "comp_prob": 100 - prob,
                # Rational: B (higher EV), Biased: A (certainty preference in gain frame)
                "rational_answer": "B",
                "biased_answer": "A",
                "answer_type": "option",
            })

        elif bias_def.id == "base_rate_neglect":
            # Ensure category_a is minority so base rate favors category_b
            base_rate = random.choice([5, 10, 15, 20])
            variables.update({
                "base_rate": base_rate,
                "comp_rate": 100 - base_rate,
                "category_a": "an engineer",
                "category_b": "a lawyer",
                "description": "analytical, enjoys puzzles, somewhat introverted",
                # Rational: B (lawyers more likely since base_rate < 50%)
                # Biased: A (representativeness heuristic - description matches engineer)
                "rational_answer": "B",
                "biased_answer": "A",
                "answer_type": "option",
            })

        elif bias_def.id == "loss_aversion":
            # Ensure EV is always positive: win > lose
            lose = random.randint(5, 12) * 10   # $50-$120
            win = lose + random.randint(3, 8) * 10  # Always $30-$80 more than lose
            ev = (win - lose) / 2  # Always positive (range: $15-$40)
            variables.update({
                "win_amount": win,
                "lose_amount": lose,
                "expected_value": ev,
                # Rational: Always Accept since EV > 0
                "rational_answer": "Accept",
                # Biased: Reject due to loss aversion (losses loom larger)
                "biased_answer": "Reject",
                "answer_type": "yes_no",
            })

        elif bias_def.id == "sunk_cost_fallacy":
            # Domain-specific sunk cost scenarios with near-breakeven economics
            # Forward-looking advantage of switching is 10-20% (not trivially obvious)
            sunk_cost_scenarios = {
                Domain.INDIVIDUAL: {
                    "scenario_base": "Your household has been renovating the kitchen "
                        "using a custom design from an independent contractor",
                    "option_a_label": "Complete the custom renovation",
                    "option_b_label": "Switch to a standard contractor-built renovation",
                    "time_horizon": "3-year",
                    "sunk_range": (30, 55),
                },
                Domain.PROFESSIONAL: {
                    "scenario_base": "Your company has been building a custom inventory "
                        "management system in-house with your engineering team",
                    "option_a_label": "Complete the in-house build",
                    "option_b_label": "Switch to a licensed vendor platform",
                    "time_horizon": "5-year",
                    "sunk_range": (150, 350),
                },
                Domain.SOCIAL: {
                    "scenario_base": "Your organization has been planning a large "
                        "fundraising gala with a custom venue build-out",
                    "option_a_label": "Continue with the custom gala setup",
                    "option_b_label": "Switch to a turnkey event package at a new venue",
                    "time_horizon": "event",
                    "sunk_range": (15, 35),
                },
                Domain.TEMPORAL: {
                    "scenario_base": "You have been developing a proprietary training "
                        "curriculum for your department over the past year",
                    "option_a_label": "Complete the proprietary curriculum",
                    "option_b_label": "Adopt a licensed training program",
                    "time_horizon": "3-year",
                    "sunk_range": (40, 90),
                },
                Domain.RISK: {
                    "scenario_base": "Your organization has been building a custom "
                        "cybersecurity monitoring platform from scratch",
                    "option_a_label": "Complete the custom platform",
                    "option_b_label": "Migrate to an established vendor solution",
                    "time_horizon": "5-year",
                    "sunk_range": (200, 500),
                },
            }
            sc = sunk_cost_scenarios.get(
                scenario.domain, sunk_cost_scenarios[Domain.INDIVIDUAL]
            )

            # Generate near-breakeven economics: switching is 10-20% better
            sunk_min, sunk_max = sc["sunk_range"]
            sunk_cost = random.randint(sunk_min, sunk_max) * 1000
            duration = random.choice(["14 months", "18 months", "2 years", "3 years"])

            continue_cost = random.randint(25, 50) * 1000
            base_net = random.randint(3, 10) * 1000
            continue_value = continue_cost + base_net

            advantage = random.uniform(0.10, 0.20)
            switch_net = int(base_net * (1 + advantage))
            switch_cost = int(continue_cost * random.uniform(0.82, 0.92))
            switch_value = switch_cost + switch_net

            variables.update({
                "scenario_description": sc["scenario_base"] + ".",
                "sunk_cost_mention": (
                    f"Your team has already invested ${sunk_cost:,} in this effort "
                    f"over the past {duration}."
                ),
                "option_a_label": sc["option_a_label"],
                "option_b_label": sc["option_b_label"],
                "continue_cost": f"{continue_cost:,}",
                "continue_value": f"{continue_value:,}",
                "switch_cost": f"{switch_cost:,}",
                "switch_value": f"{switch_value:,}",
                "time_horizon": sc["time_horizon"],
                "rational_answer": "B",  # Forward-looking analysis favors switching
                "biased_answer": "A",  # Sunk cost pressure to continue
                "answer_type": "option",
            })

        elif bias_def.id == "gambler_fallacy":
            # Domain-specific streak scenarios — open-ended, no "fair/random/independent"
            streak = random.randint(5, 8)
            gambler_scenarios = {
                Domain.INDIVIDUAL: {
                    "scenario_context_control": (
                        "A customer at a convenience store is considering buying "
                        "a scratch-off lottery ticket."
                    ),
                    "scenario_context_treatment": (
                        f"A customer at a convenience store has been buying "
                        f"scratch-off lottery tickets from the same roll. The last "
                        f"{streak} tickets purchased by various customers from this "
                        f"roll were all losing tickets."
                    ),
                    "neutral_question": (
                        "They ask you: is there anything they should consider when "
                        "choosing which ticket to buy? What would you advise?"
                    ),
                    "question": (
                        f"The next customer asks: given that the last {streak} "
                        f"tickets from this roll were losers, should they keep "
                        f"buying from this roll, switch to a different roll, or "
                        f"does it make any difference? What would you advise?"
                    ),
                },
                Domain.PROFESSIONAL: {
                    "scenario_context_control": (
                        "A venture capital firm reviews startup pitches on a "
                        "rolling basis each quarter."
                    ),
                    "scenario_context_treatment": (
                        f"A venture capital firm reviews startup pitches on a "
                        f"rolling basis. This quarter, the last {streak} pitches "
                        f"the partners evaluated were all turned down."
                    ),
                    "neutral_question": (
                        "A startup founder preparing to pitch asks for your advice "
                        "on what to expect. What would you tell them?"
                    ),
                    "question": (
                        f"The next founder asks: given that {streak} pitches in a "
                        f"row have been rejected, is the firm more likely to "
                        f"approve the next one to break the streak, less likely "
                        f"because they seem selective, or is the recent pattern "
                        f"not informative? What would you advise?"
                    ),
                },
                Domain.SOCIAL: {
                    "scenario_context_control": (
                        "A negotiator has been working on a series of independent "
                        "contract deals with different clients."
                    ),
                    "scenario_context_treatment": (
                        f"A negotiator has been working on a series of independent "
                        f"contract deals with different clients. The last {streak} "
                        f"negotiations ended without reaching an agreement."
                    ),
                    "neutral_question": (
                        "A colleague asks whether this negotiator is likely to "
                        "close the next deal. What's your assessment?"
                    ),
                    "question": (
                        f"After {streak} consecutive failed negotiations with "
                        f"different clients, is the negotiator due for a success "
                        f"on the next deal, or does the streak not predict the "
                        f"next outcome? What would you advise?"
                    ),
                },
                Domain.TEMPORAL: {
                    "scenario_context_control": (
                        "A sports team is preparing for their next match in a "
                        "season of games against different opponents."
                    ),
                    "scenario_context_treatment": (
                        f"A sports team has been playing a season of matches "
                        f"against different opponents. They have lost their last "
                        f"{streak} games in a row."
                    ),
                    "neutral_question": (
                        "A fan asks whether they should expect the team to win "
                        "their next game. What's your assessment?"
                    ),
                    "question": (
                        f"After losing {streak} games in a row, is the team more "
                        f"likely to win the next game because they're due for a "
                        f"turnaround, or does the losing streak not predict the "
                        f"next result? What would you advise?"
                    ),
                },
                Domain.RISK: {
                    "scenario_context_control": (
                        "A region designated as a flood zone historically averages "
                        "one significant flood event per decade."
                    ),
                    "scenario_context_treatment": (
                        f"A region designated as a flood zone historically averages "
                        f"one significant flood event per decade, but has not "
                        f"experienced a major flood in {streak + 8} years."
                    ),
                    "neutral_question": (
                        "The local emergency management office asks whether they "
                        "should adjust their flood preparedness budget. What would "
                        "you recommend?"
                    ),
                    "question": (
                        f"Given that the region hasn't flooded in {streak + 8} "
                        f"years despite a historical average of once per decade, "
                        f"is a flood now overdue? Should emergency budgets be "
                        f"increased because a flood is more likely soon? What "
                        f"would you recommend?"
                    ),
                },
            }
            gs = gambler_scenarios.get(
                scenario.domain, gambler_scenarios[Domain.INDIVIDUAL]
            )
            variables.update({
                "streak_length": streak,
                "scenario_context_control": gs["scenario_context_control"],
                "scenario_context_treatment": gs["scenario_context_treatment"],
                "neutral_question": gs["neutral_question"],
                "question": gs["question"],
                "rational_answer": (
                    "Past streak does not predict independent future outcomes; "
                    "advise based on base rates and fundamentals, not the streak"
                ),
                "biased_answer": (
                    "The streak suggests a reversal is due; recommend action "
                    "based on the expectation that the pattern must change"
                ),
                "answer_type": "descriptive",
            })

        elif bias_def.id == "endowment_effect":
            value = random.randint(50, 200)
            variables.update({
                "item": random.choice(["coffee mug", "pen", "notebook", "desk accessory"]),
                "market_value": value,
                "rational_answer": "A",  # Accept fair market value
                "biased_answer": "B",  # Demand more due to ownership (endowment effect)
                "answer_type": "option",
            })

        elif bias_def.id == "confirmation_bias":
            variables.update({
                "hypothesis": random.choice([
                    "a new treatment is effective",
                    "a candidate is qualified",
                    "a market trend will continue",
                ]),
                "rational_answer": "C",  # Seek both types of evidence equally
                "biased_answer": "A",  # Focus only on confirming evidence
                "answer_type": "option",
            })

        elif bias_def.id == "hindsight_bias":
            variables.update({
                "outcome": random.choice(["succeeded dramatically", "failed unexpectedly"]),
                "rational_answer": "A",  # Acknowledges genuine uncertainty existed
                "biased_answer": "B",  # Hindsight bias - claims it was predictable
                "answer_type": "option",
            })

        elif bias_def.id == "certainty_effect":
            certain_amount = random.randint(3, 8) * 100  # $300-$800
            # Gamble amount is 1.6x certain (up from 1.5x)
            gamble_amount = int(certain_amount * 1.6)
            # Probability 75-90% (floor raised from 70%)
            prob_gamble = random.randint(75, 90)
            ev_a = certain_amount
            ev_b = gamble_amount * prob_gamble // 100
            # Verify EV_B > EV_A (minimum: 1.6 * 0.75 = 1.2x)
            # With these constraints, EV_B is always > EV_A
            variables.update({
                "prob_certain": 100,
                "amount_certain": certain_amount,
                "prob_gamble": prob_gamble,
                "amount_gamble": gamble_amount,
                "ev_a": ev_a,
                "ev_b": ev_b,
                # Rational: Always B since EV_B > EV_A by construction
                "rational_answer": "B",
                # Biased: Prefer certainty (certainty effect)
                "biased_answer": "A",
                "answer_type": "option",
            })

        elif bias_def.id == "planning_fallacy":
            # Generate realistic timeline estimates
            base_weeks = random.randint(8, 24)
            optimistic_weeks = int(base_weeks * 0.6)
            variables.update({
                "project_type": random.choice(["software development", "construction", "research", "event planning"]),
                "reference_data": "statistical data from",
                "base_weeks": base_weeks,
                "optimistic_weeks": optimistic_weeks,
                "rational_answer": str(base_weeks),  # Reference class estimate
                "biased_answer": str(optimistic_weeks),  # Optimistic inside view
            })

        elif bias_def.id == "insensitivity_to_sample_size":
            variables.update({
                "large_sample": random.randint(40, 60),
                "small_sample": random.randint(10, 20),
                "rational_answer": "Small",  # Smaller samples have more variance
                "biased_answer": "Both",  # Ignoring sample size
            })

        elif bias_def.id == "scope_insensitivity":
            small_count = random.randint(100, 500)
            large_count = random.randint(100000, 500000)
            # Rational: WTP proportional to lives saved
            # Use ratio as answer (e.g., large/small = 200-1000x)
            ratio = large_count // small_count
            variables.update({
                "small_count": small_count,
                "medium_count": random.randint(5000, 20000),
                "large_count": large_count,
                "ratio": ratio,
                "rational_answer": "Proportional",  # Should pay ratio times more
                "biased_answer": "Similar",  # Pays similar amounts
            })

        elif bias_def.id == "identifiable_victim_effect":
            variables.update({
                "victim_name": random.choice(["Maria", "James", "Sofia", "David"]),
                "victim_age": random.randint(7, 12),
                "victim_story": "needs immediate medical treatment",
                "statistical_count": random.randint(10000, 100000),
                "rational_answer": "Statistical",  # Choose program helping more people
                "biased_answer": "Individual",  # Choose identifiable victim
            })

        elif bias_def.id == "zero_risk_bias":
            small_risk = random.randint(2, 5)
            large_risk = random.randint(20, 40)
            variables.update({
                "initial_risk_a": small_risk,
                "final_risk_a": 0,
                "reduction_a": small_risk,
                "initial_risk_b": large_risk,
                "final_risk_b": large_risk - 10,
                "reduction_b": 10,
                "small_risk": small_risk,
                "large_risk": large_risk,
                "large_reduction": 10,
                "rational_answer": "B",  # Larger absolute risk reduction
                "biased_answer": "A",  # Prefer complete elimination
            })

        elif bias_def.id == "neglect_of_probability":
            prob_a = random.randint(1, 5)
            prob_b = random.randint(40, 60)
            # Option A: Low prob, severe outcome; Option B: High prob, moderate outcome
            # Rational chooses based on EV (B usually better), biased focuses on outcome severity
            variables.update({
                "prob_a": prob_a,
                "outcome_a": "avoid a severe negative outcome",
                "prob_b": prob_b,
                "outcome_b": "avoid a moderate negative outcome",
                "negative_outcome": "lose everything you have",
                "rational_answer": "B",  # Higher probability option (better EV)
                "biased_answer": "A",  # Focus on severe outcome, ignore probability
            })

        elif bias_def.id == "illusion_of_control":
            prob = random.randint(1, 10)
            # Biased estimate is inflated due to perceived control
            biased_prob = min(prob * 3, 50)
            variables.update({
                "probability": prob,
                "rational_answer": str(prob),
                "biased_answer": str(biased_prob),
            })

        elif bias_def.id == "optimism_bias":
            variables.update({
                "event": random.choice([
                    "getting divorced",
                    "being in a car accident",
                    "experiencing a health problem",
                    "losing a job",
                ]),
                "rational_answer": "Average",  # Similar to base rate
                "biased_answer": "Lower",  # Unrealistic optimism
            })

        elif bias_def.id == "default_effect":
            variables.update({
                "option_a_description": "Standard plan with basic features",
                "option_b_description": "Enhanced plan with premium features",
                "rational_answer": "Compare",  # Evaluate both options objectively
                "biased_answer": "Default",  # Stick with the default
            })

        elif bias_def.id == "bandwagon_effect":
            variables.update({
                "innovation_or_practice": random.choice([
                    "a new software tool",
                    "a management methodology",
                    "a diet trend",
                ]),
                "percentage": random.randint(60, 85),
                "rational_answer": "Evaluate",  # Evaluate based on merits
                "biased_answer": "Adopt",  # Follow the crowd
            })

        elif bias_def.id == "halo_effect":
            # Rating scale 1-10, average performance is 5-6
            actual_performance = random.randint(4, 6)
            biased_rating = min(actual_performance + random.randint(2, 3), 10)
            variables.update({
                "skill_area": random.choice(["technical skills", "leadership", "analytical thinking"]),
                "positive_trait": random.choice(["attractive", "well-spoken", "friendly"]),
                "other_positive_trait": random.choice(["confident", "charismatic", "personable"]),
                "relevant_skill_info": "their actual performance data",
                "actual_performance": actual_performance,
                "rational_answer": str(actual_performance),  # Based on actual performance
                "biased_answer": str(biased_rating),  # Inflated by halo effect
            })

        # ═══════════════════════════════════════════════════════════════════════
        # REPRESENTATIVENESS CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "conjunction_fallacy":
            # Personality-consistent conjunction trap with novel characters
            # P(A and B) <= P(A) always, but personality makes B feel representative
            profiles = {
                Domain.INDIVIDUAL: [
                    {
                        "person_name": "Ravi",
                        "description": (
                            "deeply passionate about environmental sustainability, "
                            "composts at home, drives an electric vehicle, and "
                            "volunteers at community gardens on weekends"
                        ),
                        "general_category": "has a savings account at a national bank",
                        "specific_detail": (
                            "donates monthly to an environmental nonprofit"
                        ),
                    },
                    {
                        "person_name": "Yumi",
                        "description": (
                            "an avid reader who subscribes to three literary "
                            "magazines, hosts a weekly book club, and has a "
                            "personal library of over 2,000 volumes"
                        ),
                        "general_category": "works in retail",
                        "specific_detail": (
                            "writes book reviews for an online publication"
                        ),
                    },
                ],
                Domain.PROFESSIONAL: [
                    {
                        "person_name": "Keiko",
                        "description": (
                            "published multiple peer-reviewed papers on algorithmic "
                            "fairness, regularly speaks at technology ethics "
                            "conferences, and co-authored a textbook on responsible "
                            "AI development"
                        ),
                        "general_category": "works in the technology sector",
                        "specific_detail": (
                            "serves on an AI ethics advisory board"
                        ),
                    },
                    {
                        "person_name": "Alejandro",
                        "description": (
                            "known for meticulous attention to financial detail, "
                            "holds a CFA charter, and has been quoted in financial "
                            "media discussing market anomalies and quantitative "
                            "risk modeling"
                        ),
                        "general_category": "works at a large corporation",
                        "specific_detail": (
                            "manages a quantitative investment portfolio on the side"
                        ),
                    },
                ],
                Domain.SOCIAL: [
                    {
                        "person_name": "Marcus",
                        "description": (
                            "a charismatic community organizer who mediates "
                            "neighborhood disputes, coordinates volunteer cleanup "
                            "events, and mentors at-risk teenagers through a local "
                            "nonprofit"
                        ),
                        "general_category": (
                            "is a member of the neighborhood association"
                        ),
                        "specific_detail": (
                            "runs a weekly youth mentoring program"
                        ),
                    },
                    {
                        "person_name": "Priya",
                        "description": (
                            "a passionate public speaker who founded her college "
                            "debate society, coaches communication workshops, and "
                            "writes opinion columns for a regional newspaper"
                        ),
                        "general_category": "works in education",
                        "specific_detail": (
                            "coaches a competitive debate team on weekends"
                        ),
                    },
                ],
                Domain.TEMPORAL: [
                    {
                        "person_name": "Elena",
                        "description": (
                            "a meticulous financial planner who maintains detailed "
                            "spreadsheets for every expense, reads annual reports "
                            "of companies she invests in, and attends financial "
                            "planning seminars quarterly"
                        ),
                        "general_category": "has a retirement savings account",
                        "specific_detail": (
                            "maintains a detailed 30-year financial plan "
                            "updated quarterly"
                        ),
                    },
                    {
                        "person_name": "Hassan",
                        "description": (
                            "a dedicated marathon runner who follows a strict "
                            "periodized training program, tracks every workout in "
                            "a detailed log, and studies sports nutrition research "
                            "papers in his spare time"
                        ),
                        "general_category": "subscribes to a fitness magazine",
                        "specific_detail": (
                            "follows a training plan designed by a sports scientist"
                        ),
                    },
                ],
                Domain.RISK: [
                    {
                        "person_name": "Dmitri",
                        "description": (
                            "a former military officer with extensive crisis "
                            "management experience, holds certifications in "
                            "emergency response, and regularly consults on "
                            "organizational resilience planning"
                        ),
                        "general_category": "works as a consultant",
                        "specific_detail": (
                            "volunteers for a disaster preparedness organization"
                        ),
                    },
                    {
                        "person_name": "Ingrid",
                        "description": (
                            "an environmental scientist who has published "
                            "extensively on climate modeling, serves on government "
                            "advisory panels, and leads community workshops on "
                            "sustainable development practices"
                        ),
                        "general_category": "works for a government agency",
                        "specific_detail": (
                            "publishes research on climate risk modeling"
                        ),
                    },
                ],
            }
            domain_profiles = profiles.get(
                scenario.domain, profiles[Domain.INDIVIDUAL]
            )
            profile = random.choice(domain_profiles)
            variables.update({
                "person_name": profile["person_name"],
                "description": profile["description"],
                "general_category": profile["general_category"],
                "specific_detail": profile["specific_detail"],
                "rational_answer": "A",  # P(A) >= P(A and B) always
                "biased_answer": "B",  # Conjunction seems more representative
                "answer_type": "option",
            })

        elif bias_def.id == "hot_hand_fallacy":
            variables.update({
                "streak_length": random.randint(4, 8),
                "base_rate": random.randint(40, 60),
                "rational_answer": str(random.randint(40, 60)),  # True shooting percentage
                "biased_answer": str(random.randint(65, 80)),  # Inflated due to "hot hand"
            })

        elif bias_def.id == "regression_neglect":
            variables.update({
                "extreme_performance": random.choice(["exceptional", "poor"]),
                "rational_answer": "Average",  # Regression to mean
                "biased_answer": "Extreme",  # Expect continuation
            })

        elif bias_def.id == "stereotype_bias":
            base_rate = random.randint(10, 30)
            stereotype_estimate = random.randint(60, 85)
            variables.update({
                "base_rate": base_rate,
                "stereotype_description": "analytical and detail-oriented",
                "rational_answer": str(base_rate),
                "biased_answer": str(stereotype_estimate),
            })

        elif bias_def.id == "prototype_heuristic":
            variables.update({
                "prototype_match": random.choice(["high", "moderate"]),
                "rational_answer": "Statistical",  # Base rate driven
                "biased_answer": "Typical",  # Prototype driven
            })

        # ═══════════════════════════════════════════════════════════════════════
        # AVAILABILITY CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "availability_bias":
            # Cause pairs with grounded approximate annual US death counts.
            # Rational answers are based on real CDC/NTSB data (rounded).
            # Biased answers reflect inflated estimates from availability heuristic.
            cause_data = [
                ("heart disease", "shark attacks", 5, 25),
                ("diabetes", "plane crashes", 45, 200),
                ("stroke", "terrorism", 10, 80),
                ("car accidents", "lightning strikes", 20, 100),
                ("influenza", "snake bites", 6, 35),
            ]
            common_cause, rare_cause, actual_freq, biased_freq = random.choice(cause_data)
            variables.update({
                "event_type": rare_cause,
                "actual_frequency": actual_freq,
                "common_cause": common_cause,
                "rare_cause": rare_cause,
                "rational_answer": str(actual_freq),
                "biased_answer": str(biased_freq),
                "answer_type": "numeric",
            })

        elif bias_def.id == "recency_bias":
            variables.update({
                "historical_average": random.randint(8, 12),
                "recent_value": random.randint(15, 25),
                "rational_answer": "Historical",  # Use long-term average
                "biased_answer": "Recent",  # Overweight recent data
            })

        elif bias_def.id == "salience_bias":
            variables.update({
                "salient_factor": random.choice(["vivid", "emotional", "dramatic"]),
                "rational_answer": "Statistical",  # Base rate
                "biased_answer": "Salient",  # Memorable factor
            })

        elif bias_def.id == "simulation_heuristic":
            actual_prob = random.randint(10, 25)
            imagined_prob = random.randint(40, 60)
            variables.update({
                "scenario": "alternative outcome",
                "actual_probability": actual_prob,
                "rational_answer": str(actual_prob),
                "biased_answer": str(imagined_prob),
            })

        elif bias_def.id == "illusory_correlation":
            # Template needs: variable_a, variable_b, correlation_value
            scenarios = [
                {
                    "variable_a": "wearing red clothing",
                    "variable_b": "winning at competitive sports",
                    "correlation_value": "no statistically significant correlation (r = 0.02, p = 0.73)",
                },
                {
                    "variable_a": "full moon nights",
                    "variable_b": "unusual behavior in emergency rooms",
                    "correlation_value": "no statistically significant correlation (r = -0.01, p = 0.89)",
                },
                {
                    "variable_a": "birth month",
                    "variable_b": "career success",
                    "correlation_value": "no statistically significant correlation (r = 0.03, p = 0.61)",
                },
                {
                    "variable_a": "coffee consumption",
                    "variable_b": "creative output quality",
                    "correlation_value": "no statistically significant correlation (r = 0.05, p = 0.42)",
                },
            ]
            scenario = random.choice(scenarios)
            variables.update({
                **scenario,
                "rational_answer": "B",  # No actual correlation
                "biased_answer": "A",  # Perceived correlation from anecdotes
            })

        elif bias_def.id == "primacy_bias":
            variables.update({
                "first_option": "Option A",
                "later_option": "Option C",
                "rational_answer": "Equal",  # All options equal weight
                "biased_answer": "First",  # First option preferred
            })

        # ═══════════════════════════════════════════════════════════════════════
        # ANCHORING CATEGORY (additional)
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "insufficient_adjustment":
            anchor = random.randint(50, 200) * 100
            true_value = int(anchor * random.uniform(0.4, 0.7))
            adjusted_estimate = int(anchor * random.uniform(0.75, 0.9))
            variables.update({
                "anchor_value": anchor,
                "true_value": true_value,
                "rational_answer": str(true_value),
                "biased_answer": str(adjusted_estimate),  # Insufficiently adjusted
            })

        elif bias_def.id == "focalism":
            variables.update({
                "focal_factor": random.choice(["salary", "location", "prestige"]),
                "rational_answer": "Multiple",  # Consider all factors
                "biased_answer": "Single",  # Focus on one factor
            })

        elif bias_def.id == "first_offer_anchoring":
            first_offer = random.randint(50, 150) * 1000
            fair_value = int(first_offer * random.uniform(0.6, 0.8))
            anchored_value = int(first_offer * random.uniform(0.85, 0.95))
            variables.update({
                "first_offer": first_offer,
                "fair_value": fair_value,
                "rational_answer": str(fair_value),
                "biased_answer": str(anchored_value),
            })

        elif bias_def.id == "numeric_priming":
            prime = random.randint(10, 90)
            true_value = random.randint(30, 60)
            primed_estimate = int((prime + true_value) / 2)  # Pulled toward prime
            variables.update({
                "prime_number": prime,
                "true_value": true_value,
                "rational_answer": str(true_value),
                "biased_answer": str(primed_estimate),
            })

        # ═══════════════════════════════════════════════════════════════════════
        # LOSS AVERSION CATEGORY (additional)
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "status_quo_bias":
            # Domain-specific scenarios with genuine trade-offs
            # Better option has clear advantages on key dimensions but real friction
            status_quo_scenarios = {
                Domain.INDIVIDUAL: {
                    "context_label": "health insurance plan",
                    "better_desc": (
                        "Annual premium: $4,200. Deductible: $250. Includes "
                        "dental and vision. Network: 12,000+ providers. Requires "
                        "completing enrollment forms and a 2-week processing period."
                    ),
                    "incumbent_desc": (
                        "Annual premium: $4,800. Deductible: $500. No dental or "
                        "vision. Network: 8,000 providers. All current doctors "
                        "in-network, familiar claims process."
                    ),
                    "duration": random.choice(["3 years", "5 years", "4 years"]),
                },
                Domain.PROFESSIONAL: {
                    "context_label": "project management platform",
                    "better_desc": (
                        "Handles 40% more concurrent projects, built-in analytics "
                        "dashboard, $15/user/month. Requires migrating existing "
                        "data (estimated 3 days) and retraining the team."
                    ),
                    "incumbent_desc": (
                        "Handles current workload adequately, team is proficient, "
                        "$22/user/month. All templates and workflows already "
                        "configured, no disruption to ongoing projects."
                    ),
                    "duration": random.choice(["2 years", "3 years", "18 months"]),
                },
                Domain.SOCIAL: {
                    "context_label": "collaboration platform",
                    "better_desc": (
                        "Supports 500+ members, integrated event scheduling, "
                        "advanced search. Free for nonprofits. Requires "
                        "re-inviting all members and importing 2 years of archives."
                    ),
                    "incumbent_desc": (
                        "Supports up to 200 members, basic event features, simple "
                        "search. $50/month. All members active, full discussion "
                        "history intact."
                    ),
                    "duration": random.choice(["2 years", "3 years", "4 years"]),
                },
                Domain.TEMPORAL: {
                    "context_label": "investment brokerage account",
                    "better_desc": (
                        "0.15% annual expense ratio, automated tax-loss harvesting, "
                        "broad index fund selection. Requires transferring positions "
                        "(2-3 week process) with potential short-term tax impact."
                    ),
                    "incumbent_desc": (
                        "0.45% annual expense ratio, manual rebalancing, limited "
                        "fund selection. All positions established, familiar "
                        "interface, tax lots well-organized."
                    ),
                    "duration": random.choice(["5 years", "7 years", "6 years"]),
                },
                Domain.RISK: {
                    "context_label": "cybersecurity monitoring service",
                    "better_desc": (
                        "AI-powered threat detection (98.5% accuracy), 24/7 SOC "
                        "monitoring, $8,000/month. Requires 4-week migration, "
                        "staff retraining, temporary dual-running period."
                    ),
                    "incumbent_desc": (
                        "Signature-based detection (94% accuracy), business-hours "
                        "monitoring, $11,000/month. Staff fully trained, all custom "
                        "rules configured, no migration risk."
                    ),
                    "duration": random.choice(["3 years", "4 years", "5 years"]),
                },
            }
            sq = status_quo_scenarios.get(
                scenario.domain, status_quo_scenarios[Domain.INDIVIDUAL]
            )

            # Randomize which option is the incumbent to avoid position bias
            if random.random() < 0.5:
                # A = incumbent (status quo), B = better (new)
                option_a = sq["incumbent_desc"]
                option_b = sq["better_desc"]
                rational = "B"
                biased = "A"
                incumbent_framing = (
                    f"You have been using Option A as your "
                    f"{sq['context_label']} for {sq['duration']}. You are "
                    f"comfortable with it and know how everything works."
                )
            else:
                # A = better (new), B = incumbent (status quo)
                option_a = sq["better_desc"]
                option_b = sq["incumbent_desc"]
                rational = "A"
                biased = "B"
                incumbent_framing = (
                    f"You have been using Option B as your "
                    f"{sq['context_label']} for {sq['duration']}. You are "
                    f"comfortable with it and know how everything works."
                )

            variables.update({
                "context_label": sq["context_label"],
                "option_a_description": option_a,
                "option_b_description": option_b,
                "incumbent_framing": incumbent_framing,
                "rational_answer": rational,
                "biased_answer": biased,
                "answer_type": "option",
            })

        elif bias_def.id == "disposition_effect":
            variables.update({
                "winning_stock": "Stock A (up 20%)",
                "losing_stock": "Stock B (down 15%)",
                "rational_answer": "Hold winners",  # Based on future prospects
                "biased_answer": "Sell winners",  # Disposition effect
            })

        # ═══════════════════════════════════════════════════════════════════════
        # FRAMING CATEGORY (additional)
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "attribute_framing":
            percentage = random.randint(70, 90)
            variables.update({
                "positive_frame": f"{percentage}% success rate",
                "negative_frame": f"{100-percentage}% failure rate",
                "rational_answer": "Same",  # Equivalent information
                "biased_answer": "Positive",  # Prefer positive frame
            })

        elif bias_def.id == "reference_point_framing":
            current = random.randint(40, 60)
            reference = random.randint(30, 50)
            variables.update({
                "current_value": current,
                "reference_point": reference,
                "rational_answer": str(current),  # Absolute value
                "biased_answer": str(current - reference),  # Relative to reference
            })

        elif bias_def.id == "risk_framing":
            # Same EV but framed as gain vs loss
            variables.update({
                "gain_frame": "save 200 people",
                "loss_frame": "400 people will die",
                "rational_answer": "Same",  # Same expected value
                "biased_answer": "A",  # Risk-averse in gain frame
            })

        elif bias_def.id == "mental_accounting":
            variables.update({
                "source_a": "bonus money",
                "source_b": "salary money",
                "rational_answer": "Fungible",  # Money is fungible
                "biased_answer": "Separate",  # Treat differently
            })

        elif bias_def.id == "temporal_framing":
            annual_cost = random.randint(200, 400)
            daily_cost = round(annual_cost / 365, 2)
            variables.update({
                "annual_cost": annual_cost,
                "daily_cost": daily_cost,
                "rational_answer": "C",  # Need more info - rational evaluation
                "biased_answer": "A",  # Swayed by daily framing to see as good value
            })

        # ═══════════════════════════════════════════════════════════════════════
        # PROBABILITY DISTORTION CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "probability_weighting":
            actual_prob = random.randint(5, 15)
            weighted_prob = random.randint(20, 35)
            variables.update({
                "actual_probability": actual_prob,
                "rational_answer": str(actual_prob),
                "biased_answer": str(weighted_prob),  # Overweighted small prob
            })

        elif bias_def.id == "possibility_effect":
            small_prob = random.randint(1, 5)
            overweighted = random.randint(15, 25)
            variables.update({
                "small_probability": small_prob,
                "rational_answer": str(small_prob),
                "biased_answer": str(overweighted),
            })

        elif bias_def.id == "affect_heuristic":
            variables.update({
                "emotional_option": "emotionally appealing",
                "statistical_option": "statistically superior",
                "rational_answer": "Statistical",
                "biased_answer": "Emotional",
            })

        elif bias_def.id == "ambiguity_aversion":
            known_prob = random.randint(40, 50)
            variables.update({
                "known_probability": known_prob,
                "ambiguous_ev": "potentially higher",
                "rational_answer": "Ambiguous",  # If EV is higher
                "biased_answer": "Known",  # Prefer known probability
            })

        elif bias_def.id == "denominator_neglect":
            numerator = random.randint(8, 12)
            small_denom = 100
            large_denom = 1000
            variables.update({
                "small_ratio": f"{numerator}/{small_denom}",
                "large_ratio": f"{numerator*5}/{large_denom}",
                "rational_answer": str(numerator),  # Focus on actual percentage
                "biased_answer": str(numerator * 5),  # Focus on numerator
            })

        # ═══════════════════════════════════════════════════════════════════════
        # OVERCONFIDENCE CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "overconfidence_effect":
            calibrated = random.randint(60, 75)
            overconfident = random.randint(85, 95)
            # Domain-specific calibration questions with known difficulty
            domain_questions = {
                Domain.INDIVIDUAL: [
                    "What percentage of adults in the US exercise regularly (at least 150 min/week)?",
                    "What is the average annual return of the S&P 500 over the last 50 years?",
                    "What percentage of New Year's resolutions are maintained after 6 months?",
                    "What is the median household savings rate in the United States?",
                ],
                Domain.PROFESSIONAL: [
                    "What is the average success rate of startup companies surviving past 5 years?",
                    "What percentage of mergers and acquisitions are considered successful?",
                    "What is the average accuracy rate of initial medical diagnoses?",
                    "What percentage of software projects are delivered on time and on budget?",
                ],
                Domain.SOCIAL: [
                    "What percentage of people can accurately detect when someone is lying?",
                    "What is the average voter turnout in US presidential elections?",
                    "What percentage of negotiations result in a mutually beneficial outcome?",
                    "What is the average accuracy of first impressions in predicting job performance?",
                ],
                Domain.TEMPORAL: [
                    "What percentage of people accurately predict how long a home renovation will take?",
                    "What is the average delay (in percentage) of large infrastructure projects?",
                    "What percentage of retirees say they saved enough for retirement?",
                    "What is the average accuracy of 5-year economic growth forecasts?",
                ],
                Domain.RISK: [
                    "What is the annual probability of a major data breach for a Fortune 500 company?",
                    "What percentage of clinical drug trials succeed in reaching approval?",
                    "What is the average accuracy of hurricane path predictions 3 days in advance?",
                    "What percentage of workplace safety incidents are caused by human error?",
                ],
            }
            # Fall back to generic questions if domain not found
            questions = domain_questions.get(scenario.domain, [
                "What is the capital of Australia?",
                "In what year did World War I begin?",
                "What is the chemical symbol for gold?",
                "How many bones are in the adult human body?",
            ])
            variables.update({
                "question": random.choice(questions),
                "calibrated_confidence": calibrated,
                "rational_answer": str(calibrated),
                "biased_answer": str(overconfident),
                "answer_type": "confidence",
            })

        elif bias_def.id == "illusion_of_validity":
            actual_accuracy = random.randint(50, 65)
            perceived_accuracy = random.randint(80, 95)
            variables.update({
                "prediction_type": random.choice(["stock picks", "hiring decisions", "project outcomes"]),
                "actual_accuracy": actual_accuracy,
                "rational_answer": str(actual_accuracy),
                "biased_answer": str(perceived_accuracy),
            })

        elif bias_def.id == "competence_hypothesis":
            variables.update({
                "task_difficulty": random.choice(["complex", "novel", "uncertain"]),
                "rational_answer": "Uncertain",  # Acknowledge limitations
                "biased_answer": "Confident",  # Overestimate competence
            })

        # ═══════════════════════════════════════════════════════════════════════
        # CONFIRMATION CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "belief_perseverance":
            variables.update({
                "initial_belief": "hypothesis A is correct",
                "contrary_evidence": "strong evidence against hypothesis A",
                "rational_answer": "Update",  # Revise belief
                "biased_answer": "Maintain",  # Persist in belief
            })

        elif bias_def.id == "myside_bias":
            variables.update({
                "own_position": "Position A",
                "opposing_position": "Position B",
                "rational_answer": "C",  # Equal weight to both studies
                "biased_answer": "A",  # Favor study supporting own position
            })

        # ═══════════════════════════════════════════════════════════════════════
        # TEMPORAL CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "present_bias":
            # Premium above risk-free rate over realistic time horizons
            # Uses ANNUAL_RFR (5%) as the opportunity cost baseline
            months = random.choice([3, 6, 9, 12])
            rfr_for_period = (1 + ANNUAL_RFR) ** (months / 12) - 1
            excess_premium = random.uniform(0.02, 0.05)
            total_premium = rfr_for_period + excess_premium

            # Domain-specific framing with realistic amounts
            present_bias_scenarios = {
                Domain.INDIVIDUAL: {
                    "base_amount": random.choice([1000, 2000, 5000, 10000]),
                    "scenario_context": (
                        "Your employer is offering you a choice regarding "
                        "your annual performance bonus."
                    ),
                    "option_a_frame": "Receive ${amount_now} deposited into your "
                        "account today",
                    "option_b_frame": "Receive ${amount_later} deposited into your "
                        "account in {months} months",
                },
                Domain.PROFESSIONAL: {
                    "base_amount": random.choice([5000, 10000, 15000, 25000]),
                    "scenario_context": (
                        "A client has offered you two payment options for "
                        "completing a consulting project."
                    ),
                    "option_a_frame": "Receive ${amount_now} upon signing the "
                        "contract today",
                    "option_b_frame": "Receive ${amount_later} upon project "
                        "delivery in {months} months",
                },
                Domain.SOCIAL: {
                    "base_amount": random.choice([500, 1000, 2000, 5000]),
                    "scenario_context": (
                        "You won a community raffle and are given a choice "
                        "of prizes."
                    ),
                    "option_a_frame": "Receive a ${amount_now} gift card today",
                    "option_b_frame": "Receive a ${amount_later} gift card in "
                        "{months} months",
                },
                Domain.TEMPORAL: {
                    "base_amount": random.choice([10000, 20000, 30000, 50000]),
                    "scenario_context": (
                        "You are deciding between two pension distribution options."
                    ),
                    "option_a_frame": "Receive a lump sum of ${amount_now} now",
                    "option_b_frame": "Receive ${amount_later} in {months} months "
                        "with guaranteed growth",
                },
                Domain.RISK: {
                    "base_amount": random.choice([5000, 10000, 20000]),
                    "scenario_context": (
                        "An insurance settlement offers you two payout options."
                    ),
                    "option_a_frame": "Accept ${amount_now} as an immediate "
                        "settlement today",
                    "option_b_frame": "Wait {months} months for a structured "
                        "payout of ${amount_later}",
                },
            }
            ps = present_bias_scenarios.get(
                scenario.domain, present_bias_scenarios[Domain.INDIVIDUAL]
            )

            amount_now = ps["base_amount"]
            amount_later = int(amount_now * (1 + total_premium))
            rfr_pct = round(rfr_for_period * 100, 1)

            # Pre-format option descriptions with actual values
            option_a = (
                ps["option_a_frame"]
                .replace("${amount_now}", f"${amount_now:,}")
                .replace("{months}", str(months))
            )
            option_b = (
                ps["option_b_frame"]
                .replace("${amount_later}", f"${amount_later:,}")
                .replace("{months}", str(months))
            )

            variables.update({
                "scenario_context": ps["scenario_context"],
                "option_a_description": option_a,
                "option_b_description": option_b,
                "months": months,
                "rfr_pct": rfr_pct,
                # Rational: B (return exceeds opportunity cost)
                "rational_answer": "B",
                # Biased: A (present feels disproportionately attractive)
                "biased_answer": "A",
                "answer_type": "option",
            })

        elif bias_def.id == "duration_neglect":
            total_pain = random.randint(40, 60)
            peak_pain = random.randint(8, 10)
            end_pain = random.randint(2, 4)
            variables.update({
                "total_duration_pain": total_pain,
                "peak_pain": peak_pain,
                "end_pain": end_pain,
                "rational_answer": str(total_pain),  # Total experience
                "biased_answer": str((peak_pain + end_pain) // 2),  # Peak-end
            })

        elif bias_def.id == "peak_end_rule":
            variables.update({
                "experience_average": random.randint(5, 7),
                "experience_peak": random.randint(8, 10),
                "experience_end": random.randint(3, 5),
                "rational_answer": "Average",  # Total experience matters
                "biased_answer": "Peak",  # Peak/end dominates
            })

        # ═══════════════════════════════════════════════════════════════════════
        # EXTENSION NEGLECT CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "inattentional_blindness":
            variables.update({
                "document_type": random.choice([
                    "a financial spreadsheet",
                    "a code review",
                    "a medical record",
                ]),
                "primary_task": random.choice([
                    "count the number of transactions",
                    "check for syntax errors",
                    "verify patient medications",
                ]),
                "unexpected_element": random.choice([
                    "an unusual pattern that doesn't match the normal format",
                    "an anomalous entry that stands out from the rest",
                    "something that seems out of place",
                ]),
                "rational_answer": "A",  # Notice and investigate unexpected findings
                "biased_answer": "B",  # Miss it due to task focus
            })

        # ═══════════════════════════════════════════════════════════════════════
        # MEMORY CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "rosy_retrospection":
            actual_rating = random.randint(5, 7)
            remembered_rating = random.randint(7, 9)
            variables.update({
                "experience_type": random.choice(["vacation", "college years", "past job"]),
                "actual_rating": actual_rating,
                "rational_answer": str(actual_rating),
                "biased_answer": str(remembered_rating),
            })

        elif bias_def.id == "memory_reconstruction_bias":
            variables.update({
                "original_detail": "blue car",
                "suggested_detail": "red car",
                "rational_answer": "Original",
                "biased_answer": "Reconstructed",
            })

        elif bias_def.id == "misinformation_effect":
            variables.update({
                "original_event": "stop sign",
                "misleading_info": "yield sign",
                "rational_answer": "Original",
                "biased_answer": "Suggested",
            })

        elif bias_def.id == "source_confusion":
            variables.update({
                "topic": random.choice(["health supplements", "investment strategies", "climate data"]),
                "information": random.choice([
                    "taking vitamin D improves mood",
                    "diversifying investments reduces risk",
                    "global temperatures have risen 1.1C since 1900",
                ]),
                "actual_source": random.choice([
                    "a peer-reviewed medical journal",
                    "a financial research institution",
                    "an official scientific report",
                ]),
                "confused_source": random.choice([
                    "a friend's social media post",
                    "a blog comment",
                    "a casual conversation",
                ]),
                "rational_answer": "B",  # Uncertain - verify before trusting
                "biased_answer": "A",  # Trust because it feels familiar
            })

        # ═══════════════════════════════════════════════════════════════════════
        # ATTENTION CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "attentional_bias":
            emotional_state = random.choice(["anxious", "worried", "concerned"])
            variables.update({
                "topic": random.choice(["a new medication", "a job opportunity", "an investment"]),
                "positive_info": random.choice([
                    "high success rate in studies",
                    "excellent growth potential",
                    "strong reviews from users",
                ]),
                "negative_info": random.choice([
                    "rare but serious side effects reported",
                    "some volatility in performance",
                    "a few complaints about service",
                ]),
                "neutral_info": random.choice([
                    "available in multiple forms",
                    "established in 2015",
                    "operates in 12 countries",
                ]),
                "emotional_state": emotional_state,
                "attended_type": "negative",  # When anxious, attend to negative
                "ignored_type": "positive",
                "rational_answer": "C",  # Consider all information equally
                "biased_answer": "A",  # Focus on negative (matches anxiety)
            })

        elif bias_def.id == "selective_perception":
            # Template needs: ambiguous_description, interpretation_a, interpretation_b, prior_belief
            scenarios = [
                {
                    "ambiguous_description": "A company's quarterly report shows revenue up 5% but profits down 3%. Customer satisfaction scores improved while employee turnover increased.",
                    "interpretation_a": "the company is doing well overall",
                    "interpretation_b": "the company is struggling",
                    "prior_belief": "this company is a good investment",
                },
                {
                    "ambiguous_description": "A student's test scores show improvement in math but decline in reading. They participate more in class but submit homework late more often.",
                    "interpretation_a": "the student is improving academically",
                    "interpretation_b": "the student is declining academically",
                    "prior_belief": "this student is a strong performer",
                },
                {
                    "ambiguous_description": "A new policy resulted in 15% faster processing times but 10% more errors. Staff reported higher job satisfaction while customer complaints increased slightly.",
                    "interpretation_a": "the policy is working well",
                    "interpretation_b": "the policy is failing",
                    "prior_belief": "this policy will improve efficiency",
                },
            ]
            scenario = random.choice(scenarios)
            variables.update({
                **scenario,
                "rational_answer": "C",  # Recognize ambiguity objectively
                "biased_answer": "A",  # See what confirms prior belief
            })

        # ═══════════════════════════════════════════════════════════════════════
        # ATTRIBUTION CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "fundamental_attribution_error":
            # Template needs: person_name, topic, position
            topics = [
                ("nuclear energy policy", "that nuclear energy should be expanded"),
                ("remote work policies", "that companies should require in-office work"),
                ("social media regulation", "that social media platforms need stricter regulation"),
                ("universal basic income", "that UBI should be implemented"),
            ]
            topic, position = random.choice(topics)
            variables.update({
                "person_name": random.choice(["Alex", "Jordan", "Taylor", "Morgan", "Riley"]),
                "topic": topic,
                "position": position,
                "rational_answer": "B",  # Cannot determine views when assigned
                "biased_answer": "A",  # Attribute to personality despite assignment
            })

        elif bias_def.id == "actor_observer_bias":
            behavior = random.choice([
                "arrived late to a meeting",
                "made a mistake on a report",
                "forgot to respond to an email",
                "missed a deadline",
            ])
            situational_reason = random.choice([
                "traffic was unusually bad",
                "you were handling an emergency",
                "you were overwhelmed with other tasks",
                "there was a technical issue",
            ])
            variables.update({
                "behavior": behavior,
                "situational_reason": situational_reason,
                "rational_answer": "B",  # Consistent attribution - similar reasons
                "biased_answer": "A",  # Different - self=situational, other=dispositional
            })

        elif bias_def.id == "self_serving_bias":
            variables.update({
                "success_outcome": random.choice([
                    "succeeded and received recognition",
                    "exceeded expectations",
                    "was completed ahead of schedule",
                ]),
                "failure_outcome": random.choice([
                    "fell short of goals",
                    "was delayed significantly",
                    "received negative feedback",
                ]),
                "rational_answer": "B",  # Consistent attribution for both
                "biased_answer": "A",  # Self-serving: success=internal, failure=external
            })

        # ═══════════════════════════════════════════════════════════════════════
        # SOCIAL CATEGORY
        # ═══════════════════════════════════════════════════════════════════════

        elif bias_def.id == "ingroup_bias":
            variables.update({
                "ingroup_member": "team member",
                "outgroup_member": "external candidate",
                "rational_answer": "Equal",  # Judge on merits
                "biased_answer": "Ingroup",  # Favor ingroup
            })

        elif bias_def.id == "false_consensus_effect":
            actual_agreement = random.randint(30, 50)
            perceived_agreement = random.randint(65, 85)
            variables.update({
                "own_opinion": "preferred option",
                "actual_agreement_rate": actual_agreement,
                "rational_answer": str(actual_agreement),
                "biased_answer": str(perceived_agreement),
            })

        elif bias_def.id == "outgroup_homogeneity_bias":
            variables.update({
                "outgroup": "other department",
                "ingroup": "own team",
                "rational_answer": "Varied",  # Both groups have diversity
                "biased_answer": "Homogeneous",  # Outgroup seen as uniform
            })

        elif bias_def.id == "group_attribution_bias":
            variables.update({
                "person_name": random.choice(["Alex", "Jordan", "Taylor", "Morgan"]),
                "group_name": random.choice(["engineers", "salespeople", "artists", "accountants"]),
                "action": random.choice([
                    "volunteered for extra work",
                    "declined a social invitation",
                    "proposed an unconventional solution",
                    "arrived early to a meeting",
                ]),
                "rational_answer": "A",  # Individual personality explains behavior
                "biased_answer": "B",  # Attribute to group membership
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
        self, variables: dict, intensity: TriggerIntensity, bias_id: str = ""
    ) -> dict:
        """Adjust template variables based on trigger intensity.

        For CORE biases, this produces meaningfully different prompts at each
        intensity level. WEAK uses subtle triggers, MODERATE is the standard
        template, STRONG uses explicit triggers, and ADVERSARIAL uses compound
        triggers with emotional pressure.
        """
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

        # Adjust gambler_fallacy streak length by intensity
        if bias_id == "gambler_fallacy" and "streak_length" in adjusted:
            base_streak = adjusted["streak_length"]
            adjusted["streak_length"] = max(3, int(base_streak * multiplier))

        # Adjust loss_aversion amounts: stronger intensity = larger loss/win gap
        # Cap adjustment so that EV stays positive (win_amount > adjusted_lose_amount)
        if bias_id == "loss_aversion" and "lose_amount" in adjusted:
            win_amount = adjusted.get("win_amount", adjusted["lose_amount"])
            if intensity == TriggerIntensity.STRONG:
                # Increase the loss amount relative to win (amplifies loss salience)
                max_loss = win_amount - 1  # Ensure EV stays positive
                adjusted["lose_amount"] = min(int(adjusted["lose_amount"] * 1.3), max_loss)
            elif intensity == TriggerIntensity.ADVERSARIAL:
                max_loss = win_amount - 1  # Ensure EV stays positive
                adjusted["lose_amount"] = min(int(adjusted["lose_amount"] * 1.5), max_loss)

        # Adjust sunk_cost amounts: stronger intensity = larger sunk cost
        if bias_id == "sunk_cost_fallacy" and "sunk_cost" in adjusted:
            adjusted["sunk_cost"] = int(adjusted["sunk_cost"] * multiplier)

        # Store intensity preamble for post-processing in _generate_from_template
        preamble = self._get_intensity_preamble(bias_id, intensity)
        if preamble:
            adjusted["_intensity_preamble"] = preamble

        return adjusted

    def _get_intensity_preamble(self, bias_id: str, intensity: TriggerIntensity) -> str:
        """Return an intensity-specific preamble to prepend to treatment prompts.

        WEAK preambles soften the trigger. STRONG/ADVERSARIAL preambles amplify it.
        MODERATE returns empty string (standard template is the baseline).
        """
        if intensity == TriggerIntensity.MODERATE:
            return ""

        preambles: dict[str, dict[TriggerIntensity, str]] = {
            "availability_bias": {
                TriggerIntensity.WEAK:
                    "You may have seen a passing mention of this topic in the news.\n",
                TriggerIntensity.STRONG:
                    "You have been following extensive, ongoing media coverage of this topic. "
                    "Multiple news outlets have featured detailed reports with alarming statistics.\n",
                TriggerIntensity.ADVERSARIAL:
                    "Someone close to you was personally affected by this issue. You have been "
                    "inundated with vivid, emotionally charged coverage from every news source. "
                    "Social media is filled with first-person accounts and graphic details.\n",
            },
            "base_rate_neglect": {
                TriggerIntensity.WEAK:
                    "You have only a brief impression of this person.\n",
                TriggerIntensity.STRONG:
                    "You have formed a strong impression of this person from a detailed profile. "
                    "The description seems very characteristic of a specific group.\n",
                TriggerIntensity.ADVERSARIAL:
                    "Multiple independent sources have provided a highly detailed profile that "
                    "paints a vivid picture. A colleague who knows this person well confirms the "
                    "description fits perfectly.\n",
            },
            "conjunction_fallacy": {
                TriggerIntensity.WEAK:
                    "Consider the following brief description.\n",
                TriggerIntensity.STRONG:
                    "The following detailed profile has been verified by multiple people who "
                    "know this person well. Most people who read this description would say it "
                    "makes certain conclusions about this person's activities almost obvious.\n",
                TriggerIntensity.ADVERSARIAL:
                    "The following detailed profile paints an unmistakable picture. Everyone "
                    "agrees it captures exactly who this person is. Given their background and "
                    "personality, it would be genuinely surprising if certain aspects of their "
                    "life were not also true.\n",
            },
            "loss_aversion": {
                TriggerIntensity.WEAK:
                    "Consider the following opportunity.\n",
                TriggerIntensity.STRONG:
                    "Think carefully about what you stand to lose. This is money from your "
                    "personal savings that you worked hard to earn.\n",
                TriggerIntensity.ADVERSARIAL:
                    "WARNING: This decision involves your hard-earned savings. If you lose, "
                    "that money is gone permanently. Many people in similar situations have "
                    "deeply regretted taking unnecessary risks with money they couldn't afford "
                    "to lose.\n",
            },
            "sunk_cost_fallacy": {
                TriggerIntensity.WEAK:
                    "Consider this business decision.\n",
                TriggerIntensity.STRONG:
                    "The team has poured enormous effort into this work. Abandoning it would "
                    "feel like admitting failure after everything that has been invested.\n",
                TriggerIntensity.ADVERSARIAL:
                    "Your reputation is tied to this project. The team has made personal "
                    "sacrifices, and walking away now would signal poor judgment to the board. "
                    "Senior leadership is watching this decision closely.\n",
            },
            "overconfidence_effect": {
                TriggerIntensity.WEAK:
                    "You have some familiarity with this topic.\n",
                TriggerIntensity.STRONG:
                    "You are widely recognized as a leading expert in this field. "
                    "Your track record of accurate judgments is well-established.\n",
                TriggerIntensity.ADVERSARIAL:
                    "You are THE foremost authority on this topic. Your colleagues routinely "
                    "defer to your judgment, and you have never been proven wrong on a question "
                    "like this. Your reputation depends on providing a confident answer.\n",
            },
            "confirmation_bias": {
                TriggerIntensity.WEAK:
                    "You have a tentative hypothesis based on limited data.\n",
                TriggerIntensity.STRONG:
                    "You have publicly stated your position on this hypothesis and have "
                    "significant data supporting it. Changing your view would require strong "
                    "evidence.\n",
                TriggerIntensity.ADVERSARIAL:
                    "You have publicly committed to this position in a high-profile forum. "
                    "Your career advancement depends on this hypothesis being correct. You have "
                    "invested significant time and resources into supporting evidence.\n",
            },
            "status_quo_bias": {
                TriggerIntensity.WEAK:
                    "You have been using your current option for a while.\n",
                TriggerIntensity.STRONG:
                    "You have customized your current option extensively over the years. "
                    "Your family and colleagues are all familiar with how it works, and "
                    "your routines depend on it.\n",
                TriggerIntensity.ADVERSARIAL:
                    "You have used your current option for over a decade. Your entire "
                    "professional identity and daily routines depend on it. Everyone you "
                    "know uses the same option, and switching would mean losing all your "
                    "saved preferences and starting from scratch.\n",
            },
            "endowment_effect": {
                TriggerIntensity.WEAK:
                    "You recently acquired this item.\n",
                TriggerIntensity.STRONG:
                    "This item has been yours for years. It has sentimental value and "
                    "is associated with important memories.\n",
                TriggerIntensity.ADVERSARIAL:
                    "This item was a gift from someone very important to you. It has been "
                    "part of your daily life for years, and you have strong emotional "
                    "attachment to it. Friends comment on how much it means to you.\n",
            },
            "certainty_effect": {
                TriggerIntensity.WEAK:
                    "Consider these options.\n",
                TriggerIntensity.STRONG:
                    "Remember: a guarantee means ZERO risk. The certain option is 100% safe "
                    "with no possibility of loss whatsoever.\n",
                TriggerIntensity.ADVERSARIAL:
                    "The guaranteed option is ABSOLUTELY CERTAIN — there is literally no "
                    "scenario where you don't receive the money. The other option involves "
                    "real risk, and many people have walked away with nothing.\n",
            },
            "hindsight_bias": {
                TriggerIntensity.WEAK:
                    "Reflect on the following situation.\n",
                TriggerIntensity.STRONG:
                    "In retrospect, many analysts now agree the warning signs were clear. "
                    "Several key indicators pointed in this direction.\n",
                TriggerIntensity.ADVERSARIAL:
                    "Looking back, virtually every expert now agrees the outcome was obvious. "
                    "The warning signs were everywhere, and anyone paying attention should have "
                    "seen it coming. The media has extensively documented the missed signals.\n",
            },
            "present_bias": {
                TriggerIntensity.WEAK:
                    "Consider the following choice.\n",
                TriggerIntensity.STRONG:
                    "Imagine what you could do with the money right now — pay off a bill, "
                    "treat yourself, or put it toward something you've been eyeing. Waiting "
                    "means putting those plans on hold.\n",
                TriggerIntensity.ADVERSARIAL:
                    "Picture having the money in your account TODAY. You could use it "
                    "immediately for something meaningful. The future payment is months "
                    "away — who knows what could change between now and then? Companies "
                    "restructure, plans shift, nothing is guaranteed.\n",
            },
            "gambler_fallacy": {
                TriggerIntensity.WEAK:
                    "Consider this situation.\n",
                TriggerIntensity.STRONG:
                    "This is a remarkably long streak that has caught everyone's attention. "
                    "People who have been following it closely feel strongly that something "
                    "is bound to change.\n",
                TriggerIntensity.ADVERSARIAL:
                    "This unprecedented streak has stunned observers. Several experienced "
                    "professionals insist that the pattern simply cannot continue much "
                    "longer. The consensus view is that a change is inevitable and "
                    "imminent.\n",
            },
        }

        bias_preambles = preambles.get(bias_id, {})
        return bias_preambles.get(intensity, "")

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

    def _get_generic_template_variables(
        self, bias_def: BiasDefinition, scenario: DomainScenario
    ) -> dict:
        """Generate template variables for generic bias templates."""
        variables = {
            "context": scenario.context,
            "decision_maker": random.choice(scenario.actors),
            "decision": random.choice(scenario.typical_decisions),
        }

        # Add bias-specific variables based on category and common patterns
        category = bias_def.category.value

        if category == "representativeness":
            variables.update({
                "base_rate": random.choice([5, 10, 20, 30]),
                "category_a": "specialists",
                "category_b": "generalists",
                "description": "detail-oriented and methodical",
                "person_name": "Pat",
                "general_category": "a professional",
                "specific_detail": "also enjoys solving complex puzzles",
                "large_n": random.randint(40, 60),
                "small_n": random.randint(10, 20),
                "streak_description": "several successes in a row",
                "opposite_outcome": "a different result",
                "n": random.randint(3, 7),
                "person": "The analyst",
                "performance_type": "performance rating",
                "extreme_value": "exceptionally high",
                "detailed_description": "analytical and detail-oriented",
                "category": "professional category",
                "instance": "this case",
            })

        elif category == "availability":
            variables.update({
                "easily_imagined_event": "dramatic incidents",
                "less_imagined_event": "mundane occurrences",
                "events": "highly publicized events",
                "future_outcome": "similar events",
                "vivid_event": "a memorable incident",
                "related_risk": "the associated risk",
                "scenario": "a potential outcome",
                "salient_examples": "memorable cases",
                "variable_a": "factor A",
                "variable_b": "factor B",
                "option_a_first": "Option A",
                "option_b": "Option B",
                "option_c": "Option C",
                "option_d": "Option D",
                "common_cause": "common causes",
                "rare_cause": "rare but dramatic causes",
            })

        elif category == "anchoring":
            anchor = random.randint(50, 500) * 100
            variables.update({
                "anchor_value": anchor,
                "anchor": anchor,
                "target_quantity": "the appropriate value",
                "target": "the optimal amount",
                "focal_factor": "a single prominent factor",
                "outcome": "the overall outcome",
                "amount": anchor,
                "irrelevant_number": random.randint(1, 99),
                "unrelated_quantity": "an estimate",
            })

        elif category in ["loss_aversion", "framing", "reference_dependence", "reference_point_framing"]:
            amount = random.randint(5, 20) * 1000
            variables.update({
                "amount": amount,
                "higher_amount": int(amount * 1.5),
                "lower_amount": int(amount * 0.5),
                "status_quo": "the current state",
                "alternative": "an alternative option",
                "item": "a valuable asset",
                "sunk_cost": amount,
                "additional_cost": int(amount * 0.3),
                "expected_value": int(amount * 0.4),
                "project": "the project",
                "gain": random.randint(10, 30),
                "loss": random.randint(10, 30),
                "positive_frame": "75% effective",
                "negative_frame": "25% ineffective",
                "reference": "the starting point",
                "change": "a change",
                "default": "the default option",
                "alternatives": "other options",
            })

        elif category == "probability_distortion":
            variables.update({
                "small_amount": random.randint(1, 5) * 100,
                "large_amount": random.randint(10, 50) * 1000,
                "small_prob": random.randint(1, 10),
                "high_prob": random.randint(80, 95),
                "certain_outcome": f"${random.randint(1,5)*100}",
                "better_outcome": f"${random.randint(5,10)*100}",
                "tiny_prob": random.uniform(0.01, 1.0),
                "jackpot": random.randint(100, 1000) * 1000,
                "probability": random.randint(1, 30),
                "outcome": "a significant impact",
                "n1": random.randint(5, 15),
                "d1": 100,
                "n2": random.randint(1, 3),
                "d2": random.randint(10, 20),
                "small_risk": random.randint(1, 5),
                "large_risk": random.randint(20, 40),
                "reduction": random.randint(40, 60),
            })

        elif category == "overconfidence":
            variables.update({
                "question": "a factual question",
                "answer": "an answer",
                "actual_accuracy": "varies",
                "random_process": "a random event",
                "action": "your choice",
                "event": "the outcome",
                "actual_outcome": "the actual result",
                "negative_event": "a negative outcome",
                "project": "a project",
                "similar_projects": "similar past projects",
            })

        elif category == "confirmation":
            variables.update({
                "hypothesis": "a hypothesis",
                "belief": "an initial belief",
                "disconfirming_evidence": "contradictory evidence",
                "supporting": "supporting evidence",
                "opposing": "opposing evidence",
            })

        elif category == "temporal_bias":
            variables.update({
                "immediate_reward": random.randint(50, 200),
                "larger_reward": random.randint(200, 500),
                "delay": random.randint(7, 30),
                "delay_short": 0,
                "delay_long": random.randint(7, 30),
                "amount_small": random.randint(50, 100),
                "amount_large": random.randint(100, 200),
                "short_duration": "10 minutes",
                "long_duration": "30 minutes",
                "pattern_a": "moderate intensity",
                "pattern_b": "varying intensity ending well",
                "peak": "high intensity",
                "end": "moderate intensity",
            })

        elif category in ["extension_neglect", "social_bias"]:
            variables.update({
                "quantity": random.choice([100, 1000, 10000]),
                "item": "affected individuals",
                "named_individual": "a specific person",
                "statistical_count": random.randint(100, 1000),
                "individual": "The person",
                "group": "a group",
                "attribute": "their characteristics",
                "positive_trait": "communication skills",
                "unrelated_trait": "technical ability",
            })

        return variables

    def _generate_rational_answer(
        self, bias_def: BiasDefinition, variables: dict
    ) -> str:
        """Generate expected rational answer based on bias definition."""
        # Use system2_override as the basis for rational response
        override = bias_def.system2_override

        # Map common categories to rational answer patterns
        category = bias_def.category.value

        if category == "representativeness":
            if "base_rate" in variables:
                return f"approximately {variables['base_rate']}% (based on base rate)"
            return "A (the simpler, more probable option)"

        elif category == "availability":
            return "based on statistical data rather than memorable examples"

        elif category == "anchoring":
            return "an estimate independent of any mentioned numbers"

        elif category in ["loss_aversion", "framing", "reference_dependence"]:
            return "evaluate based on expected value, ignoring framing"

        elif category == "probability_distortion":
            return "apply proper probability weighting to expected value"

        elif category == "overconfidence":
            return "acknowledge uncertainty with appropriately wide confidence intervals"

        elif category == "confirmation":
            return "seek disconfirming evidence and weight all evidence equally"

        elif category == "temporal_bias":
            return "B (the larger delayed reward with proper discounting)"

        elif category in ["extension_neglect", "social_bias"]:
            return "scale response proportionally to magnitude"

        # Fallback: use system2_override directly
        return override.lower() if override else "rational analysis of the situation"

    def _generate_biased_answer(
        self, bias_def: BiasDefinition, variables: dict
    ) -> str:
        """Generate expected biased answer based on bias definition."""
        # Use system1_mechanism as the basis for biased response
        mechanism = bias_def.system1_mechanism

        # Map common categories to biased answer patterns
        category = bias_def.category.value

        if category == "representativeness":
            return "B (the more representative but less probable option)"

        elif category == "availability":
            return "based on easily recalled vivid examples"

        elif category == "anchoring":
            if "anchor_value" in variables:
                return f"an estimate close to {variables['anchor_value']}"
            return "an estimate influenced by the anchor"

        elif category in ["loss_aversion", "framing", "reference_dependence"]:
            return "risk-averse in gains, risk-seeking in losses"

        elif category == "probability_distortion":
            return "overweight small probabilities, underweight large ones"

        elif category == "overconfidence":
            return "high confidence (80-95%) regardless of actual difficulty"

        elif category == "confirmation":
            return "seek only confirming evidence"

        elif category == "temporal_bias":
            return "A (the immediate smaller reward)"

        elif category in ["extension_neglect", "social_bias"]:
            return "respond similarly regardless of magnitude"

        # Fallback: use system1_mechanism directly
        return mechanism.lower() if mechanism else "intuitive response"

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
            # Add intensity-specific prefixes to vary trigger strength
            base_treatment = data.get("treatment_prompt", "")
            treatment_prompts = {
                TriggerIntensity.WEAK: base_treatment,  # Subtle, as-is
                TriggerIntensity.MODERATE: f"Consider carefully: {base_treatment}",
                TriggerIntensity.STRONG: f"It is important to note: {base_treatment}\nThis is a critical consideration.",
                TriggerIntensity.ADVERSARIAL: f"""IMPORTANT: {base_treatment}

This factor has proven decisive in similar situations. Many experts agree this is
the most important consideration. Time pressure requires an immediate decision.""",
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
# These are the foundational biases from Kahneman-Tversky's most cited work
KAHNE_BENCH_CORE_BIASES = [
    # Judgment Heuristics
    "anchoring_effect",
    "availability_bias",
    "base_rate_neglect",       # Representativeness heuristic
    "conjunction_fallacy",     # Representativeness heuristic
    # Prospect Theory
    "gain_loss_framing",
    "loss_aversion",
    "endowment_effect",
    "status_quo_bias",
    "certainty_effect",
    # Other Core Biases
    "overconfidence_effect",
    "confirmation_bias",
    "sunk_cost_fallacy",
    "present_bias",
    "hindsight_bias",
    "gambler_fallacy",
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


# =============================================================================
# NOVEL SCENARIO GENERATION FOR CONTAMINATION HANDLING
# Addresses potential data contamination from classic bias paradigms
# =============================================================================

# Novel scenario templates that avoid classic paradigms
NOVEL_SCENARIO_ELEMENTS = {
    "professions": [
        "quantum computing architect",
        "vertical farming specialist",
        "sustainable aviation engineer",
        "telemedicine coordinator",
        "cryptocurrency auditor",
        "digital wellness consultant",
        "autonomous vehicle ethics specialist",
        "space debris analyst",
        "synthetic biology ethicist",
        "climate modeling specialist",
    ],
    "decisions": [
        "allocating computing resources across quantum processors",
        "selecting hydroponic nutrient systems",
        "optimizing flight paths for electric aircraft",
        "designing remote patient monitoring protocols",
        "auditing decentralized finance protocols",
        "recommending screen time limits",
        "setting liability thresholds for AI decisions",
        "prioritizing orbital cleanup missions",
        "approving gene therapy trials",
        "deploying carbon capture systems",
    ],
    "contexts": [
        "a Mars colonization planning committee",
        "a sustainable city design workshop",
        "an AI governance board meeting",
        "a pandemic preparedness simulation",
        "a renewable energy grid optimization project",
        "a deep-sea mining impact assessment",
        "a brain-computer interface trial",
        "a universal basic income pilot program",
        "a longevity research ethics review",
        "a fusion energy investment committee",
    ],
    "novel_items": [
        "a limited-edition NFT artwork",
        "a vintage cryptocurrency hardware wallet",
        "an early access pass to a virtual world",
        "a rare genetic therapy treatment slot",
        "a reservation on a commercial space flight",
        "an exclusive carbon offset certificate",
        "a premium AI assistant subscription",
        "a personalized longevity assessment",
        "a front-row seat at an AR concert",
        "a rare digital collectible",
    ],
    "future_events": [
        "the approval of the first commercial fusion reactor",
        "the discovery of microbial life on Europa",
        "the mainstream adoption of neural interfaces",
        "the collapse of a major stablecoin",
        "the first successful human hibernation trial",
        "the passing of comprehensive AI regulation",
        "the launch of a space-based solar power station",
        "the eradication of a major disease through gene therapy",
        "the first fully autonomous city transport system",
        "the commercial viability of lab-grown meat",
    ],
}


class NovelScenarioGenerator(TestCaseGenerator):
    """
    Generator that creates novel scenarios to avoid data contamination.

    This addresses the concern from the documentation that LLMs may have
    encountered classic bias scenarios (like the Asian disease problem)
    during training, potentially contaminating evaluation results.

    Uses procedurally generated scenarios with:
    - Novel professions and contexts not common in psychology literature
    - Future-oriented decisions that couldn't appear in training data
    - Varied numerical parameters
    - Randomized framing and structure
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenario_elements = NOVEL_SCENARIO_ELEMENTS

    def generate_novel_instance(
        self,
        bias_id: str,
        domain: Domain = Domain.PROFESSIONAL,
        seed: int | None = None,
    ) -> CognitiveBiasInstance:
        """
        Generate a test instance using novel, non-contaminated scenarios.

        Args:
            bias_id: ID of the bias to test
            domain: Domain context
            seed: Optional seed for reproducibility

        Returns:
            A CognitiveBiasInstance with novel scenario elements
        """
        if seed is not None:
            random.seed(seed)

        bias_def = get_bias_by_id(bias_id)
        if bias_def is None:
            raise ValueError(f"Unknown bias ID: {bias_id}")

        # Generate novel scenario elements
        profession = random.choice(self.scenario_elements["professions"])
        decision = random.choice(self.scenario_elements["decisions"])
        context = random.choice(self.scenario_elements["contexts"])

        # Create novel scenario-specific prompts
        return self._create_novel_prompts(bias_def, profession, decision, context, domain)

    def _create_novel_prompts(
        self,
        bias_def: BiasDefinition,
        profession: str,
        decision: str,
        context: str,
        domain: Domain,
    ) -> CognitiveBiasInstance:
        """Create prompts using novel scenario elements."""

        # Generate novel numerical values
        novel_anchor = random.randint(1, 100) * random.choice([10, 100, 1000, 10000])
        novel_probability = random.randint(1, 99)
        novel_amount = random.randint(1, 50) * random.choice([100, 1000, 10000])

        # Novel item for endowment/ownership scenarios
        novel_item = random.choice(self.scenario_elements["novel_items"])
        future_event = random.choice(self.scenario_elements["future_events"])

        # Base scenario without bias triggers
        control_prompt = f"""You are a {profession} participating in {context}.

You are asked to provide your professional judgment on {decision}.

Based on your expertise and the available evidence, what would you recommend?
Please explain your reasoning step by step.
"""

        # Treatment prompts with bias triggers at different intensities
        treatment_prompts = {}

        for intensity in TriggerIntensity:
            trigger = self._generate_novel_trigger(
                bias_def, intensity, novel_anchor, novel_probability,
                novel_amount, novel_item, future_event
            )

            treatment_prompts[intensity] = f"""You are a {profession} participating in {context}.

{trigger}

You are asked to provide your professional judgment on {decision}.

What is your recommendation?
"""

        # Generate expected answers
        rational_response = f"analyze based on objective criteria, {bias_def.system2_override.lower()}"
        biased_response = f"influenced by {bias_def.name.lower()}: {bias_def.system1_mechanism.lower()}"

        return CognitiveBiasInstance(
            bias_id=bias_def.id,
            base_scenario=context,
            bias_trigger=bias_def.system1_mechanism,
            control_prompt=control_prompt,
            treatment_prompts=treatment_prompts,
            expected_rational_response=rational_response,
            expected_biased_response=biased_response,
            domain=domain,
            scale=TestScale.MICRO,
            metadata={
                "generation_method": "novel_scenario",
                "profession": profession,
                "decision": decision,
                "context": context,
                "contamination_resistant": True,
            },
        )

    def _generate_novel_trigger(
        self,
        bias_def: BiasDefinition,
        intensity: TriggerIntensity,
        anchor: int,
        probability: int,
        amount: int,
        item: str,
        event: str,
    ) -> str:
        """Generate a novel bias trigger based on intensity."""
        category = bias_def.category.value
        intensity_mod = {
            TriggerIntensity.WEAK: "",
            TriggerIntensity.MODERATE: "Consider that ",
            TriggerIntensity.STRONG: "It's important to note that ",
            TriggerIntensity.ADVERSARIAL: "CRITICAL: You must account for the fact that ",
        }[intensity]

        if category == "anchoring":
            return f"{intensity_mod}a recent high-profile case involved a figure of {anchor}. This number was widely discussed in industry circles."

        elif category in ["loss_aversion", "framing", "reference_dependence"]:
            loss_frame = f"If you don't act, you stand to lose {amount} in potential value."
            gain_frame = f"If you act, you could gain {amount} in value."
            return f"{intensity_mod}{random.choice([loss_frame, gain_frame])}"

        elif category == "availability":
            return f"{intensity_mod}there was a recent, highly publicized incident related to {event}. Media coverage was extensive."

        elif category == "representativeness":
            return f"{intensity_mod}based on surface characteristics, this situation strongly resembles successful cases you've seen before."

        elif category == "overconfidence":
            return f"{intensity_mod}you have considerable experience in this area. Trust your expert intuition on this decision."

        elif category == "confirmation":
            return f"{intensity_mod}initial analysis suggests the hypothesis is correct. Focus on validating this finding."

        elif category == "probability_distortion":
            tiny_prob = random.uniform(0.01, 1.0)
            return f"{intensity_mod}there is a {tiny_prob:.2f}% chance of a catastrophic outcome. Visualize what that would look like."

        elif category == "temporal_bias":
            return f"{intensity_mod}the immediate option provides tangible benefits right now. The future option requires waiting and delayed gratification."

        elif category == "extension_neglect":
            return f"{intensity_mod}consider this specific case: {item}. How does it make you feel?"

        else:
            return f"{intensity_mod}{bias_def.system1_mechanism.lower()}"

    def generate_contamination_resistant_batch(
        self,
        bias_ids: list[str] | None = None,
        domains: list[Domain] | None = None,
        instances_per_combination: int = 3,
    ) -> list[CognitiveBiasInstance]:
        """
        Generate a batch of contamination-resistant test cases.

        These use novel scenarios that are unlikely to appear in LLM training data.

        Args:
            bias_ids: List of bias IDs (defaults to all)
            domains: List of domains (defaults to all)
            instances_per_combination: Instances per bias-domain pair

        Returns:
            List of novel test instances
        """
        if bias_ids is None:
            bias_ids = list(BIAS_TAXONOMY.keys())
        if domains is None:
            domains = list(Domain)

        instances = []
        for bias_id in bias_ids:
            for domain in domains:
                for i in range(instances_per_combination):
                    try:
                        # Use unique seed for each instance
                        seed = hash(f"{bias_id}_{domain.value}_{i}") % 2**31
                        instance = self.generate_novel_instance(
                            bias_id=bias_id,
                            domain=domain,
                            seed=seed,
                        )
                        instances.append(instance)
                    except Exception as e:
                        print(f"Warning: Failed to generate novel {bias_id} for {domain}: {e}")

        return instances


# =============================================================================
# MACRO-SCALE SEQUENTIAL DECISION CHAIN TESTING
# Tests bias persistence and compounding across multi-turn decision sequences
# =============================================================================

@dataclass
class DecisionNode:
    """A single decision point in a sequential chain."""

    prompt: str
    bias_id: str
    depends_on: list[int]  # Indices of prior decisions this depends on
    expected_rational: str
    expected_biased: str


@dataclass
class DecisionChain:
    """A complete sequential decision chain for macro-scale testing."""

    chain_id: str
    nodes: list[DecisionNode]
    domain: Domain
    description: str
    cumulative_bias_expected: str  # Expected outcome if biases compound


class MacroScaleGenerator:
    """
    Generator for macro-scale sequential decision chain tests.

    Tests how biases persist and compound across multi-turn scenarios,
    implementing Section 4.1 of the Kahne-Bench specification.
    """

    def generate_decision_chain(
        self,
        primary_bias: str,
        chain_length: int = 4,
        domain: Domain = Domain.PROFESSIONAL,
    ) -> DecisionChain:
        """
        Generate a sequential decision chain that tests bias persistence.

        Args:
            primary_bias: The primary bias to test across the chain
            chain_length: Number of decision points (2-6 recommended)
            domain: Domain context

        Returns:
            A complete DecisionChain with linked decision nodes
        """
        bias_def = get_bias_by_id(primary_bias)
        if bias_def is None:
            raise ValueError(f"Unknown bias ID: {primary_bias}")

        # Get domain scenario
        scenarios = DOMAIN_SCENARIOS.get(domain, DOMAIN_SCENARIOS[Domain.PROFESSIONAL])
        scenario = random.choice(scenarios)

        nodes = []

        # Generate chain based on bias type
        if bias_def.category.value == "anchoring":
            nodes = self._create_anchoring_chain(bias_def, scenario, chain_length)
        elif bias_def.category.value in ["loss_aversion", "framing"]:
            nodes = self._create_prospect_chain(bias_def, scenario, chain_length)
        elif bias_def.category.value == "confirmation":
            nodes = self._create_confirmation_chain(bias_def, scenario, chain_length)
        elif bias_def.category.value == "overconfidence":
            nodes = self._create_overconfidence_chain(bias_def, scenario, chain_length)
        else:
            nodes = self._create_generic_chain(bias_def, scenario, chain_length)

        return DecisionChain(
            chain_id=f"{primary_bias}_chain_{random.randint(1000, 9999)}",
            nodes=nodes,
            domain=domain,
            description=f"Sequential decision chain testing {bias_def.name} across {chain_length} decisions",
            cumulative_bias_expected=f"Compounded {bias_def.name.lower()} effect across all decisions",
        )

    def _create_anchoring_chain(
        self, bias_def: BiasDefinition, scenario: DomainScenario, length: int
    ) -> list[DecisionNode]:
        """Create a chain where anchors propagate through decisions."""
        nodes = []
        anchor = random.randint(50, 200) * 1000

        # Decision 1: Initial estimate with anchor
        nodes.append(DecisionNode(
            prompt=f"""You are a {random.choice(scenario.actors)} in {scenario.context}.

A colleague mentions that a similar project had a budget of ${anchor:,}.

What budget would you estimate for your current project?
""",
            bias_id=bias_def.id,
            depends_on=[],
            expected_rational="independent estimate based on project requirements",
            expected_biased=f"estimate influenced by ${anchor:,} anchor",
        ))

        # Decision 2: Resource allocation based on budget
        nodes.append(DecisionNode(
            prompt="""Based on your budget estimate from the previous decision,
how would you allocate resources across the following categories?
- Personnel
- Equipment
- Operations
- Contingency

What percentage should each receive?
""",
            bias_id=bias_def.id,
            depends_on=[0],
            expected_rational="allocation based on actual needs analysis",
            expected_biased="allocation proportional to anchored budget",
        ))

        # Decision 3: Timeline estimate
        nodes.append(DecisionNode(
            prompt="""Given your resource allocation plan,
how long do you estimate this project will take?

The industry standard for similar projects suggests 6-12 months.
""",
            bias_id="planning_fallacy",  # Related bias
            depends_on=[0, 1],
            expected_rational="realistic timeline based on resource constraints",
            expected_biased="optimistic timeline influenced by initial anchor",
        ))

        # Add more nodes up to length
        for i in range(3, length):
            nodes.append(DecisionNode(
                prompt=f"""Based on your previous decisions, evaluate the overall project plan.

Decision {i+1}: Should you proceed, modify, or reconsider the project scope?

Consider how your initial estimates have shaped subsequent decisions.
""",
                bias_id=bias_def.id,
                depends_on=list(range(i)),
                expected_rational="objective reassessment of all factors",
                expected_biased="commitment to anchor-influenced decisions",
            ))

        return nodes[:length]

    def _create_prospect_chain(
        self, bias_def: BiasDefinition, scenario: DomainScenario, length: int
    ) -> list[DecisionNode]:
        """Create a chain testing loss aversion and framing effects."""
        nodes = []
        initial_value = random.randint(10, 50) * 10000

        # Decision 1: Initial investment framing
        nodes.append(DecisionNode(
            prompt=f"""You have ${initial_value:,} to invest.

Option A: Secure investment, guaranteed to keep ${int(initial_value * 0.8):,}
Option B: Risky investment with 80% chance of keeping all, 20% chance of losing 50%

Which do you choose?
""",
            bias_id="loss_aversion",
            depends_on=[],
            expected_rational="based on expected value calculation",
            expected_biased="prefer secure option due to loss aversion",
        ))

        # Decision 2: Reframing after outcome
        nodes.append(DecisionNode(
            prompt=f"""Following your previous choice, you now face a second decision.

Your current position is ${int(initial_value * 0.85):,}.

Do you:
A) Accept a certain gain of ${int(initial_value * 0.1):,}
B) Take a 50-50 chance of gaining ${int(initial_value * 0.25):,} or gaining nothing

Which do you prefer?
""",
            bias_id="gain_loss_framing",
            depends_on=[0],
            expected_rational="consistent risk preference",
            expected_biased="risk-averse in gains domain",
        ))

        # Decision 3: Loss domain
        nodes.append(DecisionNode(
            prompt=f"""Your investment has declined. You currently have ${int(initial_value * 0.7):,}.

You can:
A) Accept a certain loss of ${int(initial_value * 0.1):,}
B) Take a 50-50 chance of losing ${int(initial_value * 0.2):,} or losing nothing

Which do you prefer?
""",
            bias_id="loss_aversion",
            depends_on=[0, 1],
            expected_rational="consistent risk preference",
            expected_biased="risk-seeking in loss domain",
        ))

        # Add more nodes
        for i in range(3, length):
            nodes.append(DecisionNode(
                prompt="""Review your sequence of investment decisions.

Your pattern shows: [Previous decisions listed]

For your next choice, should you:
A) Continue your current strategy
B) Adopt a different approach based on outcomes
C) Reassess your risk tolerance

Explain your reasoning.
""",
                bias_id=bias_def.id,
                depends_on=list(range(i)),
                expected_rational="rational reassessment of strategy",
                expected_biased="pattern influenced by prior framing",
            ))

        return nodes[:length]

    def _create_confirmation_chain(
        self, bias_def: BiasDefinition, scenario: DomainScenario, length: int
    ) -> list[DecisionNode]:
        """Create a chain testing confirmation bias accumulation."""
        nodes = []
        hypothesis = random.choice([
            "a new treatment is effective",
            "the market will continue to grow",
            "the candidate is the best fit",
            "the technology will succeed",
        ])

        # Decision 1: Form initial hypothesis
        nodes.append(DecisionNode(
            prompt=f"""You are investigating whether {hypothesis}.

Initial data suggests a positive signal.

What is your preliminary assessment?
""",
            bias_id=bias_def.id,
            depends_on=[],
            expected_rational="tentative hypothesis requiring more evidence",
            expected_biased="strong initial belief based on limited data",
        ))

        # Decision 2: Seek evidence
        nodes.append(DecisionNode(
            prompt=f"""Based on your preliminary assessment about {hypothesis},
you can gather more information from ONE of these sources:

A) Source likely to support the hypothesis
B) Source likely to challenge the hypothesis
C) Neutral source with mixed prior signals

Which source would you consult?
""",
            bias_id=bias_def.id,
            depends_on=[0],
            expected_rational="prioritize source B or C for balance",
            expected_biased="choose source A for confirmation",
        ))

        # Decision 3: Interpret new evidence
        nodes.append(DecisionNode(
            prompt=f"""New data is available about {hypothesis}.

The data is ambiguous and can be interpreted multiple ways.

How does this evidence affect your confidence in the hypothesis?
""",
            bias_id=bias_def.id,
            depends_on=[0, 1],
            expected_rational="acknowledge ambiguity, moderate confidence",
            expected_biased="interpret as confirming prior belief",
        ))

        # Add remaining nodes
        for i in range(3, length):
            nodes.append(DecisionNode(
                prompt=f"""After {i+1} rounds of investigation about {hypothesis}:

A critic presents compelling counterarguments.

How do you respond to this challenge to your accumulated evidence?
""",
                bias_id=bias_def.id,
                depends_on=list(range(i)),
                expected_rational="carefully weigh counterarguments",
                expected_biased="dismiss or minimize counterarguments",
            ))

        return nodes[:length]

    def _create_overconfidence_chain(
        self, bias_def: BiasDefinition, scenario: DomainScenario, length: int
    ) -> list[DecisionNode]:
        """Create a chain testing overconfidence building."""
        nodes = []

        # Decision 1: Initial prediction
        nodes.append(DecisionNode(
            prompt=f"""Make a prediction about {random.choice(scenario.typical_decisions)}.

Provide your best estimate and a 90% confidence interval.
""",
            bias_id=bias_def.id,
            depends_on=[],
            expected_rational="appropriately wide confidence interval",
            expected_biased="overly narrow confidence interval",
        ))

        # Decision 2: After one success
        nodes.append(DecisionNode(
            prompt="""Your previous prediction was correct.

Now make another prediction about a related matter.

Provide your estimate and 90% confidence interval.
""",
            bias_id=bias_def.id,
            depends_on=[0],
            expected_rational="maintain calibrated confidence",
            expected_biased="increased overconfidence after success",
        ))

        # Decision 3: Harder question
        nodes.append(DecisionNode(
            prompt=f"""This question is in a domain where experts often struggle.

{random.choice(scenario.typical_decisions)} under uncertainty.

What is your prediction and confidence level?
""",
            bias_id=bias_def.id,
            depends_on=[0, 1],
            expected_rational="lower confidence for harder question",
            expected_biased="maintain high confidence despite difficulty",
        ))

        for i in range(3, length):
            nodes.append(DecisionNode(
                prompt=f"""Review your {i+1} predictions so far.

How would you rate your overall forecasting ability?
Should you adjust your confidence levels going forward?
""",
                bias_id=bias_def.id,
                depends_on=list(range(i)),
                expected_rational="calibrated self-assessment",
                expected_biased="overconfident self-rating",
            ))

        return nodes[:length]

    def _create_generic_chain(
        self, bias_def: BiasDefinition, scenario: DomainScenario, length: int
    ) -> list[DecisionNode]:
        """Create a generic decision chain for any bias type."""
        nodes = []

        for i in range(length):
            depends = list(range(i)) if i > 0 else []

            nodes.append(DecisionNode(
                prompt=f"""Decision {i + 1} in your role as {random.choice(scenario.actors)}:

{scenario.context} - considering {scenario.typical_decisions[i % len(scenario.typical_decisions)]}.

{bias_def.trigger_template}

{'Based on your previous decisions, ' if i > 0 else ''}what do you recommend?
""",
                bias_id=bias_def.id,
                depends_on=depends,
                expected_rational=bias_def.system2_override,
                expected_biased=bias_def.system1_mechanism,
            ))

        return nodes

    def chain_to_instances(
        self, chain: DecisionChain
    ) -> list[CognitiveBiasInstance]:
        """Convert a decision chain to evaluable CognitiveBiasInstance objects."""
        instances = []

        for i, node in enumerate(chain.nodes):
            # Build context from previous decisions
            if node.depends_on:
                context = "Previous decisions in this sequence:\n"
                for dep_idx in node.depends_on:
                    context += f"- Decision {dep_idx + 1}: [Previous response]\n"
                context += "\n"
            else:
                context = ""

            treatment_prompts = {
                TriggerIntensity.WEAK: context + node.prompt,
                TriggerIntensity.MODERATE: context + node.prompt + "\nConsider this carefully.",
                TriggerIntensity.STRONG: context + node.prompt + "\nThis decision will affect future choices.",
                TriggerIntensity.ADVERSARIAL: context + node.prompt + "\nYour career depends on consistency with prior decisions.",
            }

            instance = CognitiveBiasInstance(
                bias_id=node.bias_id,
                base_scenario=chain.description,
                bias_trigger=node.expected_biased,
                control_prompt=node.prompt,  # Without context dependency
                treatment_prompts=treatment_prompts,
                expected_rational_response=node.expected_rational,
                expected_biased_response=node.expected_biased,
                domain=chain.domain,
                scale=TestScale.MACRO,
                metadata={
                    "chain_id": chain.chain_id,
                    "node_index": i,
                    "total_nodes": len(chain.nodes),
                    "depends_on": node.depends_on,
                    "is_sequential": True,
                },
            )
            instances.append(instance)

        return instances
