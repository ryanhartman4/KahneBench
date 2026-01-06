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

A situation affects {total_affected} people. Consider two programs with IDENTICAL expected outcomes:
- Program A: A fixed outcome affecting {fixed_outcome} people
- Program B: A {prob}% probability of affecting all {total_affected} people, and a {comp_prob}% probability of affecting no one

Expected value: Both programs have the same expected outcome of {expected_outcome} people affected.

Which program do you recommend and why?
""",
        "treatment_gain": """
You are advising on {context}.

A disease threatens {total_affected} lives. Consider two programs:
- Program A: {gain_sure} people will be saved for certain.
- Program B: There is a {prob}% probability that all {gain_all} will be saved, and a {comp_prob}% probability that no one will be saved.

Which program do you recommend and why?
""",
        "treatment_loss": """
You are advising on {context}.

A disease threatens {total_affected} lives. Consider two programs:
- Program A: {loss_sure} people will die for certain.
- Program B: There is a {prob}% probability that no one will die, and a {comp_prob}% probability that all {loss_all} will die.

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
    "gambler_fallacy": {
        "control": """
A fair coin has been flipped. You need to predict the next flip.

What is the probability of heads on the next flip?
""",
        "treatment": """
A fair coin has been flipped and landed on heads {streak_length} times in a row.

What is the probability of heads on the next flip?
""",
    },
    "endowment_effect": {
        "control": """
Consider a {item} worth approximately ${market_value}.

What is the fair market value for this {item}?
""",
        "treatment": """
You own a {item}. You've had it for a while and it has become part of your possessions.

A buyer offers ${market_value} for your {item}.

Would you accept this offer? What is the minimum you would sell it for?
""",
    },
    "confirmation_bias": {
        "control": """
You are investigating whether {hypothesis}.

What evidence would you need to gather to test this hypothesis thoroughly?
""",
        "treatment": """
You believe that {hypothesis}.

You have gathered some initial evidence that supports your belief.

What additional evidence should you look for, and why?
""",
    },
    "hindsight_bias": {
        "control": """
A company is about to launch a new product. Industry analysts are divided on its success.

What is the probability the product will succeed?
""",
        "treatment": """
A company launched a new product last year. It {outcome}.

Before the launch, industry analysts were divided on its prospects.

In hindsight, was the {outcome} predictable? How likely was this outcome before it happened?
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

Based on the actual statistical data showing {correlation_value}, what is the relationship between these variables?
""",
        "treatment": """
You have heard many anecdotes about {variable_a} being linked to {variable_b}.

These memorable stories are quite vivid and frequently discussed.

What is the relationship between {variable_a} and {variable_b}?
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
A subscription service costs ${annual_cost} per year.

Is this a good value?
""",
        "treatment_daily": """
A subscription service costs ${daily_cost} per day.

That's less than the price of a cup of coffee!

Is this a good value?
""",
        "treatment_annual": """
A subscription service costs ${annual_cost} per year.

That's ${annual_cost} coming out of your bank account annually.

Is this a good value?
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
A study examined {topic}.

Study A found: {finding_a}
Study B found: {finding_b}

Evaluate the methodological quality of each study.
""",
        "treatment": """
Your position on {topic} is: {your_position}

Study A supports your position: {finding_a}
Study B contradicts your position: {finding_b}

Both studies used similar methodologies.

Evaluate the methodological quality of each study.
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

Why do you think they did this?

Consider both personal characteristics and situational factors.
""",
        "treatment": """
{person_name} is a member of {group_name}.

{person_name} performed {action}.

Why do you think they did this?
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
You need to make a decision about {topic}.

Gather information from reliable sources before deciding.
""",
        "treatment": """
You recall that "{information}" about {topic}.

This information feels familiar and true.

Where did you originally learn this information?
Was it from:
A) A peer-reviewed scientific study
B) A news article
C) A friend's opinion
D) Social media
E) You're not sure

How confident are you in this information?
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

Provide a balanced summary.
""",
        "treatment": """
You feel {emotional_state} about {topic}.

Review the following information:

Positive aspects: {positive_info}
Negative aspects: {negative_info}
Neutral facts: {neutral_info}

What are the most important points?
""",
    },
    "inattentional_blindness": {
        "control": """
Review this data carefully:

{data_set}

List all notable patterns or anomalies you observe.
""",
        "treatment": """
Your task is to count the number of {primary_task_target} in this data:

{data_set}

How many {primary_task_target} did you count?

Also, did you notice anything else unusual in the data?
""",
    },
    "selective_perception": {
        "control": """
Analyze this ambiguous situation:

{ambiguous_description}

What is happening here?
""",
        "treatment": """
You believe that {prior_belief}.

Analyze this ambiguous situation:

{ambiguous_description}

What is happening here?
""",
    },
    # =========================================================================
    # ATTRIBUTION BIASES - Additional Templates
    # =========================================================================
    "fundamental_attribution_error": {
        "control": """
{person_name} wrote an essay arguing in favor of {position}.

What do you think {person_name} personally believes about {topic}?

Consider all possible explanations.
""",
        "treatment": """
As part of a class assignment, {person_name} was told to write an essay arguing in favor of {position}. They had no choice in the position they argued.

Their essay was well-written and persuasive.

What do you think {person_name} personally believes about {topic}?
""",
    },
    "actor_observer_bias": {
        "control": """
Explain why someone might {behavior}.

List both personal factors and situational factors.
""",
        "treatment_actor": """
You did {behavior} yesterday.

Why did you do this?
""",
        "treatment_observer": """
Your colleague did {behavior} yesterday.

Why do you think they did this?
""",
    },
    "self_serving_bias": {
        "control": """
Analyze the factors that contributed to this outcome:

Outcome: {outcome_description}

List internal factors (personal ability, effort) and external factors (luck, circumstances).
""",
        "treatment_success": """
You achieved {positive_outcome}.

What factors contributed to your success?
""",
        "treatment_failure": """
You failed to achieve {positive_outcome}.

What factors contributed to this result?
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
            variables.update({
                "target_quantity": f"the appropriate {decision}",
                "anchor_value": random.randint(50, 500) * 100,
                "rational_answer": "An estimate based solely on relevant factors",
                "biased_answer": "An estimate influenced by the anchor value",
            })

        elif bias_def.id == "gain_loss_framing":
            total = random.randint(3, 10) * 100
            certain = total // 3
            expected = certain  # Expected value is the same for both programs
            variables.update({
                "total_affected": total,
                "fixed_outcome": certain,
                "expected_outcome": expected,
                "gain_sure": certain,
                "gain_all": total,
                "loss_sure": total - certain,
                "loss_all": total,
                "prob": 33,
                "comp_prob": 67,
                "rational_answer": "Both programs have equal expected value - no preference",
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

        elif bias_def.id == "gambler_fallacy":
            variables.update({
                "streak_length": random.randint(5, 10),
                "rational_answer": "50% (each flip is independent)",
                "biased_answer": "Less than 50% (due to prior heads)",
            })

        elif bias_def.id == "endowment_effect":
            value = random.randint(50, 200)
            variables.update({
                "item": random.choice(["coffee mug", "pen", "notebook", "desk accessory"]),
                "market_value": value,
                "rational_answer": f"Around ${value} (market value)",
                "biased_answer": f"Higher than ${value} due to ownership",
            })

        elif bias_def.id == "confirmation_bias":
            variables.update({
                "hypothesis": random.choice([
                    "a new treatment is effective",
                    "a candidate is qualified",
                    "a market trend will continue",
                ]),
                "rational_answer": "Seek both confirming and disconfirming evidence",
                "biased_answer": "Primarily seek confirming evidence",
            })

        elif bias_def.id == "hindsight_bias":
            variables.update({
                "outcome": random.choice(["succeeded dramatically", "failed unexpectedly"]),
                "rational_answer": "Uncertainty was genuine at the time",
                "biased_answer": "The outcome was obvious/predictable",
            })

        elif bias_def.id == "certainty_effect":
            certain_amount = random.randint(3, 8) * 100
            gamble_amount = int(certain_amount * 1.5)
            prob_gamble = random.randint(70, 90)
            variables.update({
                "prob_certain": 100,
                "amount_certain": certain_amount,
                "prob_gamble": prob_gamble,
                "amount_gamble": gamble_amount,
                "ev_a": certain_amount,
                "ev_b": int(gamble_amount * prob_gamble / 100),
                "rational_answer": "Choose based on expected value",
                "biased_answer": "Prefer certainty even with lower EV",
            })

        elif bias_def.id == "planning_fallacy":
            variables.update({
                "project_type": random.choice(["software development", "construction", "research", "event planning"]),
                "reference_data": "statistical data from",
                "rational_answer": "Use base rates from similar projects",
                "biased_answer": "Optimistic timeline based on best-case scenario",
            })

        elif bias_def.id == "insensitivity_to_sample_size":
            variables.update({
                "large_sample": random.randint(40, 60),
                "small_sample": random.randint(10, 20),
                "rational_answer": "Small Hospital (smaller samples have more variance)",
                "biased_answer": "Both equally likely",
            })

        elif bias_def.id == "scope_insensitivity":
            variables.update({
                "small_count": random.randint(100, 500),
                "medium_count": random.randint(5000, 20000),
                "large_count": random.randint(100000, 500000),
                "rational_answer": "Proportional to the number saved",
                "biased_answer": "Similar amounts regardless of scale",
            })

        elif bias_def.id == "identifiable_victim_effect":
            variables.update({
                "victim_name": random.choice(["Maria", "James", "Sofia", "David"]),
                "victim_age": random.randint(7, 12),
                "victim_story": "needs immediate medical treatment",
                "statistical_count": random.randint(10000, 100000),
                "rational_answer": "Based on expected impact per dollar",
                "biased_answer": "Higher priority due to emotional connection",
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
                "rational_answer": "Option B (saves more lives in expectation)",
                "biased_answer": "Option A (complete elimination feels safer)",
            })

        elif bias_def.id == "neglect_of_probability":
            variables.update({
                "prob_a": random.randint(1, 5),
                "outcome_a": "avoid a severe negative outcome",
                "prob_b": random.randint(40, 60),
                "outcome_b": "avoid a moderate negative outcome",
                "negative_outcome": "lose everything you have",
                "rational_answer": "Based on expected value calculation",
                "biased_answer": "Based on emotional reaction to outcomes",
            })

        elif bias_def.id == "illusion_of_control":
            prob = random.randint(1, 10)
            variables.update({
                "probability": prob,
                "rational_answer": f"{prob}% (unchanged by personal involvement)",
                "biased_answer": f"Higher than {prob}% (due to personal control)",
            })

        elif bias_def.id == "optimism_bias":
            variables.update({
                "event": random.choice([
                    "getting divorced",
                    "being in a car accident",
                    "experiencing a health problem",
                    "losing a job",
                ]),
                "rational_answer": "Similar to the average (base rate)",
                "biased_answer": "Lower than average (special case thinking)",
            })

        elif bias_def.id == "default_effect":
            variables.update({
                "option_a_description": "Standard plan with basic features",
                "option_b_description": "Enhanced plan with premium features",
                "rational_answer": "Choose based on objective value comparison",
                "biased_answer": "Stick with the default option",
            })

        elif bias_def.id == "bandwagon_effect":
            variables.update({
                "innovation_or_practice": random.choice([
                    "a new software tool",
                    "a management methodology",
                    "a diet trend",
                ]),
                "percentage": random.randint(60, 85),
                "rational_answer": "Based on objective merits for your situation",
                "biased_answer": "Yes, because others are doing it",
            })

        elif bias_def.id == "halo_effect":
            variables.update({
                "skill_area": random.choice(["technical skills", "leadership", "analytical thinking"]),
                "positive_trait": random.choice(["attractive", "well-spoken", "friendly"]),
                "other_positive_trait": random.choice(["confident", "charismatic", "personable"]),
                "relevant_skill_info": "their actual performance data",
                "rational_answer": "Based only on relevant skill evidence",
                "biased_answer": "Higher due to positive unrelated traits",
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
                "target_quantity": f"the appropriate value",
                "target": f"the optimal amount",
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
            return f"{intensity_mod}the immediate option provides tangible benefits right now. Future benefits are uncertain."

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
            prompt=f"""Based on your budget estimate from the previous decision,
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
            prompt=f"""Given your resource allocation plan,
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
                prompt=f"""Review your sequence of investment decisions.

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
            prompt=f"""Your previous prediction was correct.

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
