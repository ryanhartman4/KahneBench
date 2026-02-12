"""
Complete taxonomy of 69 cognitive biases for Kahne-Bench.

Each bias is grounded in the Kahneman-Tversky research program, with clear
links to dual-process theory and prospect theory mechanisms.

The framework extends beyond the original 50-bias specification to include
69 well-documented biases across 16 categories:

Bias categories (16 total, 69 biases):
- Representativeness Heuristic (8 biases)
- Availability Heuristic (6 biases)
- Anchoring (5 biases)
- Loss Aversion / Prospect Theory (5 biases)
- Framing Effects (7 biases) - includes mental_accounting
- Probability Distortion (7 biases) - includes affect_heuristic
- Overconfidence (5 biases)
- Confirmation Bias (3 biases)
- Temporal Biases (3 biases)
- Extension Neglect (4 biases)
- Memory Biases (4 biases)
- Attention Biases (3 biases)
- Attribution Biases (3 biases) - NEW
- Uncertainty Judgment (3 biases) - NEW
- Social Biases (5 biases) - expanded with ingroup_bias, false_consensus, outgroup_homogeneity
"""

from kahne_bench.core import BiasDefinition, BiasCategory


# =============================================================================
# REPRESENTATIVENESS HEURISTIC BIASES (1-8)
# Judging probability by similarity to prototypes/stereotypes
# =============================================================================

REPRESENTATIVENESS_BIASES = [
    BiasDefinition(
        id="base_rate_neglect",
        name="Base Rate Neglect",
        category=BiasCategory.REPRESENTATIVENESS,
        description="Ignoring prior probability information in favor of specific case details",
        theoretical_basis="Kahneman & Tversky (1973) - Demonstrated with engineer/lawyer problem",
        system1_mechanism="Pattern matching to stereotypes overrides statistical thinking",
        system2_override="Explicitly calculate Bayesian posterior probabilities",
        classic_paradigm="Engineer/Lawyer problem: Given base rates and personality description",
        trigger_template="Given that {base_rate}% of the population are {category_a}, consider {vivid_description}...",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="conjunction_fallacy",
        name="Conjunction Fallacy",
        category=BiasCategory.REPRESENTATIVENESS,
        description="Judging P(A and B) > P(A) when B adds representativeness",
        theoretical_basis="Tversky & Kahneman (1983) - The Linda Problem",
        system1_mechanism="Representativeness makes conjunction seem more typical",
        system2_override="Apply probability axiom: P(A∩B) ≤ P(A)",
        classic_paradigm="Linda Problem: feminist bank teller vs bank teller",
        trigger_template="{person_description} Which is more likely: (A) {general_category} or (B) {general_category} and {specific_detail}?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="insensitivity_to_sample_size",
        name="Insensitivity to Sample Size",
        category=BiasCategory.REPRESENTATIVENESS,
        description="Expecting small samples to be representative of population",
        theoretical_basis="Kahneman & Tversky (1972) - Law of small numbers",
        system1_mechanism="Representativeness ignores sample size in pattern assessment",
        system2_override="Apply statistical reasoning about sampling variability",
        classic_paradigm="Hospital problem: Which hospital has more days with >60% boys?",
        trigger_template="Hospital A delivers {large_n} babies/day, Hospital B delivers {small_n}/day. Which has more extreme days?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="gambler_fallacy",
        name="Gambler's Fallacy",
        category=BiasCategory.REPRESENTATIVENESS,
        description="Expecting random sequences to 'balance out' locally",
        theoretical_basis="Tversky & Kahneman (1971) - Belief in law of small numbers",
        system1_mechanism="Short sequences expected to represent global frequencies",
        system2_override="Recognize independence of random events",
        classic_paradigm="After 5 heads in a row, tails is 'due'",
        trigger_template="After {streak_description}, what would you recommend regarding {next_outcome}?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="hot_hand_fallacy",
        name="Hot Hand Fallacy",
        category=BiasCategory.REPRESENTATIVENESS,
        description="Perceiving streaks as evidence of momentum in random processes",
        theoretical_basis="Gilovich, Vallone & Tversky (1985) - Basketball shooting analysis",
        system1_mechanism="Pattern detection in random sequences",
        system2_override="Test for actual statistical dependence",
        classic_paradigm="Basketball player 'on fire' - expecting continued success",
        trigger_template="{person} has succeeded in the last {n} attempts. What is the probability of success on the next attempt?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="regression_neglect",
        name="Regression to Mean Neglect",
        category=BiasCategory.REPRESENTATIVENESS,
        description="Failing to expect extreme values to regress toward the mean",
        theoretical_basis="Kahneman & Tversky (1973) - Flight instructor study",
        system1_mechanism="Causal thinking applied to statistical regression",
        system2_override="Recognize regression as statistical necessity",
        classic_paradigm="Praise after good performance seems to cause decline",
        trigger_template="After an extreme {performance_type} of {extreme_value}, predict the next {performance_type}.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="stereotype_bias",
        name="Stereotype Bias",
        category=BiasCategory.REPRESENTATIVENESS,
        description="Over-relying on category stereotypes for individual judgments",
        theoretical_basis="Kahneman & Tversky (1973) - Personality prediction studies",
        system1_mechanism="Prototype matching dominates individuating information",
        system2_override="Weight base rates and individual evidence appropriately",
        classic_paradigm="Predicting profession from personality description",
        trigger_template="{detailed_description}. What is the most likely {category} for this person?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="prototype_heuristic",
        name="Prototype/Typicality Heuristic",
        category=BiasCategory.REPRESENTATIVENESS,
        description="Judging category membership by similarity to prototype",
        theoretical_basis="Rosch (1978) extended by Kahneman's attribute substitution",
        system1_mechanism="Substituting 'how typical?' for 'how probable?'",
        system2_override="Consider base rates and diagnostic value of features",
        classic_paradigm="Is a penguin a 'real' bird?",
        trigger_template="Rate how typical {instance} is of {category}, then estimate probability of {instance} being {category}.",
        is_kt_core=True,
    ),
]

# =============================================================================
# AVAILABILITY HEURISTIC BIASES (9-14)
# Judging frequency/probability by ease of recall
# =============================================================================

AVAILABILITY_BIASES = [
    BiasDefinition(
        id="availability_bias",
        name="Availability Bias",
        category=BiasCategory.AVAILABILITY,
        description="Estimating frequency/probability by ease of mental retrieval",
        theoretical_basis="Tversky & Kahneman (1973) - Availability: A heuristic for frequency",
        system1_mechanism="Fluency of recall substitutes for actual frequency",
        system2_override="Seek objective statistics rather than relying on memory",
        classic_paradigm="Letters: More words starting with K or with K as 3rd letter?",
        trigger_template="Estimate the frequency of {easily_imagined_event} versus {less_imagined_event}.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="recency_bias",
        name="Recency Bias",
        category=BiasCategory.AVAILABILITY,
        description="Overweighting recent events in probability judgments",
        theoretical_basis="Extension of availability heuristic - recent = easily recalled",
        system1_mechanism="Recent events more available in memory",
        system2_override="Consider full historical data, not just recent events",
        classic_paradigm="Stock market: Recent trends predict future",
        trigger_template="Given recent {events}, predict the likelihood of {future_outcome}."
    ),
    BiasDefinition(
        id="salience_bias",
        name="Salience Bias",
        category=BiasCategory.AVAILABILITY,
        description="Overweighting vivid, emotionally striking information",
        theoretical_basis="Kahneman (2011) - Emotional events are highly available",
        system1_mechanism="Vivid memories easily retrieved, seem more frequent",
        system2_override="Correct for emotional impact on memory accessibility",
        classic_paradigm="Fear of flying vs. driving despite statistics",
        trigger_template="After {vivid_event}, estimate the probability of {related_risk}.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="simulation_heuristic",
        name="Simulation Heuristic",
        category=BiasCategory.AVAILABILITY,
        description="Judging probability by ease of mentally simulating scenarios",
        theoretical_basis="Kahneman & Tversky (1982) - Simulation and counterfactuals",
        system1_mechanism="Easily imagined scenarios seem more likely",
        system2_override="Distinguish imaginability from actual probability",
        classic_paradigm="Near-miss scenarios in lottery (almost won!)",
        trigger_template="Imagine {scenario}. How likely is this to occur?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="illusory_correlation",
        name="Illusory Correlation",
        category=BiasCategory.AVAILABILITY,
        description="Perceiving relationships between unrelated variables",
        theoretical_basis="Hamilton & Gifford (1976), extended by availability research",
        system1_mechanism="Distinctive pairings are memorable, create false associations",
        system2_override="Check actual covariation statistics",
        classic_paradigm="Believing minorities commit more crimes due to media salience",
        trigger_template="Based on {salient_examples}, estimate the correlation between {variable_a} and {variable_b}."
    ),
    BiasDefinition(
        id="primacy_bias",
        name="Primacy Bias",
        category=BiasCategory.AVAILABILITY,
        description="Overweighting information presented first",
        theoretical_basis="Documented in LLMs by recent research; links to availability",
        system1_mechanism="First information sets context and is highly available",
        system2_override="Deliberately consider all information equally",
        classic_paradigm="First impressions in interviews; first options in lists",
        trigger_template="Consider options: {option_a_first}, {option_b}, {option_c}, {option_d}. Which do you prefer?"
    ),
]

# =============================================================================
# ANCHORING BIASES (15-19)
# Insufficient adjustment from initial values
# =============================================================================

ANCHORING_BIASES = [
    BiasDefinition(
        id="anchoring_effect",
        name="Anchoring Effect",
        category=BiasCategory.ANCHORING,
        description="Estimates biased toward an initial reference value",
        theoretical_basis="Tversky & Kahneman (1974) - Anchoring and adjustment",
        system1_mechanism="Initial value activates related concepts; adjustment insufficient",
        system2_override="Generate estimate independently before considering anchor",
        classic_paradigm="UN percentage in Africa after random wheel spin",
        trigger_template="The number {anchor} was mentioned. Now estimate {target_quantity}.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="insufficient_adjustment",
        name="Insufficient Adjustment",
        category=BiasCategory.ANCHORING,
        description="Stopping adjustment from anchor too early",
        theoretical_basis="Epley & Gilovich (2006) - Anchoring and adjustment mechanism",
        system1_mechanism="Adjustment is effortful; System 1 stops when plausible reached",
        system2_override="Continue adjustment until confident in independence from anchor",
        classic_paradigm="Self-generated anchors in estimation tasks",
        trigger_template="Starting from {anchor}, estimate {target}. Consider if you've adjusted enough."
    ),
    BiasDefinition(
        id="focalism",
        name="Focalism",
        category=BiasCategory.ANCHORING,
        description="Focusing too heavily on a single piece of information",
        theoretical_basis="Wilson et al. (2000) - Related to anchoring mechanisms",
        system1_mechanism="Focal information dominates attention and judgment",
        system2_override="Systematically consider multiple relevant factors",
        classic_paradigm="Predicting future happiness based on single life change",
        trigger_template="Given {focal_factor}, predict {outcome}. What else should you consider?"
    ),
    BiasDefinition(
        id="first_offer_anchoring",
        name="First Offer Anchoring",
        category=BiasCategory.ANCHORING,
        description="Negotiation outcomes anchored by first offer",
        theoretical_basis="Galinsky & Mussweiler (2001) - Negotiation anchoring",
        system1_mechanism="First number sets psychological reference point",
        system2_override="Prepare independent valuation before negotiation",
        classic_paradigm="Salary negotiations anchored by initial offer",
        trigger_template="The first offer in this negotiation was {amount}. What would be a fair final agreement?"
    ),
    BiasDefinition(
        id="numeric_priming",
        name="Numeric Priming/Incidental Anchoring",
        category=BiasCategory.ANCHORING,
        description="Irrelevant numbers influencing subsequent estimates",
        theoretical_basis="Wilson et al. (1996) - Incidental anchoring effects",
        system1_mechanism="Any activated number can serve as implicit anchor",
        system2_override="Recognize potential for incidental anchoring",
        classic_paradigm="Social security number affecting price estimates",
        trigger_template="Consider the number {irrelevant_number}. Now estimate {unrelated_quantity}."
    ),
]

# =============================================================================
# PROSPECT THEORY - LOSS AVERSION (20-24)
# Losses loom larger than equivalent gains
# =============================================================================

LOSS_AVERSION_BIASES = [
    BiasDefinition(
        id="loss_aversion",
        name="Loss Aversion",
        category=BiasCategory.LOSS_AVERSION,
        description="Losses psychologically weigh ~2x more than equivalent gains",
        theoretical_basis="Kahneman & Tversky (1979) - Core prospect theory finding",
        system1_mechanism="Negative outcomes trigger stronger emotional response",
        system2_override="Calculate expected value objectively",
        classic_paradigm="Rejecting 50/50 bet to win $150 or lose $100",
        trigger_template="Choose: (A) Sure gain of {amount}, or (B) 50% chance to gain {higher_amount} or lose {lower_amount}?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="endowment_effect",
        name="Endowment Effect",
        category=BiasCategory.LOSS_AVERSION,
        description="Valuing owned items more than equivalent unowned items",
        theoretical_basis="Thaler (1980), Kahneman et al. (1990) - Mug experiments",
        system1_mechanism="Selling = loss of owned item; loss aversion inflates price",
        system2_override="Consider opportunity cost of not selling",
        classic_paradigm="Mug owners demand ~2x what buyers will pay",
        trigger_template="You own {item}. At what price would you sell it? If you didn't own it, what would you pay?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="status_quo_bias",
        name="Status Quo Bias",
        category=BiasCategory.LOSS_AVERSION,
        description="Preferring current state over alternatives due to loss framing",
        theoretical_basis="Samuelson & Zeckhauser (1988) - Status quo as reference point",
        system1_mechanism="Change = potential losses from current position",
        system2_override="Evaluate all options from neutral reference point",
        classic_paradigm="Sticking with default options even when suboptimal",
        trigger_template="Current situation: {status_quo}. Alternative: {alternative}. Which do you prefer?"
    ),
    BiasDefinition(
        id="sunk_cost_fallacy",
        name="Sunk Cost Fallacy",
        category=BiasCategory.LOSS_AVERSION,
        description="Continuing investments due to prior irrecoverable costs",
        theoretical_basis="Arkes & Blumer (1985), related to loss aversion",
        system1_mechanism="Abandoning = crystallizing prior losses",
        system2_override="Ignore sunk costs; decide based on future value only",
        classic_paradigm="Watching boring movie because ticket was expensive",
        trigger_template="You've invested {sunk_cost} in {project}. It will require {additional_cost} to complete, or you could switch to {alternative}. Which option do you recommend?"
    ),
    BiasDefinition(
        id="disposition_effect",
        name="Disposition Effect",
        category=BiasCategory.LOSS_AVERSION,
        description="Selling winners too early and holding losers too long",
        theoretical_basis="Shefrin & Statman (1985), derived from prospect theory",
        system1_mechanism="Selling losers = realizing losses; hold to avoid pain",
        system2_override="Evaluate each position on future prospects only",
        classic_paradigm="Investor portfolio decisions",
        trigger_template="Stock A is up {gain}%. Stock B is down {loss}%. Which do you sell to raise cash?"
    ),
]

# =============================================================================
# PROSPECT THEORY - FRAMING EFFECTS (25-30)
# Decisions affected by how options are presented
# =============================================================================

FRAMING_BIASES = [
    BiasDefinition(
        id="gain_loss_framing",
        name="Gain-Loss Framing",
        category=BiasCategory.FRAMING,
        description="Risk preferences reverse based on gain vs loss framing",
        theoretical_basis="Tversky & Kahneman (1981) - Asian Disease Problem",
        system1_mechanism="Gains trigger risk aversion; losses trigger risk seeking",
        system2_override="Compute expected values; recognize framing manipulation",
        classic_paradigm="Asian Disease Problem: 200 saved vs 400 die",
        trigger_template="Program A: {gain_frame}. Program B: {equivalent_gamble}. Which do you choose?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="attribute_framing",
        name="Attribute Framing",
        category=BiasCategory.FRAMING,
        description="Single attributes evaluated differently based on positive/negative frame",
        theoretical_basis="Levin & Gaeth (1988) - Beef labeling studies",
        system1_mechanism="Positive/negative labels trigger different associations",
        system2_override="Translate frames to common metric before evaluating",
        classic_paradigm="75% lean vs 25% fat beef",
        trigger_template="Product is {positive_frame}. How would you rate it? (Compare: {negative_frame})"
    ),
    BiasDefinition(
        id="reference_point_framing",
        name="Reference Point Framing",
        category=BiasCategory.REFERENCE_DEPENDENCE,  # Directly reflects K&T reference dependence concept
        description="Outcomes coded as gains/losses relative to arbitrary reference",
        theoretical_basis="Kahneman & Tversky (1979) - Reference dependence in prospect theory",
        system1_mechanism="Reference point determines gain/loss coding",
        system2_override="Evaluate final wealth states, not changes",
        classic_paradigm="Tax 'bonus' vs 'refund' framing",
        trigger_template="Starting from {reference}, outcome is {change}. Evaluate this outcome.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="default_effect",
        name="Default Effect",
        category=BiasCategory.FRAMING,
        description="Tendency to accept pre-selected options",
        theoretical_basis="Johnson & Goldstein (2003) - Organ donation defaults",
        system1_mechanism="Default = implicit recommendation + loss aversion for change",
        system2_override="Actively evaluate whether default suits your preferences",
        classic_paradigm="Organ donation opt-in vs opt-out countries",
        trigger_template="The default option is {default}. Alternatives are {alternatives}. What do you choose?"
    ),
    BiasDefinition(
        id="risk_framing",
        name="Risk Framing",
        category=BiasCategory.FRAMING,
        description="Risk perception altered by probability vs frequency formats",
        theoretical_basis="Gigerenzer & Hoffrage (1995) - Natural frequencies",
        system1_mechanism="Percentages harder to process than natural frequencies",
        system2_override="Convert to common format before comparing risks",
        classic_paradigm="1 in 1000 chance vs 0.1% probability",
        trigger_template="Risk A: {percentage_format}. Risk B: {frequency_format}. Which seems more concerning?"
    ),
    BiasDefinition(
        id="temporal_framing",
        name="Temporal Framing",
        category=BiasCategory.FRAMING,
        description="Time period framing affects value perception",
        theoretical_basis="Related to mental accounting and prospect theory",
        system1_mechanism="Daily vs yearly costs processed as different magnitudes",
        system2_override="Normalize all time periods for fair comparison",
        classic_paradigm="$1/day vs $365/year",
        trigger_template="Cost is {daily_frame} per day, equivalent to {yearly_frame} per year. How expensive does this seem?"
    ),
]

# =============================================================================
# PROBABILITY DISTORTION (31-36)
# Systematic misweighting of probabilities
# =============================================================================

PROBABILITY_BIASES = [
    BiasDefinition(
        id="probability_weighting",
        name="Probability Weighting",
        category=BiasCategory.PROBABILITY_DISTORTION,
        description="Overweighting small probabilities, underweighting moderate/large ones",
        theoretical_basis="Kahneman & Tversky (1979) - Probability weighting function in PT",
        system1_mechanism="Small probabilities overweighted; certainty effect for high probs",
        system2_override="Use objective probabilities in calculations",
        classic_paradigm="Lottery tickets and insurance purchases",
        trigger_template="Option A: Certain {small_amount}. Option B: {small_prob}% chance of {large_amount}. Choose.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="certainty_effect",
        name="Certainty Effect",
        category=BiasCategory.PROBABILITY_DISTORTION,
        description="Disproportionate preference for certain over probable outcomes",
        theoretical_basis="Allais (1953), integrated into prospect theory",
        system1_mechanism="Certainty eliminates anxiety; probabilistic outcomes feel risky",
        system2_override="Calculate expected values without overweighting certainty",
        classic_paradigm="Allais Paradox: Preference reversals with certain options",
        trigger_template="Option A: {certain_outcome} for sure. Option B: {high_prob}% chance of {better_outcome}. Choose.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="possibility_effect",
        name="Possibility Effect",
        category=BiasCategory.PROBABILITY_DISTORTION,
        description="Overweighting of small but non-zero probabilities",
        theoretical_basis="Kahneman & Tversky (1979) - Explains lottery purchases",
        system1_mechanism="Non-zero probability feels like 'real chance'",
        system2_override="Weight tiny probabilities appropriately in EV calculation",
        classic_paradigm="Buying lottery tickets despite negative expected value",
        trigger_template="There is a {tiny_prob}% chance of winning {jackpot}. How much would you pay for this lottery ticket?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="neglect_of_probability",
        name="Neglect of Probability",
        category=BiasCategory.PROBABILITY_DISTORTION,
        description="Focusing on outcomes while ignoring their probabilities",
        theoretical_basis="Sunstein (2002), related to affect heuristic",
        system1_mechanism="Emotional outcomes dominate; probability fades to background",
        system2_override="Force explicit probability estimation before evaluation",
        classic_paradigm="Fear of terrorism vs car accidents",
        trigger_template="Event has {probability}% chance and would result in {outcome}. How worried should you be?"
    ),
    BiasDefinition(
        id="denominator_neglect",
        name="Denominator Neglect",
        category=BiasCategory.PROBABILITY_DISTORTION,
        description="Focusing on numerator while ignoring denominator of ratios",
        theoretical_basis="Reyna & Brainerd (2008), related to probability distortion",
        system1_mechanism="Numerator is salient; denominator requires calculation",
        system2_override="Explicitly consider both parts of ratio",
        classic_paradigm="7 in 100 vs 1 in 10 preference",
        trigger_template="Which is more likely: {n1} in {d1} or {n2} in {d2}?"
    ),
    BiasDefinition(
        id="zero_risk_bias",
        name="Zero Risk Bias",
        category=BiasCategory.PROBABILITY_DISTORTION,
        description="Preference for eliminating risk completely over larger reductions",
        theoretical_basis="Baron et al. (1993), related to certainty effect",
        system1_mechanism="Zero is qualitatively different from small number",
        system2_override="Calculate actual risk reduction magnitudes",
        classic_paradigm="Preferring to eliminate one small risk over reducing larger risk",
        trigger_template="Option A: Eliminate {small_risk}% risk completely. Option B: Reduce {large_risk}% risk by {reduction}%. Choose."
    ),
]

# =============================================================================
# OVERCONFIDENCE (37-41)
# Excessive certainty in one's judgments
# =============================================================================

OVERCONFIDENCE_BIASES = [
    BiasDefinition(
        id="overconfidence_effect",
        name="Overconfidence Effect",
        category=BiasCategory.OVERCONFIDENCE,
        description="Subjective confidence exceeds objective accuracy",
        theoretical_basis="Lichtenstein et al. (1982) - Calibration studies",
        system1_mechanism="Coherent narrative feels like accurate understanding",
        system2_override="Track calibration; seek disconfirming evidence",
        classic_paradigm="Confidence intervals too narrow; 90% confidence correct <50%",
        trigger_template="Answer: {answer}. How confident are you (0-100%)? {actual_accuracy}"
    ),
    BiasDefinition(
        id="planning_fallacy",
        name="Planning Fallacy",
        category=BiasCategory.OVERCONFIDENCE,
        description="Underestimating time, costs, and risks for future projects",
        theoretical_basis="Kahneman & Tversky (1979), detailed in Kahneman (2011)",
        system1_mechanism="Inside view focuses on specifics, ignores base rates",
        system2_override="Use outside view: compare to similar past projects",
        classic_paradigm="Sydney Opera House: 6 years estimate, 16 years actual",
        trigger_template="Estimate time to complete {project}. Consider past {similar_projects}.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="illusion_of_control",
        name="Illusion of Control",
        category=BiasCategory.OVERCONFIDENCE,
        description="Belief in ability to control random or uncontrollable outcomes",
        theoretical_basis="Langer (1975) - Dice throwing studies",
        system1_mechanism="Agency feelings extend beyond actual influence",
        system2_override="Distinguish skill from chance outcomes",
        classic_paradigm="Rolling dice harder for higher numbers",
        trigger_template="In {random_process}, how much can your {action} influence the outcome?"
    ),
    BiasDefinition(
        id="hindsight_bias",
        name="Hindsight Bias",
        category=BiasCategory.OVERCONFIDENCE,
        description="'Knew-it-all-along' effect after learning outcomes",
        theoretical_basis="Fischhoff (1975) - Outcome knowledge distorts memory",
        system1_mechanism="Outcome integrates into narrative as inevitable",
        system2_override="Record predictions before outcomes; compare honestly",
        classic_paradigm="Post-election: 'I always knew they would win'",
        trigger_template="Before knowing outcome: What did you predict for {event}? After: {actual_outcome}"
    ),
    BiasDefinition(
        id="optimism_bias",
        name="Optimism Bias",
        category=BiasCategory.OVERCONFIDENCE,
        description="Believing oneself less likely to experience negative events",
        theoretical_basis="Weinstein (1980) - Unrealistic optimism studies",
        system1_mechanism="Positive self-image generalizes to predictions",
        system2_override="Compare to objective base rates for similar people",
        classic_paradigm="Smokers underestimating personal cancer risk",
        trigger_template="Estimate your personal probability of {negative_event}. Population base rate: {base_rate}%"
    ),
]

# =============================================================================
# CONFIRMATION BIAS (42-44)
# Seeking/interpreting information to confirm existing beliefs
# =============================================================================

CONFIRMATION_BIASES = [
    BiasDefinition(
        id="confirmation_bias",
        name="Confirmation Bias",
        category=BiasCategory.CONFIRMATION,
        description="Seeking information that confirms existing beliefs",
        theoretical_basis="Wason (1960) - 2-4-6 task",
        system1_mechanism="Confirming evidence is more fluently processed",
        system2_override="Actively seek disconfirming evidence",
        classic_paradigm="2-4-6 task: Testing rule by confirming examples only",
        trigger_template="Your hypothesis is {hypothesis}. What evidence would you seek to test it?"
    ),
    BiasDefinition(
        id="belief_perseverance",
        name="Belief Perseverance",
        category=BiasCategory.CONFIRMATION,
        description="Maintaining beliefs after evidence has been discredited",
        theoretical_basis="Ross et al. (1975) - Debriefing ineffectiveness",
        system1_mechanism="Initial belief creates mental model resistant to update",
        system2_override="Consider evidence strength when updating beliefs",
        classic_paradigm="Fake feedback studies: beliefs persist after debriefing",
        trigger_template="You believed {belief}. Evidence now shows {disconfirming_evidence}. What do you now believe?"
    ),
    BiasDefinition(
        id="myside_bias",
        name="Myside Bias",
        category=BiasCategory.CONFIRMATION,
        description="Evaluating evidence according to prior beliefs",
        theoretical_basis="Stanovich et al. (2013) - Related to confirmation bias",
        system1_mechanism="Prior beliefs act as filter for evidence evaluation",
        system2_override="Evaluate evidence quality independent of conclusion",
        classic_paradigm="Asymmetric skepticism about evidence for/against one's views",
        trigger_template="Evidence for your position: {supporting}. Against: {opposing}. Rate the quality of each."
    ),
]

# =============================================================================
# TEMPORAL BIASES (45-47)
# Biases related to time perception and intertemporal choice
# =============================================================================

TEMPORAL_BIASES = [
    BiasDefinition(
        id="present_bias",
        name="Present Bias/Hyperbolic Discounting",
        category=BiasCategory.TEMPORAL_BIAS,
        description="Disproportionate preference for immediate over future rewards",
        theoretical_basis="Laibson (1997) - Quasi-hyperbolic discounting",
        system1_mechanism="Present is vivid and certain; future is abstract",
        system2_override="Apply consistent discount rate across time periods",
        classic_paradigm="Preferring $100 today over $110 tomorrow, but $110 in 31 days over $100 in 30",
        trigger_template="Choose: {immediate_reward} now, or {larger_reward} in {delay}?"
    ),
    BiasDefinition(
        id="duration_neglect",
        name="Duration Neglect",
        category=BiasCategory.TEMPORAL_BIAS,
        description="Ignoring experience duration in retrospective evaluation",
        theoretical_basis="Kahneman et al. (1993) - Colonoscopy studies",
        system1_mechanism="Peak and end moments dominate memory, not duration",
        system2_override="Consider total experienced utility",
        classic_paradigm="Longer painful procedure preferred if end is less painful",
        trigger_template="Experience A: {short_duration} with {pattern_a}. Experience B: {long_duration} with {pattern_b}. Rate each.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="peak_end_rule",
        name="Peak-End Rule",
        category=BiasCategory.TEMPORAL_BIAS,
        description="Evaluating experiences by peak intensity and ending",
        theoretical_basis="Kahneman et al. (1993) - Memory vs experienced utility",
        system1_mechanism="Peak and end are most memorable; summarize experience",
        system2_override="Weight all moments equally in evaluation",
        classic_paradigm="Adding painful time if it improves the ending",
        trigger_template="Experience peaked at {peak} and ended at {end}. Rate the overall experience.",
        is_kt_core=True,
    ),
]

# =============================================================================
# EXTENSION NEGLECT / SCOPE INSENSITIVITY (48-51)
# Ignoring sample size, scope, and extensional attributes
# Includes social biases related to group judgments
# =============================================================================

EXTENSION_NEGLECT_BIASES = [
    BiasDefinition(
        id="scope_insensitivity",
        name="Scope Insensitivity",
        category=BiasCategory.EXTENSION_NEGLECT,
        description="Willingness to pay insensitive to magnitude of good",
        theoretical_basis="Kahneman & Knetsch (1992) - WTP for birds study",
        system1_mechanism="Emotional response similar regardless of scope",
        system2_override="Scale valuation proportionally with quantity",
        classic_paradigm="Similar WTP to save 2,000, 20,000, or 200,000 birds",
        trigger_template="How much would you pay to save {quantity} {item}?",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="identifiable_victim_effect",
        name="Identifiable Victim Effect",
        category=BiasCategory.EXTENSION_NEGLECT,
        description="Greater concern for identified individuals than statistics",
        theoretical_basis="Small et al. (2007) - Related to scope insensitivity",
        system1_mechanism="Individual story evokes emotion; statistics don't",
        system2_override="Consider total impact regardless of identifiability",
        classic_paradigm="Donations higher for named child than statistics",
        trigger_template="Help {named_individual} or program affecting {statistical_count} people?"
    ),
    BiasDefinition(
        id="group_attribution_bias",
        name="Group Attribution Bias",
        category=BiasCategory.SOCIAL_BIAS,
        description="Attributing group characteristics to individual members",
        theoretical_basis="Pettigrew (1979) - Ultimate attribution error",
        system1_mechanism="Group prototype applied to individuals",
        system2_override="Consider individual variation within groups",
        classic_paradigm="Judging individual based on group stereotypes",
        trigger_template="{individual} belongs to {group}. Predict their {attribute}."
    ),
    BiasDefinition(
        id="halo_effect",
        name="Halo Effect",
        category=BiasCategory.SOCIAL_BIAS,
        description="Overgeneralization of positive traits from one domain to unrelated domains",
        theoretical_basis="Thorndike (1920), extended by Nisbett & Wilson (1977)",
        system1_mechanism="Positive impression in one area creates positive expectations in others",
        system2_override="Evaluate each attribute independently based on specific evidence",
        classic_paradigm="Attractive people judged as more competent and honest",
        trigger_template="{person} excels at {positive_trait}. Evaluate their likely {unrelated_trait}."
    ),
]


# =============================================================================
# MEMORY BIASES (4 biases)
# Systematic distortions in how information is encoded, stored, and recalled
# =============================================================================

MEMORY_BIASES = [
    BiasDefinition(
        id="rosy_retrospection",
        name="Rosy Retrospection",
        category=BiasCategory.MEMORY_BIAS,
        description="Tendency to recall past events more positively than they actually were",
        theoretical_basis="Mitchell et al. (1997) - Studies on vacation experiences",
        system1_mechanism="Emotional fading allows positive aspects to dominate memory reconstruction",
        system2_override="Use contemporaneous records and objective data when evaluating past events",
        classic_paradigm="Vacations remembered more positively than rated during the trip",
        trigger_template="Recall your experience with {past_event}. How would you rate it compared to similar {current_event}?"
    ),
    BiasDefinition(
        id="source_confusion",
        name="Source Confusion",
        category=BiasCategory.MEMORY_BIAS,
        description="Misattributing the source of a memory to the wrong context or person",
        theoretical_basis="Johnson et al. (1993) - Source monitoring framework",
        system1_mechanism="Familiarity overrides source attribution during retrieval",
        system2_override="Explicitly verify source attribution before acting on remembered information",
        classic_paradigm="Remembering information but forgetting where it was learned",
        trigger_template="You recall {information}. Where did you originally learn this, and how certain are you?"
    ),
    BiasDefinition(
        id="misinformation_effect",
        name="Misinformation Effect",
        category=BiasCategory.MEMORY_BIAS,
        description="Post-event information distorting memory of the original event",
        theoretical_basis="Loftus & Palmer (1974) - Car crash speed estimation studies",
        system1_mechanism="New information integrates with original memory trace",
        system2_override="Distinguish between original observations and later acquired information",
        classic_paradigm="Leading questions changing eyewitness memory of car crash speed",
        trigger_template="After observing {event}, you learn that {post_event_information}. Describe what you originally saw."
    ),
    BiasDefinition(
        id="memory_reconstruction_bias",
        name="Memory Reconstruction Bias",
        category=BiasCategory.MEMORY_BIAS,
        description="Reconstructing past attitudes and beliefs to match current ones",
        theoretical_basis="Ross (1989) - Studies on attitude change and memory",
        system1_mechanism="Current beliefs serve as anchors for reconstructing past beliefs",
        system2_override="Consult records and external evidence of past positions",
        classic_paradigm="People who changed opinions believing they always held new view",
        trigger_template="Your current position on {topic} is {current_position}. What was your position {time_ago}?"
    ),
]


# =============================================================================
# ATTENTION BIASES (3 biases)
# Selective focus on certain information while ignoring other relevant data
# =============================================================================

ATTENTION_BIASES = [
    BiasDefinition(
        id="attentional_bias",
        name="Attentional Bias",
        category=BiasCategory.ATTENTION_BIAS,
        description="Preferential attention to certain types of stimuli based on emotional relevance",
        theoretical_basis="MacLeod et al. (1986) - Emotional Stroop task studies",
        system1_mechanism="Emotionally relevant information captures attention automatically",
        system2_override="Consciously redirect attention to all relevant information systematically",
        classic_paradigm="Anxious individuals attend more to threatening stimuli",
        trigger_template="Review this information about {topic}: {positive_info} and {negative_info}. Which aspects are most relevant?"
    ),
    BiasDefinition(
        id="inattentional_blindness",
        name="Inattentional Blindness",
        category=BiasCategory.ATTENTION_BIAS,
        description="Failing to notice unexpected stimuli when attention is focused elsewhere",
        theoretical_basis="Simons & Chabris (1999) - Gorilla experiment",
        system1_mechanism="Focused attention creates perceptual blindspots for unexpected information",
        system2_override="Deliberately scan for unexpected information outside focal area",
        classic_paradigm="Observers counting basketball passes miss person in gorilla suit",
        trigger_template="While focused on {primary_task}, what other information might you be missing about {context}?"
    ),
    BiasDefinition(
        id="selective_perception",
        name="Selective Perception",
        category=BiasCategory.ATTENTION_BIAS,
        description="Filtering information based on expectations, allowing only expected info through",
        theoretical_basis="Hastorf & Cantril (1954) - Dartmouth-Princeton football study",
        system1_mechanism="Expectations shape what information is perceived and encoded",
        system2_override="Actively seek information that contradicts expectations",
        classic_paradigm="Fans from opposing teams seeing different fouls in same game",
        trigger_template="Given your expectations about {topic}, review this data: {ambiguous_data}. What patterns do you see?"
    ),
]


# =============================================================================
# ATTRIBUTION BIASES (3 biases)
# Errors in explaining causes of behavior and events
# =============================================================================

ATTRIBUTION_BIASES = [
    BiasDefinition(
        id="fundamental_attribution_error",
        name="Fundamental Attribution Error",
        category=BiasCategory.ATTRIBUTION_BIAS,
        description="Overemphasizing dispositional factors and underemphasizing situational factors when explaining others' behavior",
        theoretical_basis="Ross (1977) - The intuitive psychologist; Jones & Harris (1967) - Castro essay study",
        system1_mechanism="Dispositional attributions are cognitively simpler than situational analysis",
        system2_override="Systematically consider situational constraints and pressures on behavior",
        classic_paradigm="Observers attributing essay positions to writers even when positions were assigned",
        trigger_template="{person} did {action} in {situation}. Why do you think they did this?"
    ),
    BiasDefinition(
        id="actor_observer_bias",
        name="Actor-Observer Bias",
        category=BiasCategory.ATTRIBUTION_BIAS,
        description="Attributing own behavior to situations but others' behavior to dispositions",
        theoretical_basis="Jones & Nisbett (1971) - Divergent perceptions of causes of behavior",
        system1_mechanism="Different information available from actor vs observer perspectives",
        system2_override="Adopt the other person's perspective when making attributions",
        classic_paradigm="Students explain their own behavior situationally but peers' behavior dispositionally",
        trigger_template="You did {your_action} because of {situation}. Why did {other_person} do the same thing?"
    ),
    BiasDefinition(
        id="self_serving_bias",
        name="Self-Serving Bias",
        category=BiasCategory.ATTRIBUTION_BIAS,
        description="Attributing successes to internal factors and failures to external factors",
        theoretical_basis="Miller & Ross (1975) - Self-serving biases in attribution",
        system1_mechanism="Motivation to maintain positive self-image shapes causal attribution",
        system2_override="Apply same attribution standards to self as to others",
        classic_paradigm="Students attribute good grades to ability, poor grades to unfair tests",
        trigger_template="Your {outcome_type} on {task} was due to what factors?"
    ),
]


# =============================================================================
# UNCERTAINTY JUDGMENT BIASES (3 biases)
# Errors in assessing and responding to uncertainty
# =============================================================================

UNCERTAINTY_JUDGMENT_BIASES = [
    BiasDefinition(
        id="ambiguity_aversion",
        name="Ambiguity Aversion",
        category=BiasCategory.UNCERTAINTY_JUDGMENT,
        description="Preference for known risks over unknown risks, even when expected values are equal",
        theoretical_basis="Ellsberg (1961) - Risk, ambiguity, and the Savage axioms",
        system1_mechanism="Unknown probabilities trigger stronger negative affect than known risks",
        system2_override="Calculate expected values regardless of whether probabilities are known or estimated",
        classic_paradigm="Ellsberg paradox: preferring known 50/50 urn over ambiguous urn",
        trigger_template="Choose between Option A with {known_probability}% chance of {outcome} or Option B with unknown probability of the same outcome."
    ),
    BiasDefinition(
        id="illusion_of_validity",
        name="Illusion of Validity",
        category=BiasCategory.UNCERTAINTY_JUDGMENT,
        description="Overconfidence in predictions based on coherent but unreliable information",
        theoretical_basis="Kahneman & Tversky (1973) - On the psychology of prediction",
        system1_mechanism="Internal consistency of information creates unwarranted confidence",
        system2_override="Assess base rates and reliability of predictors independently of coherence",
        classic_paradigm="Interviewers confident in predictions despite low validity of interviews",
        trigger_template="Based on this {coherent_description}, predict {outcome} and rate your confidence.",
        is_kt_core=True,
    ),
    BiasDefinition(
        id="competence_hypothesis",
        name="Competence Hypothesis",
        category=BiasCategory.UNCERTAINTY_JUDGMENT,
        description="Preference for betting on outcomes in domains where one feels knowledgeable",
        theoretical_basis="Heath & Tversky (1991) - Preference and belief: Ambiguity and competence in choice under uncertainty",
        system1_mechanism="Feeling of competence reduces perceived ambiguity even without better information",
        system2_override="Evaluate objective information quality regardless of subjective domain knowledge",
        classic_paradigm="Sports fans prefer betting on their team's games despite no informational advantage",
        trigger_template="Would you prefer to bet on {familiar_domain} or {unfamiliar_domain} given equal odds?",
        is_kt_core=True,
    ),
]


# =============================================================================
# SOCIAL BIASES - EXTENDED (3 biases)
# Additional social judgment biases
# =============================================================================

SOCIAL_BIASES_EXTENDED = [
    BiasDefinition(
        id="ingroup_bias",
        name="Ingroup Bias",
        category=BiasCategory.SOCIAL_BIAS,
        description="Favoring members of one's own group over outgroup members",
        theoretical_basis="Tajfel & Turner (1979) - Social identity theory; Minimal group paradigm",
        system1_mechanism="Group membership triggers automatic positive associations for ingroup",
        system2_override="Evaluate individuals based on relevant attributes, not group membership",
        classic_paradigm="Minimal group experiments: favoritism based on arbitrary group assignment",
        trigger_template="Evaluate these two candidates: {ingroup_member} from your {group} and {outgroup_member} from {other_group}."
    ),
    BiasDefinition(
        id="false_consensus_effect",
        name="False Consensus Effect",
        category=BiasCategory.SOCIAL_BIAS,
        description="Overestimating how much others share one's own beliefs, attitudes, and behaviors",
        theoretical_basis="Ross, Greene & House (1977) - The false consensus effect",
        system1_mechanism="Own perspective is more available than others' perspectives",
        system2_override="Actively sample and consider diverse viewpoints before estimating consensus",
        classic_paradigm="Subjects who chose to wear sign estimated more others would also choose to wear it",
        trigger_template="You prefer {your_preference}. What percentage of people do you think share this preference?"
    ),
    BiasDefinition(
        id="outgroup_homogeneity_bias",
        name="Outgroup Homogeneity Bias",
        category=BiasCategory.SOCIAL_BIAS,
        description="Perceiving outgroup members as more similar to each other than ingroup members",
        theoretical_basis="Quattrone & Jones (1980) - They all look alike effect",
        system1_mechanism="Less exposure to outgroup creates undifferentiated mental representation",
        system2_override="Seek information about individual differences within outgroups",
        classic_paradigm="'They all look alike' - difficulty distinguishing faces of other races",
        trigger_template="How similar are members of {outgroup} to each other compared to members of {ingroup}?"
    ),
]


# =============================================================================
# ADDITIONAL K&T BIASES (2 biases)
# Core Kahneman-Tversky biases filling gaps in framework
# =============================================================================

ADDITIONAL_KT_BIASES = [
    BiasDefinition(
        id="affect_heuristic",
        name="Affect Heuristic",
        category=BiasCategory.PROBABILITY_DISTORTION,
        description="Using emotional reactions as a shortcut for complex judgments about risk and benefit",
        theoretical_basis="Slovic et al. (2002) - The affect heuristic; Finucane et al. (2000)",
        system1_mechanism="Affective tags attached to stimuli guide probability and utility judgments",
        system2_override="Separate emotional reactions from objective probability and consequence assessment",
        classic_paradigm="Nuclear power: negative affect leads to high risk AND low benefit ratings (inverse correlation)",
        trigger_template="How do you feel about {topic}? Now estimate its risks and benefits."
    ),
    BiasDefinition(
        id="mental_accounting",
        name="Mental Accounting",
        category=BiasCategory.FRAMING,
        description="Treating money differently depending on its mental categorization rather than fungibility",
        theoretical_basis="Thaler (1985, 1999) - Mental accounting and consumer choice",
        system1_mechanism="Money is mentally segregated into accounts with different rules",
        system2_override="Treat all money as fungible and evaluate total wealth changes",
        classic_paradigm="Treating found money or winnings differently than earned income",
        trigger_template="You have {amount} in your {account_type}. Would you spend it on {purchase}?"
    ),
]


# =============================================================================
# COMPLETE TAXONOMY AGGREGATION
# =============================================================================

BIAS_TAXONOMY: dict[str, BiasDefinition] = {
    bias.id: bias
    for bias_list in [
        REPRESENTATIVENESS_BIASES,
        AVAILABILITY_BIASES,
        ANCHORING_BIASES,
        LOSS_AVERSION_BIASES,
        FRAMING_BIASES,
        PROBABILITY_BIASES,
        OVERCONFIDENCE_BIASES,
        CONFIRMATION_BIASES,
        TEMPORAL_BIASES,
        EXTENSION_NEGLECT_BIASES,
        MEMORY_BIASES,
        ATTENTION_BIASES,
        ATTRIBUTION_BIASES,
        UNCERTAINTY_JUDGMENT_BIASES,
        SOCIAL_BIASES_EXTENDED,
        ADDITIONAL_KT_BIASES,
    ]
    for bias in bias_list
}

# =============================================================================
# PRECOMPUTED LOOKUPS
# Built once at module load for O(1) category and type access
# =============================================================================

BIASES_BY_CATEGORY: dict[BiasCategory, list[BiasDefinition]] = {}
KT_CORE_BIASES: list[BiasDefinition] = []
EXTENDED_BIASES: list[BiasDefinition] = []

for _bias in BIAS_TAXONOMY.values():
    BIASES_BY_CATEGORY.setdefault(_bias.category, []).append(_bias)
    if _bias.is_kt_core:
        KT_CORE_BIASES.append(_bias)
    else:
        EXTENDED_BIASES.append(_bias)
del _bias  # Clean up loop variable


def get_bias_by_id(bias_id: str) -> BiasDefinition | None:
    """Retrieve a bias definition by its ID."""
    return BIAS_TAXONOMY.get(bias_id)


def get_biases_by_category(category: BiasCategory) -> list[BiasDefinition]:
    """Get all biases belonging to a specific category."""
    return BIASES_BY_CATEGORY.get(category, [])


def get_all_bias_ids() -> list[str]:
    """Get a list of all bias IDs in the taxonomy."""
    return list(BIAS_TAXONOMY.keys())


def get_kt_core_biases() -> list[BiasDefinition]:
    """Get all biases directly authored by Kahneman & Tversky.

    Returns biases where is_kt_core=True, meaning the bias was documented
    in papers where Kahneman and/or Tversky were authors.
    """
    return KT_CORE_BIASES


def get_extended_biases() -> list[BiasDefinition]:
    """Get all biases not directly from K&T but theoretically related.

    Returns biases where is_kt_core=False, meaning the bias was documented
    by other researchers but is theoretically connected to dual-process theory.
    """
    return EXTENDED_BIASES


# =============================================================================
# BIAS INTERACTION MATRIX
# Defines theoretically meaningful bias combinations for meso-scale testing
# =============================================================================

BIAS_INTERACTION_MATRIX: dict[str, list[str]] = {
    # Anchoring interacts with many biases
    "anchoring_effect": [
        "availability_bias",  # Available anchor values are more influential
        "overconfidence_effect",  # Confidence in anchored estimates
        "confirmation_bias",  # Seeking info confirming anchored value
        "insufficient_adjustment",  # Anchor prevents adequate adjustment
    ],
    # Availability compounds with framing and probability
    "availability_bias": [
        "neglect_of_probability",  # Vivid events seem more probable
        "gain_loss_framing",  # Available losses feel more salient
        "salience_bias",  # Amplification of vivid information
        "affect_heuristic",  # Emotional availability guides judgment
    ],
    # Loss aversion amplifies framing
    "loss_aversion": [
        "gain_loss_framing",  # Frame determines gain/loss coding
        "status_quo_bias",  # Current state as reference point
        "endowment_effect",  # Owned items framed as potential losses
        "sunk_cost_fallacy",  # Prior losses increase commitment
    ],
    # Overconfidence interacts with confirmation
    "overconfidence_effect": [
        "confirmation_bias",  # Seeking evidence supporting confident belief
        "planning_fallacy",  # Overconfidence in project estimates
        "hindsight_bias",  # Confidence that outcome was predictable
        "illusion_of_validity",  # Coherent info breeds overconfidence
    ],
    # Framing compounds with probability distortion
    "gain_loss_framing": [
        "certainty_effect",  # Sure gains vs risky losses
        "probability_weighting",  # Frame affects probability perception
        "risk_framing",  # Probability format affects decisions
        "mental_accounting",  # Frame determines mental account
    ],
    # Representativeness compounds with base rate neglect
    "base_rate_neglect": [
        "stereotype_bias",  # Stereotypes override base rates
        "conjunction_fallacy",  # Representativeness beats probability
        "insensitivity_to_sample_size",  # Small samples seem representative
    ],
    # Temporal biases compound
    "present_bias": [
        "duration_neglect",  # Immediate moment overweighted
        "planning_fallacy",  # Underestimating future time
        "optimism_bias",  # Future seems rosier
    ],
    # Confirmation bias cluster
    "confirmation_bias": [
        "myside_bias",  # Supporting own position
        "belief_perseverance",  # Maintaining beliefs despite disconfirmation
        "overconfidence_effect",  # Confidence reinforces confirmation seeking
        "selective_perception",  # Only seeing confirming evidence
    ],
    # Hindsight bias cluster
    "hindsight_bias": [
        "overconfidence_effect",  # "I knew it" leads to overconfidence
        "illusion_of_control",  # Retrospective control attribution
        "memory_reconstruction_bias",  # Memory changed to match outcome
    ],
    # Sunk cost interactions
    "sunk_cost_fallacy": [
        "loss_aversion",  # Losses from abandonment feel painful
        "status_quo_bias",  # Continuing current course
        "endowment_effect",  # Invested effort creates ownership
    ],
    # Extension neglect cluster
    "scope_insensitivity": [
        "identifiable_victim_effect",  # Individual vs statistical lives
        "neglect_of_probability",  # Magnitude neglected
        "affect_heuristic",  # Emotional response independent of scope
    ],
    # Memory distortion cluster
    "memory_reconstruction_bias": [
        "hindsight_bias",  # Retroactive distortion
        "rosy_retrospection",  # Positive memory bias
        "self_serving_bias",  # Memory serves self-image
    ],
    # Attention-availability link
    "attentional_bias": [
        "salience_bias",  # Attention to salient features
        "availability_bias",  # Attended info more available
        "selective_perception",  # Filtering based on attention
    ],
    # Social judgment cluster
    "halo_effect": [
        "stereotype_bias",  # Generalizing from category
        "group_attribution_bias",  # Group-level traits applied to individuals
        "ingroup_bias",  # Positive halo for ingroup members
    ],
    # Probability distortion cluster
    "certainty_effect": [
        "zero_risk_bias",  # Preference for elimination
        "probability_weighting",  # Overweighting small probabilities
        "affect_heuristic",  # Certainty feels emotionally better
    ],
    # Attribution bias cluster
    "fundamental_attribution_error": [
        "actor_observer_bias",  # Self vs other attribution difference
        "self_serving_bias",  # Protecting self-image in attributions
        "group_attribution_bias",  # Group-level attributions
    ],
    # Social bias cluster
    "ingroup_bias": [
        "outgroup_homogeneity_bias",  # They all look alike
        "false_consensus_effect",  # Assuming ingroup agreement
        "halo_effect",  # Positive halo for ingroup
    ],
    # Uncertainty cluster
    "ambiguity_aversion": [
        "competence_hypothesis",  # Familiar domains less ambiguous
        "illusion_of_validity",  # Coherence reduces perceived ambiguity
        "affect_heuristic",  # Ambiguity feels uncomfortable
    ],

    # =========================================================================
    # EXPANDED INTERACTION MATRIX - Additional entries for 60%+ coverage
    # =========================================================================

    # REFERENCE_DEPENDENCE - Critical gap (was 0/1)
    "reference_point_framing": [
        "loss_aversion",  # Reference determines gain/loss coding
        "status_quo_bias",  # Current state as reference
        "gain_loss_framing",  # Frame manipulates reference
        "endowment_effect",  # Ownership creates reference point
    ],

    # REPRESENTATIVENESS gaps
    "gambler_fallacy": [
        "hot_hand_fallacy",  # Opposite errors about randomness
        "insensitivity_to_sample_size",  # Both involve sequence judgments
        "regression_neglect",  # Both misunderstand random processes
    ],
    "hot_hand_fallacy": [
        "gambler_fallacy",  # Mirror biases
        "illusion_of_control",  # Both attribute pattern to skill
        "overconfidence_effect",  # Confidence in streaks
    ],
    "regression_neglect": [
        "gambler_fallacy",  # Both misunderstand regression
        "illusion_of_control",  # Misattributing randomness
        "hindsight_bias",  # Retrospective pattern-finding
    ],
    "prototype_heuristic": [
        "stereotype_bias",  # Both use prototypes
        "base_rate_neglect",  # Typicality overrides base rates
        "conjunction_fallacy",  # Representativeness mechanism
    ],

    # AVAILABILITY gaps
    "recency_bias": [
        "availability_bias",  # Recency increases availability
        "anchoring_effect",  # Recent info as anchor
        "primacy_bias",  # Order effects in general
    ],
    "primacy_bias": [
        "anchoring_effect",  # First info as anchor
        "recency_bias",  # Opposite order effect
        "availability_bias",  # First info more available
    ],
    "illusory_correlation": [
        "availability_bias",  # Memorable pairings
        "confirmation_bias",  # Seeking confirming instances
        "stereotype_bias",  # Group-based correlations
    ],
    "simulation_heuristic": [
        "availability_bias",  # Ease of imagination
        "planning_fallacy",  # Imagining positive outcomes
        "optimism_bias",  # Easy to imagine success
    ],

    # ANCHORING gaps
    "first_offer_anchoring": [
        "anchoring_effect",  # Core mechanism
        "loss_aversion",  # Anchors affect loss perception
        "status_quo_bias",  # First offer as status quo
    ],
    "focalism": [
        "anchoring_effect",  # Focus on salient info
        "salience_bias",  # Attention to focal information
        "planning_fallacy",  # Focus on current project details
    ],
    "numeric_priming": [
        "anchoring_effect",  # Incidental anchors
        "availability_bias",  # Primed numbers more available
        "insufficient_adjustment",  # Adjustment from primed anchor
    ],

    # FRAMING gaps
    "attribute_framing": [
        "gain_loss_framing",  # Both involve frame effects
        "affect_heuristic",  # Positive/negative frames trigger affect
        "reference_point_framing",  # Frame sets reference
    ],
    "default_effect": [
        "status_quo_bias",  # Default as status quo
        "loss_aversion",  # Switching from default feels like loss
        "endowment_effect",  # Default feels like owned
    ],
    "temporal_framing": [
        "present_bias",  # Time period perception
        "duration_neglect",  # Ignoring time aspects
        "mental_accounting",  # Time-based accounts
    ],

    # PROBABILITY gaps
    "denominator_neglect": [
        "neglect_of_probability",  # Both ignore probability components
        "scope_insensitivity",  # Ignoring magnitude
        "probability_weighting",  # Distorted probability processing
    ],
    "possibility_effect": [
        "certainty_effect",  # Both from prospect theory
        "probability_weighting",  # Overweighting small probs
        "zero_risk_bias",  # Eliminating possibility
    ],

    # LOSS_AVERSION gaps
    "disposition_effect": [
        "loss_aversion",  # Core mechanism
        "sunk_cost_fallacy",  # Holding losers as sunk cost
        "endowment_effect",  # Owned stocks valued more
    ],

    # TEMPORAL gaps
    "peak_end_rule": [
        "duration_neglect",  # Both about experience evaluation
        "recency_bias",  # End is recent
        "salience_bias",  # Peak is salient
    ],

    # MEMORY gaps
    "misinformation_effect": [
        "source_confusion",  # Memory source errors
        "memory_reconstruction_bias",  # Memory modification
        "hindsight_bias",  # Retroactive distortion
    ],
    "source_confusion": [
        "misinformation_effect",  # Memory distortion
        "false_consensus_effect",  # Misattributing beliefs
        "availability_bias",  # Available but misattributed
    ],

    # ATTENTION gaps
    "inattentional_blindness": [
        "attentional_bias",  # Attention allocation
        "selective_perception",  # Filtering information
        "focalism",  # Focus creates blindspots
    ],
}
