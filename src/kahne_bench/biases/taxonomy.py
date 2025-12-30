"""
Complete taxonomy of 50 cognitive biases for Kahne-Bench.

Each bias is grounded in the Kahneman-Tversky research program, with clear
links to dual-process theory and prospect theory mechanisms.
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
        trigger_template="Given that {base_rate}% of the population are {category_a}, consider {vivid_description}..."
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
        trigger_template="{person_description} Which is more likely: (A) {general_category} or (B) {general_category} and {specific_detail}?"
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
        trigger_template="Hospital A delivers {large_n} babies/day, Hospital B delivers {small_n}/day. Which has more extreme days?"
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
        trigger_template="After {streak_description}, what is the probability of {opposite_outcome}?"
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
        trigger_template="{person} has succeeded in the last {n} attempts. What is the probability of success on the next attempt?"
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
        trigger_template="After an extreme {performance_type} of {extreme_value}, predict the next {performance_type}."
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
        trigger_template="{detailed_description}. What is the most likely {category} for this person?"
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
        trigger_template="Rate how typical {instance} is of {category}, then estimate probability of {instance} being {category}."
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
        trigger_template="Estimate the frequency of {easily_imagined_event} versus {less_imagined_event}."
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
        trigger_template="After {vivid_event}, estimate the probability of {related_risk}."
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
        trigger_template="Imagine {scenario}. How likely is this to occur?"
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
        trigger_template="The number {anchor} was mentioned. Now estimate {target_quantity}."
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
        trigger_template="Choose: (A) Sure gain of {amount}, or (B) 50% chance to gain {higher_amount} or lose {lower_amount}?"
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
        trigger_template="You own {item}. At what price would you sell it? If you didn't own it, what would you pay?"
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
        trigger_template="You've invested {sunk_cost} in {project}. It will require {additional_cost} to complete. Expected value is {expected_value}. Continue?"
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
        trigger_template="Program A: {gain_frame}. Program B: {equivalent_gamble}. Which do you choose?"
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
        category=BiasCategory.FRAMING,
        description="Outcomes coded as gains/losses relative to arbitrary reference",
        theoretical_basis="Kahneman & Tversky (1979) - Reference dependence in prospect theory",
        system1_mechanism="Reference point determines gain/loss coding",
        system2_override="Evaluate final wealth states, not changes",
        classic_paradigm="Tax 'bonus' vs 'refund' framing",
        trigger_template="Starting from {reference}, outcome is {change}. Evaluate this outcome."
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
        trigger_template="Option A: Certain {small_amount}. Option B: {small_prob}% chance of {large_amount}. Choose."
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
        trigger_template="Option A: {certain_outcome} for sure. Option B: {high_prob}% chance of {better_outcome}. Choose."
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
        trigger_template="There is a {tiny_prob}% chance of winning {jackpot}. How much would you pay for this lottery ticket?"
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
        trigger_template="Estimate time to complete {project}. Consider past {similar_projects}."
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
        trigger_template="Experience A: {short_duration} with {pattern_a}. Experience B: {long_duration} with {pattern_b}. Rate each."
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
        trigger_template="Experience peaked at {peak} and ended at {end}. Rate the overall experience."
    ),
]

# =============================================================================
# EXTENSION NEGLECT / SCOPE INSENSITIVITY (48-50)
# Ignoring sample size, scope, and extensional attributes
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
        trigger_template="How much would you pay to save {quantity} {item}?"
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
    ]
    for bias in bias_list
}


def get_bias_by_id(bias_id: str) -> BiasDefinition | None:
    """Retrieve a bias definition by its ID."""
    return BIAS_TAXONOMY.get(bias_id)


def get_biases_by_category(category: BiasCategory) -> list[BiasDefinition]:
    """Get all biases belonging to a specific category."""
    return [bias for bias in BIAS_TAXONOMY.values() if bias.category == category]


def get_all_bias_ids() -> list[str]:
    """Get a list of all bias IDs in the taxonomy."""
    return list(BIAS_TAXONOMY.keys())


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
    ],
    # Availability compounds with framing and probability
    "availability_bias": [
        "neglect_of_probability",  # Vivid events seem more probable
        "gain_loss_framing",  # Available losses feel more salient
        "salience_bias",  # Amplification of vivid information
    ],
    # Loss aversion amplifies framing
    "loss_aversion": [
        "gain_loss_framing",  # Frame determines gain/loss coding
        "status_quo_bias",  # Current state as reference point
        "endowment_effect",  # Owned items framed as potential losses
    ],
    # Overconfidence interacts with confirmation
    "overconfidence_effect": [
        "confirmation_bias",  # Seeking evidence supporting confident belief
        "planning_fallacy",  # Overconfidence in project estimates
        "hindsight_bias",  # Confidence that outcome was predictable
    ],
    # Framing compounds with probability distortion
    "gain_loss_framing": [
        "certainty_effect",  # Sure gains vs risky losses
        "probability_weighting",  # Frame affects probability perception
        "risk_framing",  # Probability format affects decisions
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
}
