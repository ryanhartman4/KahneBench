"""
Core data structures for Kahne-Bench.

This module defines the fundamental types that represent cognitive bias tests,
including the CognitiveBiasInstance class that encapsulates all elements of
a single test case.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BiasCategory(Enum):
    """
    Categories of cognitive biases based on Kahneman-Tversky dual-process theory.

    These categories reflect the underlying cognitive mechanisms that produce biases,
    organized around key concepts from Prospect Theory and heuristics research.
    """
    # Judgment heuristics - mental shortcuts that often lead to systematic errors
    REPRESENTATIVENESS = "representativeness"  # Judging by similarity to prototypes
    AVAILABILITY = "availability"  # Judging by ease of recall
    ANCHORING = "anchoring"  # Over-reliance on initial information

    # Prospect theory - how people evaluate gains and losses
    LOSS_AVERSION = "loss_aversion"  # Losses loom larger than gains
    FRAMING = "framing"  # Decisions affected by presentation
    REFERENCE_DEPENDENCE = "reference_dependence"  # Outcomes evaluated relative to reference points

    # Probability and uncertainty
    PROBABILITY_DISTORTION = "probability_distortion"  # Misweighting probabilities
    UNCERTAINTY_JUDGMENT = "uncertainty_judgment"  # Errors in assessing uncertainty

    # Memory and attention
    MEMORY_BIAS = "memory_bias"  # Systematic distortions in recall
    ATTENTION_BIAS = "attention_bias"  # Selective focus on certain information

    # Social and attribution
    SOCIAL_BIAS = "social_bias"  # Biases in social judgments
    ATTRIBUTION_BIAS = "attribution_bias"  # Errors in explaining causes

    # Overconfidence and calibration
    OVERCONFIDENCE = "overconfidence"  # Excessive certainty in judgments
    CONFIRMATION = "confirmation"  # Seeking confirming evidence

    # Temporal reasoning
    TEMPORAL_BIAS = "temporal_bias"  # Biases related to time perception

    # Extension and scope
    EXTENSION_NEGLECT = "extension_neglect"  # Ignoring sample size and scope


class Domain(Enum):
    """
    Real-world domains for ecological validity in bias testing.

    Testing biases across multiple domains ensures the benchmark measures
    how biases manifest in practical, high-stakes situations.
    """
    INDIVIDUAL = "individual"  # Personal finance, consumer choice, lifestyle
    PROFESSIONAL = "professional"  # Managerial, medical, legal decisions
    SOCIAL = "social"  # Negotiation, persuasion, collaboration
    TEMPORAL = "temporal"  # Long-term planning, investments, delayed gratification
    RISK = "risk"  # Policy, technology, environmental uncertainty


class TestScale(Enum):
    """
    Multi-scale testing methodology for comprehensive bias evaluation.

    Each scale provides a different lens on cognitive bias manifestation.
    """
    MICRO = "micro"  # Single isolated bias, control vs treatment
    MESO = "meso"  # Multiple bias interactions in complex scenarios
    MACRO = "macro"  # Bias persistence across sequential decisions
    META = "meta"  # Self-correction and debiasing capacity


class TriggerIntensity(Enum):
    """
    Intensity levels for bias triggers.

    Varying intensity enables calculation of the Bias Magnitude Score (BMS)
    by measuring response changes across different trigger strengths.
    """
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    ADVERSARIAL = "adversarial"  # Compound bias pressure from multiple triggers


class TemporalCondition(Enum):
    """
    Temporal conditions for testing bias dynamics.

    These conditions simulate different reasoning modes and track
    how biases evolve over time.
    """
    IMMEDIATE = "immediate"  # Fast System 1 response
    DELIBERATIVE = "deliberative"  # Extended System 2 reasoning
    PERSISTENT = "persistent"  # Bias tested across prompt sequences
    ADAPTIVE = "adaptive"  # Response to corrective feedback


class ExpertiseLevel(Enum):
    """
    Expertise levels for context sensitivity testing.

    Tests how domain expertise affects bias expression, as documented
    in Section 4.2 of the Kahne-Bench specification.
    """
    NOVICE = "novice"  # Minimal domain knowledge
    INTERMEDIATE = "intermediate"  # Some domain experience
    EXPERT = "expert"  # Deep domain expertise
    AUTHORITY = "authority"  # Recognized authority in the field


class Formality(Enum):
    """
    Formality levels for context sensitivity testing.

    Tests how setting formality affects bias expression.
    """
    CASUAL = "casual"  # Informal conversation
    PROFESSIONAL = "professional"  # Standard business setting
    FORMAL = "formal"  # High-stakes formal setting
    ACADEMIC = "academic"  # Scholarly/research context


class Stakes(Enum):
    """
    Stakes levels for context sensitivity testing.

    Tests how perceived stakes affect bias expression.
    """
    LOW = "low"  # Minor consequences
    MODERATE = "moderate"  # Moderate impact
    HIGH = "high"  # Significant consequences
    CRITICAL = "critical"  # Life-or-death or catastrophic consequences


@dataclass
class ContextSensitivityConfig:
    """
    Configuration for context sensitivity testing.

    Captures the contextual factors that may influence bias expression,
    as specified in Section 4.2 of the Kahne-Bench documentation.
    """
    expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    formality: Formality = Formality.PROFESSIONAL
    stakes: Stakes = Stakes.MODERATE

    def get_expertise_prefix(self) -> str:
        """Get role description based on expertise level."""
        prefixes = {
            ExpertiseLevel.NOVICE: "You are new to this field with minimal experience. You are learning and may need to rely on basic principles.",
            ExpertiseLevel.INTERMEDIATE: "You have several years of experience in this domain and solid foundational knowledge.",
            ExpertiseLevel.EXPERT: "You are a seasoned expert with decades of experience. You have deep knowledge and have seen many similar situations.",
            ExpertiseLevel.AUTHORITY: "You are a world-renowned authority in this field. Your judgment is highly respected and your decisions set precedent.",
        }
        return prefixes[self.expertise_level]

    def get_formality_framing(self) -> str:
        """Get situational framing based on formality level."""
        framings = {
            Formality.CASUAL: "In a casual conversation with a colleague,",
            Formality.PROFESSIONAL: "In a standard business meeting,",
            Formality.FORMAL: "In a formal boardroom presentation to senior executives,",
            Formality.ACADEMIC: "In an academic peer-review context requiring rigorous analysis,",
        }
        return framings[self.formality]

    def get_stakes_emphasis(self) -> str:
        """Get stakes emphasis text."""
        emphases = {
            Stakes.LOW: "This decision has minor implications.",
            Stakes.MODERATE: "This decision has moderate consequences for the stakeholders involved.",
            Stakes.HIGH: "This is a high-stakes decision with significant financial and reputational consequences.",
            Stakes.CRITICAL: "This is a critical decision with potentially catastrophic or life-altering consequences. Every detail matters.",
        }
        return emphases[self.stakes]


@dataclass
class BiasDefinition:
    """
    Complete definition of a cognitive bias with theoretical grounding.
    """
    id: str
    name: str
    category: BiasCategory
    description: str
    theoretical_basis: str  # Link to K&T research
    system1_mechanism: str  # How System 1 produces this bias
    system2_override: str  # How System 2 can correct it
    classic_paradigm: str  # Famous experimental demonstration
    trigger_template: str  # Template for creating bias triggers


@dataclass
class CognitiveBiasInstance:
    """
    A single test case for evaluating a cognitive bias.

    This class encapsulates all elements needed for multi-scale testing,
    cross-domain validation, and metric calculation.

    Attributes:
        bias_id: Identifier for the bias being tested
        base_scenario: Neutral, unbiased context for the decision
        bias_trigger: The manipulation designed to elicit the bias
        control_prompt: Baseline version for rational response
        treatment_prompts: Variants with bias triggers at different intensities
        expected_rational_response: The objectively correct answer
        expected_biased_response: The response a biased agent would give
        domain: The real-world context for ecological validity
        scale: The testing scale (micro, meso, macro, meta)
        cross_domain_variants: Adapted scenarios for other domains
        debiasing_prompts: Prompts to engage System 2 reasoning
        interaction_biases: For meso-scale compound bias testing
        context_config: Context sensitivity configuration (expertise, formality, stakes)
        metadata: Additional information about the test case
    """
    bias_id: str
    base_scenario: str
    bias_trigger: str
    control_prompt: str
    treatment_prompts: dict[TriggerIntensity, str]
    expected_rational_response: str
    expected_biased_response: str
    domain: Domain
    scale: TestScale = TestScale.MICRO
    cross_domain_variants: dict[Domain, str] = field(default_factory=dict)
    debiasing_prompts: list[str] = field(default_factory=list)
    interaction_biases: list[str] = field(default_factory=list)  # For meso-scale
    context_config: ContextSensitivityConfig | None = None  # Context sensitivity settings
    metadata: dict = field(default_factory=dict)

    def get_treatment(self, intensity: TriggerIntensity) -> str:
        """Get the treatment prompt for a specific intensity level."""
        return self.treatment_prompts.get(intensity, self.control_prompt)

    def has_debiasing(self) -> bool:
        """Check if this instance includes debiasing prompts."""
        return len(self.debiasing_prompts) > 0

    def is_compound(self) -> bool:
        """Check if this is a compound (meso-scale) test with multiple biases."""
        return len(self.interaction_biases) > 0

    def apply_context_sensitivity(
        self,
        prompt: str,
        config: ContextSensitivityConfig | None = None,
    ) -> str:
        """
        Apply context sensitivity framing to a prompt.

        This implements Section 4.2 of the Kahne-Bench specification,
        testing how expertise, formality, and stakes affect bias expression.

        Args:
            prompt: The base prompt to enhance
            config: Context configuration (uses instance config if not provided)

        Returns:
            Enhanced prompt with context sensitivity framing
        """
        cfg = config or self.context_config
        if cfg is None:
            return prompt

        context_frame = f"""{cfg.get_expertise_prefix()}

{cfg.get_formality_framing()} you are asked to make a decision.

{cfg.get_stakes_emphasis()}

"""
        return context_frame + prompt

    def get_context_variant(
        self,
        intensity: TriggerIntensity,
        expertise: ExpertiseLevel | None = None,
        formality: Formality | None = None,
        stakes: Stakes | None = None,
    ) -> str:
        """
        Get a treatment prompt with specific context sensitivity settings.

        Args:
            intensity: Trigger intensity level
            expertise: Override expertise level
            formality: Override formality level
            stakes: Override stakes level

        Returns:
            Context-enhanced treatment prompt
        """
        base_prompt = self.get_treatment(intensity)

        # Build config from overrides or defaults
        cfg = ContextSensitivityConfig(
            expertise_level=expertise or (
                self.context_config.expertise_level
                if self.context_config else ExpertiseLevel.INTERMEDIATE
            ),
            formality=formality or (
                self.context_config.formality
                if self.context_config else Formality.PROFESSIONAL
            ),
            stakes=stakes or (
                self.context_config.stakes
                if self.context_config else Stakes.MODERATE
            ),
        )

        return self.apply_context_sensitivity(base_prompt, cfg)


@dataclass
class TestResult:
    """
    Result of running a single bias test on an LLM.
    """
    instance: CognitiveBiasInstance
    model_id: str
    condition: str  # "control" or intensity level
    prompt_used: str
    model_response: str
    extracted_answer: str
    response_time_ms: float
    confidence_stated: Optional[float] = None  # If model states confidence
    is_biased: Optional[bool] = None  # Determined by evaluation
    bias_score: Optional[float] = None  # Magnitude of bias exhibited
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationSession:
    """
    A complete evaluation session for a model across multiple test cases.
    """
    session_id: str
    model_id: str
    model_config: dict
    test_instances: list[CognitiveBiasInstance]
    results: list[TestResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
