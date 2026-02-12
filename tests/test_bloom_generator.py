"""Tests for the BLOOM-style agentic scenario generator."""

from __future__ import annotations

import pytest

from kahne_bench.core import (
    CognitiveBiasInstance,
    Domain,
    TestScale,
    TriggerIntensity,
)
from kahne_bench.biases.taxonomy import get_bias_by_id
from kahne_bench.engines.bloom_generator import (
    BloomBiasGenerator,
    BiasUnderstanding,
    GeneratedScenario,
)


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------

MOCK_UNDERSTANDING_RESPONSE = """\
<behavioral_markers>
- Estimate is disproportionately close to the anchor value
- Insufficient adjustment from initial reference point
- Numeric judgments shift toward irrelevant numbers
- Final answer correlates with presented anchor
- Respondent fails to recognize anchor influence
</behavioral_markers>

<trigger_patterns>
- Presenting an irrelevant number before asking for an estimate
- Mentioning a high or low reference price before valuation
- Providing a random percentage before a probability question
- Starting negotiation with an extreme opening offer
- Showing a previous (outdated) figure before requesting a current estimate
</trigger_patterns>

<resistance_factors>
- Explicit instruction to ignore irrelevant numbers
- Prompting for independent estimation before revealing anchor
- Providing base-rate statistics alongside the anchor
</resistance_factors>

<variation_dimensions>
- Numerical magnitude of anchor (small vs large)
- Domain expertise required
- Plausibility of the anchor value
- Time pressure on the respondent
</variation_dimensions>"""

MOCK_IDEATION_RESPONSE = """\
<scenario>
<description>Salary estimation for a software engineer role</description>
<answer_type>numeric</answer_type>
<control_prompt>
A company is hiring a senior software engineer. Based on market data, \
the average salary for this role in the region is $130,000. What is a \
fair salary offer for this position?

Answer:
</control_prompt>
<treatment_prompt>
A company is hiring a senior software engineer. A recruiter mentions \
that a previous candidate was offered $190,000 for a similar role at \
a competitor. Based on market data, the average salary for this role \
in the region is $130,000. What is a fair salary offer for this position?

Answer:
</treatment_prompt>
<bias_trigger>The $190,000 anchor from the competitor offer</bias_trigger>
<expected_rational>130000</expected_rational>
<expected_biased>160000</expected_biased>
</scenario>

<scenario>
<description>Estimating project completion time with deadline anchor</description>
<answer_type>numeric</answer_type>
<control_prompt>
A development team needs to build a new authentication module. \
Based on similar past projects, how many weeks will it take?

Answer:
</control_prompt>
<treatment_prompt>
A development team needs to build a new authentication module. \
The VP mentioned the board expects delivery in 3 weeks. Based on \
similar past projects, how many weeks will it take?

Answer:
</treatment_prompt>
<bias_trigger>The 3-week deadline anchor from the VP</bias_trigger>
<expected_rational>8</expected_rational>
<expected_biased>4</expected_biased>
</scenario>"""


MOCK_IDEATION_MALFORMED = """\
<scenario>
<description>Good scenario</description>
<answer_type>option</answer_type>
<control_prompt>Choose an investment. Answer:</control_prompt>
<treatment_prompt>A guru suggests Option A. Choose. Answer:</treatment_prompt>
<bias_trigger>Guru suggestion</bias_trigger>
<expected_rational>Option B</expected_rational>
<expected_biased>Option A</expected_biased>
</scenario>

<scenario>
<description>Missing prompts</description>
</scenario>

<scenario>
<description>Identical answers</description>
<answer_type>yes_no</answer_type>
<control_prompt>Should you invest? Answer:</control_prompt>
<treatment_prompt>Experts say yes. Should you invest? Answer:</treatment_prompt>
<bias_trigger>Expert appeal</bias_trigger>
<expected_rational>yes</expected_rational>
<expected_biased>yes</expected_biased>
</scenario>"""


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockProvider:
    """Records calls and returns canned responses."""

    def __init__(self, responses: list[str] | None = None):
        self.calls: list[dict] = []
        self._responses = list(responses or [])
        self._index = 0

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if self._responses:
            resp = self._responses[self._index % len(self._responses)]
            self._index += 1
            return resp
        return ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anchoring_def():
    """Return the anchoring_effect BiasDefinition."""
    bias = get_bias_by_id("anchoring_effect")
    assert bias is not None
    return bias


@pytest.fixture
def sample_understanding():
    """A pre-built BiasUnderstanding for tests that skip stage 1."""
    return BiasUnderstanding(
        bias_id="anchoring_effect",
        behavioral_markers=[
            "Estimate is close to anchor",
            "Insufficient adjustment",
            "Numeric shift toward irrelevant number",
        ],
        trigger_patterns=[
            "Presenting irrelevant number before estimate",
            "High reference price before valuation",
        ],
        resistance_factors=[
            "Explicit instruction to ignore anchor",
            "Independent estimation first",
        ],
        variation_dimensions=[
            "Anchor magnitude",
            "Domain expertise",
        ],
        raw_understanding=MOCK_UNDERSTANDING_RESPONSE,
    )


@pytest.fixture
def sample_scenario():
    """A pre-built GeneratedScenario for conversion tests."""
    return GeneratedScenario(
        scenario_id="abcd1234",
        description="Salary estimation for a software engineer role",
        control_prompt=(
            "A company is hiring a senior software engineer. "
            "Based on market data, the average salary is $130,000. "
            "What is a fair offer?\n\nAnswer:"
        ),
        treatment_prompt=(
            "A company is hiring a senior software engineer. "
            "A recruiter mentions a competitor offered $190,000. "
            "Based on market data, the average salary is $130,000. "
            "What is a fair offer?\n\nAnswer:"
        ),
        bias_trigger="The $190,000 anchor from the competitor offer",
        expected_rational="130000",
        expected_biased="160000",
        domain=Domain.PROFESSIONAL,
        answer_type="numeric",
    )


# ---------------------------------------------------------------------------
# Tests: _parse_understanding
# ---------------------------------------------------------------------------


class TestParseUnderstanding:
    """Tests for _parse_understanding (stage 1 parsing)."""

    def test_parse_understanding_valid(self, anchoring_def):
        gen = BloomBiasGenerator(provider=MockProvider())
        result = gen._parse_understanding(
            "anchoring_effect", MOCK_UNDERSTANDING_RESPONSE,
        )

        assert isinstance(result, BiasUnderstanding)
        assert result.bias_id == "anchoring_effect"
        assert len(result.behavioral_markers) >= 3
        assert "Estimate is disproportionately close to the anchor value" in (
            result.behavioral_markers
        )
        assert len(result.trigger_patterns) >= 3
        assert len(result.resistance_factors) >= 2
        assert len(result.variation_dimensions) >= 2
        assert result.raw_understanding == MOCK_UNDERSTANDING_RESPONSE

    def test_parse_understanding_partial(self):
        """Missing tags should produce default single-element lists."""
        gen = BloomBiasGenerator(provider=MockProvider())
        partial = "<behavioral_markers>\n- One marker\n</behavioral_markers>"
        result = gen._parse_understanding("test_bias", partial)

        assert result.behavioral_markers == ["One marker"]
        # Missing tags get defaults
        assert result.trigger_patterns == ["contextual trigger"]
        assert result.resistance_factors == ["explicit instruction"]
        assert result.variation_dimensions == ["context domain"]


# ---------------------------------------------------------------------------
# Tests: _parse_scenarios
# ---------------------------------------------------------------------------


class TestParseScenarios:
    """Tests for _parse_scenarios (stage 2 parsing)."""

    def test_parse_scenarios_valid(self):
        gen = BloomBiasGenerator(provider=MockProvider())
        scenarios = gen._parse_scenarios(
            MOCK_IDEATION_RESPONSE, Domain.PROFESSIONAL,
        )
        assert len(scenarios) == 2
        for s in scenarios:
            assert isinstance(s, GeneratedScenario)
            assert s.domain == Domain.PROFESSIONAL
            assert len(s.control_prompt) > 0
            assert len(s.treatment_prompt) > 0
            assert s.expected_rational != s.expected_biased

    def test_parse_scenarios_malformed(self):
        """Malformed scenarios (missing prompts) are silently skipped."""
        gen = BloomBiasGenerator(provider=MockProvider())
        scenarios = gen._parse_scenarios(
            MOCK_IDEATION_MALFORMED, Domain.INDIVIDUAL,
        )
        # Only the first valid scenario survives (second has missing
        # prompts, third has identical answers)
        assert len(scenarios) == 1
        assert scenarios[0].description == "Good scenario"

    def test_parse_scenarios_filters_identical_answers(self):
        """Scenarios where rational == biased are filtered out."""
        gen = BloomBiasGenerator(provider=MockProvider())
        xml = (
            "<scenario>"
            "<description>Same answers</description>"
            "<answer_type>numeric</answer_type>"
            "<control_prompt>What is X? Answer:</control_prompt>"
            "<treatment_prompt>With anchor, what is X? Answer:"
            "</treatment_prompt>"
            "<bias_trigger>anchor</bias_trigger>"
            "<expected_rational>50</expected_rational>"
            "<expected_biased>50</expected_biased>"
            "</scenario>"
        )
        scenarios = gen._parse_scenarios(xml, Domain.PROFESSIONAL)
        assert len(scenarios) == 0


# ---------------------------------------------------------------------------
# Tests: scenario_to_instance
# ---------------------------------------------------------------------------


class TestScenarioToInstance:
    """Tests for scenario_to_instance conversion."""

    def test_scenario_to_instance(
        self, anchoring_def, sample_scenario,
    ):
        gen = BloomBiasGenerator(provider=MockProvider())
        instance = gen.scenario_to_instance(
            sample_scenario, anchoring_def,
        )
        assert isinstance(instance, CognitiveBiasInstance)
        assert instance.bias_id.endswith("_anchoring_effect")
        assert instance.domain == Domain.PROFESSIONAL
        assert instance.scale == TestScale.MICRO

    def test_instance_has_all_intensities(
        self, anchoring_def, sample_scenario,
    ):
        gen = BloomBiasGenerator(provider=MockProvider())
        instance = gen.scenario_to_instance(
            sample_scenario, anchoring_def,
        )
        for intensity in TriggerIntensity:
            assert intensity in instance.treatment_prompts
            assert len(instance.treatment_prompts[intensity]) > 0

    def test_instance_has_debiasing(
        self, anchoring_def, sample_scenario,
    ):
        gen = BloomBiasGenerator(provider=MockProvider())
        instance = gen.scenario_to_instance(
            sample_scenario, anchoring_def, include_debiasing=True,
        )
        assert len(instance.debiasing_prompts) == 3

    def test_instance_no_debiasing(
        self, anchoring_def, sample_scenario,
    ):
        gen = BloomBiasGenerator(provider=MockProvider())
        instance = gen.scenario_to_instance(
            sample_scenario, anchoring_def, include_debiasing=False,
        )
        assert len(instance.debiasing_prompts) == 0

    def test_instance_metadata(
        self, anchoring_def, sample_scenario,
    ):
        gen = BloomBiasGenerator(provider=MockProvider())
        instance = gen.scenario_to_instance(
            sample_scenario, anchoring_def,
        )
        assert instance.metadata["generation_method"] == "bloom"
        assert instance.metadata["answer_type"] == "numeric"
        assert "scenario_description" in instance.metadata


# ---------------------------------------------------------------------------
# Tests: intensity variant helpers
# ---------------------------------------------------------------------------


class TestIntensityVariants:
    """Tests for _soften_treatment, _intensify_treatment, etc."""

    def test_soften_treatment(self):
        gen = BloomBiasGenerator(provider=MockProvider())
        original = "You must definitely choose the clearly best option."
        softened = gen._soften_treatment(original, "control text")
        assert "Consider the following scenario." in softened
        assert "might want to" in softened
        assert "possibly" in softened
        assert "arguably" in softened

    def test_intensify_treatment(self):
        gen = BloomBiasGenerator(provider=MockProvider())
        original = "Choose an option. Answer:"
        intensified = gen._intensify_treatment(original)
        assert "important decision" in intensified
        assert original in intensified
        assert "reliable sources" in intensified

    def test_adversarial_treatment(self, anchoring_def):
        gen = BloomBiasGenerator(provider=MockProvider())
        original = "Choose an option. Answer:"
        adversarial = gen._adversarial_treatment(
            original, anchoring_def,
        )
        assert "Pay close attention" in adversarial
        assert original in adversarial
        assert "Research consistently shows" in adversarial
        assert "Most experts" in adversarial


# ---------------------------------------------------------------------------
# Tests: understand_bias (async, calls provider)
# ---------------------------------------------------------------------------


class TestUnderstandBias:
    """Tests for the async understand_bias stage 1."""

    @pytest.mark.asyncio
    async def test_understand_bias_calls_provider(self, anchoring_def):
        mock = MockProvider(responses=[MOCK_UNDERSTANDING_RESPONSE])
        gen = BloomBiasGenerator(provider=mock, temperature=0.8)

        result = await gen.understand_bias(anchoring_def)

        assert len(mock.calls) == 1
        call = mock.calls[0]
        assert anchoring_def.name in call["prompt"]
        assert call["max_tokens"] == 2048
        assert call["temperature"] == 0.8
        assert isinstance(result, BiasUnderstanding)
        assert result.bias_id == "anchoring_effect"


# ---------------------------------------------------------------------------
# Tests: generate_scenarios (async, calls provider)
# ---------------------------------------------------------------------------


class TestGenerateScenarios:
    """Tests for the async generate_scenarios stage 2."""

    @pytest.mark.asyncio
    async def test_generate_scenarios_calls_provider(
        self, anchoring_def, sample_understanding,
    ):
        mock = MockProvider(responses=[MOCK_IDEATION_RESPONSE])
        gen = BloomBiasGenerator(
            provider=mock, temperature=0.6, max_tokens=2048,
        )

        scenarios = await gen.generate_scenarios(
            sample_understanding, anchoring_def, Domain.SOCIAL, 3,
        )

        assert len(mock.calls) == 1
        call = mock.calls[0]
        assert "social" in call["prompt"].lower()
        assert "3" in call["prompt"]
        assert call["max_tokens"] == 2048
        assert call["temperature"] == 0.6
        assert len(scenarios) == 2  # Two valid scenarios in mock


# ---------------------------------------------------------------------------
# Tests: generate_batch (async, full pipeline)
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    """Tests for the full pipeline."""

    @pytest.mark.asyncio
    async def test_generate_batch(self):
        mock = MockProvider(responses=[
            MOCK_UNDERSTANDING_RESPONSE,
            MOCK_IDEATION_RESPONSE,
        ])
        gen = BloomBiasGenerator(provider=mock, num_scenarios=2)

        instances = await gen.generate_batch(
            bias_ids=["anchoring_effect"],
            domains=[Domain.PROFESSIONAL],
            scenarios_per_bias=2,
        )

        # 1 understanding call + 1 ideation call = 2 calls
        assert len(mock.calls) == 2
        # Two valid scenarios parsed from mock response
        assert len(instances) == 2
        for inst in instances:
            assert isinstance(inst, CognitiveBiasInstance)
            assert inst.metadata["generation_method"] == "bloom"

    @pytest.mark.asyncio
    async def test_generate_batch_skips_unknown_bias(self):
        mock = MockProvider(responses=[])
        gen = BloomBiasGenerator(provider=mock)

        instances = await gen.generate_batch(
            bias_ids=["nonexistent_xyz"],
        )
        assert instances == []
        assert len(mock.calls) == 0

    @pytest.mark.asyncio
    async def test_generate_batch_multiple_domains(self):
        mock = MockProvider(responses=[
            MOCK_UNDERSTANDING_RESPONSE,
            MOCK_IDEATION_RESPONSE,
            MOCK_IDEATION_RESPONSE,
        ])
        gen = BloomBiasGenerator(provider=mock)

        instances = await gen.generate_batch(
            bias_ids=["anchoring_effect"],
            domains=[Domain.PROFESSIONAL, Domain.SOCIAL],
            scenarios_per_bias=2,
        )

        # 1 understanding + 2 ideation (one per domain) = 3 calls
        assert len(mock.calls) == 3
        # 2 scenarios per domain * 2 domains = 4
        assert len(instances) == 4


# ---------------------------------------------------------------------------
# Tests: XML helper methods
# ---------------------------------------------------------------------------


class TestXmlHelpers:
    """Tests for _extract_tag and _extract_list."""

    def test_extract_tag(self):
        gen = BloomBiasGenerator(provider=MockProvider())
        text = "<foo>  hello world  </foo>"
        assert gen._extract_tag("foo", text) == "hello world"

    def test_extract_tag_missing(self):
        gen = BloomBiasGenerator(provider=MockProvider())
        assert gen._extract_tag("missing", "no tags here") is None

    def test_extract_tag_multiline(self):
        gen = BloomBiasGenerator(provider=MockProvider())
        text = "<prompt>\nLine 1\nLine 2\n</prompt>"
        assert gen._extract_tag("prompt", text) == "Line 1\nLine 2"

    def test_extract_list(self):
        gen = BloomBiasGenerator(provider=MockProvider())
        text = (
            "<items>\n"
            "- First item\n"
            "- Second item\n"
            "- Third item\n"
            "</items>"
        )
        result = gen._extract_list("items", text)
        assert result == ["First item", "Second item", "Third item"]

    def test_extract_list_empty_tag(self):
        gen = BloomBiasGenerator(provider=MockProvider())
        assert gen._extract_list("missing", "no tags") == []

    def test_extract_list_skips_blank_lines(self):
        gen = BloomBiasGenerator(provider=MockProvider())
        text = "<items>\n- One\n\n- Two\n</items>"
        result = gen._extract_list("items", text)
        assert result == ["One", "Two"]
