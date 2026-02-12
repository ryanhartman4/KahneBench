"""Tests for the test quality meta-metrics module."""

from __future__ import annotations

import pytest

from kahne_bench.core import (
    CognitiveBiasInstance,
    Domain,
    TriggerIntensity,
)
from kahne_bench.engines.quality import (
    QualityJudge,
    QualityScores,
    QualityReport,
    _extract_xml_tag,
)


# ---------------------------------------------------------------------------
# Mock provider and response fixtures
# ---------------------------------------------------------------------------

MOCK_HIGH_QUALITY_RESPONSE = (
    "<realism>8</realism>\n"
    "<realism_justification>Realistic financial scenario.</realism_justification>\n"
    "<elicitation_difficulty>7</elicitation_difficulty>\n"
    "<elicitation_justification>Moderate difficulty with subtle anchor."
    "</elicitation_justification>\n"
    "<detection_awareness>3</detection_awareness>\n"
    "<detection_justification>Not obviously a bias test."
    "</detection_justification>"
)

MOCK_LOW_QUALITY_RESPONSE = (
    "<realism>2</realism>\n"
    "<realism_justification>Contrived example.</realism_justification>\n"
    "<elicitation_difficulty>2</elicitation_difficulty>\n"
    "<elicitation_justification>Very obvious trigger.</elicitation_justification>\n"
    "<detection_awareness>9</detection_awareness>\n"
    "<detection_justification>Clearly a bias test.</detection_justification>"
)


class MockLLMProvider:
    """Mock provider returning pre-configured responses."""

    def __init__(self, response: str = MOCK_HIGH_QUALITY_RESPONSE):
        self.response = response
        self.call_count = 0

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.call_count += 1
        return self.response


def _make_instance(
    bias_id: str = "anchoring_effect",
    domain: Domain = Domain.INDIVIDUAL,
) -> CognitiveBiasInstance:
    """Create a minimal test instance."""
    return CognitiveBiasInstance(
        bias_id=bias_id,
        base_scenario="A financial decision scenario.",
        bias_trigger="An initial price anchor.",
        control_prompt="What would you estimate the price to be?",
        treatment_prompts={
            TriggerIntensity.WEAK: "The initial listing was $500k. Estimate the price.",
            TriggerIntensity.MODERATE: "The initial listing was $500k. Estimate the price.",
            TriggerIntensity.STRONG: "The initial listing was $500k. Estimate the price.",
            TriggerIntensity.ADVERSARIAL: "The listing was $500k. Estimate the price.",
        },
        expected_rational_response="$350,000",
        expected_biased_response="$480,000",
        domain=domain,
    )


# ---------------------------------------------------------------------------
# QualityScores tests
# ---------------------------------------------------------------------------


class TestOverallQualityComputation:
    """Tests for the composite quality formula."""

    def test_overall_quality_computation(self):
        """Verify formula: realism*0.3 + difficulty*0.3 + subtlety*0.4."""
        # realism=6, difficulty=8, awareness=4
        # 6/10*0.3 + 8/10*0.3 + (10-4)/10*0.4 = 0.18 + 0.24 + 0.24 = 0.66
        result = QualityScores.compute_overall(6.0, 8.0, 4.0)
        assert abs(result - 0.66) < 1e-9

    def test_overall_quality_perfect_score(self):
        """realism=10, difficulty=10, awareness=1 -> ~0.96."""
        result = QualityScores.compute_overall(10.0, 10.0, 1.0)
        # 1.0*0.3 + 1.0*0.3 + 9/10*0.4 = 0.3 + 0.3 + 0.36 = 0.96
        assert abs(result - 0.96) < 1e-9

    def test_overall_quality_worst_score(self):
        """realism=1, difficulty=1, awareness=10 -> 0.06."""
        result = QualityScores.compute_overall(1.0, 1.0, 10.0)
        # 0.1*0.3 + 0.1*0.3 + 0.0*0.4 = 0.03 + 0.03 + 0.0 = 0.06
        assert abs(result - 0.06) < 1e-9


# ---------------------------------------------------------------------------
# XML parsing tests
# ---------------------------------------------------------------------------


class TestXMLParsing:
    """Tests for XML tag extraction."""

    def test_parse_valid_xml(self):
        """All 6 XML tags correctly extracted."""
        assert _extract_xml_tag(MOCK_HIGH_QUALITY_RESPONSE, "realism") == "8"
        assert (
            _extract_xml_tag(
                MOCK_HIGH_QUALITY_RESPONSE, "elicitation_difficulty"
            )
            == "7"
        )
        assert (
            _extract_xml_tag(
                MOCK_HIGH_QUALITY_RESPONSE, "detection_awareness"
            )
            == "3"
        )
        assert (
            _extract_xml_tag(
                MOCK_HIGH_QUALITY_RESPONSE, "realism_justification"
            )
            == "Realistic financial scenario."
        )
        assert (
            _extract_xml_tag(
                MOCK_HIGH_QUALITY_RESPONSE, "elicitation_justification"
            )
            == "Moderate difficulty with subtle anchor."
        )
        assert (
            _extract_xml_tag(
                MOCK_HIGH_QUALITY_RESPONSE, "detection_justification"
            )
            == "Not obviously a bias test."
        )

    def test_parse_missing_tag(self):
        """Missing tag returns None."""
        assert _extract_xml_tag("no tags here", "realism") is None


# ---------------------------------------------------------------------------
# QualityJudge tests
# ---------------------------------------------------------------------------


class TestQualityJudgeAssessment:
    """Tests for single-instance quality assessment."""

    @pytest.mark.asyncio
    async def test_assess_instance_high_quality(self):
        """Scores parsed correctly for a high-quality test."""
        provider = MockLLMProvider(MOCK_HIGH_QUALITY_RESPONSE)
        judge = QualityJudge(provider=provider)
        instance = _make_instance()

        scores = await judge.assess_instance(instance)

        assert scores.realism == 8.0
        assert scores.elicitation_difficulty == 7.0
        assert scores.detection_awareness == 3.0
        assert scores.overall_quality == pytest.approx(
            QualityScores.compute_overall(8.0, 7.0, 3.0)
        )
        assert scores.instance_id == "anchoring_effect_individual"

    @pytest.mark.asyncio
    async def test_assess_instance_low_quality(self):
        """Low realism + high detection -> low overall."""
        provider = MockLLMProvider(MOCK_LOW_QUALITY_RESPONSE)
        judge = QualityJudge(provider=provider)
        instance = _make_instance()

        scores = await judge.assess_instance(instance)

        assert scores.realism == 2.0
        assert scores.detection_awareness == 9.0
        assert scores.overall_quality < 0.4

    @pytest.mark.asyncio
    async def test_parse_missing_required_tag_returns_defaults(self):
        """Missing required tag returns default scores instead of crashing."""
        provider = MockLLMProvider("<realism>5</realism>")
        judge = QualityJudge(provider=provider)
        instance = _make_instance()

        scores = await judge.assess_instance(instance)
        assert scores.realism == 5.0
        assert scores.elicitation_difficulty == 5.0
        assert scores.overall_quality == 0.5

    def test_parse_quality_response_raises_on_missing_tag(self):
        """_parse_quality_response still raises ValueError for missing tags."""
        judge = QualityJudge(provider=MockLLMProvider(""))
        with pytest.raises(ValueError, match="elicitation_difficulty"):
            judge._parse_quality_response("<realism>5</realism>")

    @pytest.mark.asyncio
    async def test_score_clamping(self):
        """Values outside 1-10 are clamped to bounds."""
        response = (
            "<realism>15</realism>\n"
            "<realism_justification>Too high.</realism_justification>\n"
            "<elicitation_difficulty>-3</elicitation_difficulty>\n"
            "<elicitation_justification>Too low.</elicitation_justification>\n"
            "<detection_awareness>0</detection_awareness>\n"
            "<detection_justification>Below min.</detection_justification>"
        )
        provider = MockLLMProvider(response)
        judge = QualityJudge(provider=provider)
        instance = _make_instance()

        scores = await judge.assess_instance(instance)

        assert scores.realism == 10.0
        assert scores.elicitation_difficulty == 1.0
        assert scores.detection_awareness == 1.0

    def test_instance_id_format(self):
        """Instance ID is 'bias_id_domain_value'."""
        instance = _make_instance(
            bias_id="loss_aversion", domain=Domain.PROFESSIONAL
        )
        expected_id = "loss_aversion_professional"
        assert (
            f"{instance.bias_id}_{instance.domain.value}" == expected_id
        )


# ---------------------------------------------------------------------------
# Batch assessment tests
# ---------------------------------------------------------------------------


class TestBatchAssessment:
    """Tests for batch quality assessment."""

    @pytest.mark.asyncio
    async def test_batch_assessment(self):
        """Batch runs on sampled subset and returns report."""
        provider = MockLLMProvider(MOCK_HIGH_QUALITY_RESPONSE)
        judge = QualityJudge(
            provider=provider, sample_rate=1.0
        )
        instances = [
            _make_instance(domain=Domain.INDIVIDUAL),
            _make_instance(domain=Domain.PROFESSIONAL),
            _make_instance(domain=Domain.SOCIAL),
        ]

        report = await judge.assess_batch(instances)

        assert report.total_instances == 3
        assert report.assessed_instances == 3
        assert len(report.instance_scores) == 3
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_sampling_rate(self):
        """With 0.5 rate, ~50% of 20 instances assessed."""
        provider = MockLLMProvider(MOCK_HIGH_QUALITY_RESPONSE)
        judge = QualityJudge(
            provider=provider, sample_rate=0.5
        )
        instances = [
            _make_instance(domain=Domain.INDIVIDUAL)
            for _ in range(20)
        ]

        report = await judge.assess_batch(instances)

        assert report.total_instances == 20
        assert report.assessed_instances == 10
        assert provider.call_count == 10

    @pytest.mark.asyncio
    async def test_report_mean_scores(self):
        """Aggregate means computed correctly."""
        provider = MockLLMProvider(MOCK_HIGH_QUALITY_RESPONSE)
        judge = QualityJudge(
            provider=provider, sample_rate=1.0
        )
        instances = [
            _make_instance(domain=Domain.INDIVIDUAL),
            _make_instance(domain=Domain.PROFESSIONAL),
        ]

        report = await judge.assess_batch(instances)

        # All responses are the same mock, so means == individual values
        assert report.mean_realism == pytest.approx(8.0)
        assert report.mean_elicitation_difficulty == pytest.approx(7.0)
        assert report.mean_detection_awareness == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_quality_distribution(self):
        """Report correctly counts high/medium/low."""
        # High quality mock: overall = 8*0.3/10 + 7*0.3/10 + 7*0.4/10
        #   = 0.24 + 0.21 + 0.28 = 0.73 -> high
        provider = MockLLMProvider(MOCK_HIGH_QUALITY_RESPONSE)
        judge = QualityJudge(
            provider=provider, sample_rate=1.0
        )
        instances = [_make_instance()]

        report = await judge.assess_batch(instances)

        # overall = 8/10*0.3 + 7/10*0.3 + 7/10*0.4 = 0.24 + 0.21 + 0.28 = 0.73
        assert report.quality_distribution["high"] == 1
        assert report.quality_distribution["medium"] == 0
        assert report.quality_distribution["low"] == 0


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


class TestFilterLowQuality:
    """Tests for low-quality instance filtering."""

    def test_filter_low_quality(self):
        """Low quality instances removed correctly."""
        judge = QualityJudge(
            provider=MockLLMProvider(), quality_threshold=0.4
        )
        instances = [
            _make_instance(domain=Domain.INDIVIDUAL),
            _make_instance(domain=Domain.PROFESSIONAL),
        ]

        report = QualityReport(
            total_instances=2,
            assessed_instances=2,
            mean_realism=5.0,
            mean_elicitation_difficulty=5.0,
            mean_detection_awareness=5.0,
            mean_overall_quality=0.35,
            low_quality_instances=["anchoring_effect_individual"],
            quality_distribution={"high": 0, "medium": 1, "low": 1},
            instance_scores=[],
        )

        filtered = judge.filter_low_quality(instances, report)

        assert len(filtered) == 1
        assert filtered[0].domain == Domain.PROFESSIONAL

    def test_filter_preserves_unassessed(self):
        """Instances not assessed are NOT filtered."""
        judge = QualityJudge(
            provider=MockLLMProvider(), quality_threshold=0.4
        )
        instances = [
            _make_instance(domain=Domain.INDIVIDUAL),
            _make_instance(domain=Domain.PROFESSIONAL),
            _make_instance(domain=Domain.SOCIAL),
        ]

        # Only one instance was assessed and flagged as low quality
        report = QualityReport(
            total_instances=3,
            assessed_instances=1,
            mean_realism=2.0,
            mean_elicitation_difficulty=2.0,
            mean_detection_awareness=9.0,
            mean_overall_quality=0.2,
            low_quality_instances=["anchoring_effect_individual"],
            quality_distribution={"high": 0, "medium": 0, "low": 1},
            instance_scores=[],
        )

        filtered = judge.filter_low_quality(instances, report)

        # Only the flagged one is removed; professional and social kept
        assert len(filtered) == 2
        domains = {inst.domain for inst in filtered}
        assert Domain.PROFESSIONAL in domains
        assert Domain.SOCIAL in domains
