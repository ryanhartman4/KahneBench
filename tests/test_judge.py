"""Tests for the LLM-as-Judge fallback scoring module."""

import pytest

from kahne_bench.engines.judge import (
    LLMJudge,
    JudgeResult,
    JUDGE_SCORING_PROMPT,
    _extract_xml_tag,
)


class MockJudgeProvider:
    """Mock LLM provider that returns pre-configured responses."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []

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
        return self.response


# Shared kwargs for judge.score() calls
_SCORE_KWARGS = dict(
    bias_name="anchoring_effect",
    bias_description="Estimates biased toward initial reference value",
    system1_mechanism="Initial value activates related concepts",
    control_prompt="Estimate the population of Chicago.",
    treatment_prompt=(
        "A recent survey mentioned 10 million. "
        "Estimate the population of Chicago."
    ),
    expected_rational="2700000",
    expected_biased="5000000",
    answer_type="numeric",
)


def _make_xml(
    answer: str = "Option B",
    bias_score: str = "0.85",
    confidence: str = "0.9",
    justification: str = "The model chose the biased option.",
) -> str:
    """Build a well-formed XML judge response."""
    return (
        f"<extracted_answer>{answer}</extracted_answer>\n"
        f"<bias_score>{bias_score}</bias_score>\n"
        f"<confidence>{confidence}</confidence>\n"
        f"<justification>{justification}</justification>"
    )


# ---------- Core scoring tests ----------


@pytest.mark.asyncio
async def test_score_biased_response():
    """Judge correctly identifies a biased response (bias_score > 0.5)."""
    provider = MockJudgeProvider(_make_xml(bias_score="0.85"))
    judge = LLMJudge(provider=provider)
    result = await judge.score(
        **_SCORE_KWARGS,
        model_response="Based on the information, I estimate about 5 million.",
    )
    assert result.is_biased is True
    assert result.bias_score == 0.85


@pytest.mark.asyncio
async def test_score_rational_response():
    """Judge correctly identifies a rational response (bias_score < 0.5)."""
    provider = MockJudgeProvider(_make_xml(bias_score="0.1"))
    judge = LLMJudge(provider=provider)
    result = await judge.score(
        **_SCORE_KWARGS,
        model_response="Chicago has roughly 2.7 million residents.",
    )
    assert result.is_biased is False
    assert result.bias_score == 0.1


@pytest.mark.asyncio
async def test_score_ambiguous_response():
    """Judge handles middle-ground responses (bias_score ~ 0.5)."""
    provider = MockJudgeProvider(_make_xml(bias_score="0.5"))
    judge = LLMJudge(provider=provider)
    result = await judge.score(
        **_SCORE_KWARGS,
        model_response="Hard to say; perhaps around 3.5 million.",
    )
    # 0.5 is NOT > 0.5, so is_biased should be False
    assert result.is_biased is False
    assert result.bias_score == 0.5


# ---------- XML parsing tests ----------


def test_parse_valid_xml():
    """All XML tags correctly extracted."""
    provider = MockJudgeProvider("")
    judge = LLMJudge(provider=provider)
    result = judge._parse_judge_response(_make_xml(
        answer="Option A",
        bias_score="0.3",
        confidence="0.8",
        justification="Rational reasoning observed.",
    ))
    assert result.extracted_answer == "Option A"
    assert result.bias_score == 0.3
    assert result.confidence == 0.8
    assert result.justification == "Rational reasoning observed."


def test_parse_missing_bias_score_tag():
    """Raises ValueError when required bias_score tag is missing."""
    provider = MockJudgeProvider("")
    judge = LLMJudge(provider=provider)
    xml = (
        "<extracted_answer>A</extracted_answer>\n"
        "<confidence>0.9</confidence>\n"
        "<justification>Some reason.</justification>"
    )
    with pytest.raises(ValueError, match="bias_score"):
        judge._parse_judge_response(xml)


def test_parse_missing_confidence_tag():
    """Raises ValueError when required confidence tag is missing."""
    provider = MockJudgeProvider("")
    judge = LLMJudge(provider=provider)
    xml = (
        "<extracted_answer>A</extracted_answer>\n"
        "<bias_score>0.5</bias_score>\n"
        "<justification>Some reason.</justification>"
    )
    with pytest.raises(ValueError, match="confidence"):
        judge._parse_judge_response(xml)


def test_parse_missing_justification_tag():
    """Raises ValueError when required justification tag is missing."""
    provider = MockJudgeProvider("")
    judge = LLMJudge(provider=provider)
    xml = (
        "<extracted_answer>A</extracted_answer>\n"
        "<bias_score>0.5</bias_score>\n"
        "<confidence>0.9</confidence>"
    )
    with pytest.raises(ValueError, match="justification"):
        judge._parse_judge_response(xml)


# ---------- Clamping tests ----------


def test_bias_score_clamping():
    """Values outside [0, 1] are clamped."""
    provider = MockJudgeProvider("")
    judge = LLMJudge(provider=provider)

    # Above 1.0
    result = judge._parse_judge_response(_make_xml(bias_score="1.5"))
    assert result.bias_score == 1.0

    # Below 0.0
    result = judge._parse_judge_response(_make_xml(bias_score="-0.3"))
    assert result.bias_score == 0.0


def test_confidence_clamping():
    """Values outside [0, 1] are clamped."""
    provider = MockJudgeProvider("")
    judge = LLMJudge(provider=provider)

    result = judge._parse_judge_response(_make_xml(confidence="2.0"))
    assert result.confidence == 1.0

    result = judge._parse_judge_response(_make_xml(confidence="-1.0"))
    assert result.confidence == 0.0


def test_non_numeric_bias_score_defaults_to_zero():
    """Non-numeric bias_score text is clamped to 0.0."""
    provider = MockJudgeProvider("")
    judge = LLMJudge(provider=provider)
    result = judge._parse_judge_response(_make_xml(bias_score="not a number"))
    assert result.bias_score == 0.0


# ---------- Prompt construction and provider interaction ----------


@pytest.mark.asyncio
async def test_prompt_includes_all_fields():
    """Verify the prompt sent to the provider contains all required context."""
    provider = MockJudgeProvider(_make_xml())
    judge = LLMJudge(provider=provider)
    await judge.score(
        **_SCORE_KWARGS,
        model_response="Some response text.",
    )
    sent_prompt = provider.calls[0]["prompt"]

    assert "anchoring_effect" in sent_prompt
    assert "Estimates biased toward initial reference value" in sent_prompt
    assert "Initial value activates related concepts" in sent_prompt
    assert "Estimate the population of Chicago." in sent_prompt
    assert "10 million" in sent_prompt
    assert "2700000" in sent_prompt
    assert "5000000" in sent_prompt
    assert "numeric" in sent_prompt
    assert "Some response text." in sent_prompt


@pytest.mark.asyncio
async def test_provider_called_with_correct_params():
    """Verify temperature and max_tokens are passed correctly."""
    provider = MockJudgeProvider(_make_xml())
    judge = LLMJudge(provider=provider, temperature=0.3, max_tokens=512)
    await judge.score(
        **_SCORE_KWARGS,
        model_response="response",
    )
    assert provider.calls[0]["temperature"] == 0.3
    assert provider.calls[0]["max_tokens"] == 512


# ---------- JudgeResult field tests ----------


@pytest.mark.asyncio
async def test_judge_result_fields():
    """Verify JudgeResult dataclass fields are all populated."""
    provider = MockJudgeProvider(_make_xml(
        answer="5 million",
        bias_score="0.7",
        confidence="0.85",
        justification="Anchored to initial value.",
    ))
    judge = LLMJudge(provider=provider)
    result = await judge.score(
        **_SCORE_KWARGS,
        model_response="I think about 5 million.",
    )
    assert isinstance(result, JudgeResult)
    assert result.extracted_answer == "5 million"
    assert result.bias_score == 0.7
    assert result.is_biased is True
    assert result.confidence == 0.85
    assert result.justification == "Anchored to initial value."
    assert result.scoring_method == "llm_judge"


@pytest.mark.asyncio
async def test_scoring_method_always_llm_judge():
    """Verify scoring_method is always 'llm_judge'."""
    provider = MockJudgeProvider(_make_xml(bias_score="0.0"))
    judge = LLMJudge(provider=provider)
    result = await judge.score(
        **_SCORE_KWARGS,
        model_response="Perfectly rational answer.",
    )
    assert result.scoring_method == "llm_judge"


# ---------- Edge cases ----------


@pytest.mark.asyncio
async def test_empty_model_response():
    """Handle empty/whitespace model responses gracefully."""
    provider = MockJudgeProvider(_make_xml(
        answer="unknown",
        bias_score="0.0",
        confidence="0.2",
        justification="Model produced no substantive answer.",
    ))
    judge = LLMJudge(provider=provider)
    result = await judge.score(
        **_SCORE_KWARGS,
        model_response="   ",
    )
    # The judge still produces a result; the empty response is passed through
    assert isinstance(result, JudgeResult)
    assert result.extracted_answer == "unknown"


def test_missing_extracted_answer_defaults_to_unknown():
    """When extracted_answer tag is absent, default to 'unknown'."""
    provider = MockJudgeProvider("")
    judge = LLMJudge(provider=provider)
    xml = (
        "<bias_score>0.5</bias_score>\n"
        "<confidence>0.8</confidence>\n"
        "<justification>Reason.</justification>"
    )
    result = judge._parse_judge_response(xml)
    assert result.extracted_answer == "unknown"


def test_empty_extracted_answer_defaults_to_unknown():
    """When extracted_answer tag is empty, default to 'unknown'."""
    provider = MockJudgeProvider("")
    judge = LLMJudge(provider=provider)
    xml = (
        "<extracted_answer>   </extracted_answer>\n"
        "<bias_score>0.5</bias_score>\n"
        "<confidence>0.8</confidence>\n"
        "<justification>Reason.</justification>"
    )
    result = judge._parse_judge_response(xml)
    assert result.extracted_answer == "unknown"


def test_extract_xml_tag_helper():
    """Verify the _extract_xml_tag function works correctly."""
    text = "<foo>hello world</foo>"
    assert _extract_xml_tag(text, "foo") == "hello world"
    assert _extract_xml_tag(text, "bar") is None


def test_extract_xml_tag_multiline():
    """Verify _extract_xml_tag handles multiline content."""
    text = "<justification>Line one.\nLine two.</justification>"
    assert _extract_xml_tag(text, "justification") == "Line one.\nLine two."
