# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kahne-Bench is a cognitive bias benchmark framework for evaluating Large Language Models, grounded in Kahneman-Tversky dual-process theory. It tests 69 cognitive biases across 5 ecological domains with 6 advanced metrics.

**Quick Start:** See `examples/basic_usage.py` for a complete demo with mock provider, or `examples/openai_evaluation.py` for production usage.

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests
PYTHONPATH=src uv run pytest

# Run a single test file
PYTHONPATH=src uv run pytest tests/test_generator.py

# Run a specific test
PYTHONPATH=src uv run pytest tests/test_generator.py::TestTestCaseGenerator::test_generate_instance_returns_valid_instance

# Run with verbose output
PYTHONPATH=src uv run pytest -v

# Run the basic demo
PYTHONPATH=src uv run python examples/basic_usage.py

# Run OpenAI evaluation (requires OPENAI_API_KEY)
PYTHONPATH=src uv run python examples/openai_evaluation.py --model gpt-5.2 --tier core
```

### CLI Commands

```bash
# List all 69 biases
PYTHONPATH=src uv run kahne-bench list-biases

# List all bias categories
PYTHONPATH=src uv run kahne-bench list-categories

# Show detailed bias information
PYTHONPATH=src uv run kahne-bench describe anchoring_effect

# Generate test instances
PYTHONPATH=src uv run kahne-bench generate --bias anchoring_effect --domain INDIVIDUAL

# Generate compound (meso-scale) test instances
PYTHONPATH=src uv run kahne-bench generate-compound --bias anchoring_effect --bias availability_bias

# Run full evaluation pipeline (requires API key)
PYTHONPATH=src uv run kahne-bench evaluate -i test_cases.json -p openai -m gpt-5.2

# Generate cognitive fingerprint report
PYTHONPATH=src uv run kahne-bench report fingerprint.json

# Show framework information
PYTHONPATH=src uv run kahne-bench info
```

### Code Quality

```bash
# Format code (line-length: 100)
black src/ tests/ examples/

# Lint (target: py310)
ruff check src/ tests/

# Type check (strict mode)
mypy src/
```

## Architecture

### Core Data Flow

1. **Bias Taxonomy** (`biases/taxonomy.py`) → Defines 69 biases with theoretical grounding
2. **Test Generation** (`engines/generator.py`) → Creates test instances from templates + domain scenarios
3. **LLM Evaluation** (`engines/evaluator.py`) → Runs tests via async provider protocol
4. **Metric Calculation** (`metrics/core.py`) → Computes 6 metrics from results

### Key Abstractions

**LLMProvider Protocol** (`engines/evaluator.py`): Any LLM can be tested by implementing:
```python
async def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str
```

**Built-in Providers** (`engines/evaluator.py`):
- `OpenAIProvider`: Uses AsyncOpenAI client, configurable model (also used for Fireworks via OpenAI-compatible API)
- `AnthropicProvider`: Uses AsyncAnthropic client
- `XAIProvider`: Uses xai-sdk, sync wrapped with `asyncio.to_thread()`
- `GeminiProvider`: Uses google-genai, sync wrapped with `asyncio.to_thread()`

## Frontier Models (January 2026)

This section tracks the models to benchmark. Models with extended thinking/reasoning capabilities are tested twice: once with minimal reasoning budget and once with full reasoning.

### Models to Benchmark

| # | Model | Provider | Model ID | Reasoning Variants |
|---|-------|----------|----------|-------------------|
| 1 | **Claude Opus 4.5** | Anthropic | `claude-opus-4-5-20251101` | 0 thinking budget, full thinking budget |
| 2 | **GPT-5.2** | OpenAI | `gpt-5.2-2025-12-11` | No reasoning effort, high reasoning effort |
| 3 | **GLM 4.7** | Fireworks | `accounts/fireworks/models/glm-4p7` | - |
| 4 | **MiniMax M2P1** | Fireworks | `accounts/fireworks/models/minimax-m2p1` | - |
| 5 | **Gemini 3 Pro** | Google | `gemini-3-pro-preview` | - |
| 6 | **Claude Sonnet 4.5** | Anthropic | `claude-sonnet-4-5-20250929` | - |
| 7 | **DeepSeek V3.2** | Fireworks | `accounts/fireworks/models/deepseek-v3p2` | - |
| 8 | **Kimi K2** | Fireworks | `accounts/fireworks/models/kimi-k2-thinking` | - |
| 9 | **Grok 4.1 Fast** | xAI | `grok-4-1-fast-reasoning` | - |

### Provider Support Status

| Provider | CLI Flag | Status | API Compatibility | Models |
|----------|----------|--------|-------------------|--------|
| Anthropic | `-p anthropic` | ✅ Ready | Native SDK | `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929` |
| OpenAI | `-p openai` | ✅ Ready | Native SDK | `gpt-5.2-2025-12-11` |
| Fireworks | `-p fireworks` | ✅ Ready | OpenAI-compatible | `glm-4p7`, `minimax-m2p1`, `deepseek-v3p2`, `kimi-k2-thinking` |
| xAI | `-p xai` | ✅ Ready | Native SDK (`xai-sdk`) | `grok-4-1-fast-reasoning` |
| Google | `-p gemini` | ✅ Ready | Native SDK (`google-genai`) | `gemini-3-pro-preview` |

### API Configuration

```python
# Fireworks (GLM, MiniMax, DeepSeek, Kimi) - OpenAI-compatible
from openai import AsyncOpenAI
client = AsyncOpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1"
)
provider = OpenAIProvider(client=client, model="accounts/fireworks/models/glm-4p7")

# xAI Grok - Native SDK (pip install xai-sdk)
from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600,  # Extended timeout for reasoning models
)
chat = client.chat.create(model="grok-4-1-fast-reasoning")
chat.append(system("You are a helpful assistant."))
chat.append(user(prompt))
response = chat.sample()
# response.content contains the answer

# Google Gemini - Native SDK (pip install google-genai)
from google import genai

client = genai.Client()  # Uses GOOGLE_API_KEY env var
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents=prompt,
)
# response.text contains the answer
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # Claude Opus 4.5, Claude Sonnet 4.5
export OPENAI_API_KEY="sk-..."           # GPT-5.2
export FIREWORKS_API_KEY="fw_..."        # GLM 4.7, MiniMax, DeepSeek V3.2, Kimi K2
export GOOGLE_API_KEY="..."              # Gemini 3 Pro (after implementing)
export XAI_API_KEY="xai-..."             # Grok 4.1 Fast
```

### Recommended Evaluation Commands

```bash
# Claude Opus 4.5 (no thinking)
PYTHONPATH=src uv run kahne-bench evaluate -p anthropic -m claude-opus-4-5-20251101 --trials 3

# Claude Opus 4.5 (full thinking) - requires extended_thinking support
PYTHONPATH=src uv run kahne-bench evaluate -p anthropic -m claude-opus-4-5-20251101 --extended-thinking --trials 3

# GPT-5.2 (no reasoning)
PYTHONPATH=src uv run kahne-bench evaluate -p openai -m gpt-5.2-2025-12-11 --trials 3

# GPT-5.2 (high reasoning) - requires reasoning_effort support
PYTHONPATH=src uv run kahne-bench evaluate -p openai -m gpt-5.2-2025-12-11 --reasoning-effort high --trials 3

# Claude Sonnet 4.5
PYTHONPATH=src uv run kahne-bench evaluate -p anthropic -m claude-sonnet-4-5-20250929 --trials 3

# GLM 4.7 (via Fireworks)
PYTHONPATH=src uv run kahne-bench evaluate -p fireworks \
    -m accounts/fireworks/models/glm-4p7 --trials 3

# MiniMax M2P1 (via Fireworks)
PYTHONPATH=src uv run kahne-bench evaluate -p fireworks \
    -m accounts/fireworks/models/minimax-m2p1 --trials 3

# DeepSeek V3.2 (via Fireworks)
PYTHONPATH=src uv run kahne-bench evaluate -p fireworks \
    -m accounts/fireworks/models/deepseek-v3p2 --trials 3

# Kimi K2 (via Fireworks)
PYTHONPATH=src uv run kahne-bench evaluate -p fireworks \
    -m accounts/fireworks/models/kimi-k2-thinking --trials 3

# Grok 4.1 Fast Reasoning (via xAI)
PYTHONPATH=src uv run kahne-bench evaluate -p xai -m grok-4-1-fast-reasoning --trials 3
```

### Model Notes

| Model | Notes |
|-------|-------|
| Claude Opus 4.5 | Extended thinking available via `budget_tokens` parameter |
| GPT-5.2 | Reasoning effort available via `reasoning_effort` parameter (low/medium/high) |
| Kimi K2 | Thinking model - includes chain-of-thought by default |
| Grok 4.1 Fast | Uses `xai-sdk`; sync SDK wrapped with `asyncio.to_thread()` |
| DeepSeek V3.2 | MoE architecture, cost-effective |
| Gemini 3 Pro | Uses `google-genai`; sync SDK wrapped with `asyncio.to_thread()` |

**CognitiveBiasInstance** (`core.py`): Central test case type containing:
- Control prompt (baseline, no bias trigger)
- Treatment prompts keyed by `TriggerIntensity` (WEAK, MODERATE, STRONG, ADVERSARIAL)
- Expected rational and biased responses
- Optional debiasing prompts for meta-scale testing
- Cross-domain variants for efficient multi-domain testing

**TestResult** (`core.py`): Evaluation output containing:
- Model response and extracted answer
- `is_biased` flag and `bias_score` (0-1 magnitude)
- `confidence_stated` for metacognition analysis
- Response timing and metadata

**EvaluationConfig** (`engines/evaluator.py`): Key configuration options:
- `num_trials`: Trials per condition (default: 3, used for RCI calculation)
- `intensities`: Which trigger levels to test
- `include_control` / `include_debiasing`: Enable/disable conditions
- `max_concurrent_requests`: Semaphore-based concurrency control (default: 50)

**Benchmark Tiers** (`engines/generator.py`):
- CORE: 15 foundational biases for quick evaluation
- EXTENDED: All 69 biases
- INTERACTION: Bias pairs for compound effect testing

### The 6 Metrics

| Metric | Class | Purpose |
|--------|-------|---------|
| BMS | `BiasMagnitudeScore` | Strength of bias (weighted by trigger intensity) |
| BCI | `BiasConsistencyIndex` | Cross-domain consistency + systematic prevalence |
| BMP | `BiasMitigationPotential` | System 2 override capacity with debiasing prompts |
| HAS | `HumanAlignmentScore` | Comparison to human baselines from research literature |
| RCI | `ResponseConsistencyIndex` | Trial-to-trial variance (distinguishes noise from systematic bias) |
| CAS | `CalibrationAwarenessScore` | Metacognitive accuracy (confidence vs actual performance) |

### Module Dependencies

```
kahne_bench/
├── core.py              # Core types + context sensitivity (no dependencies)
├── cli.py               # Click-based CLI with evaluate/report/generate commands
├── biases/
│   └── taxonomy.py      # 69 BiasDefinition instances (depends on core)
├── engines/
│   ├── generator.py     # TestCaseGenerator, NovelScenarioGenerator, MacroScaleGenerator
│   ├── evaluator.py     # BiasEvaluator, TemporalEvaluator, ContextSensitivityEvaluator
│   ├── judge.py         # LLMJudge fallback scoring with XML parsing
│   ├── compound.py      # Meso-scale testing (depends on generator)
│   ├── bloom_generator.py # LLM-driven BLOOM scenario generation
│   ├── variation.py     # Prompt variation and robustness testing
│   ├── quality.py       # Test quality assessment with LLM judge
│   ├── conversation.py  # Multi-turn conversational bias evaluation
│   └── robustness.py    # Adversarial testing
├── metrics/
│   └── core.py          # All 6 metric classes + MetricCalculator
└── utils/
    ├── io.py            # JSON/CSV export functions
    └── diversity.py     # Dataset validation (self-BLEU, ROUGE)
```

## Bias Taxonomy

### 16 Bias Categories

| Category | Description | Example Biases |
|----------|-------------|----------------|
| REPRESENTATIVENESS | Judging by similarity to prototypes | base_rate_neglect, conjunction_fallacy |
| AVAILABILITY | Judging by ease of recall | availability_bias, recency_bias |
| ANCHORING | Over-reliance on initial information | anchoring_effect, insufficient_adjustment |
| LOSS_AVERSION | Losses loom larger than gains | loss_aversion, endowment_effect |
| FRAMING | Decisions affected by presentation | gain_loss_framing, attribute_framing, default_effect |
| REFERENCE_DEPENDENCE | Outcomes evaluated relative to reference points | reference_point_framing |
| PROBABILITY_DISTORTION | Misweighting probabilities | probability_weighting, certainty_effect |
| UNCERTAINTY_JUDGMENT | Errors in assessing uncertainty | overconfidence_effect, illusion_of_control |
| MEMORY_BIAS | Systematic distortions in recall | hindsight_bias, rosy_retrospection |
| ATTENTION_BIAS | Selective focus on certain information | salience_bias, focalism |
| SOCIAL_BIAS | Biases in social judgments | stereotype_bias, ingroup_bias |
| ATTRIBUTION_BIAS | Errors in explaining causes | fundamental_attribution_error |
| OVERCONFIDENCE | Excessive certainty in judgments | planning_fallacy, illusion_of_validity |
| CONFIRMATION | Seeking confirming evidence | confirmation_bias, belief_perseverance |
| TEMPORAL_BIAS | Biases related to time perception | present_bias, duration_neglect |
| EXTENSION_NEGLECT | Ignoring sample size and scope | scope_insensitivity, identifiable_victim_effect |

**Note:** All 69 biases with full definitions (K&T theoretical basis, System 1 mechanism, System 2 override, classic paradigm) are in `biases/taxonomy.py`.

## Ecological Domains

| Domain | Description | Typical Decisions |
|--------|-------------|-------------------|
| INDIVIDUAL | Personal finance, consumer choice, lifestyle | Investment, purchases, health |
| PROFESSIONAL | Managerial, medical, legal decisions | Hiring, diagnosis, case assessment |
| SOCIAL | Negotiation, persuasion, collaboration | Offers, influence, team dynamics |
| TEMPORAL | Long-term planning, delayed gratification | Retirement, project timelines |
| RISK | Policy, technology, environmental uncertainty | Safety protocols, innovation |

## Test Scales & Intensities

### Test Scales

| Scale | Purpose |
|-------|---------|
| MICRO | Single isolated bias, control vs treatment comparison |
| MESO | Multiple bias interactions in complex scenarios |
| MACRO | Bias persistence across sequential related decisions |
| META | Self-correction and debiasing capacity testing |

### Trigger Intensities

| Intensity | Weight | Rationale |
|-----------|--------|-----------|
| WEAK | 2.0x | High susceptibility if subtle triggers cause bias |
| MODERATE | 1.0x | Baseline standard trigger |
| STRONG | 0.67x | Expected that strong pressure causes deviation |
| ADVERSARIAL | 0.5x | Compound triggers, lowest weight |

**Philosophy:** The weighting reflects susceptibility, not trigger strength. A model vulnerable to weak anchors is more biased than one requiring strong pressure.

## Temporal Testing

**TemporalCondition** enum (`core.py`):
- `IMMEDIATE`: Instant response, System 1 dominant
- `DELIBERATIVE`: With explicit reflection time
- `PERSISTENT`: Bias stability across sequential prompts
- `ADAPTIVE`: Pre/post feedback comparison

**TemporalEvaluator** (`engines/evaluator.py`): Extends BiasEvaluator with:
- `evaluate_persistent()`: Tests bias evolution over sequential decisions (5 rounds default)
- `evaluate_adaptive()`: Pre/post feedback testing for learning effects

**ContextSensitivityEvaluator** (`engines/evaluator.py`): Tests how context affects bias:
- `evaluate_context_sensitivity()`: Tests all context combinations (6 preset configs)
- `evaluate_expertise_gradient()`: Isolates expertise level effects (NOVICE → AUTHORITY)
- `evaluate_stakes_gradient()`: Isolates stakes level effects (LOW → CRITICAL)

## Advanced Generators

**NovelScenarioGenerator** (`engines/generator.py`): Contamination-resistant testing:
- Uses futuristic professions (quantum computing architect, space debris analyst, etc.)
- Uses novel contexts unlikely in training data (Mars colonization, AI governance boards)
- `generate_novel_instance()`: Single contamination-resistant test
- `generate_contamination_resistant_batch()`: Batch generation for all bias-domain pairs

**MacroScaleGenerator** (`engines/generator.py`): Sequential decision chain testing:
- `generate_decision_chain()`: Creates multi-turn bias persistence tests
- Bias-specific chain generators for anchoring, prospect theory, confirmation, overconfidence
- `DecisionNode` and `DecisionChain` dataclasses for structured chain representation

## Context Sensitivity

**Context Types** (`core.py`):
- `ExpertiseLevel`: NOVICE, INTERMEDIATE, EXPERT, AUTHORITY
- `Formality`: CASUAL, PROFESSIONAL, FORMAL, ACADEMIC
- `Stakes`: LOW, MODERATE, HIGH, CRITICAL

**ContextSensitivityConfig** (`core.py`): Wraps prompts with context framing:
- `get_expertise_prefix()`: Generates role descriptions
- `get_formality_framing()`: Generates setting descriptions
- `get_stakes_emphasis()`: Generates stakes descriptions

**CognitiveBiasInstance** methods for context:
- `apply_context_sensitivity()`: Wraps prompts with context framing
- `get_context_variant()`: Gets treatment with specific context overrides

## Testing Patterns

- Tests use `pytest` with `pytest-asyncio` for async evaluator tests
- Generator tests use `seed=42` for reproducibility
- Test files mirror source structure: `test_generator.py` tests `engines/generator.py`
- No `conftest.py` - fixtures are defined locally within test files
- Temporary file tests use try/finally for cleanup

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `test_evaluator.py` | ~112 | Extraction, scoring, frame-aware, temporal, context |
| `test_generator.py` | ~71 | Instance generation, batch, tiers, intensity, framing |
| `test_advanced_generators.py` | ~63 | Novel scenarios, macro chains |
| `test_metrics.py` | ~55 | All 6 metrics, MetricCalculator, guardrails |
| `test_io.py` | ~26 | JSON/CSV export/import roundtrips |
| `test_taxonomy.py` | ~24 | Bias definitions, interaction matrix |
| `test_conversation.py` | ~20 | Multi-turn conversational evaluation |
| `test_variation.py` | ~20 | Robustness and variation dimensions |
| `test_bloom_generator.py` | ~19 | BLOOM LLM-driven generation |
| `test_judge.py` | ~11 | LLM judge fallback scoring |
| `test_quality.py` | ~9 | Test quality assessment |
| `test_integration.py` | ~6 | End-to-end workflows |

**Total:** ~548 tests

## Key Design Decisions

1. **Intensity weighting in BMS**: Weak triggers causing bias are weighted higher (2.0x) than strong triggers (0.67x) - a model susceptible to weak triggers is more biased than one only affected by strong pressure

2. **Human baselines** in `metrics/core.py` (`HUMAN_BASELINES` dict): Research-backed susceptibility rates from K&T literature for 40+ biases, enabling human-AI alignment scoring

3. **Template-based generation**: `BIAS_TEMPLATES` and `DOMAIN_SCENARIOS` in generator.py enable consistent test creation across all bias×domain combinations

4. **Async-first evaluation**: `BiasEvaluator.evaluate_batch()` is async with rate limiting built in (`requests_per_minute` config)

5. **Placeholder answers**: Expected answers starting with `[` are treated as non-evaluable (default score: 0.5)

## Dependencies

### Core
- `openai>=1.0.0` - OpenAI API client (also used for Fireworks)
- `anthropic>=0.18.0` - Anthropic API client
- `xai-sdk>=0.1.0` - xAI/Grok API client
- `google-genai>=0.1.0` - Google Gemini API client
- `pandas>=2.0.0` - Data analysis
- `numpy>=1.24.0` - Numerical computing
- `rich>=13.0.0` - Terminal output formatting
- `click>=8.0.0` - CLI framework

### Development
- `pytest>=7.0.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Linting
- `mypy>=1.0.0` - Type checking

**Python:** 3.10, 3.11, 3.12 supported

## Examples

| File | Purpose |
|------|---------|
| `examples/basic_usage.py` | Full demo with MockProvider - taxonomy, generation, evaluation, metrics |
| `examples/openai_evaluation.py` | Production usage with CLI args: `--model`, `--tier`, `--domains`, `--trials` |

**Environment:** Set `OPENAI_API_KEY` for OpenAI examples, `ANTHROPIC_API_KEY` for Anthropic.

**Note:** No CI/CD configuration exists currently. Run tests locally before committing.
