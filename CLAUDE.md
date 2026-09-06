# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Kahne-Bench is a cognitive bias benchmark for Large Language Models grounded in Kahneman-Tversky dual-process theory: 69 biases, 5 ecological domains, 6 metrics. The benchmark is deprecated as of July 2026 because the newest frontier models saturated it. The leaderboard in `README.md` is final. The code stays runnable for reproduction.

## Development Commands

```bash
uv sync --group dev                     # editable install of the package plus dev tools
uv run pytest                           # full suite (649 tests, about 20 seconds)
uv run pytest tests/test_generator.py   # one file
uv run pytest tests/test_generator.py::TestTestCaseGenerator::test_generate_instance_returns_valid_instance
uv run ruff check .                     # lint
uv run ruff format .                    # format (line length 100)
uv run mypy src/                        # advisory only; not run in CI
uv run python examples/basic_usage.py   # end-to-end demo with a mock provider
uv run python scripts/verify_readme_results.py   # README tables vs results/ fingerprints
```

CI (`.github/workflows/ci.yml`) runs ruff check, ruff format --check, pytest, and the README verification script on Python 3.10 and 3.12.

`pyproject.toml` sets `pythonpath = ["src"]` for pytest. Everything else relies on the editable install from `uv sync`, which is a `.pth` file in the venv pointing at `src/`.

**macOS trap.** If the checkout lives in an iCloud-synced folder (Desktop or Documents with iCloud Drive on), iCloud marks dot-prefixed paths and everything under them as hidden a few minutes after they are written, and current CPython releases (the check is present in 3.10.19 and 3.14.1) silently skip hidden `.pth` files. The symptom is `ModuleNotFoundError: No module named 'kahne_bench'` from a venv that `uv pip show` says is fine, recurring after every `uv run` that rebuilds the package. Fix: `chflags -R nohidden .venv`, or keep the venv outside the synced tree with `UV_PROJECT_ENVIRONMENT=~/.venvs/kahne-bench`. Do not paper over it with `PYTHONPATH=src`.

## Evaluation Workflow

Two steps: generate test cases, then evaluate a model against them.

```bash
uv run kahne-bench generate --tier core --seed 42 -o test_cases.json
uv run kahne-bench evaluate -i test_cases.json -p anthropic -m claude-sonnet-4-5 -n 3 \
  -o results.json -f fingerprint.json --tier core
```

- `core_tests.json` at the repo root is the fixed input every leaderboard run used: core tier, 15 biases, 5 domains, 4,725 evaluations per model at 3 trials.
- `run/NN_<model>.sh` records the exact command for each leaderboard model. Shared flags live in `run/common.sh`. Verbose logs go to `run/logs/` (gitignored).
- The `evaluate` command's LLM-judge fallback defaults to `--judge-provider anthropic --judge-model claude-haiku-4-5`, so `ANTHROPIC_API_KEY` is required unless you override both.
- Rich progress output is buffered until the batch finishes. Pass `--verbose` for timestamped per-instance log lines instead.
- Outputs land in `results/`. Fingerprints (`fingerprint_*.json`, about 30KB each) are tracked. Raw results (`results_*.json`, about 10MB each) are gitignored.

### Results JSON key fields

- `model_response` (not `response`): the model's full reply
- `extracted_answer`: parsed answer (A/B/C or descriptive)
- `is_biased` / `bias_score`: scoring output
- `condition`: `control`, `treatment_weak|moderate|strong`, or `debiasing_0|1|2`
- `domain`: one of the 5 ecological domains

## Architecture

Data flow: `biases/taxonomy.py` defines the biases. `engines/generator.py` builds test instances from prompt templates and domain scenarios. `engines/evaluator.py` runs them through a provider and scores the responses. `metrics/core.py` turns the results into a `CognitiveFingerprintReport`.

```
src/kahne_bench/
├── core.py              # CognitiveBiasInstance, TestResult, enums, LLMProvider protocol, context sensitivity
├── cli.py               # Click CLI; run `kahne-bench --help` for the command list
├── biases/taxonomy.py   # 69 BiasDefinition instances in 16 categories, plus the interaction matrix
├── engines/
│   ├── generator.py     # BIAS_TEMPLATES, DOMAIN_SCENARIOS, TestCaseGenerator, tiers, NovelScenarioGenerator, MacroScaleGenerator
│   ├── evaluator.py     # Provider clients, AnswerExtractor, BiasEvaluator, TemporalEvaluator, ContextSensitivityEvaluator
│   ├── judge.py         # LLM judge fallback scoring
│   ├── compound.py      # Meso-scale bias interaction tests
│   ├── bloom_generator.py  # LLM-driven scenario generation (not used for the leaderboard)
│   ├── conversation.py  # Multi-turn conversational evaluation
│   ├── quality.py       # Test quality assessment
│   ├── robustness.py    # Adversarial testing
│   └── variation.py     # Prompt variation
├── metrics/core.py      # BMS, BCI, BMP, HAS, RCI, CAS, HUMAN_BASELINES, MetricCalculator
└── utils/               # io.py (JSON/CSV import and export), diversity.py (dataset validation)
```

### Key abstractions

- **`LLMProvider`** (`core.py`): any object with `async def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str`. Built-ins in `engines/evaluator.py`: `OpenAIProvider` (also serves Fireworks through `base_url`), `AnthropicProvider`, `XAIProvider`, `GeminiProvider`. The last two wrap sync SDKs in `asyncio.to_thread`.
- **`CognitiveBiasInstance`** (`core.py`): one test case. Control prompt, treatment prompts keyed by `TriggerIntensity`, expected rational and biased responses, optional debiasing prompts.
- **`TestResult`** (`core.py`): one model response with `extracted_answer`, `is_biased`, `bias_score`, `confidence_stated`.
- **`EvaluationConfig`** (`engines/evaluator.py`): `num_trials` (default 3), `intensities`, `include_control`, `include_debiasing`, `max_concurrent_requests` (default 50), rate-limit retry settings.
- **Tiers** (`engines/generator.py`): `KahneBenchTier.CORE` (15 biases), `EXTENDED` (all 69), `INTERACTION` (bias pairs).

### Key design decisions

1. **BMS intensity weights** (`DEFAULT_INTENSITY_WEIGHTS` in `metrics/core.py`): WEAK 2.0, MODERATE 1.0, STRONG 0.67, ADVERSARIAL 0.5. A model that bites on a weak trigger is more biased than one that needs strong pressure. The weights are design choices, not empirically calibrated (see `docs/LIMITATIONS.md`).
2. **Human baselines** (`HUMAN_BASELINES` in `metrics/core.py`): literature-derived susceptibility rates used by the Human Alignment Score.
3. **Placeholder answers**: expected answers starting with `[` are non-evaluable. `score_response` returns `(None, None)` for them, and such results count toward the per-bias `unknown_rate` rather than as biased or unbiased.
4. **Async evaluation** bounded by a semaphore (`max_concurrent_requests`) rather than a fixed rate limit.

## Provider Gotchas

The two Anthropic rows each produced a clean-looking 0.0% fingerprint before they were caught, and the gpt-5 token cap returned empty answers for entire biases. Treat a 0.0% overall susceptibility, or a bias whose answers are all empty, as a bug until proven otherwise.

| Model | Behavior | Handling in `engines/evaluator.py` |
|---|---|---|
| `claude-opus-4-7`, `claude-opus-4-8`, `claude-fable-5` | Reject `temperature` with HTTP 400 "temperature is deprecated for this model" | `AnthropicProvider.complete` omits the parameter for models on an explicit prefix denylist. Older Claude models still accept it. Runs without `temperature` sample non-deterministically, which can inflate RCI. |
| `claude-fable-5` | Reasoning model: `response.content` is `[ThinkingBlock, TextBlock]`, so `content[0].text` raises `AttributeError` | `complete()` concatenates the blocks whose `type == "text"`. Thinking tokens bill as output. |
| `gpt-5*` | Chat completions reject explicit `temperature`; `max_completion_tokens` caps reasoning plus output and starved visible answers | `OpenAIProvider.complete` omits both for `gpt-5*` models and lets them self-bound. |
| Grok, Gemini | Sync SDKs | `XAIProvider` and `GeminiProvider` wrap calls in `asyncio.to_thread`. |

## Testing Patterns

- `pytest` with `pytest-asyncio` in strict mode for async evaluator tests.
- Generator tests use `seed=42` for reproducibility.
- Test files mirror source files: `tests/test_generator.py` covers `engines/generator.py`.
- No `conftest.py`; fixtures live in the test files that use them.
- `TestResult` in `core.py` sets `__test__ = False` so pytest does not try to collect it.

## Bias Test Quality

A zero score on a bias deserves investigation before it is reported as resistance. Four failure modes: training contamination (famous examples the model memorized, such as the Linda problem), dominant options (one answer is objectively right with no trade-off), unrealistic parameters (the numbers make the rational choice trivial), and structurally untestable biases (they need embodied or temporal cognition, so the zero is legitimate).

## Completed Evaluations (core tier)

| Model | Fingerprint generated | Fingerprint |
|---|---|---|
| Claude Sonnet 4.5 | 2026-02-14 | `results/fingerprint_sonnet45.json` (pilot run) |
| GPT-5.2 | 2026-02-14 | `results/fingerprint_gpt52.json` |
| Claude Haiku 4.5 | 2026-02-14 | `results/fingerprint_haiku45.json` |
| Claude Opus 4.6 | 2026-02-14 | `results/fingerprint_opus46.json` |
| Grok 4.1 Fast | 2026-02-15 | `results/fingerprint_grok.json` |
| Claude Sonnet 4.6 | 2026-02-17 | `results/fingerprint_sonnet46.json` |
| GPT-5.4 | 2026-03-10 | `results/fingerprint_gpt54.json` |
| Claude Opus 4.7 | 2026-04-17 | `results/fingerprint_opus47.json` |
| GPT-5.5 | 2026-04-24 | `results/fingerprint_gpt55.json` |
| Claude Opus 4.8 | 2026-05-31 | `results/fingerprint_opus48.json` |
| Claude Fable 5 | 2026-06-09 | `results/fingerprint_fable5.json` |

To publish a result to the website, run `scripts/fingerprint_to_website.py` (its docstring lists the field-mapping gotchas) and `scripts/export_samples_for_website.py`.
