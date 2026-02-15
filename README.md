# Kahne-Bench

A cognitive bias benchmark framework for evaluating Large Language Models, grounded in Kahneman-Tversky dual-process theory.

Website: https://www.kahnebench.com/

## Overview

Kahne-Bench evaluates 69 cognitive biases across 5 ecological domains using multi-scale testing and 6 specialized metrics.

### Key Features

- **69 Cognitive Biases**: Complete taxonomy based on Kahneman-Tversky research
- **5 Ecological Domains**: Individual, Professional, Social, Temporal, Risk
- **Multi-Scale Testing**: Micro, Meso, Macro, and Meta scales
- **6 Metrics**: BMS, BCI, BMP, HAS, RCI, CAS
- **Bias Interaction Matrix**: Test compound effects between biases
- **Benchmark Tiers**: Core (15), Extended (69), and Interaction tiers
- **Data Export**: JSON, CSV, and human-readable reports

## What's Included

- Bias taxonomy (69 biases), categories, and interaction matrix
- Test generation: standard templates, novel contamination-resistant scenarios, compound (meso) scenarios, and macro decision chains
- Evaluation engines: standard evaluator plus temporal dynamics, context sensitivity, and multi-turn conversational evaluators
- BLOOM generation: LLM-driven scenario generation for richer test cases
- Quality assessment: automated test quality evaluation with LLM judge
- Robustness tooling: prompt variation, paraphrase consistency, debiasing variants, self-help debiasing prompts
- Metrics: six metrics and cognitive fingerprint reporting
- CLI commands, examples, and JSON/CSV import/export utilities

## Sample Results: Claude Sonnet 4.5 (Core Tier)

Evaluated 2026-02-12 using the core tier (15 foundational biases), 3 trials per condition, 4,725 total evaluations.

**Overall Bias Susceptibility: 13.17%**

| Bias | BMS Score | Human Baseline | Direction |
|------|:---------:|:--------------:|:---------:|
| Endowment Effect | 0.827 | 0.65 | Over-human |
| Gain-Loss Framing | 0.420 | 0.72 | Under |
| Status Quo Bias | 0.140 | 0.62 | Under |
| Certainty Effect | 0.112 | 0.72 | Under |
| Hindsight Bias | 0.107 | 0.65 | Under |
| Sunk Cost Fallacy | 0.101 | 0.55 | Under |
| Anchoring Effect | 0.078 | 0.65 | Under |
| Availability Bias | 0.068 | 0.60 | Under |
| Loss Aversion | 0.060 | 0.70 | Over-human |
| Overconfidence | 0.039 | 0.75 | Under |
| Gambler's Fallacy | 0.014 | 0.45 | Under |
| Confirmation Bias | 0.011 | 0.72 | Under |
| Base Rate Neglect | 0.000 | 0.68 | Under |
| Conjunction Fallacy | 0.000 | 0.85 | Under |
| Present Bias | 0.000 | 0.70 | Under |

**Key findings:**
- **Endowment effect** is the top vulnerability (0.827) but is 100% debiasable with prompting
- **Loss aversion** is the only *systematic* bias — consistent across all 5 domains and only partially mitigable (67.5%)
- **Conjunction fallacy** and **present bias** score 0.000 due to genuine structural resistance — LLMs have internalized the probability axiom (conjunction) and lack temporal salience (present bias)
- 13 of 15 biases show the model is *less* biased than humans; only loss aversion and endowment effect exceed human baselines

Full results: `deprecated_results/sonnet45_core_fingerprint_v2.json`

## Limitations (Read Before Use)

Kahne-Bench is a research framework, not a validated psychometric instrument. Results are best used for relative comparisons between models, not absolute claims about human-like bias.

- No direct human validation data for these exact prompts; baselines are literature-derived and may be outdated or population-biased.
- Metric weights (e.g., BMS intensity weights) are design choices, not empirically calibrated.
- Template-based prompts and heuristic answer extraction can misclassify or miss responses.
- Some expected answers are placeholders or context-dependent, which can yield neutral scores.

See `docs/LIMITATIONS.md` for full details.

## Reproducibility Checklist

- [ ] Record benchmark tier and exact bias list (core/extended/interaction).
- [ ] Fix random seeds for generation and any shuffling.
- [ ] Export and version the generated instances JSON used in the run.
- [ ] Log model ID, provider, temperature, max_tokens, and number of trials.
- [ ] Record prompt variants (control/treatment intensities/debiasing).
- [ ] Capture code version (git commit) and config file or CLI args.
- [ ] Report runtime environment (Python version, dependency lockfile hash).
- [ ] Publish results and fingerprint JSON or CSV alongside the above metadata.

## Installation

```bash
# Install from GitHub with uv (recommended)
uv pip install "git+https://github.com/ryanhartman4/KahneBench.git"

# Install from GitHub with pip
pip install "git+https://github.com/ryanhartman4/KahneBench.git"

# Local development from GitHub
git clone https://github.com/ryanhartman4/KahneBench.git
cd KahneBench
uv sync
uv sync --group dev
```

## Environment Variables

For the CLI `evaluate` command, the LLM judge fallback defaults to Anthropic Haiku 4.5 (`--judge-provider anthropic --judge-model claude-haiku-4-5`), so `ANTHROPIC_API_KEY` is required unless you override judge settings.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # Needed for default CLI judge (claude-haiku-4-5)
export OPENAI_API_KEY="sk-..."           # OpenAI models (default CLI model: gpt-5.2)
export FIREWORKS_API_KEY="fw_..."        # Fireworks models (default CLI model: kimi-k2p5)
export GOOGLE_API_KEY="..."              # Google models (default CLI model: gemini-3-pro-preview)
export XAI_API_KEY="xai-..."             # xAI models (default CLI model: grok-4-1-fast-reasoning)
```

## Quick Start

```bash
# Run the demo
PYTHONPATH=src uv run python examples/basic_usage.py

# Run a full evaluation with OpenAI
export OPENAI_API_KEY="your-openai-key"
PYTHONPATH=src uv run python examples/openai_evaluation.py --model gpt-5.2 --tier core
```

---

## Usage Guide

### 1. Explore the Bias Taxonomy

```python
from kahne_bench import (
    BIAS_TAXONOMY,
    get_bias_by_id,
    get_biases_by_category,
    BiasCategory,
)

# See all 69 biases
print(f"Total biases: {len(BIAS_TAXONOMY)}")

# Get a specific bias
anchoring = get_bias_by_id("anchoring_effect")
print(f"Name: {anchoring.name}")
print(f"Description: {anchoring.description}")
print(f"Theoretical basis: {anchoring.theoretical_basis}")
print(f"System 1 mechanism: {anchoring.system1_mechanism}")
print(f"System 2 override: {anchoring.system2_override}")

# Get biases by category
framing_biases = get_biases_by_category(BiasCategory.FRAMING)
for bias in framing_biases:
    print(f"- {bias.name}")
```

### 2. Generate Test Cases

```python
from kahne_bench import TestCaseGenerator, Domain, TestScale, TriggerIntensity

generator = TestCaseGenerator(seed=42)

# Generate a single test instance
instance = generator.generate_instance(
    bias_id="anchoring_effect",
    domain=Domain.PROFESSIONAL,
    scale=TestScale.MICRO,
    include_debiasing=True,  # Include debiasing prompts
)

# Access the prompts
print(instance.control_prompt)                          # Baseline (no bias trigger)
print(instance.get_treatment(TriggerIntensity.WEAK))     # Weak trigger
print(instance.get_treatment(TriggerIntensity.MODERATE)) # Moderate trigger
print(instance.get_treatment(TriggerIntensity.STRONG))   # Strong trigger

# Access debiasing prompts
if instance.has_debiasing():
    for i, prompt in enumerate(instance.debiasing_prompts):
        print(f"Debiasing strategy {i+1}: {prompt[:100]}...")
```

### 3. Generate Test Batches

```python
# Generate batch for multiple biases and domains
batch = generator.generate_batch(
    bias_ids=["anchoring_effect", "loss_aversion", "gain_loss_framing"],
    domains=[Domain.PROFESSIONAL, Domain.INDIVIDUAL],
    instances_per_combination=3,
)
print(f"Generated {len(batch)} test instances")
# Output: Generated 18 test instances (3 biases × 2 domains × 3 instances)
```

### 4. Use Benchmark Tiers

```python
from kahne_bench.engines.generator import get_tier_biases, KahneBenchTier

# Core tier: 15 foundational biases for quick evaluation
core_biases = get_tier_biases(KahneBenchTier.CORE)
print(f"Core tier: {len(core_biases)} biases")

# Extended tier: All 69 biases for comprehensive evaluation
extended_biases = get_tier_biases(KahneBenchTier.EXTENDED)
print(f"Extended tier: {len(extended_biases)} biases")

# Interaction tier: Bias pairs for compound effect testing
interaction_biases = get_tier_biases(KahneBenchTier.INTERACTION)
print(f"Interaction tier: {len(interaction_biases)} biases")
```

### 5. Run Evaluations with Your LLM

```python
import asyncio
from openai import AsyncOpenAI
from kahne_bench import BiasEvaluator, TriggerIntensity
from kahne_bench.engines.evaluator import EvaluationConfig

# Define your LLM provider (must have async complete() method)
class OpenAIProvider:
    def __init__(self, client, model="gpt-5.2"):
        self.client = client
        self.model = model

    async def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        uses_completion_tokens = self.model.startswith(("gpt-5", "o3", "o1", "chatgpt-"))
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        if uses_completion_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

# Configure evaluation
config = EvaluationConfig(
    num_trials=5,  # Trials per condition (for consistency measurement)
    intensities=[TriggerIntensity.MODERATE, TriggerIntensity.STRONG],
    include_control=True,
    include_debiasing=True,
)

client = AsyncOpenAI()
provider = OpenAIProvider(client, model="gpt-5.2")
evaluator = BiasEvaluator(provider, config)

# Run evaluation
async def evaluate():
    session = await evaluator.evaluate_batch(
        instances=batch,
        model_id="gpt-5.2",
        progress_callback=lambda i, n: print(f"Progress: {i}/{n}"),
    )
    return session

session = asyncio.run(evaluate())
print(f"Completed {len(session.results)} evaluations")
```

### 6. Calculate Metrics

```python
from kahne_bench import MetricCalculator

calculator = MetricCalculator()
report = calculator.calculate_all_metrics("gpt-5.2", session.results)

# Summary statistics
print(f"Overall Bias Susceptibility: {report.overall_bias_susceptibility:.2%}")
print(f"Most Susceptible: {report.most_susceptible_biases[:3]}")
print(f"Most Resistant: {report.most_resistant_biases[:3]}")
print(f"Human-Like Biases: {report.human_like_biases}")
print(f"AI-Specific Biases: {report.ai_specific_biases}")

# Per-bias metrics
for bias_id in report.biases_tested:
    bms = report.magnitude_scores[bias_id]
    has = report.human_alignments[bias_id]
    print(f"\n{bias_id}:")
    print(f"  Magnitude: {bms.overall_magnitude:.3f}")
    print(f"  Human Alignment: {has.alignment_score:.3f} ({has.bias_direction})")
```

### 7. Export Results

```python
from kahne_bench.utils import (
    export_instances_to_json,
    import_instances_from_json,
    export_results_to_json,
    export_results_to_csv,
    export_session_to_json,
    export_fingerprint_to_json,
    export_fingerprint_to_csv,
    generate_summary_report,
)

# Export test instances (for reproducibility)
export_instances_to_json(batch, "test_instances.json")

# Export results
export_results_to_json(session.results, "results.json")
export_results_to_csv(session.results, "results.csv")

# Export full session
export_session_to_json(session, "session.json", include_prompts=True)

# Export cognitive fingerprint
export_fingerprint_to_json(report, "fingerprint.json")
export_fingerprint_to_csv(report, "fingerprint.csv")

# Generate human-readable report
summary = generate_summary_report(report)
print(summary)
```

### 8. Compound Bias Testing (Meso-Scale)

```python
from kahne_bench.engines.compound import CompoundTestGenerator

compound_gen = CompoundTestGenerator()

# Generate compound test with interacting biases
instance = compound_gen.generate_compound_instance(
    primary_bias="anchoring_effect",
    secondary_biases=["availability_bias", "overconfidence_effect"],
    domain=Domain.PROFESSIONAL,
    interaction_type="amplifying",  # or "attenuating"
)

print(f"Primary: {instance.bias_id}")
print(f"Interactions: {instance.interaction_biases}")
print(f"Scale: {instance.scale.value}")  # "meso"
```

---

## Bias Taxonomy

The 69 biases are organized into 16 categories based on underlying cognitive mechanisms:

| Category | Count | Key Biases |
|----------|-------|------------|
| Representativeness | 8 | Base rate neglect, Conjunction fallacy |
| Availability | 6 | Availability bias, Recency bias, Primacy bias |
| Anchoring | 5 | Anchoring effect, Insufficient adjustment |
| Loss Aversion | 5 | Loss aversion, Endowment effect, Sunk cost |
| Framing | 6 | Gain-loss framing, Mental accounting |
| Reference Dependence | 1 | Reference point framing |
| Probability Distortion | 7 | Certainty effect, Affect heuristic |
| Overconfidence | 5 | Overconfidence, Planning fallacy |
| Confirmation | 3 | Confirmation bias, Belief perseverance |
| Temporal | 3 | Present bias, Duration neglect |
| Extension Neglect | 2 | Scope insensitivity, Identifiable victim |
| Memory | 4 | Rosy retrospection, Source confusion |
| Attention | 3 | Attentional bias, Selective perception |
| Attribution | 3 | Fundamental attribution error, Self-serving bias |
| Uncertainty Judgment | 3 | Ambiguity aversion, Illusion of validity |
| Social Bias | 5 | Ingroup bias, False consensus, Halo effect |

---

## The 6 Metrics

| Metric | Abbreviation | What It Measures |
|--------|--------------|------------------|
| **Bias Magnitude Score** | BMS | How strongly the model exhibits a bias (0-1) |
| **Bias Consistency Index** | BCI | How consistently bias appears across domains |
| **Bias Mitigation Potential** | BMP | How well debiasing prompts reduce bias |
| **Human Alignment Score** | HAS | How closely model biases match human patterns |
| **Response Consistency Index** | RCI | Variance across multiple identical trials |
| **Calibration Awareness Score** | CAS | Whether model knows when it's being biased |

---

## Testing Scales

| Scale | Description | Use Case |
|-------|-------------|----------|
| **Micro** | Single isolated bias with control vs treatment | Basic bias detection |
| **Meso** | Multiple bias interactions in complex scenarios | Compound effect analysis |
| **Macro** | Bias persistence across sequential decisions | Temporal dynamics |
| **Meta** | Self-correction and debiasing capacity | Metacognitive evaluation |

---

## CLI Usage

```bash
# List all biases
kahne-bench list-biases

# List all bias categories
kahne-bench list-categories

# Get info about a specific bias
kahne-bench describe anchoring_effect

# Generate test cases
kahne-bench generate --bias anchoring_effect --bias loss_aversion --output tests.json

# Generate compound (interaction) tests
kahne-bench generate-compound --domain professional

# Run full evaluation pipeline (requires API key)
kahne-bench evaluate -i test_cases.json -p openai -m gpt-5.2 --judge-provider openai --judge-model gpt-5.2

# Run evaluation with verbose logging (shows per-instance progress, API timing, scoring details)
kahne-bench evaluate -i test_cases.json -p openai -m gpt-5.2 --verbose

# Generate cognitive fingerprint report
kahne-bench report fingerprint.json

# Assess test quality
kahne-bench assess-quality -i test_cases.json

# Generate BLOOM scenarios
kahne-bench generate-bloom --bias anchoring_effect

# Run conversational evaluation
kahne-bench evaluate-conversation -i test_cases.json -p openai -m gpt-5.2

# Show framework information
kahne-bench info
```

The `evaluate` command supports `--verbose` for detailed logging output. When enabled, the Rich progress bar is replaced with timestamped log lines showing per-instance progress, API call timing, answer extraction, scoring results, and LLM judge fallback events.

Note: The CLI covers listing/description, generation, compound tests, evaluation, quality assessment, BLOOM generation, and conversational evaluation. Advanced evaluators are Python API only: `TemporalEvaluator`, `ContextSensitivityEvaluator`, `MacroScaleGenerator`, `RobustnessTester`, and `ContrastiveRobustnessTester`.

---

## Project Structure

```
src/kahne_bench/
├── __init__.py          # Main exports
├── core.py              # Core data structures
├── cli.py               # Command-line interface
├── biases/
│   ├── __init__.py
│   └── taxonomy.py      # 69-bias taxonomy
├── engines/
│   ├── __init__.py
│   ├── generator.py     # Test case generation
│   ├── evaluator.py     # LLM evaluation
│   ├── judge.py         # LLM judge fallback scoring
│   ├── compound.py      # Compound bias testing
│   ├── bloom_generator.py # LLM-driven BLOOM scenario generation
│   ├── variation.py     # Prompt variation and robustness testing
│   ├── quality.py       # Test quality assessment
│   ├── conversation.py  # Multi-turn conversational evaluation
│   └── robustness.py    # Adversarial testing
├── metrics/
│   ├── __init__.py
│   └── core.py          # 6 advanced metrics
└── utils/
    ├── __init__.py
    ├── diversity.py     # Dataset validation
    └── io.py            # Export/import utilities
```

---

## Examples

### Basic Demo
```bash
PYTHONPATH=src uv run python examples/basic_usage.py
```
Demonstrates taxonomy exploration, test generation, evaluation with a mock provider, and metrics calculation.

### OpenAI Evaluation
```bash
export OPENAI_API_KEY="your-openai-key"
PYTHONPATH=src uv run python examples/openai_evaluation.py --model gpt-5.2 --tier core
```

Options:
- `--model`, `-m`: Model name (default in script: gpt-5.2)
- `--tier`, `-t`: Benchmark tier - core, extended, or interaction (default: core)
- `--domains`, `-d`: Domains to test (default: professional, individual)
- `--trials`, `-n`: Trials per condition (default: 3)
- `--output`, `-o`: Output file prefix (default: evaluation)

Example with extended tier:
```bash
PYTHONPATH=src uv run python examples/openai_evaluation.py --model gpt-5.2 --tier extended --trials 5
```

---

## References

- Kahneman, D. (2011). *Thinking, Fast and Slow*
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk

### Related Benchmarks

- Koo, R., et al. (2024). Benchmarking Cognitive Biases in Large Language Models as Evaluators. *ACL 2024 Findings*. [arXiv:2309.17012](https://arxiv.org/abs/2309.17012)
- Coda-Forno, J., et al. (2024). CogBench: A large language model walks into a psychology lab. *ICML 2024*. [arXiv:2402.18225](https://arxiv.org/abs/2402.18225)
- Malberg, S., et al. (2024). A Comprehensive Evaluation of Cognitive Biases in LLMs. [arXiv:2410.15413](https://arxiv.org/abs/2410.15413)
- Echterhoff, J., et al. (2024). Cognitive Bias in Decision-Making with LLMs. *EMNLP 2024 Findings*. [arXiv:2403.00811](https://arxiv.org/abs/2403.00811)

## License

MIT License
