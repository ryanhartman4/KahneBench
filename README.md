# Kahne-Bench

A comprehensive cognitive bias benchmark framework for evaluating Large Language Models, grounded in the dual-process theory of Kahneman and Tversky.

## Overview

Kahne-Bench provides systematic evaluation of 69 cognitive biases across 5 ecological domains, with multi-scale testing methodology and 6 advanced metrics for deep analysis of LLM cognitive patterns.

### Key Features

- **69 Cognitive Biases**: Complete taxonomy based on Kahneman-Tversky research
- **5 Ecological Domains**: Individual, Professional, Social, Temporal, Risk
- **Multi-Scale Testing**: Micro, Meso, Macro, and Meta scales
- **6 Advanced Metrics**: BMS, BCI, BMP, HAS, RCI, CAS
- **Bias Interaction Matrix**: Test compound effects between biases
- **Benchmark Tiers**: Core (15), Extended (69), and Interaction tiers
- **Data Export**: JSON, CSV, and human-readable reports

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
# Using uv (recommended)
uv sync

# Using pip
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the demo
PYTHONPATH=src python examples/basic_usage.py

# Run a full evaluation with OpenAI
export OPENAI_API_KEY="your-api-key"
PYTHONPATH=src python examples/openai_evaluation.py --model gpt-5.2 --tier core
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
from kahne_bench import BiasEvaluator
from kahne_bench.engines.evaluator import EvaluationConfig

# Define your LLM provider (must have async complete() method)
class OpenAIProvider:
    def __init__(self, client, model="gpt-5.2"):
        self.client = client
        self.model = model

    async def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
            instructions="You are a helpful assistant.",
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return response.output_text

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
| Framing | 7 | Gain-loss framing, Mental accounting |
| Probability Distortion | 7 | Certainty effect, Affect heuristic |
| Overconfidence | 5 | Overconfidence, Planning fallacy |
| Confirmation | 3 | Confirmation bias, Belief perseverance |
| Temporal | 3 | Present bias, Duration neglect |
| Extension Neglect | 4 | Scope insensitivity, Identifiable victim |
| Memory | 4 | Hindsight bias, Rosy retrospection |
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

# Get info about a specific bias
kahne-bench describe anchoring_effect

# Generate test cases
kahne-bench generate --bias anchoring_effect --bias loss_aversion --output tests.json

# Generate compound (interaction) tests
kahne-bench generate-compound --domain professional
```

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
│   ├── compound.py      # Compound bias testing
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
PYTHONPATH=src python examples/basic_usage.py
```
Demonstrates taxonomy exploration, test generation, evaluation with a mock provider, and metrics calculation.

### OpenAI Evaluation
```bash
export OPENAI_API_KEY="your-api-key"
PYTHONPATH=src python examples/openai_evaluation.py --model gpt-5.2 --tier core
```

Options:
- `--model`, `-m`: Model name (default: gpt-5.2)
- `--tier`, `-t`: Benchmark tier - core, extended, or interaction (default: core)
- `--domains`, `-d`: Domains to test (default: professional, individual)
- `--trials`, `-n`: Trials per condition (default: 3)
- `--output`, `-o`: Output file prefix (default: evaluation)

Example with extended tier:
```bash
python examples/openai_evaluation.py --model gpt-5.2 --tier extended --trials 5
```

---

## References

- Kahneman, D. (2011). *Thinking, Fast and Slow*
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk

## License

MIT License
