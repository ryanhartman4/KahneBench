# Kahne-Bench

A comprehensive cognitive bias benchmark framework for evaluating Large Language Models, grounded in the dual-process theory of Kahneman and Tversky.

## Overview

Kahne-Bench provides systematic evaluation of 50 cognitive biases across 5 ecological domains, with multi-scale testing methodology and 6 advanced metrics for deep analysis of LLM cognitive patterns.

### Key Features

- **50 Cognitive Biases**: Complete taxonomy based on Kahneman-Tversky research
- **5 Ecological Domains**: Individual, Professional, Social, Temporal, Risk
- **Multi-Scale Testing**: Micro, Meso, Macro, and Meta scales
- **6 Advanced Metrics**: BMS, BCI, BMP, HAS, RCI, CAS
- **Bias Interaction Matrix**: Test compound effects between biases
- **Temporal Dynamics**: Track bias evolution over decision sequences
- **Adversarial Robustness**: Paraphrase and debiasing tests

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### CLI Usage

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

### Python API

```python
from kahne_bench import (
    TestCaseGenerator,
    BiasEvaluator,
    MetricCalculator,
    Domain,
    BIAS_TAXONOMY,
)

# Generate test cases
generator = TestCaseGenerator(seed=42)
instances = generator.generate_batch(
    bias_ids=["anchoring_effect", "loss_aversion"],
    domains=[Domain.PROFESSIONAL],
    instances_per_combination=5,
)

# Run evaluation (with your LLM provider)
evaluator = BiasEvaluator(your_llm_provider)
session = await evaluator.evaluate_batch(instances, model_id="your-model")

# Calculate metrics
calculator = MetricCalculator()
report = calculator.calculate_all_metrics(session.model_id, session.results)

print(f"Overall bias susceptibility: {report.overall_bias_susceptibility:.2f}")
print(f"Most susceptible biases: {report.most_susceptible_biases}")
```

## Bias Taxonomy

The 50 biases are organized into categories based on underlying cognitive mechanisms:

| Category | Count | Key Biases |
|----------|-------|------------|
| Representativeness | 8 | Base rate neglect, Conjunction fallacy |
| Availability | 6 | Availability bias, Recency bias, Primacy bias |
| Anchoring | 5 | Anchoring effect, Insufficient adjustment |
| Loss Aversion | 5 | Loss aversion, Endowment effect, Sunk cost |
| Framing | 6 | Gain-loss framing, Attribute framing |
| Probability Distortion | 6 | Certainty effect, Probability weighting |
| Overconfidence | 5 | Overconfidence, Planning fallacy |
| Confirmation | 3 | Confirmation bias, Belief perseverance |
| Temporal | 3 | Present bias, Duration neglect |
| Extension Neglect | 3 | Scope insensitivity, Identifiable victim |

## Advanced Metrics

### Bias Magnitude Score (BMS)
Quantifies the strength of a bias by measuring deviation from rational baseline.

### Bias Consistency Index (BCI)
Measures how consistently a bias appears across different domains.

### Bias Mitigation Potential (BMP)
Assesses the model's ability to overcome bias with debiasing prompts.

### Human Alignment Score (HAS)
Compares model bias patterns to established human baselines.

### Response Consistency Index (RCI)
Measures variance across multiple identical trials.

### Calibration Awareness Score (CAS)
Assesses whether the model recognizes when it's being biased.

## Testing Scales

- **Micro**: Single isolated bias with control vs treatment
- **Meso**: Multiple bias interactions in complex scenarios
- **Macro**: Bias persistence across sequential decisions
- **Meta**: Self-correction and debiasing capacity

## Project Structure

```
src/kahne_bench/
├── __init__.py         # Main exports
├── core.py             # Core data structures
├── cli.py              # Command-line interface
├── biases/
│   ├── __init__.py
│   └── taxonomy.py     # 50-bias taxonomy
├── engines/
│   ├── __init__.py
│   ├── generator.py    # Test case generation
│   ├── evaluator.py    # LLM evaluation
│   ├── compound.py     # Compound bias testing
│   └── robustness.py   # Adversarial testing
├── metrics/
│   ├── __init__.py
│   └── core.py         # 6 advanced metrics
├── domains/
│   └── __init__.py
└── utils/
    ├── __init__.py
    └── diversity.py    # Dataset validation
```

## References

- Kahneman, D. (2011). *Thinking, Fast and Slow*
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk

## License

MIT License
