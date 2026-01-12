# Kahne-Bench: Agent Guide

This guide helps AI agents run cognitive bias benchmarks on themselves or other LLMs.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set your API key (pick one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export FIREWORKS_API_KEY="fw_..."
export XAI_API_KEY="xai-..."
export GOOGLE_API_KEY="..."

# 3. Generate test cases (small test)
PYTHONPATH=src uv run kahne-bench generate \
  --bias anchoring_effect \
  --domain professional \
  --instances 1 \
  -o test_cases.json

# 4. Run evaluation
PYTHONPATH=src uv run kahne-bench evaluate \
  -i test_cases.json \
  -p openai \
  -m gpt-5.2-2025-12-11 \
  --trials 1

# 5. View results
cat fingerprint.json
```

## Providers & Models

| Provider | Flag | Environment Variable | Example Models |
|----------|------|---------------------|----------------|
| OpenAI | `-p openai` | `OPENAI_API_KEY` | `gpt-5.2-2025-12-11`, `gpt-4o` |
| Anthropic | `-p anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929` |
| Fireworks | `-p fireworks` | `FIREWORKS_API_KEY` | `accounts/fireworks/models/glm-4p7`, `accounts/fireworks/models/deepseek-v3p2` |
| xAI | `-p xai` | `XAI_API_KEY` | `grok-4-1-fast-reasoning` |
| Google | `-p gemini` | `GOOGLE_API_KEY` | `gemini-3-pro-preview` |
| Mock | `-p mock` | (none) | `mock-model` (for testing) |

## Test Scope Options

### Benchmark Tiers
```bash
--tier core       # 15 foundational biases (quick, ~5 min)
--tier extended   # All 69 biases (comprehensive, ~30 min)
--tier interaction # Bias pairs for compound effects
```

### Specific Biases
```bash
--bias anchoring_effect
--bias loss_aversion
--bias confirmation_bias
# Can specify multiple --bias flags
```

### Domains
```bash
--domain individual    # Personal finance, consumer choice
--domain professional  # Managerial, medical, legal
--domain social        # Negotiation, persuasion
--domain temporal      # Long-term planning
--domain risk          # Policy, technology decisions
```

### Trials (for statistical significance)
```bash
--trials 1   # Quick test
--trials 3   # Default, good balance
--trials 5   # More reliable metrics
```

## Generate Commands

```bash
# Single bias, quick test
PYTHONPATH=src uv run kahne-bench generate \
  --bias anchoring_effect \
  -o test_cases.json

# Core tier (15 biases)
PYTHONPATH=src uv run kahne-bench generate \
  --tier core \
  -o test_cases.json

# Extended tier (69 biases)
PYTHONPATH=src uv run kahne-bench generate \
  --tier extended \
  -o test_cases.json

# Multiple specific biases
PYTHONPATH=src uv run kahne-bench generate \
  --bias anchoring_effect \
  --bias loss_aversion \
  --bias confirmation_bias \
  --bias overconfidence_effect \
  -o test_cases.json
```

## Evaluate Commands

```bash
# Basic evaluation
PYTHONPATH=src uv run kahne-bench evaluate \
  -i test_cases.json \
  -p <provider> \
  -m <model> \
  --trials 3

# With custom output files
PYTHONPATH=src uv run kahne-bench evaluate \
  -i test_cases.json \
  -p anthropic \
  -m claude-opus-4-5-20251101 \
  --trials 3 \
  -o results.json \
  -f fingerprint.json
```

## Understanding Output

### Fingerprint File (cognitive profile)

```json
{
  "model_id": "gpt-5.2-2025-12-11",
  "summary": {
    "overall_bias_susceptibility": 0.15,  // 0-1, lower is better
    "most_susceptible_biases": ["anchoring_effect", "availability_bias"],
    "most_resistant_biases": ["confirmation_bias", "loss_aversion"]
  },
  "magnitude_scores": { ... },      // BMS: Strength of bias per trigger
  "consistency_indices": { ... },   // BCI: Cross-domain consistency
  "mitigation_potentials": { ... }, // BMP: How well debiasing works
  "human_alignments": { ... },      // HAS: Comparison to human baselines
  "response_consistencies": { ... }, // RCI: Trial-to-trial variance
  "calibration_scores": { ... }     // CAS: Metacognitive accuracy
}
```

### Key Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| **overall_bias_susceptibility** | 0-1 | Overall bias vulnerability (lower = better) |
| **magnitude (BMS)** | 0-1 | How strongly bias manifests |
| **alignment_score (HAS)** | 0-1 | Similarity to human bias patterns |
| **direction** | "over"/"under" | More or less biased than humans |
| **consistency_score (RCI)** | 0-1 | Response stability across trials |

### Results File (raw data)

Each evaluation produces a result object:
```json
{
  "bias_id": "anchoring_effect",
  "condition": "treatment_weak",  // control, treatment_weak/moderate/strong, debiasing_0/1/2
  "prompt_used": "...",
  "model_response": "...",
  "extracted_answer": "100",
  "is_biased": true,
  "bias_score": 0.7
}
```

## Example: Full Benchmark Run

```bash
# 1. Generate core tier tests
PYTHONPATH=src uv run kahne-bench generate --tier core -o core_tests.json

# 2. Run on your model
PYTHONPATH=src uv run kahne-bench evaluate \
  -i core_tests.json \
  -p anthropic \
  -m claude-opus-4-5-20251101 \
  --trials 3 \
  -o results_claude.json \
  -f fingerprint_claude.json

# 3. Check results
python3 -c "
import json
with open('fingerprint_claude.json') as f:
    data = json.load(f)
print(f\"Overall Susceptibility: {data['summary']['overall_bias_susceptibility']:.1%}\")
print(f\"Most Susceptible: {data['summary']['most_susceptible_biases'][:3]}\")
print(f\"Most Resistant: {data['summary']['most_resistant_biases'][:3]}\")
"
```

## Comparing Models

```bash
# Run same tests on multiple models
for provider_model in "openai:gpt-5.2-2025-12-11" "anthropic:claude-opus-4-5-20251101" "gemini:gemini-3-pro-preview"; do
  provider="${provider_model%%:*}"
  model="${provider_model##*:}"
  echo "Testing $model..."
  PYTHONPATH=src uv run kahne-bench evaluate \
    -i core_tests.json \
    -p "$provider" \
    -m "$model" \
    --trials 3 \
    -o "results_${provider}.json" \
    -f "fingerprint_${provider}.json"
done

# Compare results
python3 -c "
import json, glob
for f in sorted(glob.glob('fingerprint_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    print(f\"{data['model_id']}: {data['summary']['overall_bias_susceptibility']:.1%}\")
"
```

## List Available Biases

```bash
# List all 69 biases
PYTHONPATH=src uv run kahne-bench list-biases

# Get details on a specific bias
PYTHONPATH=src uv run kahne-bench describe anchoring_effect
```

## Troubleshooting

### Missing API Key
```
Error: OPENAI_API_KEY environment variable not set
```
Solution: Export your API key before running.

### Model Not Found
Check the exact model ID with your provider's documentation.

### Rate Limiting
Add `--trials 1` to reduce API calls during testing.

## Cost Estimation

| Tier | Biases | Calls/Model | Est. Cost (GPT-5.2) |
|------|--------|-------------|---------------------|
| Single bias | 1 | ~21 | ~$0.50 |
| Core | 15 | ~315 | ~$8-15 |
| Extended | 69 | ~1,450 | ~$35-70 |

*Costs vary by model. Fireworks models are ~10x cheaper than OpenAI/Anthropic.*
