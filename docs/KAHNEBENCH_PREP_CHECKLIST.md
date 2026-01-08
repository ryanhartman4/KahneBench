# KahneBench Preparation Checklist

Checklist for running KahneBench cognitive bias evaluations on leading LLMs.

---

## Pre-Flight Checklist (18 Steps)

### Setup (Steps 1-4)

- [ ] **1. Install KahneBench**
  ```bash
  uv sync                                    # Install dependencies
  PYTHONPATH=src uv run kahne-bench info     # Verify installation
  ```

- [ ] **2. Obtain API keys** for providers you plan to test:
  | Provider | Get Key At | Env Variable |
  |----------|------------|--------------|
  | OpenAI | platform.openai.com | `OPENAI_API_KEY` |
  | Anthropic | console.anthropic.com | `ANTHROPIC_API_KEY` |
  | Google | aistudio.google.com | `GOOGLE_API_KEY` |
  | Fireworks | fireworks.ai | `FIREWORKS_API_KEY` |

- [ ] **3. Set environment variables**
  ```bash
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."
  export FIREWORKS_API_KEY="..."
  # Add others as needed
  ```

- [ ] **4. Test API connectivity** (one per provider)
  ```bash
  # OpenAI
  curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" | head -20
  ```

### Planning (Steps 5-7)

- [ ] **5. Choose benchmark tier and estimate costs**

  | Tier | Biases | API Calls/Model | Est. Cost (GPT-4o) | Est. Cost (GPT-4o-mini) |
  |------|--------|-----------------|-------------------|------------------------|
  | CORE | 15 | ~900 | ~$2.50 | ~$0.15 |
  | EXTENDED | 69 | ~4,140 | ~$11.50 | ~$0.70 |

  *Costs vary by provider. See Appendix A for full cost matrix.*

- [ ] **6. Select models to evaluate**

  Recommended priority order (cheapest first):
  1. `gpt-4o-mini` (OpenAI) - pipeline validation
  2. `claude-3-5-haiku-20241022` (Anthropic)
  3. `accounts/fireworks/models/llama-v3p1-70b-instruct` (Fireworks)
  4. `gpt-4o` (OpenAI)
  5. `claude-sonnet-4-20250514` (Anthropic)
  6. Additional models as budget allows

- [ ] **7. Set budget ceiling**: $______ (include 20% buffer for retries)

### Validation (Steps 8-11)

- [ ] **8. Generate test cases**
  ```bash
  # CORE tier (15 biases × 5 domains = 75 instances)
  PYTHONPATH=src uv run python -c "
  from kahne_bench.engines.generator import TestCaseGenerator, get_tier_biases, KahneBenchTier
  gen = TestCaseGenerator(seed=42)
  instances = gen.generate_batch(bias_ids=get_tier_biases(KahneBenchTier.CORE), instances_per_combination=1)
  gen.export_to_json(instances, 'test_cases.json')
  print(f'Generated {len(instances)} instances')
  "

  # Or for EXTENDED tier (69 biases × 5 domains = 345 instances):
  # PYTHONPATH=src uv run kahne-bench generate --seed 42 --instances 1 --output test_cases.json
  ```

- [ ] **9. Validate pipeline with mock provider**
  ```bash
  PYTHONPATH=src uv run kahne-bench evaluate \
    -i test_cases.json -p mock -n 1 \
    -o mock_results.json -f mock_fingerprint.json
  ```

- [ ] **10. Validate with real API** (single bias, 1 trial)
  ```bash
  PYTHONPATH=src uv run kahne-bench generate \
    --bias anchoring_effect --domain professional --instances 1 --output validation.json

  PYTHONPATH=src uv run kahne-bench evaluate \
    -i validation.json -p openai -m gpt-4o-mini -n 1 -o validation_results.json
  ```

- [ ] **11. Verify validation output** - check `validation_results.json` has responses and extracted answers

### Execution (Steps 12-15)

- [ ] **12. Create results directory**
  ```bash
  mkdir -p results logs
  ```

- [ ] **13. Start persistent terminal session**
  ```bash
  tmux new -s kahnebench   # or screen -S kahnebench
  ```

- [ ] **14. Run evaluations** for each model
  ```bash
  PYTHONPATH=src uv run kahne-bench evaluate \
    -i test_cases.json \
    -p openai \
    -m gpt-4o \
    -n 3 \
    -o results/gpt-4o_results.json \
    -f results/gpt-4o_fingerprint.json \
    2>&1 | tee logs/gpt-4o.log
  ```

  Repeat for each model, changing `-p`, `-m`, and output filenames.

- [ ] **15. Backup results** after each model completes
  ```bash
  cp results/*.json backups/
  ```

### Analysis (Steps 16-18)

- [ ] **16. Generate reports** for each model
  ```bash
  PYTHONPATH=src uv run kahne-bench report results/gpt-4o_fingerprint.json
  ```

- [ ] **17. Compare models** - review key metrics:
  - Overall Bias Susceptibility (0-1, lower = better)
  - Most/least susceptible biases
  - BMS (magnitude), BCI (consistency), RCI (response variance)

- [ ] **18. Archive final results** to permanent storage

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `API_KEY not set` | `export OPENAI_API_KEY=...` |
| Rate limit errors | Add `--trials 1` or wait and retry |
| `Model not found` | Verify exact model ID string |
| Many `None` extractions | Check response format in logs |

---

## Appendix A: Cost Matrix (Est. per Model)

| Provider | Model | CORE Cost | EXTENDED Cost |
|----------|-------|-----------|---------------|
| OpenAI | gpt-4o | ~$2.50 | ~$11.50 |
| OpenAI | gpt-4o-mini | ~$0.15 | ~$0.70 |
| OpenAI | o1 | ~$15 | ~$70 |
| Anthropic | claude-sonnet-4 | ~$3.50 | ~$16 |
| Anthropic | claude-3-5-haiku | ~$1.00 | ~$4.50 |
| Fireworks | llama-3.1-70b | ~$0.80 | ~$3.70 |

## Appendix B: Provider Model IDs

```
# OpenAI
gpt-4o, gpt-4o-mini, o1, o3-mini

# Anthropic
claude-sonnet-4-20250514, claude-3-5-haiku-20241022

# Fireworks (OpenAI-compatible, use base_url)
accounts/fireworks/models/llama-v3p1-70b-instruct
accounts/fireworks/models/llama-v3p1-405b-instruct
accounts/fireworks/models/deepseek-v3
```

## Appendix C: Custom Provider Template

For providers using OpenAI-compatible APIs (Fireworks, Together, xAI):

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1"
)
# Use with OpenAIProvider class from kahne_bench.engines.evaluator
```

---

*Last updated: 2026-01-08*
