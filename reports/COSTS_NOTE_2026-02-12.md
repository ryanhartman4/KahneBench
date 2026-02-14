# LLM Pricing Note (From Chat)

Date captured: 2026-02-12
Source: User-provided pricing in chat (not independently verified)
Unit: USD per 1M tokens

| Provider | Model | Input ($/1M) | Output ($/1M) |
|---|---|---:|---:|
| xAI | (unspecified) | 0.20 | 0.50 |
| Anthropic | Opus 4.6 | 5.00 | 25.00 |
| Anthropic | Sonnet 4.5 | 3.00 | 15.00 |
| Anthropic | Haiku 4.5 | 1.00 | 5.00 |
| Google | (unspecified) | 2.00 | 12.00 |
| Fireworks | Kimi K2.5 | 0.60 | 3.00 |
| Fireworks | GLM 4.6 | 0.60 | 2.20 |
| OpenAI | GPT-5.2 | 1.75 | 14.00 |

## Notes
- Fireworks Kimi designation in this project context was corrected to `kimi-k2p5`.
- These figures are intended as planning inputs for benchmark cost estimates.

## Core Run Shape (Current Repo Defaults)

- Core biases: 15
- Domains: 5
- Instances per bias-domain: 3
- Conditions per instance: 7 (`control + weak/moderate/strong + 3 debiasing`)
- Trials per condition: 3
- Total requests per model run: `15 * 5 * 3 * 7 * 3 = 4725`

## Cost Formula

`cost = (input_tokens_total/1e6 * input_rate) + (output_tokens_total/1e6 * output_rate)`

With per-request averages `Tin` and `Tout`:

- `input_tokens_total = 4725 * Tin`
- `output_tokens_total = 4725 * Tout`
- `cost = 0.004725 * (Tin * in_rate + Tout * out_rate)`

## Example Estimate (Tin=500, Tout=200 per request)

| Provider | Model | Estimated Cost (USD) |
|---|---|---:|
| xAI | (unspecified) | 0.95 |
| Anthropic | Haiku 4.5 | 7.09 |
| Anthropic | Sonnet 4.5 | 21.26 |
| Anthropic | Opus 4.6 | 35.44 |
| Google | (unspecified) | 16.07 |
| Fireworks | Kimi K2.5 | 4.25 |
| Fireworks | GLM 4.6 | 3.50 |
| OpenAI | GPT-5.2 | 17.36 |

### 5-Model Example Total

Example set:
- OpenAI GPT-5.2
- Anthropic Sonnet 4.5
- Google (unspecified)
- Fireworks Kimi K2.5
- xAI (unspecified)

Estimated total: **$59.90**

### All 8 Models Total

Using the same request count and token assumptions (`Tin=500`, `Tout=200`):

Estimated total: **$105.90**

## Practical Cost Range (All 8 Models)

- Lower token usage: ~$56
- Medium: ~$106
- Higher: ~$160+
- Very high output case near max (`1024` output avg): ~$378-$379

## Budget Risk Notes

- Judge fallback can materially increase cost if unknown-rate is high.
- Output tokens dominate spend for expensive output-priced models.
- To reduce bill risk: lower `max_tokens`, run a pilot sample first, and project from provider token logs before full run.
