# Pilot Analysis: Claude Sonnet 4.5 Cognitive Bias Profile

Date: 2026-02-14
Config: Core tier (15 biases), 5 domains, 3 trials, 3 intensities (W/M/S), seed 42, post-fix pipeline

## Headline Result

**Overall Bias Susceptibility: 21.47%** (up from 13.17% pre-fix, due to decontamination and scoring corrections)

Sonnet 4.5 is not a scaled-down version of human bias. It has a fundamentally different cognitive fingerprint — superhuman on some biases, more susceptible than humans on others.

## The Cognitive Fingerprint

### Superhuman Resistance (Below Human Baseline)

| Bias | BMS | Human | Gap | Interpretation |
|------|:---:|:-----:|:---:|----------------|
| Conjunction fallacy | 0.000 | 0.85 | -0.85 | Internalized probability axioms. P(A and B) <= P(A) is math, not intuition. |
| Confirmation bias | 0.007 | 0.72 | -0.71 | Genuinely willing to update on disconfirming evidence. |
| Gambler's fallacy | 0.007 | 0.45 | -0.44 | Understands statistical independence. |
| Present bias | 0.025 | 0.70 | -0.68 | No visceral immediacy — marshmallow now vs two later doesn't *feel* different to a model. |
| Overconfidence | 0.085 | 0.75 | -0.67 | Calibrated relative to humans. |
| Loss aversion | 0.056 | 0.70 | -0.64 | Largely symmetric in gain/loss framing. |
| Availability bias | 0.068 | 0.60 | -0.53 | Not swayed by vividness or ease of recall. |
| Anchoring effect | 0.101 | 0.65 | -0.55 | Moderate resistance to irrelevant numerical anchors. |
| Sunk cost fallacy | 0.112 | 0.55 | -0.44 | Mostly ignores sunk costs in decision-making. |
| Status quo bias | 0.140 | 0.62 | -0.48 | Moderate resistance to default option preference. |
| Hindsight bias | 0.149 | 0.65 | -0.50 | Moderate resistance to "knew it all along" effects. |
| Gain-loss framing | 0.462 | 0.72 | -0.26 | Partially susceptible to presentation effects. |
| Certainty effect | 0.420 | 0.72 | -0.30 | Partially susceptible to probability weighting. |

### Above-Human Susceptibility

| Bias | BMS | Human | Gap | Interpretation |
|------|:---:|:-----:|:---:|----------------|
| Base rate neglect | **0.835** | 0.68 | **+0.16** | Fails Bayesian reasoning on novel scenarios. Memorized canonical examples but didn't learn the principle. |
| Endowment effect | **0.818** | 0.65 | **+0.17** | Overvalues "possessed" items in hypotheticals. Training data correlates ownership framing with higher valuations. |

## Key Finding: The Contamination Story

The single most important finding is **base_rate_neglect going from 0.000 to 0.835** after template decontamination (PP-011).

### What happened

The pre-fix templates used the canonical Kahneman & Tversky (1973) engineers/lawyers paradigm. Frontier LLMs have seen this exact problem thousands of times in training data. Sonnet 4.5 scored 0.000 — appearing to perfectly resist the bias.

### What we found

When we replaced engineers/lawyers with novel profession pairs (forensic accountant/marketing coordinator, marine biologist/supply chain analyst, etc.) — same cognitive mechanism, novel surface features — the model's score jumped to 0.835. It was *pattern-matching the famous example*, not reasoning about base rates.

### What this means

1. **Benchmark contamination can create false negatives.** A zero score on a memorized example is not evidence of rational reasoning.
2. **Models have learned to identify bias tests**, not to avoid biased reasoning. The "debiasing" is superficial — triggered by recognizable framing, not by genuine Bayesian updating.
3. **This finding generalizes beyond base rate neglect.** Any benchmark using canonical examples from well-known psychology experiments is vulnerable to the same contamination effect. This has implications for how we evaluate LLM cognition broadly.

## The Dual-Process Interpretation

Kahneman's System 1 / System 2 framework maps onto LLM behavior in unexpected ways:

**System 1 analogs (fast, automatic):**
- LLMs exhibit System 1-like behavior on base rate neglect and endowment effect — they default to heuristic responses when examples are unfamiliar.
- But they *don't* exhibit System 1 behavior on conjunction fallacy or gambler's fallacy — the formal logic is so deeply embedded that it fires automatically.

**System 2 analogs (slow, deliberate):**
- Debiasing prompts (metacognitive instructions) can override bias on endowment effect (100% debiasable in prior runs).
- But loss aversion is only 67.5% mitigable — suggesting some biases are more structurally embedded than others.

**The asymmetry is the insight:** LLMs don't have a single "rationality level." They have a distinct profile where mathematical/logical biases are well-controlled but statistical/contextual biases remain problematic. This is the opposite of typical human profiles, where people struggle more with formal logic than with contextual reasoning.

## Metric-Level Observations

### CAS (Calibration Awareness Score)

12 of 15 biases flagged `insufficient_confidence_data=True`. Sonnet 4.5 rarely states numerical confidence in its responses. This is likely true across frontier models — they're trained to hedge linguistically rather than state probabilities.

**Implication for the paper:** CAS may need to be framed as aspirational or only reported for the 3 biases where confidence data exists. Alternatively, prompt engineering could elicit confidence statements, but that changes the naturalistic evaluation paradigm.

### RCI (Response Consistency Index)

At temperature=0 with 3 trials, RCI measures noise floor reliability, not behavioral stochasticity. The post-fix pipeline correctly labels this as `noise_floor_reliability` in metadata.

**Implication for the paper:** RCI claims should be carefully scoped. A consistency score of 0.98 means deterministic decoding is deterministic, not that the model is *choosing* to be consistent. For meaningful behavioral RCI, the protocol would need non-zero temperature and more trials.

### BMP (Bias Mitigation Potential)

Incomplete in this pilot due to 250 rate-limited debiasing evaluations. The full run needs lower concurrency or retry logic. Without BMP, one of the more unique metrics is missing. The post-fix evaluator now supports configurable rate-limit retries.

## Comparison to Pre-Fix Results

| Metric | Pre-Fix (v2) | Post-Fix | Reason for Change |
|--------|:------------:|:--------:|-------------------|
| Overall susceptibility | 13.17% | 21.47% | Decontamination + scoring fixes |
| base_rate_neglect | 0.000 | 0.835 | PP-011: template decontamination |
| certainty_effect | 0.112 | 0.420 | PP-008 debiasing baseline + PP-005 normalization |
| present_bias | 0.000 | 0.025 | Minor; possibly scoring normalization |
| endowment_effect | 0.827 | 0.818 | Stable (within noise) |
| loss_aversion | 0.060 | 0.056 | Stable (within noise) |
| conjunction_fallacy | 0.000 | 0.000 | Genuine structural resistance confirmed |

The overall increase from 13.17% to 21.47% is a **more accurate measurement**, not a regression. The pre-fix number was artificially low due to contaminated templates and scoring issues.

## Implications for Multi-Model Run

1. **Base rate neglect will be the differentiator.** The decontaminated templates will reveal which models have genuinely learned Bayesian reasoning vs which memorized canonical examples.

2. **Endowment effect susceptibility may be universal.** If multiple models score >0.8, it suggests this is a structural feature of language model training, not a model-specific weakness.

3. **Conjunction fallacy resistance may also be universal.** The probability axiom is well-represented in training data across all providers.

4. **Cross-model CAS comparison will be limited** unless some models naturally state confidence more than others. Reasoning models (Opus with thinking, GPT-5.2 with reasoning effort) might provide more confidence data.

5. **The human alignment pattern** (mostly under-human, but over-human on base rate neglect and endowment effect) may or may not generalize across architectures. This is the key empirical question.

## Questions for the Paper

1. **Framing:** Is this primarily a benchmark contribution (here's a tool for measuring LLM cognitive bias) or an empirical finding (here's what LLMs' cognitive profiles look like)?

2. **The contamination finding:** Does it warrant its own section or subsection? The 0→0.835 result is a methodological contribution independent of the bias findings.

3. **Human baselines:** These are literature-derived, not from the exact prompts used. How prominently should this limitation be stated? It affects the "over-human" / "under-human" claims.

4. **CAS and RCI:** Report as-is with caveats, or redesign the protocol before publication?

5. **N models:** Is 8 models sufficient for the claims being made, or should more be added for statistical power?
