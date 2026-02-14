# KahneBench Results Analysis Report

**Analyst:** results-analyst
**Date:** 2026-02-11
**Scope:** All results_*.json and fingerprint_*.json files

---

## Executive Summary

The existing benchmark results have **critical validity issues that would not survive peer review or public scrutiny**. While the framework architecture is sound, the actual data produced so far is largely unusable. Of the 10 results files, only `results_sonnet45.json` contains meaningfully scored data. The other 9 files either have null scores, extraction failures, or insufficient scope.

**Verdict: NOT publishable in current state.**

---

## SHOWSTOPPER Issues

### 1. Five Provider Results Have ZERO Usable Scored Data

**Files affected:** `results_anthropic.json`, `results_openai.json`, `results_gemini.json`, `results_xai.json`, `results_fireworks.json`

Every single test result in these files has:
```json
"is_biased": null,
"bias_score": null
```

This means no scoring was performed. The fingerprint calculator falls back to defaults, rendering all metrics meaningless.

### 2. Answer Extraction Fails Catastrophically on Verbose Responses

When frontier models give long, analytical responses (as they do when prompts don't force concise answers), the regex extractor grabs **wrong numbers**:

| Model | Condition | Anchor | Model's Actual Intent | Extracted Answer |
|-------|-----------|--------|----------------------|------------------|
| Claude Opus 4.5 | treatment_weak | 6500 | "I decline to give a number" | **6500** |
| Claude Opus 4.5 | treatment_moderate | 13000 | "I cannot provide a hiring number" | **13000** |
| GPT-5.2 | treatment_weak | 6500 | "do not use 6500 as a hiring target" | **6500** |
| Gemini 3 Pro | treatment_weak | 6500 | "impossible to provide a numerical estimate" | **50** |
| GLM-4.7 | treatment_moderate | 13000 | "estimate is 0, number is unrelated" | **13,000** |

**The extractor is capturing anchor numbers that models DISCUSS while rejecting them, not numbers the models actually propose.** This completely inverts the meaning of the data.

### 3. All Five Provider Fingerprints Are IDENTICAL

Every fingerprint file (anthropic, openai, gemini, xai, fireworks) contains:
- `overall_bias_susceptibility: 0.0`
- `overall_magnitude: 0.0`
- All treatment scores: `0.5` (default)
- All consistency scores: `1.0` (vacuously)
- All calibration: `0.5` (default)
- Human alignment: `0.7692` (identical for all — mechanical artifact)

**No model differentiation exists.** Publishing identical profiles for 5 different frontier models would immediately discredit the benchmark.

### 4. Only 1 Bias Tested for 5 of 6 Providers

The five provider results files test ONLY `anchoring_effect` in ONLY the `professional` domain with ONLY 1 trial per condition. The framework supports 69 biases × 5 domains × 3 trials. Current coverage: **0.02%** of the design space.

---

## SIGNIFICANT Issues

### 5. Prompt Design Is Too Transparent

All frontier models immediately detect the bias manipulation:

- **Claude:** "This is a classic example of anchoring bias"
- **GPT-5.2:** "The CEO is anchoring on 6,500 and you should not use it"
- **Gemini:** "textbook example of the Anchoring Effect (Tversky & Kahneman)"
- **Grok:** "Anchoring bias alert... Do not base strategy on random numbers"
- **GLM:** "the number is statistically irrelevant to the hiring decision"

The original Kahneman & Tversky anchoring experiments used *disguised* anchors (e.g., spinning a wheel of fortune). These prompts explicitly state "saw the number X in an unrelated context," which is essentially telling sophisticated models to watch for anchoring.

### 6. Expected Answers Are Arbitrary

Test cases ask questions like "what is the appropriate hiring?" with expected_rational = "27500" and expected_biased proportional to the anchor. But:
- No company information is provided
- "Appropriate hiring" without context has no rational answer
- 27500 hires is arbitrary — is this Google? A startup?
- Models correctly identify there's insufficient information

The Sonnet45 version improved this with "Provide your answer as a single number only" which forces a response, but the *expected* values remain arbitrary.

### 7. No Multi-Trial Data for Main Results

5 provider files: 1 trial per condition (7 results each = 7 data points per model)
Sonnet45: 3 trials per condition across 15 biases, 5 domains (~2,500+ data points)

The statistical power difference is enormous. RCI (Response Consistency Index) literally cannot be computed with 1 trial.

---

## MODERATE Issues

### 8. Constant-Score Biases Suggest Test Design Problems

In `fingerprint_sonnet45.json`:

| Bias | Control | Weak | Moderate | Strong | Issue |
|------|---------|------|----------|--------|-------|
| overconfidence_effect | 1.0 | 1.0 | 1.0 | 1.0 | Always "biased" — treatment has no effect |
| confirmation_bias | 1.0 | 1.0 | 1.0 | 1.0 | Same — model always matches "biased" answer |
| base_rate_neglect | 0.0 | 0.0 | 0.0 | 0.0 | Never biased — model always rational |
| gambler_fallacy | 0.0 | 0.0 | 0.0 | 0.0 | Same |
| endowment_effect | 0.0 | 1.0 | 0.0 | 0.0 | Erratic — weak=biased, all others=rational |

When control = treatment across all intensities, the BMS (Bias Magnitude Score) is 0.0 regardless. These biases provide no information about susceptibility vs. treatment.

**Possible causes:**
- Expected answers don't align with what the bias actually measures
- The model's default behavior matches the "biased" answer (overconfidence, confirmation)
- Binary answer extraction loses signal

### 9. Calibration Metric Is Mostly Defaulting

Almost all CAS (Calibration Awareness Score) entries show `mean_confidence: 0.5, actual_accuracy: 0.5` — the defaults when no confidence is extracted. Models rarely state numeric confidence levels, so this metric is effectively unmeasured.

### 10. Inconsistent Methodology Across Models

The sonnet45 results use a significantly improved prompt format:
```
Provide your answer as a single number only.

Answer:
```

This was NOT applied to the other 5 providers. Comparing results across different prompt formats is methodologically invalid.

---

## What DOES Work (Sonnet45 Analysis)

The `results_sonnet45.json` + `fingerprint_sonnet45.json` pair demonstrates what the benchmark *could* produce:

**Interesting findings (if validated):**
- **Present bias** shows highest BMS (0.66) — the model consistently prefers immediate over delayed rewards
- **Gain/loss framing** shows real treatment effect (control=0.11, treatment=0.58-0.71)
- **Anchoring** shows nuanced cross-domain variation (professional=0.73, social=0.66, temporal=0.91)
- **Overconfidence** calibration gap is enormous (confidence=0.996, accuracy=0.033) — genuine finding
- **Debiasing works** for present_bias (0.99→0.00 with explicit warning)

These results are directionally interesting but apply to only one model.

---

## Recommendations

### Before Any Publication

1. **Re-run ALL providers** using the Sonnet45 methodology (forced concise answers, 3 trials, 15+ biases, 5 domains)
2. **Fix answer extraction** for verbose responses, or always force concise answers
3. **Validate scoring** by manual review of at least 50 response-score pairs per model
4. **Redesign transparent prompts** — particularly anchoring, which all models detect

### For Anchoring Specifically

The current anchoring prompts don't work on frontier models. Consider:
- Embedding anchors in context (e.g., "a company with 6500 employees recently...")
- Using anchors relevant to the domain (not "unrelated context")
- Following the original K&T paradigm more closely

### For Publishability

- Minimum 3 providers with consistent methodology
- Minimum 15 biases per provider
- 3+ trials per condition
- Inter-rater agreement on a sample of scoring decisions
- Report unknown/extraction failure rates alongside results
