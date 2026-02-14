# KahneBench Pre-Publication Audit Report

**Date:** 2026-02-14
**Scope:** Full audit for ArXiv readiness before 8-model benchmark run
**Auditors:** 5-agent MECE review team + lead synthesis
**Codebase state:** 607 tests passing, ruff clean, commit ba84fbd

---

## Executive Summary

KahneBench has **genuine novelty and a solid architecture**, but **several issues must be fixed before running the 8-model benchmark** to ensure publishable results. The framework is the broadest cognitive bias benchmark for LLMs (69 biases, 6 metrics, 5 domains) with a novel susceptibility-based weighting scheme. However, two metrics are non-functional in practice, one critical design flaw affects debiasing measurement, and statistical reporting is incomplete for peer review.

**Overall Readiness: NEEDS WORK (6/10)**

| Area | Rating | Status |
|------|--------|--------|
| Evaluation Pipeline | Low Risk | Minor issues only |
| Metric Formulas | 3.5/5 | CAS broken, 2 fixes needed |
| Test Generation | Adequate | 1 high-priority design flaw |
| Reproducibility | 2.5/5 | Several gaps to close |
| Scientific Methodology | 6/10 | Statistical reporting incomplete |

---

## CRITICAL Issues (Must Fix Before Benchmark Run)

### C1. Debiasing prompts wrap the CONTROL prompt, not treatment
**File:** `generator.py:4143-4155`
**Impact:** BMP (Bias Mitigation Potential) metric is measuring debiasing of an already-unbiased prompt. Since the control has no bias trigger, debiasing it measures nothing meaningful.
**Fix:** Change debiasing prompts to wrap treatment prompts (at MODERATE intensity) to test whether explicit debiasing can overcome the bias trigger.
**Consequence if unfixed:** BMP metric is uninterpretable. Cannot make any claims about model debiasability.

### C2. CAS metric is non-functional
**File:** `metrics/core.py:857-870`
**Impact:** In the Sonnet 4.5 v2 results, only 7.2% of responses include confidence statements. CAS defaults to `calibration_score=1.0` (perfect) when no confidence data exists. This means nearly all models will appear perfectly calibrated, which is vacuously true.
**Fix:** Either (a) add explicit confidence-eliciting language to prompts ("State your confidence as a percentage"), or (b) exclude CAS from reported metrics and present it as 5 metrics, not 6.
**Consequence if unfixed:** Reporting CAS as a metric undermines credibility. A reviewer will immediately identify this.

### C3. RCI is meaningless at temperature=0.0 with 3 trials
**File:** `evaluator.py:274` (temperature=0.0), `evaluator.py:277` (num_trials=3)
**Impact:** At temperature=0.0, trial-to-trial variance is API noise (floating-point non-determinism), not meaningful cognitive variance. 3 trials provide negligible statistical power.
**Fix:** Either (a) increase to 10+ trials with temperature=0.3 for genuine stochasticity measurement, or (b) explicitly frame RCI as a "noise floor / extraction reliability" metric in the paper and acknowledge it cannot distinguish systematic from stochastic behavior.
**Consequence if unfixed:** Claims about response consistency have no statistical foundation.

---

## HIGH Priority Issues (Fix for Paper Quality)

### H1. No confidence intervals on reported BMS scores
**Impact:** BMS scores are point estimates (e.g., "0.827") without precision indicators. A reviewer will ask: "How do you know 0.827 is meaningfully different from 0.420?"
**Fix:** Add bootstrap 95% CIs across domains/trials for each BMS score. Report these in the paper's results table.

### H2. No effect sizes reported
**Impact:** Standard for psychology-adjacent research. Without Cohen's d or similar, results can't be compared across studies.
**Fix:** Report Cohen's d for control vs treatment at each intensity level.

### H3. base_rate_neglect uses classic K&T engineer/lawyer problem
**File:** `generator.py` (base_rate_neglect template)
**Impact:** Scored 0.000 in both v1 and v2 evaluations. Same contamination pattern as pre-fix conjunction_fallacy — models recognize the famous paradigm and give the Bayesian answer. This is likely a broken test, not genuine resistance.
**Fix:** Diversify characters and descriptions, similar to how conjunction_fallacy was fixed with 10 novel character profiles.

### H4. Missing ADVERSARIAL intensity in CLI
**File:** `cli.py:448-452`
**Impact:** CLI hardcodes [WEAK, MODERATE, STRONG] while `EvaluationConfig` defaults to all 4 including ADVERSARIAL. BMS aggregation weights include ADVERSARIAL (0.2 weight). The CLI's 3-intensity run produces different BMS values than the default config.
**Fix:** Either add ADVERSARIAL to the CLI, or document the 3-intensity design explicitly and verify the BMS renormalization handles 3 intensities correctly (it does — weights are renormalized).

### H5. No significance testing for key claims
**Impact:** Claims like "only loss aversion is systematic" have no inferential support. With 15 biases × 5 domains, multiple comparisons correction is needed.
**Fix:** Apply Benjamini-Hochberg FDR correction for per-bias BMS significance. At minimum, report which BMS scores are significantly > 0.

### H6. Option position not counterbalanced
**Impact:** LLMs exhibit documented position bias (primacy toward option A). If rational=A for most biases, this deflates BMS; if biased=A, it inflates BMS. This is a confound.
**Fix:** Counterbalance A/B position across trials, or report position-conditioned scores and discuss as limitation.

### H7. Incomplete configuration logging in output JSON
**File:** `evaluator.py:1001-1005`
**Impact:** `model_config` records only max_tokens, temperature, num_trials. Missing: intensities used, include_control/debiasing flags, provider name, model ID, framework version, git commit, generation seed.
**Fix:** Expand `model_config` to include all evaluation parameters needed for full reproducibility.

### H8. `generator.export_to_json()` drops fields
**File:** `generator.py:4194-4217` vs `io.py:55-91`
**Impact:** The CLI `generate` command uses `generator.export_to_json()` which omits `cross_domain_variants` and `interaction_biases`. These are silently lost in the generate→evaluate pipeline.
**Fix:** Use `io.export_instances_to_json()` in the CLI `generate` command (cli.py:262).

### H9. _accuracy_scorer substring matching is too loose
**File:** `metrics/core.py:1038-1039`
**Impact:** `"a" in "accept"` → True → returns 0.8. Single-character answers frequently trigger false matches, propagating into CAS calibration values.
**Fix:** Add minimum length guard: `min(len(extracted), len(expected)) >= 3`.

---

## MEDIUM Priority Issues (Acceptable with Caveats)

### M1. Global random state instead of isolated Random instance
**File:** `generator.py:1754-1755`
**Impact:** `random.seed(seed)` sets global state. Works for single-threaded runs but fragile.
**Fix:** Replace with `self.rng = random.Random(seed)`.

### M2. 6 of 15 core biases lack meaningful domain variation
**Impact:** availability_bias, loss_aversion, endowment_effect, certainty_effect, confirmation_bias, and hindsight_bias use cosmetically similar prompts across all 5 domains. BCI for these biases may be inflated.
**Fix:** Acknowledge in paper. Note that true cross-domain testing applies to ~9 of 15 core biases.

### M3. BCI is_systematic with single domain
**File:** `metrics/core.py:340`
**Impact:** A single domain with score > 0.5 reports `is_systematic=True`. Should require minimum 2 domains.
**Fix:** Add `len(scores_list) >= 2` guard before the is_systematic calculation.

### M4. Human baseline comparison validity
**Impact:** HUMAN_BASELINES are from different studies, populations, stimuli, and eras. Comparing LLM performance on KahneBench prompts to these rates is an approximation.
**Fix:** Already acknowledged in code comments and LIMITATIONS.md. The paper must include a dedicated paragraph stating HAS comparisons are illustrative, not inferential.

### M5. No version tracking in outputs
**Impact:** Results can't be linked to specific code versions. Previous fixes changed metrics (5 bias template fixes in commit 08e44da).
**Fix:** Add `provenance` section: `{version, git_commit, python_version, generated_at}`.

### M6. `--tier` flag is cosmetic during evaluation
**File:** `cli.py:383, 465`
**Impact:** Accepted and printed but never used to filter instances. All loaded instances are evaluated regardless of tier.
**Fix:** Either remove from evaluate command, or add instance filtering by tier.

---

## LOW Priority Issues (Informational)

| # | Issue | Location | Notes |
|---|-------|----------|-------|
| L1 | BMS ceiling at WEAK weight 2.0x | metrics/core.py:190 | Documented. Deviation > 0.5 saturates to 1.0 |
| L2 | Aggregation weight positional coupling | metrics/core.py:73 | Fragile but correct. Enum reorder would break silently |
| L3 | Prompt length confound | generator.py templates | Treatments are 20-30 words longer than controls |
| L4 | certainty_effect EV gap may be too large | generator.py template | Min 1.2x EV advantage may create ceiling effect |
| L5 | endowment_effect uses only small-value items | generator.py template | $50-200 items; larger items might trigger stronger effects |
| L6 | No fingerprint import function | io.py | Export-only; can't programmatically compare models |
| L7 | Missing CI/CD pipeline | - | Tests only run locally |
| L8 | Gemini provider doesn't pass max_tokens | evaluator.py:257 | Uses default; may affect response length |

---

## Evaluation Pipeline Assessment (Lead Analysis)

Since the pipeline auditor had a routing issue, here is the lead's direct assessment:

### Answer Extraction: LOW RISK
- Priority ordering (Answer: line → patterns → fallback) is correct
- `_has_negative_context` correctly filters anchor references using adjacent regex matching
- Confidence extraction is properly isolated from answer numbers via stripping patterns
- 100% scoring rate in v2 results (0% unknown) validates the extraction + judge pipeline

### Scoring Logic: LOW RISK
- Numeric tolerance (1% epsilon) is appropriate for the value ranges used
- `normalize_answer()` handles common synonyms correctly; 100+ mappings cover the answer space
- Descriptive answer detection correctly routes to LLM judge
- score_response returns (None, None) for unknowns, (False, 0.0)/(True, 1.0) for matches

### Provider Implementations: LOW RISK
- xAI/Gemini `asyncio.to_thread()` wrapping is correct — runs sync code in thread pool
- OpenAI model detection for `max_completion_tokens` covers gpt-5, o3, o1 prefixes
- Temperature=0.0 passed to all providers (Gemini via GenerateContentConfig)
- One gap: Gemini doesn't forward `max_tokens` — uses SDK default

### Concurrency: LOW RISK
- Semaphore correctly limits concurrent API calls
- `progress_lock` is asyncio.Lock (not threading.Lock), correct for event loop
- No deadlock risk: lock scope is minimal (counter increment + callback)
- trial_delay_ms staggering works correctly

### Frame-Aware Scoring: LOW RISK
- `_resolve_biased_answer` and `_resolve_rational_answer` handle all paths
- frame_map lookup handles missing intensity keys with fallback to gain frame

---

## What's Genuinely Publication-Ready

1. **BMS metric**: Solid. Well-documented weights, correct renormalization, good unknown handling
2. **BCI metric**: Solid (with M3 caveat). Cross-domain consistency is meaningful
3. **BMP metric**: Solid formula, but C1 must be fixed first for the metric to be interpretable
4. **HAS metric**: Solid. Normalization is mathematically sound, edge cases correct
5. **Bias taxonomy**: 69 biases under unified K&T theory — broadest cognitive bias benchmark
6. **Multi-intensity susceptibility weighting**: Novel contribution, no prior work does this
7. **Answer extraction pipeline**: Reliable (100% scoring rate with judge fallback)
8. **Code quality**: High. 607 tests, ruff clean, good documentation

---

## Recommendations for the 8-Model Benchmark Run

### Before Running (MUST DO)
1. **Fix C1**: Change debiasing prompts to wrap treatment (MODERATE intensity), not control
2. **Fix H3**: Diversify base_rate_neglect templates (like conjunction_fallacy was fixed)
3. **Fix H4**: Decide on 3 vs 4 intensities and document explicitly
4. **Fix H7**: Expand model_config logging for reproducibility
5. **Fix H8**: Use `io.export_instances_to_json()` in CLI generate command
6. **Regenerate test cases** after fixes with `seed=42`

### Before Writing Paper (SHOULD DO)
7. Add bootstrap 95% CIs for BMS scores (H1)
8. Add effect sizes — Cohen's d for control vs treatment (H2)
9. Add significance testing with FDR correction (H5)
10. Decide on CAS/RCI: exclude, redesign, or caveat heavily (C2, C3)
11. Add sensitivity analysis on intensity weights
12. Fix _accuracy_scorer substring matching (H9)
13. Fix BCI is_systematic guard (M3)

### For the Paper Text
14. Frame as "5 functional metrics" (BMS, BCI, BMP, HAS, RCI-as-noise-floor) unless CAS/RCI are redesigned
15. Acknowledge domain variation is cosmetic for 6/15 biases
16. Include human baseline comparison caveats prominently
17. Report extraction method agreement (regex vs LLM judge) as inter-rater reliability
18. Discuss prompt length and option position as potential confounds

---

## Existing Results Validity

The Sonnet 4.5 v2 results (`sonnet45_core_fingerprint_v2.json`) are **partially valid**:

| Metric | Valid? | Notes |
|--------|--------|-------|
| BMS | ✅ Yes | Scores are mathematically correct given the templates |
| BCI | ⚠️ Partially | Valid for biases with domain variation; inflated for 6 biases |
| BMP | ❌ No | Debiasing wraps control — must rerun after C1 fix |
| HAS | ✅ Yes | Cross-study caveat acknowledged |
| RCI | ⚠️ Qualified | Valid as noise floor measurement, not as cognitive consistency |
| CAS | ❌ No | 7.2% confidence data — metric is vacuous |

**Recommendation:** After fixing C1 and H3, rerun Sonnet 4.5 as a validation baseline before the full 8-model run.

---

## Appendix: Audit Coverage Matrix

| Area | Auditor | Issues Found | Report Quality |
|------|---------|:---:|---|
| Evaluation Pipeline | Lead (direct review) | 3 LOW | Comprehensive — full source read |
| Metric Math | metrics-auditor | 2 HIGH, 1 MED, 4 LOW | Excellent — formula-by-formula verification |
| Test Generation | prompt-auditor | 1 HIGH, 3 MED, 3 LOW | Excellent — per-bias template review |
| Reproducibility | reproducibility-auditor | 3 HIGH, 2 MED | Thorough — CLI/serialization analysis |
| Scientific Design | methodology-auditor | 4 BLOCKER, 6 MAJOR, 4 MINOR | Excellent — ArXiv reviewer perspective |
