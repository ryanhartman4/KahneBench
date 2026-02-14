# Metrics Correctness Review

**File reviewed:** `src/kahne_bench/metrics/core.py`
**Reviewer:** metrics-reviewer agent
**Date:** 2026-02-11
**Scope:** All 6 metrics (BMS, BCI, BMP, HAS, RCI, CAS), MetricCalculator, HUMAN_BASELINES

---

## Executive Summary

The metrics implementation is mathematically sound overall, with well-handled edge cases and a comprehensive unknown-rate tracking system. I identified **1 latent bug** that will crash if triggered, **5 moderate design issues** worth documenting for the publication, and **6 minor issues** that are acceptable but should be acknowledged as limitations.

| Severity | Count | Summary |
|----------|-------|---------|
| Critical (latent bug) | 1 | HAS crashes if any HUMAN_BASELINES entry is `None` |
| Moderate (design) | 5 | Sample vs population std mismatch in BCI; BMS cap loses info; CAS defaults inconsistent; CAS unknown_rate semantics differ; BMS sensitivity spacing incorrect for gaps |
| Minor | 6 | Positional coupling in BMS; RCI defaults; BMP substring matching; HAS fixed threshold; BCI includes control; aggregation RCI trial_count semantics |

---

## 1. Bias Magnitude Score (BMS)

### Formula
```
magnitude[i] = min(weight[i] * |treatment_mean - control_mean|, 1.0)
overall_BMS = sum(agg_w[i] * mag[i] for observed i) / sum(agg_w[i] for observed i)
```

### Verdict: Correct, with 3 minor issues

**Strengths:**
- Renormalization correctly prevents deflation when fewer than 4 intensities are present (verified by tests `TestBMSIntensityRenormalization`)
- Unknown rate tracking is comprehensive: counts all results including control and treatment
- `high_unknown_rate` flag at >50% with logger warning provides good transparency
- Default weights are well-documented with rationale and override mechanism

**Issue B1 (Minor): Aggregation weights positionally coupled to enum order**

`DEFAULT_AGGREGATION_WEIGHTS = [0.1, 0.3, 0.4, 0.2]` maps positionally via `zip(aggregation_weights, TriggerIntensity)`. This depends on Python enum iteration order matching declaration order (WEAK, MODERATE, STRONG, ADVERSARIAL). While Python guarantees this, it's fragile — reordering the enum would silently break the mapping.

*Recommendation:* Use a dict like `DEFAULT_INTENSITY_WEIGHTS` for consistency, or add a comment noting the order dependency.

**Issue B2 (Moderate): Cap at 1.0 loses information for weak triggers**

For WEAK triggers (weight=2.0), any raw deviation > 0.5 gets capped:
- deviation 0.5 -> weighted 1.0 (cap)
- deviation 0.6 -> weighted 1.2 -> capped at 1.0
- deviation 0.8 -> weighted 1.6 -> capped at 1.0

This means BMS cannot distinguish between moderate and strong deviations under weak triggers. The susceptibility weighting philosophy is correct, but the cap creates a ceiling effect for the most diagnostic intensity level.

*Impact on publication:* Should be noted as a known ceiling effect. Alternative: allow BMS > 1.0 for susceptibility interpretation, or apply cap after aggregation.

**Issue B3 (Moderate): Intensity sensitivity uses non-ordinal spacing**

Lines 202-206: When intensities are missing (e.g., only WEAK and ADVERSARIAL are present), the linear fit assigns x-values [0, 1] as if they're adjacent, when conceptually there are 2 intensities between them. This could produce misleading sensitivity slopes.

```python
x = list(range(len(magnitudes)))  # [0, 1] even for non-adjacent intensities
y = [magnitudes[i] for i in TriggerIntensity if i in magnitudes]
```

*Impact:* Low — `intensity_sensitivity` is a secondary diagnostic metric, not used in scoring. But should be noted if published.

### Edge Cases: All handled correctly
- Empty control results -> `control_mean = 0.0` via `mean([]) if [] else 0.0`
- Empty treatment list for intensity -> `treatment_mean = 0.0`
- No treatments at all -> `observed_weight_sum = 0` -> `overall = 0.0`
- All unknowns -> both means default to 0.0, magnitude = 0.0
- Single intensity -> renormalization gives correct single-intensity score

---

## 2. Bias Consistency Index (BCI)

### Formula
```
consistency = 1 - min(stdev(domain_scores) / 0.5, 1.0)
is_systematic = (count(score > 0.5) / total_domains) > 0.7
```

### Verdict: Correct, with 1 moderate issue

**Strengths:**
- Clean separation of magnitude (mean_bias_score) and uniformity (consistency_score)
- `is_systematic` provides a useful binary signal for filtering
- Handles single-domain case correctly (consistency=1.0, std=0.0)

**Issue C1 (Moderate): Sample stdev vs population max-std mismatch**

The code uses `statistics.stdev()` (sample standard deviation, divides by n-1) but normalizes by `max_possible_std = 0.5` which is the *population* standard deviation maximum for [0,1] range.

For n=2 domains with values [0, 1]:
- Sample stdev = 0.707
- `0.707 / 0.5 = 1.414` -> clamped to 1.0
- consistency = 0.0

For n=2 domains with values [0.2, 0.8]:
- Sample stdev = 0.424
- `0.424 / 0.5 = 0.849`
- consistency = 0.151

The sample stdev systematically overstates variance for small n, causing BCI to bottom out prematurely. With 2-3 domains (common in practice), even moderate divergence yields near-zero consistency.

*Impact on publication:* For 5 ecological domains, this is less severe (sample and population stdev converge). But evaluations using fewer domains may see artificially low consistency. Consider using `pstdev()` or documenting the small-n bias.

### Edge Cases: All handled correctly
- Empty domain scores -> returns defaults (mean=0.0, consistency=0.0, not systematic)
- Single domain -> stdev=0.0, consistency=1.0
- All domains identical -> stdev=0.0, consistency=1.0
- All domains have only unknown results -> domain excluded from scoring

---

## 3. Bias Mitigation Potential (BMP)

### Formula
```
effectiveness = max(0, (baseline - best_score) / baseline)  if baseline > 0
requires_warning = warning_method_avg < cot_method_avg
```

### Verdict: Correct, well-handled

**Strengths:**
- Division by zero properly guarded (`if baseline > 0`)
- `min(debiased_scores, key=debiased_scores.get)` correctly selects best method
- Effectiveness bounded to [0, 1] (max = 1.0 when best_score = 0; can't go negative due to `max(0, ...)`)
- Empty debiasing results return sensible defaults (effectiveness=0.0, requires_warning=True)

**Issue D1 (Minor): Substring-based method matching**

Lines 413-414:
```python
cot_methods = [m for m in debiased_scores if "chain" in m.lower() or "step" in m.lower()]
warning_methods = [m for m in debiased_scores if "warn" in m.lower() or "bias" in m.lower()]
```

A method named `"blockchain_analysis"` would match `"chain"`, and `"bias_correction"` would match `"bias"`. In practice this is fine since method names are framework-controlled, but the string matching is fragile.

*Recommendation:* Document expected method naming conventions, or use an enum/protocol for debiasing method types.

**Issue D2 (Minor): `requires_warning` semantics**

When no warning methods or no CoT methods are found, their average defaults to `baseline`. This means `requires_warning = baseline < baseline = False`. The default "doesn't require warning" when method names don't match the substring patterns could be misleading. If neither pattern matches, the system silently concludes no explicit warning is needed.

### Edge Cases: All handled correctly
- Zero baseline -> effectiveness = 0.0 (nothing to mitigate)
- Empty debiasing results -> method="none", effectiveness=0.0
- All unknown scores -> mean defaults to 0.0

---

## 4. Human Alignment Score (HAS)

### Formula
```
alignment = 1 - |model_rate - human_rate| / max(human_rate, 1 - human_rate)
direction = "aligned" if |diff| < 0.1, else "over"/"under"
```

### Verdict: Correct, with 1 latent bug

**Strengths:**
- Asymmetric normalization via `max(human_rate, 1 - human_rate)` is well-chosen — it penalizes divergence proportionally to the available range
- UNKNOWN_BASELINE_BIASES set with logger warning provides appropriate caution
- Custom human_baseline override parameter for flexibility
- Default 0.5 fallback for missing biases with warning

**Issue E1 (Critical/Latent): TypeError if HUMAN_BASELINES entry is None**

The type annotation `dict[str, float | None]` explicitly allows `None` values, but the calculation code does not handle them:

```python
human_rate = baselines[bias_id]  # Could be None
# ...
max_possible_diff = max(human_rate, 1 - human_rate)  # TypeError: int - NoneType
```

Currently no entries are `None` (verified: all 69 biases have float values), so this never triggers. But the type annotation invites future contributors to add `None` entries, which would crash. This is a **landmine bug**.

*Fix:* Either change the type annotation to `dict[str, float]` to prevent None entries, or add a None check:
```python
if human_rate is None:
    human_rate = 0.5
    logger.warning(f"Baseline for '{bias_id}' is None, using default 0.5")
```

**Issue E2 (Minor): Fixed 0.1 threshold for "aligned" doesn't scale**

The alignment direction uses a fixed absolute threshold of 0.1:
- For human_rate=0.05, a model at 0.14 (nearly 3x the human rate) is still "aligned"
- For human_rate=0.90, a model at 0.99 is still "aligned"

A relative threshold (e.g., 15% of baseline) would be more appropriate for extreme baselines. However, since all current baselines are in [0.45, 0.85], this is unlikely to cause problems in practice.

### Edge Cases: All handled correctly
- Missing baseline -> defaults to 0.5 with warning
- `max_possible_diff = 0` -> alignment = 1.0 (defensive, unreachable for valid inputs since max(h, 1-h) >= 0.5)
- Empty results -> model_rate = 0.0

---

## 5. Response Consistency Index (RCI)

### Formula
```
variance = stdev(scores)^2  if n > 1, else 0.0
max_variance = mean * (1 - mean)  if 0 < mean < 1, else 0.25
consistency = 1 - min(variance / max_variance, 1.0)
stability_threshold = 0.25 / trial_count
is_stable = variance < stability_threshold
```

### Verdict: Correct, with 1 minor and 1 moderate issue

**Strengths:**
- Bernoulli-based max_variance `mean * (1 - mean)` is the right theoretical maximum for binary-like scores
- Stability threshold scales correctly: `0.25 / n` tightens with more trials (at n=10: 0.025, matching the original hardcoded value per the code comment)
- Edge case at mean=0 or mean=1: correctly falls to `max_variance = 0.25`, and since variance=0 for identical scores, consistency=1.0

**Issue F1 (Moderate, same as C1): Sample variance vs Bernoulli population max**

`stdev(scores) ** 2` computes sample variance (n-1 denominator), while `max_variance = mean * (1 - mean)` is the Bernoulli population variance. For small n, sample variance overestimates, so `normalized_variance` can exceed 1.0. The `min(..., 1.0)` clamp prevents negative consistency, but this means for n=2-3, consistency scores cluster toward 0.0 even for moderate variance.

*Impact:* Similar to BCI — most significant when trial count is low.

**Issue F2 (Minor): Empty results default to consistency=1.0, is_stable=True**

When no valid scores exist (empty results or all unknown), the metric returns `consistency=1.0` and `is_stable=True` with `trial_count=0`. This means "no data" is indistinguishable from "perfectly consistent." A more honest representation would be `consistency=0.5` or a separate `has_sufficient_data` flag.

*Impact on publication:* If some biases have no valid trials, they appear maximally consistent in the fingerprint. The `trial_count=0` field is available to filter these, but the consumer must know to check it.

### Edge Cases: All handled correctly
- Empty scores -> defaults (see F2 above for concern)
- Single trial -> variance=0.0, consistency=1.0, stable=True
- All identical scores -> variance=0.0 regardless of n
- Mean at 0 or 1 boundary -> max_variance=0.25 (fallback)

---

## 6. Calibration Awareness Score (CAS)

### Formula
```
calibration_error = |mean_confidence - mean_accuracy|
calibration_score = 1 - min(calibration_error, 1.0)
overconfident = mean_confidence > mean_accuracy + 0.1
metacognitive_gap = max(0, mean_confidence - mean_accuracy)
```

### Verdict: Correct, with 2 moderate issues

**Strengths:**
- Clean separation of calibration (conf vs accuracy) from bias (conf vs bias)
- `overconfident` flag with 0.1 threshold is a useful clinical-style indicator
- `metacognitive_gap` captures one-directional overconfidence magnitude

**Issue G1 (Moderate): Default calibration_score is inconsistent with formula**

When no results have confidence statements, the defaults are:
```python
mean_confidence=0.5, actual_accuracy=0.5, calibration_error=0.0, calibration_score=0.5
```

But `calibration_score = 1 - calibration_error` with `calibration_error=0.0` should give `calibration_score=1.0`, not 0.5. The hardcoded 0.5 serves as a "don't know" signal, but it's inconsistent with the formula and could confuse consumers who expect `calibration_score = 1 - calibration_error` to always hold.

*Fix:* Either set `calibration_score=1.0` (formula-consistent) or add a `has_confidence_data: bool` flag and set `calibration_score=0.0` or NaN for "unknown."

**Issue G2 (Moderate): unknown_rate semantics differ from other metrics**

CAS computes `unknown_rate = results_without_confidence / total_results` — the fraction of results that lack a `confidence_stated` field. All other metrics compute `unknown_rate` as the fraction where the scorer returns `None` (failed extraction).

These measure fundamentally different things:
- Other metrics: "how many responses couldn't be evaluated at all"
- CAS: "how many responses didn't include a confidence statement"

*Impact:* If a consumer aggregates `unknown_rate` across metrics expecting uniform semantics, CAS values will be misleading. Should be documented.

### Edge Cases: All handled correctly
- No confidence statements -> defaults (see G1)
- All confidence = 0 -> calibration_error = accuracy, calibration_score depends on accuracy
- overconfident boundary: `0.6 > 0.5 + 0.1` is False (exact boundary is not overconfident) ✓

---

## 7. MetricCalculator & CognitiveFingerprintReport

### Verdict: Correct aggregation with 2 minor observations

**Strength: Per-condition RCI aggregation**

The calculator correctly groups results by condition before computing RCI (lines 1080-1112). This ensures that variance is measured across identical trials, not across different conditions. The aggregation uses:
- `avg_consistency = mean(per-condition consistencies)`
- `is_stable = all(per-condition is_stable)` (conservative AND)
- `trial_count = sum(per-condition counts)` (informational total)

This is well-designed.

**Observation M1 (Minor): BCI includes control condition results**

Lines 1056-1058 group ALL bias results (including control) by domain for BCI calculation. This means control condition scores contribute to domain averages, potentially diluting treatment-only consistency measurement.

*Whether this is intended:* Plausibly yes — BCI measures overall bias expression consistency across domains, not just treatment-induced bias. But if the intent is to measure treatment-induced bias consistency, control results should be excluded.

**Observation M2 (Minor): HAS uses all treatment intensities combined**

Line 1069: `all_treatments = [r for rs in treatments.values() for r in rs]` pools results from all intensity levels. The resulting `model_bias_rate` is an average across WEAK through ADVERSARIAL triggers. This means a model that is highly biased under ADVERSARIAL but resistant under WEAK gets an averaged rate, potentially masking intensity-specific alignment patterns.

*Recommendation for publication:* Consider reporting per-intensity HAS, or note this averaging behavior.

### CognitiveFingerprintReport.compute_summary()

- `overall_bias_susceptibility` correctly includes all biases (even high-unknown) for honest overall scoring
- `most_susceptible_biases` and `most_resistant_biases` correctly exclude high-unknown biases via `reliable_biases` filter
- `human_like_biases` uses alignment_score > 0.8 threshold
- `ai_specific_biases` uses direction == "over" (model more biased than humans)

**Minor concern:** A bias with alignment_score < 0.8 and direction == "under" (model less biased than humans) is categorized as neither human-like nor AI-specific. This seems correct — under-biased models are just less susceptible than humans.

---

## 8. HUMAN_BASELINES Review

### Coverage
- **69/69 biases** covered (verified: no missing, no extra entries)
- **3 biases** flagged in `UNKNOWN_BASELINE_BIASES` with limited empirical support
- **0 entries** are `None` (all have float values)

### Plausibility Check

| Bias | Cited Rate | Source | Assessment |
|------|-----------|--------|------------|
| conjunction_fallacy | 0.85 | Tversky & Kahneman (1983) | **Correct** — Linda problem: 85% of subjects chose conjunction |
| base_rate_neglect | 0.68 | Kahneman & Tversky (1973) | **Plausible** — engineer/lawyer problem, consistent with ~70% in literature |
| planning_fallacy | 0.80 | Kahneman & Tversky (1979) | **Plausible** — Buehler et al. found very high rates, 80% reasonable |
| anchoring_effect | 0.65 | Tversky & Kahneman (1974) | **Plausible** — effect sizes vary by paradigm, 65% reasonable |
| gain_loss_framing | 0.72 | Tversky & Kahneman (1981) | **Correct** — Asian Disease: ~72% preferred sure gain, ~78% preferred gamble for losses |
| default_effect | 0.75 | Johnson & Goldstein (2003) | **Plausible** — organ donation opt-in/opt-out difference was dramatic |
| scope_insensitivity | 0.78 | Kahneman & Knetsch (1992) | **Plausible** — embedding effect showed high rates of insensitivity |
| gambler_fallacy | 0.45 | [unspecified] | **Plausible** — lower than other biases, meta-analyses show varied rates |
| loss_aversion | 0.70 | Kahneman & Tversky (1979) | **Plausible** — 2:1 loss aversion ratio translates to high susceptibility |
| overconfidence_effect | 0.75 | Lichtenstein et al. (1982) | **Correct** — calibration studies consistently show ~75% overconfidence |
| certainty_effect | 0.72 | Allais (1953) | **Plausible** — Allais paradox replication rates consistent |
| peak_end_rule | 0.75 | Kahneman et al. (1993) | **Plausible** — cold water experiment showed strong effect |
| inattentional_blindness | 0.65 | Simons & Chabris (1999) | **Slightly high** — original gorilla study: ~50% missed. 65% may be from extended paradigms |
| hot_hand_fallacy | 0.55 | Gilovich et al. (1985) | **Plausible** — debate ongoing, moderate rate reasonable |

**Overall assessment:** Baselines are well-sourced and plausible. Most rates are within the range reported in meta-analyses. A few (like inattentional_blindness at 0.65) are on the high side compared to the most-cited study, but within the range of replication variations.

### Limitations (well-documented in code)

The extensive comment block (lines 432-453) correctly identifies 5 key limitations:
1. Different populations across studies
2. Different stimulus materials from KahneBench
3. Response format differences
4. Point estimates vs meta-analytic means
5. Cultural/temporal variation not captured

These limitations are appropriate for an arXiv publication.

---

## 9. Unknown Rate Guardrails

### Assessment: Effective and well-implemented

The unknown rate system operates at three levels:

1. **Per-result:** Scorers return `None` for failed extractions; each metric filters these out and counts them.
2. **Per-metric:** Each metric stores `unknown_rate` (0-1) in its result dataclass.
3. **Per-report:** `compute_summary()` propagates `unknown_rates_by_bias`, `high_unknown_rate_biases`, and filters rankings.

**Guardrail effectiveness:**
- `high_unknown_rate` flag at >50% triggers logger warnings
- `most_resistant_biases` and `most_susceptible_biases` exclude flagged biases
- `overall_bias_susceptibility` includes all biases (honest overall, avoids selection bias)

**One gap:** The 50% threshold is hardcoded (`UNKNOWN_THRESHOLD = 0.5`) in two places (BMS line 210, report line 901). Consider extracting to a module constant for consistency.

---

## Consolidated Recommendations for Publication

### Must Fix Before Publication

1. **HAS None-handling bug (E1):** Change type annotation to `dict[str, float]` or add None guard. This is a latent crash waiting to happen.

### Should Fix / Document

2. **BCI/RCI sample vs population stdev (C1, F1):** Either switch to `pstdev()` or note in the paper that consistency scores may be conservative for small n. This affects interpretability.

3. **CAS default inconsistency (G1):** Fix the no-data default to either be formula-consistent (1.0) or add a `has_data` flag.

4. **CAS unknown_rate semantics (G2):** Document that CAS's unknown_rate measures missing confidence statements, not failed extractions. Consider renaming to `missing_confidence_rate` for clarity.

5. **BMS cap ceiling effect (B2):** Document in the paper that WEAK trigger BMS saturates at raw deviation > 0.5. This is a design trade-off, not a bug.

### Nice to Have

6. **BMS aggregation weights as dict (B1):** Prevents silent breakage if enum is reordered.
7. **RCI empty-result semantics (F2):** Consider consistency=0.5 or `has_data` flag for no-data cases.
8. **BMS sensitivity spacing (B3):** Use ordinal intensity values [0, 1, 2, 3] for the fit, not indices of present intensities.

---

## Test Coverage Assessment

The test file (`tests/test_metrics.py`) contains **69 tests** across 14 test classes. Coverage is strong:

| Metric | Test Count | Edge Cases Covered | Assessment |
|--------|------------|-------------------|------------|
| BMS | 14 | Empty control/treatment, single intensity, renormalization, high unknown rate | Excellent |
| BCI | 6 | Empty, single domain, variable domains, all identical | Good |
| BMP | 5 | Zero baseline, best method selection, requires_warning, empty debiasing | Good |
| HAS | 5 | Missing baseline, aligned/over/under direction, boundary at 0.1 | Good |
| RCI | 5 | Perfect consistency, inconsistent, empty, single trial, stability boundary | Good |
| CAS | 4 | Well-calibrated, overconfident, no confidence, boundary at 0.1 | Good |
| Unknown handling | 9 | Per-metric unknown rate, all-unknown edge case, report tracking | Excellent |
| MetricCalculator | 2 | Default and custom scorer | Minimal |
| HUMAN_BASELINES | 4 | Coverage, range validation, unknown bias documentation | Good |

**Gap:** MetricCalculator integration tests are minimal — only 2 tests verifying it doesn't crash. No test verifies that the per-condition RCI aggregation, BCI domain grouping, or HAS intensity pooling produce correct values in an end-to-end scenario.
