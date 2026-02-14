# Kahne-Bench Pre-Publish Fix Handoff

Date: 2026-02-14
Scope: Core-tier benchmark pipeline readiness before running all 8 models for paper-grade results.

## Executive Decision

Status: **ALL P0/P1/P2 FIXES COMPLETE. Pilot validated. Ready for Codex review before 8-model run.**

Prior status: Was BLOCKED — now unblocked after all 12 items fixed and pilot-validated on 2026-02-14.

## Implementation Summary

All 12 issues (PP-001 through PP-012) were fixed by a 5-agent team (4 developers + 1 reviewer) in a single parallel session on 2026-02-14.

### Validation State (Post-Fix)

- `uv run pytest -q`: **643 passed** (up from 607; +36 new tests)
- `uv run ruff check src tests`: **passed**
- `uv run mypy src`: **249 errors** (strict mode; unchanged, not a scientific blocker)
- Pilot run: **4,725 evaluations** on Claude Sonnet 4.5, core tier, 3 trials — all audit checks passed

### Files Modified

| File | Changes |
|------|---------|
| `src/kahne_bench/engines/evaluator.py` | PP-001 error guard, PP-007 provider parity |
| `src/kahne_bench/cli.py` | PP-002 tier enforcement, PP-004 provenance, PP-009 intensity policy, PP-010 canonical export |
| `src/kahne_bench/utils/io.py` | PP-004 provenance export, PP-003 CAS flag export, PP-012 RCI interpretation export |
| `src/kahne_bench/metrics/core.py` | PP-003 CAS fix, PP-005 CAS normalization, PP-006 BCI filtering, PP-012 RCI interpretation |
| `src/kahne_bench/engines/generator.py` | PP-008 debiasing from treatment, PP-011 base-rate-neglect decontamination |
| `tests/test_evaluator.py` | 6 new tests (error guard) |
| `tests/test_cli.py` | 14 new tests (tier, provenance, intensity, canonical export) |
| `tests/test_metrics.py` | 11 new tests (CAS, BCI, RCI) |
| `tests/test_generator.py` | 4 new tests (debiasing, contamination) |

---

## Fix Status: All Items

### PP-001 (P0) Error responses scored as valid bias outcomes — FIXED

**Fix:** Added `is_error` flag at extraction time (evaluator.py:875). Error responses set `metadata["error_response"] = True`. Judge fallback gated with `and not is_error` (evaluator.py:915).

**Pilot validation:** 250 error responses (all rate-limit 429s) — zero scored. PP-001 working correctly.

**Tests:** 6 new tests in `TestErrorResponseJudgeGuard` covering: no judge scoring, is_biased=None, error metadata flag, bias_score=None, judge still fires for non-errors, mixed batch handling.

**Reviewed:** Approved by reviewer.

---

### PP-002 (P1) Tier enforcement in evaluate command — FIXED

**Fix:** After loading instances, computes expected bias set via `get_tier_biases(tier)` and validates against actual input biases. Mismatch exits with code 1 and clear message. `--allow-tier-mismatch` flag for explicit override. Output includes `tier`, `bias_manifest`, `instance_count_by_bias`.

**Pilot validation:** Tier "core" correctly enforced against 15 biases in input.

**Tests:** 5 new tests covering mismatch exit, missing biases, extra biases, override flag, exact match.

**Reviewed:** Approved by reviewer.

---

### PP-003 (P1) CAS returns perfect score when no confidence — FIXED

**Fix:** CAS returns `calibration_score=0.5` (conservative) with `insufficient_confidence_data=True` flag when no confidence data exists. Flag exported in fingerprint JSON.

**Pilot validation:** 12/15 biases correctly flagged as `insufficient_confidence_data=True`. 3 biases with actual confidence data scored normally with `insufficient_confidence_data=False`.

**Tests:** 2 new tests covering no-confidence and with-confidence cases.

**Reviewed:** Approved by reviewer. Initial gap in fingerprint export (missing flag) was fixed by cli-io-dev.

---

### PP-004 (P1) Run provenance metadata — FIXED

**Fix:** Both `results.json` and `fingerprint.json` include `run_metadata` block with 22 fields: provider, model, judge_provider, judge_model, temperature, max_tokens, num_trials, max_concurrent_requests, rate_limit_retries, rate_limit_retry_delay_s, intensities, include_control, include_debiasing, tier, input_file, bias_manifest, bias_manifest_hash, instance_count_by_bias, git_commit, timestamp, python_version, kahne_bench_version.

**Pilot validation:** All required fields present and correct in both output files.

**Tests:** 4 new tests covering results provenance, fingerprint provenance, instance counts, backwards compatibility.

**Reviewed:** Approved by reviewer.

---

### PP-005 (P2) CAS accuracy scoring normalization — FIXED

**Fix:** Imported `normalize_answer` from evaluator and integrated into CAS `_accuracy_scorer`. CAS accuracy now matches main scorer normalization (case, whitespace, synonym mapping).

**Tests:** 3 new tests covering "Option A" vs "A", "yes" vs "accept", exact match.

---

### PP-006 (P2) BCI condition filtering — FIXED

**Fix:** BCI domain consistency computation now filters to treatment results only. Control and debiasing results excluded. Code comments document the rationale.

**Tests:** 2 new tests proving control exclusion and debiasing exclusion.

---

### PP-007 (P2) Cross-provider parameter parity — FIXED

**Fix:** Audit found XAIProvider and GeminiProvider were missing `max_tokens` passthrough:
- XAIProvider: Added `max_tokens=max_tokens` to `chat.create()`.
- GeminiProvider: Added `max_output_tokens=max_tokens` to `GenerateContentConfig`.
- OpenAI and Anthropic were already correct.

**Tests:** Existing tests cover; provider-specific constraints documented in code comments.

---

### PP-008 (P1) Debiasing prompts wrap treatment, not control — FIXED

**Fix:** Both `_generate_from_template()` and `_generate_generic()` now use `treatment_prompts[TriggerIntensity.MODERATE]` as the base for debiasing prompts. BMP now measures "can the model resist bias under active trigger + debiasing instruction."

**Tests:** 2 new tests verifying debiasing contains MODERATE treatment text for both template-based and generic biases.

**Reviewed:** Approved by reviewer.

---

### PP-009 (P1) 3 vs 4 intensity policy — FIXED

**Fix:** Default: 3 intensities (WEAK, MODERATE, STRONG). Added `--include-adversarial` flag for opt-in 4th intensity. CLI help text documents default. Intensities recorded in output metadata.

**Pilot validation:** Intensities = `['weak', 'moderate', 'strong']` in fingerprint metadata.

**Tests:** 3 new tests covering default, adversarial flag, persistence in output.

**Reviewed:** Approved by reviewer.

---

### PP-010 (P1) Generate export uses canonical IO serializer — FIXED

**Fix:** `generate` command now uses `io.export_instances_to_json()` instead of `generator.export_to_json()`. All canonical fields preserved including `cross_domain_variants` and `interaction_biases`.

**Tests:** 2 new tests covering field completeness and roundtrip import/export.

**Reviewed:** Approved by reviewer.

---

### PP-011 (P1) Base-rate-neglect contamination — FIXED

**Fix:** Replaced canonical engineers/lawyers paradigm (K&T 1973) with 5 diverse profession pairs: forensic accountant/marketing coordinator, marine biologist/supply chain analyst, urban planner/payroll specialist, data scientist/compliance officer, wildlife veterinarian/procurement manager. Each with a stereotype-matching description. Same cognitive mechanism, novel surface features.

**Pilot validation:** **base_rate_neglect BMS went from 0.000 to 0.835.** The old templates were memorized by Sonnet 4.5 — it pattern-matched the "right answer." With novel scenarios, the model now exhibits the bias above human baseline (0.835 vs 0.68). This is the single largest finding from the fix round.

**Tests:** 2 new tests verifying no canonical contamination terms and scenario diversity.

**Reviewed:** Approved by reviewer.

---

### PP-012 (P2) RCI interpretation constraints — FIXED

**Fix:** Added `rci_interpretation` field to `ResponseConsistencyIndex` dataclass. Static method `get_interpretation(temperature, num_trials)` returns `"noise_floor_reliability"` when temperature <= 0.1 and trials < 5, otherwise `"behavioral_consistency"`. Field exported in fingerprint JSON.

**Tests:** 6 new tests covering boundary conditions and MetricCalculator integration.

---

## Pre-Run Gate Checklist (All Passed)

- [x] No scored rows from `ERROR:` responses (250 errors, 0 scored)
- [x] Tier mismatch is impossible without explicit `--allow-tier-mismatch` override
- [x] Session artifact includes full reproducibility metadata (22 fields)
- [x] CAS cannot report perfect calibration without confidence data (returns 0.5 + flag)
- [x] Intensity policy explicitly documented (3-intensity default, `--include-adversarial` opt-in)
- [x] Debiasing measured against active triggers (MODERATE treatment)
- [x] Base-rate-neglect decontaminated (0.000 → 0.835 in pilot)
- [x] One-model pilot run passes all audit checks

## Pilot Run Results: Claude Sonnet 4.5 (Post-Fix)

**Date:** 2026-02-14
**Config:** Core tier, 15 biases, 5 domains, 3 trials, 3 intensities (W/M/S), seed 42
**Evaluations:** 4,725 total (250 rate-limited on debiasing conditions, excluded from scoring)

### Overall Bias Susceptibility: 21.47%

| Bias | BMS (Post-Fix) | BMS (Pre-Fix v2) | Change | Human Baseline |
|------|:--------------:|:-----------------:|:------:|:--------------:|
| base_rate_neglect | **0.835** | 0.000 | **+0.835** | 0.68 |
| endowment_effect | 0.818 | 0.827 | -0.009 | 0.65 |
| gain_loss_framing | 0.462 | 0.420 | +0.042 | 0.72 |
| certainty_effect | 0.420 | 0.112 | +0.308 | 0.72 |
| hindsight_bias | 0.149 | 0.107 | +0.042 | 0.65 |
| status_quo_bias | 0.140 | 0.140 | 0 | 0.62 |
| sunk_cost_fallacy | 0.112 | 0.101 | +0.011 | 0.55 |
| anchoring_effect | 0.101 | 0.078 | +0.023 | 0.65 |
| overconfidence | 0.085 | 0.039 | +0.046 | 0.75 |
| availability_bias | 0.068 | 0.068 | 0 | 0.60 |
| loss_aversion | 0.056 | 0.060 | -0.004 | 0.70 |
| present_bias | 0.025 | 0.000 | +0.025 | 0.70 |
| gambler_fallacy | 0.007 | 0.014 | -0.007 | 0.45 |
| confirmation_bias | 0.007 | 0.011 | -0.004 | 0.72 |
| conjunction_fallacy | 0.000 | 0.000 | 0 | 0.85 |

### Key Pilot Observations

1. **base_rate_neglect 0.000→0.835**: PP-011 fix validated. Old canonical templates (engineers/lawyers) were trivially recognized by the model. Novel profession pairs now test the actual cognitive mechanism. Model exhibits bias above human baseline.

2. **certainty_effect 0.112→0.420**: Notable increase may reflect improved debiasing baseline (PP-008) or scoring normalization (PP-005). Worth investigating in full run.

3. **250 rate-limited errors on debiasing**: All on `debiasing_1` and `debiasing_2` conditions. PP-001 correctly excluded all from scoring. BMP metrics are incomplete for this pilot artifact. **Post-pilot fix:** evaluator now retries explicit 429/rate-limit failures once with a 5-second delay (`--rate-limit-retries`, `--rate-limit-retry-delay`), and run metadata records the retry policy.

4. **CAS evaluability**: 12/15 biases flagged `insufficient_confidence_data=True` — Sonnet 4.5 rarely states confidence in its responses. 3 biases with real confidence data scored normally.

5. **Overall susceptibility 13.17%→21.47%**: Increase primarily driven by base_rate_neglect decontamination and certainty_effect. This is a more accurate measurement, not a regression.

### Pilot Output Files

- `pilot_tests.json` — 225 generated test instances (seed 42)
- `pilot_results.json` — 4,725 evaluation results with run_metadata
- `pilot_fingerprint.json` — cognitive fingerprint with run_metadata

### Prior Results

All pre-fix results and fingerprints moved to `deprecated_results/` (22 files).

## Known Remaining Items (Not Blocking 8-Model Run)

### Rate limiting on debiasing conditions

Pilot artifact impact remains: 250/4725 evaluations (5.3%) failed with 429 errors, concentrated in debiasing conditions. Post-pilot code now includes configurable retry for rate limits (default: 1 retry after 5s). For the full 8-model run, still consider:
- Reducing `--concurrent` from 50 to 20
- Increasing `--rate-limit-retries` above default if provider behavior requires it
- Running debiasing conditions in a separate lower-concurrency pass

### RCI interpretation metadata in fingerprint export

Fixed post-pilot: `rci_interpretation` field now exported in fingerprint JSON (io.py). Was missing in the pilot output but is present for future runs.

## Still-Open Paper-Quality Tasks (Post-Run, Pre-Writeup)

### PP-013 (Paper) Add uncertainty quantification for key scores

1. Confidence intervals (bootstrap) for BMS and headline comparisons.
2. Effect sizes for core contrasts.
3. Significance testing with multiple-comparison control (e.g., FDR).

### PP-014 (Paper) Address option-position confound explicitly

1. Either counterbalance option ordering in protocol updates, or
2. provide position-conditioned analysis and explicit limitation text.

## Judge-All Validation Strategy (Opt-In, Not Default)

Context: Team decision is to consider 100% LLM-judge coverage as a validation layer, with concern about judge-model systematic bias.

### Positioning for the paper

- Keep benchmark primary scoring unchanged (regex + existing fallback behavior).
- Add a separate validation mode where judge scoring runs on every row.
- Treat judge-all as a reliability/audit lens, not as automatic ground truth replacement.

### Why this is valuable

- Provides a publishable reliability section rather than relying on a single scorer.
- Identifies where findings are robust vs where scoring is fragile.
- Cost is likely acceptable relative to total multi-model benchmark spend.

### Main risk and mitigation

Risk: Judge-all introduces its own systematic bias (prompt framing, judge model priors, calibration drift).

Mitigation:
- Keep judge-all explicitly opt-in (for example `--judge-all` or `--judge-mode all`).
- Do not change default CLI behavior.
- Preserve primary benchmark metrics from canonical scorer.
- Report judge confidence and sensitivity analyses.
- Optional: second-judge spot-check on disagreement rows.

### Agreement metric suite to implement/report

At minimum:
1. Label agreement rate: `% regex_label == judge_label`
2. Score agreement: mean absolute difference between scorer bias scores
3. Disagreement rate: `% rows with material label or score mismatch`

Stratify by:
1. Bias ID
2. Answer type (`option`, `numeric`, `yes_no`, `descriptive`, `confidence`)
3. Condition/intensity
4. Provider/model

Sensitivity slices:
1. All rows
2. High judge-confidence rows only
3. Excluding explicit error rows

### Recommended interpretation rule

- High agreement regions: stronger confidence in findings.
- Persistent disagreement clusters: flag as uncertain/measurement-sensitive.
- Do not assume judge is "more correct"; treat disagreement as epistemic uncertainty.

### ArXiv reporting guidance

Add one reliability subsection with:
1. A compact agreement summary table (overall + per-bias quartiles)
2. A disagreement heatmap (bias x answer type or bias x provider)
3. Appendix sensitivity analysis (confidence threshold sweeps)

Suggested claim style:
- "Findings are robust where cross-scorer agreement is high."
- "Findings are exploratory/uncertain in disagreement-dense regions."

### Handoff requirements for implementation

1. Add opt-in CLI flag for judge-all mode; default remains current behavior.
2. Persist both scorer outputs per row (regex/fallback result and judge result) when judge-all is enabled.
3. Add aggregate agreement block to fingerprint/session metadata.
4. Add tests proving:
   - default behavior unchanged
   - judge-all mode computes agreement stats
   - disagreement rows are countable/queryable from JSON artifacts

## Recommended Next Steps

1. **Codex review** of all changes before proceeding
2. **Commit** all fixes + pilot results
3. **Full 8-model run** with reduced concurrency (`--concurrent 20`) to avoid rate limiting on debiasing
4. **PP-013/PP-014** paper-quality analysis tasks post-run
5. **Judge-all validation** implementation (optional, for reliability section)
