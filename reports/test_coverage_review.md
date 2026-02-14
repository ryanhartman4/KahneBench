# Test Coverage & Quality Review

**Reviewer:** test-reviewer (automated)
**Date:** 2026-02-11
**Suite status:** 546 tests, 100% passing, 0.49s runtime

---

## 1. Test File Inventory

| Test File | Test Count | Module(s) Covered |
|-----------|-----------|-------------------|
| `test_generator.py` | ~120 | `engines/generator.py` (TestCaseGenerator, tiers, templates, biases) |
| `test_evaluator.py` | ~115 | `engines/evaluator.py` (AnswerExtractor, BiasEvaluator, TemporalEvaluator, ContextSensitivityEvaluator) |
| `test_metrics.py` | ~85 | `metrics/core.py` (all 6 metrics, MetricCalculator, CognitiveFingerprintReport) |
| `test_taxonomy.py` | ~35 | `biases/taxonomy.py` (BIAS_TAXONOMY, categories, interaction matrix, K&T core) |
| `test_io.py` | ~30 | `utils/io.py` (export/import JSON/CSV, roundtrips, error handling) |
| `test_integration.py` | ~12 | Full pipeline (generate → evaluate → metrics) |
| `test_advanced_generators.py` | ~55 | `engines/generator.py` (NovelScenarioGenerator, MacroScaleGenerator), `engines/compound.py`, `engines/robustness.py` |
| `test_variation.py` | ~25 | `engines/variation.py` (VariationGenerator, VariationRobustnessScore) |
| `test_bloom_generator.py` | ~25 | `engines/bloom_generator.py` (BloomBiasGenerator parsing and pipeline) |
| `test_conversation.py` | ~22 | `engines/conversation.py` (ConversationalEvaluator, strategies, scoring) |
| `test_quality.py` | ~15 | `engines/quality.py` (QualityJudge, QualityScores, batch assessment) |
| `test_judge.py` | ~18 | `engines/judge.py` (LLMJudge, XML parsing, prompt structure) |

## 2. Coverage Map: Tested vs Untested

### 2.1 Well-Covered Modules (>70% estimated function coverage)

| Module | Key Functions Tested | Estimated Coverage |
|--------|---------------------|-------------------|
| `engines/generator.py` — TestCaseGenerator | `generate_instance`, `generate_batch`, `_generate_from_template`, `_adjust_for_intensity`, `_generate_debiasing_prompts` | ~75% |
| `engines/evaluator.py` — AnswerExtractor | `extract()` (option, numeric, yes_no, confidence types), `extract_confidence()`, Answer: line parsing | ~90% |
| `metrics/core.py` — All 6 metrics | `calculate()` for BMS, BCI, BMP, HAS, RCI, CAS; edge cases; unknown handling | ~85% |
| `biases/taxonomy.py` | Structure validation, categories, interaction matrix, K&T core | ~90% |
| `utils/io.py` | All 8 export/import functions, roundtrips, error handling | ~85% |
| `engines/judge.py` | `score()`, `_parse_judge_response()`, prompt structure, clamping | ~85% |
| `engines/quality.py` | `assess_instance()`, `assess_batch()`, `filter_low_quality()`, XML parsing | ~80% |
| `engines/bloom_generator.py` | Parsing stages, `scenario_to_instance()`, intensity variants, `generate_batch()` | ~80% |
| `engines/conversation.py` | `evaluate_conversation()`, `_simple_score()`, `_calculate_persistence()`, strategies | ~80% |
| `engines/variation.py` | All 6 dimensions, `generate_varied_instances()`, `VariationRobustnessScore.calculate()` | ~85% |

### 2.2 Partially Covered Modules (30-70%)

| Module | What's Tested | What's Missing | Estimated Coverage |
|--------|--------------|---------------|-------------------|
| `engines/evaluator.py` — BiasEvaluator | `evaluate_instance`, `evaluate_batch`, `score_response`, error handling | `_resolve_biased_answer` frame-aware paths, `_infer_answer_type` edge cases | ~60% |
| `engines/evaluator.py` — TemporalEvaluator | `evaluate_persistent`, `evaluate_adaptive` basic paths | Error recovery in persistent chains, edge cases with 0 rounds | ~50% |
| `engines/evaluator.py` — ContextSensitivityEvaluator | `evaluate_context_sensitivity`, `evaluate_expertise_gradient`, `evaluate_stakes_gradient` | Interaction between context configs, frame-aware scoring within context | ~45% |
| `engines/generator.py` — NovelScenarioGenerator | Basic generation, batch, seed reproducibility | `_generate_novel_trigger` comprehensive coverage per bias category | ~55% |
| `engines/generator.py` — MacroScaleGenerator | Chain generation, all bias-specific chain types | `chain_to_instances()` conversion, chain nodes with actual evaluation | ~50% |
| `engines/compound.py` | All 3 interaction types, battery generation | `analyze_interaction_effects()`, `_generate_compound_rational/biased` edge cases | ~55% |
| `engines/robustness.py` | Paraphrases, debiasing variants, consistency | `ContrastiveRobustnessTester.aggregate()` | ~60% |

### 2.3 Untested Modules (0-30%)

| Module | Functions | Estimated Coverage |
|--------|-----------|-------------------|
| `cli.py` | 12 Click commands (`list-biases`, `evaluate`, `report`, `generate-bloom`, `evaluate-conversation`, etc.) | ~0% |
| `utils/diversity.py` | `calculate_bleu`, `calculate_self_bleu`, `calculate_rouge_similarity`, `validate_dataset_diversity` | ~0% |
| `core.py` — ContextSensitivityConfig | `get_expertise_prefix`, `get_formality_framing`, `get_stakes_emphasis` | ~0% |
| `core.py` — CognitiveBiasInstance | `apply_context_sensitivity`, `get_context_variant` | ~0% |

## 3. Test Quality Assessment

### 3.1 Assertion Quality: Strong

The test suite has **meaningful, behavior-verifying assertions** throughout. Key patterns:

- **Generator tests:** Verify both structural correctness (instance fields populated) AND semantic validity (rational ≠ biased, EV calculations correct, frame-specific targets). The `TestEVValueModelBiases` class runs 20-trial fuzz tests per bias — excellent.
- **Metric tests:** Test both the math (specific expected values like `bms.overall_magnitude >= 0.3`) and edge cases (empty inputs, zero baselines, single intensity).
- **Extraction tests:** Assert specific extracted values against known inputs — not just "doesn't crash."
- **Frame-aware tests (`TestGainLossFramingBothFrames`):** 9 tests that verify gain/loss frame metadata, distinct rational/biased targets per frame, and regression tests for the P0 fix. Very thorough.

### 3.2 Async Test Structure: Correct

All async tests use `@pytest.mark.asyncio` decorator correctly. Found in:
- `test_evaluator.py` (~20 async tests)
- `test_integration.py` (~5 async tests)
- `test_bloom_generator.py` (~5 async tests)
- `test_conversation.py` (~4 async tests)
- `test_quality.py` (~6 async tests)
- `test_judge.py` (~7 async tests)

No issues found with async test structure.

### 3.3 Edge Cases: Good but Gaps Exist

**Well-tested edge cases:**
- Empty inputs (empty bias list, empty domain list, empty results)
- None/null values in TestResult fields
- Boundary values (0%, 100% confidence; 0.0 and 1.0 bias scores)
- Malformed JSON import
- Missing enum values during import
- BMS renormalization when intensities are missing
- Unknown/failed extraction handling across all 6 metrics
- Score clamping in judge and quality modules

**Missing edge cases:** (see Section 6 for recommendations)

### 3.4 Tautological Tests: 2 Found

1. **`test_treatment_prompts_differ_for_multiple_biases`** (line 253): Asserts `len(unique_treatments) >= 1`, which is trivially always true for any non-empty dict. The earlier test correctly uses `>= 2`; this parametrized variant weakened the assertion. **Impact: Low** — the non-parametrized version catches the real bug.

2. **`TestMetricCalculator.test_default_scorer`** (line 400): Calls `calculate_all_metrics` and only asserts `report.model_id == "test-model"`. This verifies the calculator doesn't crash but not that metrics are correctly computed. The comment even says "Just verify it doesn't crash." **Impact: Medium** — MetricCalculator integration is only tested at the "doesn't crash" level.

### 3.5 Mock Provider Realism

**3 different mock providers are used:**

1. **`MockLLMProvider` (test_evaluator.py):** Returns a fixed string or cycles through a response list. Simple but sufficient for unit tests of extraction/scoring.

2. **`IntegrationMockProvider` (test_integration.py):** Uses `random.Random(seed)` for deterministic behavior, varies response based on prompt analysis (control vs treatment vs debiasing), includes configurable bias rates. **Good realism** for integration tests.

3. **`MockProvider` (test_bloom_generator.py, test_quality.py, test_judge.py):** Returns canned XML responses. Appropriate for testing parsing logic.

**Weakness:** No mock provider simulates realistic LLM behavior like:
- Partial/truncated responses
- Responses that don't follow the requested format
- Very long or very short responses
- Responses with markdown formatting, code blocks, or HTML
- Rate limiting / timeout behavior

For publication, these edge cases matter because real LLM outputs are unpredictable.

## 4. Frame-Aware Scoring Test Comprehensiveness

The `TestGainLossFramingBothFrames` class (9 tests) is **very comprehensive**:

| Test | What It Verifies |
|------|-----------------|
| `test_loss_frame_is_used` | STRONG/ADVERSARIAL treatments mention "will die" |
| `test_gain_frame_is_used` | WEAK/MODERATE treatments mention "will be saved" |
| `test_gain_and_loss_frames_differ` | Gain ≠ loss frame prompts |
| `test_frame_map_metadata` | Correct frame labels per intensity |
| `test_frame_specific_biased_responses_in_metadata` | gain=A, loss=B |
| `test_frame_specific_rational_responses_in_metadata` | gain_rational=B, loss_rational=A |
| `test_loss_frame_rational_differs_from_biased` | Regression test for P0 bug |
| `test_all_four_frame_targets_present` | All 4 metadata keys exist with valid values |
| (via `TestIntensityVariation`) | gain_loss_framing has ≥2 distinct treatments |

**Gap:** No test verifies that `_resolve_biased_answer` in `BiasEvaluator` correctly uses `frame_map` and frame-specific targets during actual scoring. The generator-side metadata is tested, but the evaluator-side consumption of that metadata during `score_response()` is not.

## 5. Rough Coverage Estimates by Module

| Module | Lines (est.) | Functions | Tested Functions | Line Coverage (est.) |
|--------|-------------|-----------|-----------------|---------------------|
| `core.py` | 400 | 8 | 3 | ~40% |
| `biases/taxonomy.py` | 2800 | 5 | 5 | ~85% (data validation) |
| `engines/generator.py` | 4600 | 30+ | ~20 | ~55% |
| `engines/evaluator.py` | 1600 | 25+ | ~18 | ~55% |
| `engines/compound.py` | 400 | 12 | 8 | ~55% |
| `engines/robustness.py` | 400 | 9 | 7 | ~60% |
| `engines/bloom_generator.py` | 500 | 12 | 10 | ~75% |
| `engines/conversation.py` | 400 | 8 | 7 | ~75% |
| `engines/quality.py` | 300 | 8 | 7 | ~75% |
| `engines/judge.py` | 200 | 4 | 4 | ~85% |
| `engines/variation.py` | 250 | 5 | 5 | ~85% |
| `metrics/core.py` | 1100 | 11 | 10 | ~80% |
| `utils/io.py` | 700 | 13 | 8 | ~70% |
| `utils/diversity.py` | 200 | 6 | 0 | ~0% |
| `cli.py` | 750 | 12 | 0 | ~0% |
| **Overall** | **~14,600** | **~168** | **~112** | **~55-60%** |

## 6. Top 10 Most Important Missing Tests

### Priority 1: Critical for Publication Validity

1. **`score_response()` frame-aware path** (`evaluator.py:1059`)
   - The frame-aware scoring logic in `_resolve_biased_answer` uses `frame_map` metadata to determine per-frame biased/rational targets. No test verifies this path end-to-end with `gain_loss_framing` instances through the evaluator.
   - **Why critical:** If frame-aware scoring is broken, half the framing effect measurements are invalid. The generator tests prove metadata is set correctly, but no test proves the evaluator consumes it correctly.

2. **`ContextSensitivityEvaluator` with frame-aware biases** (`evaluator.py:1333-1548`)
   - `evaluate_context_sensitivity()`, `evaluate_expertise_gradient()`, and `evaluate_stakes_gradient()` all call `evaluate_instance()` internally, but no test verifies that the intensity parameter is correctly passed through the context-evaluation path for frame-resolving biases.
   - **Why critical:** Task #7 on the task list is specifically about this issue ("Fix context/expertise/stakes evaluators to pass intensity to frame resolver"). Tests are needed to verify the fix.

3. **`MetricCalculator.calculate_all_metrics()` integration** (`metrics/core.py:1011`)
   - Currently only tested at the "doesn't crash" level. No test verifies that a realistic set of evaluation results produces correct metric values across all 6 metrics simultaneously.
   - **Why critical:** This is the final aggregation step. Correctness of the individual metrics has been verified, but their composition into a `CognitiveFingerprintReport` with correct `most_susceptible_biases`, `most_resistant_biases`, and `overall_bias_susceptibility` is untested.

### Priority 2: Important for Correctness

4. **`_resolve_biased_answer` and `_resolve_rational_answer`** (`evaluator.py:704-777`)
   - These methods handle metadata-driven answer resolution for multiple bias types. Only indirectly tested through `score_response`. Direct unit tests for each answer_type (option, yes_no, numeric, confidence) with and without metadata would be valuable.

5. **`utils/diversity.py` — all functions**
   - `calculate_bleu`, `calculate_self_bleu`, `calculate_rouge_similarity`, and `validate_dataset_diversity` are completely untested. These are used to validate test case diversity, which is a quality guarantee for the benchmark.
   - **Why important:** If dataset diversity validation is broken, published test suites might contain near-duplicate prompts.

6. **`_infer_answer_type`** (`evaluator.py:935`)
   - This method determines how to extract and score responses. Incorrect inference → wrong extraction → wrong scores. Only tested indirectly through integration paths.

7. **`TemporalEvaluator.evaluate_persistent()` with error recovery** (`evaluator.py:1158`)
   - The persistent evaluation runs 5 sequential rounds. No test verifies behavior when a provider error occurs mid-chain (e.g., round 3 of 5 fails). Does the chain continue? Does it retry?

### Priority 3: Valuable for Robustness

8. **CLI commands** (`cli.py`)
   - Zero test coverage for 12 Click commands. While CLI testing is often considered integration-level, the `evaluate` and `report` commands contain significant logic (provider creation, progress callbacks, output formatting) that could silently break.
   - At minimum: test that `list-biases` and `describe anchoring_effect` produce correct output.

9. **`export_quality_report_to_json` and `export_transcripts_to_json`** (`utils/io.py:271, 299`)
   - These export functions for the newer quality and conversation modules have no tests.

10. **`_is_descriptive_answer` static method** (`evaluator.py:1035`)
    - This determines whether expected answers are "placeholder" descriptive text (starting with `[`). Incorrect classification → all instances for that bias scored at 0.5. No direct test.

## 7. Additional Observations

### 7.1 Pytest Warnings

12 `PytestCollectionWarning` messages appear because source classes (`TestScale`, `TestCaseGenerator`, `TestResult`) have names starting with `Test`. This is cosmetic but noisy. Consider renaming or adding `__test__ = False` to these classes.

### 7.2 Test Organization

Tests are well-organized with descriptive class names and docstrings. The test class naming convention (`TestXxxYyy`) is consistent. Parametrized tests are used effectively (e.g., testing all CORE biases for intensity variation).

### 7.3 Reproducibility

Generator tests consistently use `seed=42` for reproducibility. The `IntegrationMockProvider` also uses seeded randomness. This is good practice.

### 7.4 Test Speed

546 tests in 0.49 seconds is excellent. No slow tests that could become CI bottlenecks.

### 7.5 No `conftest.py`

Fixtures are defined locally in each test file. This is fine for the current scale but could lead to duplication as the test suite grows. The `sample_instance` fixture is defined independently in 3+ files with slightly different configurations.

## 8. Summary

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Test count | **Strong** | 546 tests covering 12 test files |
| Pass rate | **Excellent** | 100% passing |
| Assertion quality | **Strong** | Meaningful assertions, not just "doesn't crash" (2 exceptions noted) |
| Async test structure | **Good** | All async tests properly decorated |
| Edge case coverage | **Good** | Most critical edge cases covered; gaps in evaluator internals |
| Frame-aware testing | **Strong** | 9 dedicated tests + intensity variation coverage |
| Mock provider realism | **Adequate** | 3 providers with varying sophistication; missing adversarial LLM output patterns |
| Module coverage | **Moderate** | ~55-60% estimated; cli.py and diversity.py at 0% |
| Tautological tests | **Minor** | 2 found, low impact |
| Publication readiness | **Moderate** | Missing tests #1-3 should be addressed before publication |
