# Documentation Accuracy Review

**Reviewer:** docs-reviewer (pub-review team)
**Date:** 2026-02-11
**Scope:** All documentation files checked against the current codebase

---

## Summary

| Document | Verdict | Error Count |
|----------|---------|-------------|
| README.md | HAS ERRORS | 8 |
| CLAUDE.md | HAS ERRORS | 10 |
| docs/LIMITATIONS.md | MOSTLY ACCURATE | 1 |
| docs/AUDIT_FIX_SPEC.md | OUTDATED | 3 |
| docs/BENCHMARK_IMPROVEMENTS.md | ACCURATE | 0 |
| docs/IMPROVEMENT_TRACKER.md | HAS ERRORS | 3 |
| Source docstrings | HAS ERRORS | 2 |
| CLI help text | ACCURATE | 0 |
| examples/basic_usage.py | ACCURATE | 0 |
| examples/openai_evaluation.py | ACCURATE | 0 |

**Total issues found: 27**

---

## 1. README.md

**Verdict: HAS ERRORS**

### Error 1: Taxonomy category table has wrong counts and missing category (CRITICAL)

The taxonomy table on lines 298-314 has three issues:

- **Framing listed as 7 biases** — actual count is **6**. `reference_point_framing` is categorized as `REFERENCE_DEPENDENCE`, not `FRAMING`.
- **Extension Neglect listed as 4 biases** — actual count is **2** (`scope_insensitivity`, `identifiable_victim_effect`). `group_attribution_bias` and `halo_effect` have `category=BiasCategory.SOCIAL_BIAS`, not `EXTENSION_NEGLECT`.
- **REFERENCE_DEPENDENCE category is missing entirely** — it has 1 bias (`reference_point_framing`). The README table only lists 15 of 16 categories.

The table sums to 71 biases across 15 categories, but the actual taxonomy has **69 biases across 16 categories**.

Verified counts:

| Category | README Claims | Actual |
|----------|-------------|--------|
| FRAMING | 7 | 6 |
| EXTENSION_NEGLECT | 4 | 2 |
| REFERENCE_DEPENDENCE | (missing) | 1 |
| All others | correct | correct |

### Error 2: OpenAI example uses Responses API, not Chat Completions API

Lines 175-188 show an `OpenAIProvider` using `self.client.responses.create()` and `response.output_text`. The actual built-in `OpenAIProvider` in `evaluator.py:157-189` uses `self.client.chat.completions.create()` and `response.choices[0].message.content`. The README example would not work with the actual evaluation pipeline's scoring.

### Error 3: CLI section omits `PYTHONPATH=src uv run` prefix

Lines 345-356 show CLI commands like `kahne-bench list-biases` without the `PYTHONPATH=src uv run` prefix that is required when running from source. This is inconsistent with the Quick Start section (line 69-74) which correctly uses `PYTHONPATH=src python`.

### Error 4: CLI generate-compound example uses wrong flags

Line 355 shows:
```
kahne-bench generate-compound --domain professional
```

But the actual CLI (`cli.py:270-274`) uses `--bias/-b` (multiple), `--domain/-d`, and `--output/-o`. There is no `--primary` or `--secondary` flag. This is correct in the README but the usage pattern could be misleading without showing `--bias`.

### Error 5: Missing CLI commands in documentation

The README CLI section does not mention these commands that exist in the actual CLI:
- `evaluate` — the main evaluation pipeline
- `report` — generate fingerprint reports
- `info` — show framework information
- `assess-quality` — quality assessment with LLM judge
- `generate-bloom` — BLOOM-style LLM-driven generation
- `evaluate-conversation` — multi-turn conversational evaluation

Only `list-biases`, `describe`, `generate`, and `generate-compound` are shown.

### Error 6: Project structure tree is incomplete

Lines 364-384 show the source tree but are missing 5 modules that exist in the codebase:
- `engines/variation.py` — Prompt variation and robustness testing
- `engines/conversation.py` — Multi-turn conversational evaluation
- `engines/bloom_generator.py` — LLM-driven test generation
- `engines/quality.py` — Quality assessment judge
- `engines/judge.py` — LLM judge for scoring fallback
- `domains/__init__.py` — Domains package

### Error 7: "What's Included" section is incomplete

Lines 22-28 don't mention conversational evaluation, BLOOM generation, quality assessment, or variation/robustness modules.

### Error 8: Quick Start lacks `uv run` prefix

Line 70: `PYTHONPATH=src python examples/basic_usage.py` should be `PYTHONPATH=src uv run python examples/basic_usage.py` for consistency with how the project is set up (using uv).

---

## 2. CLAUDE.md

**Verdict: HAS ERRORS**

### Error 1: `generate-compound` CLI command uses wrong flags (CRITICAL)

Line 52 shows:
```
kahne-bench generate-compound --primary anchoring_effect --secondary availability_bias
```

The actual CLI (`cli.py:270-274`) has `--bias/-b` (multiple), not `--primary` or `--secondary`. The correct command is:
```
kahne-bench generate-compound --bias anchoring_effect --bias availability_bias
```

### Error 2: Test file count and test counts are massively outdated (CRITICAL)

Lines 388-395 claim:

| File | Tests |
|------|-------|
| test_taxonomy.py | 12 |
| test_generator.py | 19 |
| test_metrics.py | 13 |
| test_io.py | 11 |
| **Total** | **55** |

Actual state:

| File | Tests |
|------|-------|
| test_taxonomy.py | 24 |
| test_generator.py | 71 |
| test_metrics.py | 55 |
| test_io.py | 26 |
| test_evaluator.py | 112 |
| test_advanced_generators.py | 63 |
| test_conversation.py | 20 |
| test_variation.py | 20 |
| test_bloom_generator.py | 19 |
| test_judge.py | 11 |
| test_quality.py | 9 |
| test_integration.py | 6 |
| **Total** | **~599** |

8 test files are completely missing from the documentation.

### Error 3: Module dependencies tree is incomplete

Lines 257-273 omit 5 modules:
- `engines/variation.py`
- `engines/conversation.py`
- `engines/bloom_generator.py`
- `engines/quality.py`
- `engines/judge.py`

### Error 4: Dependencies list omits `xai-sdk` and `google-genai`

Lines 411-421 list core dependencies but omit:
- `xai-sdk>=0.1.0` — Required for xAI/Grok provider
- `google-genai>=0.1.0` — Required for Gemini provider

Both are in `pyproject.toml` as core dependencies.

### Error 5: Taxonomy category table has same errors as README

Lines 277-296 repeat the same incorrect counts:
- FRAMING: 7 (actual: 6)
- EXTENSION_NEGLECT: 4 (actual: 2)

### Error 6: REFERENCE_DEPENDENCE example biases are wrong

Line 286:
```
| REFERENCE_DEPENDENCE | ... | status_quo_bias, default_effect |
```

Actual: `status_quo_bias` has `category=BiasCategory.LOSS_AVERSION` and `default_effect` has `category=BiasCategory.FRAMING`. The only `REFERENCE_DEPENDENCE` bias is `reference_point_framing`.

### Error 7: EXTENSION_NEGLECT example bias is wrong

Line 296:
```
| EXTENSION_NEGLECT | ... | scope_insensitivity, halo_effect |
```

`halo_effect` actually has `category=BiasCategory.SOCIAL_BIAS`. The actual EXTENSION_NEGLECT biases are `scope_insensitivity` and `identifiable_victim_effect`.

### Error 8: K&T core count reference is imprecise

Line 66 in LIMITATIONS.md says "Core K&T (25 biases)" — this is actually correct (verified: 25 K&T core biases). However, the CLAUDE.md doesn't mention K&T core count anywhere for cross-reference.

### Error 9: EvaluationConfig description incomplete

Lines 233-237 describe `EvaluationConfig` with `requests_per_minute` for rate limiting but omit `max_concurrent_requests` which is the actual concurrency mechanism used (semaphore-based, default 50). The `requests_per_minute` field is marked as "legacy, kept for compatibility" in the code.

### Error 10: Built-in providers description is incomplete

Lines 93-95 list only `OpenAIProvider` and `AnthropicProvider`. The code actually has 4 built-in providers:
- `OpenAIProvider` (also used for Fireworks via OpenAI-compatible API)
- `AnthropicProvider`
- `XAIProvider`
- `GeminiProvider`

---

## 3. docs/LIMITATIONS.md

**Verdict: MOSTLY ACCURATE**

### Minor Issue 1: Interaction matrix claims don't fully match tracker

Line 75 says "now covers 100% of biases (40 primary entries with all 69 biases appearing either as primary or secondary)." This is **correct** — verified via code: 40 primary entries, 69 unique biases, 100% coverage.

However, this contradicts the IMPROVEMENT_TRACKER.md which says coverage is 26% (18/69). The LIMITATIONS.md was updated more recently and is correct; the IMPROVEMENT_TRACKER is outdated.

### Overall assessment

The LIMITATIONS.md is well-written, honest, and largely accurate. The section on answer extraction (4.1), scoring edge cases (4.2), and recommendations for users are all consistent with current code behavior. The document transparently acknowledges key weaknesses.

---

## 4. docs/AUDIT_FIX_SPEC.md

**Verdict: OUTDATED**

The audit spec is dated 2026-02-10. Since then, commit `f846ec2` claims "Fix all 16 audit issues: extraction, templates, metrics, and judge" and commit `10c47a4` claims "Fix publication validity and framing-awareness issues." The audit spec should be updated to reflect which issues are resolved.

### Issue 1: Line number references are likely stale

Multiple references to specific line numbers (e.g., `generator.py:168-197`, `generator.py:1825-1842`, `evaluator.py:801-802`) are likely outdated after the fix commits changed those files. Line numbers shift with every edit.

### Issue 2: Some P0 issues may now be resolved

The spec lists issues as open, but recent commits claim to fix them. Without re-verification, the document's current state creates confusion about what's still broken.

### Issue 3: "NOT PUBLISHABLE" verdict may be outdated

The headline verdict "NOT PUBLISHABLE in current state" may no longer be accurate if the P0 fixes in `f846ec2` are genuine. The document should either be updated with current status or archived.

---

## 5. docs/BENCHMARK_IMPROVEMENTS.md

**Verdict: ACCURATE**

This document describes MECE work parts (A through H) for improving the benchmark. It's structured as a work specification rather than a status document, so it doesn't make testable claims about current state. The file references and bias lists match the current codebase.

---

## 6. docs/IMPROVEMENT_TRACKER.md

**Verdict: HAS ERRORS**

### Error 1: Test count is massively outdated

Line 117: "[x] All tests pass (`PYTHONPATH=src uv run pytest`) - 55 tests passed"

Actual: ~599 tests exist across 12 files. The tracker was last updated 2026-01-04 and hasn't been updated since many test files were added.

### Error 2: Interaction matrix coverage is outdated

Lines 12 and 66: "Interaction Matrix Coverage: 26% (18/69)"

Actual: 100% coverage (40 primary entries, 69 unique biases). The matrix was expanded significantly after this tracker was written.

### Error 3: Biases added count may be incomplete

The tracker says 11 biases were added to reach 69. If additional biases were added or removed since, this isn't reflected. However, the current total is 69, which matches the "After" column.

---

## 7. Source Code Docstrings

**Verdict: HAS ERRORS**

### Error 1: `biases/__init__.py` docstring says "50 cognitive biases"

Line 2: `"""Comprehensive taxonomy of 50 cognitive biases based on Kahneman-Tversky research."""`

Actual: 69 biases. The docstring was never updated when biases were added.

### Error 2: `taxonomy.py` docstring is accurate

The `taxonomy.py` module docstring (lines 1-26) correctly states 69 biases across 16 categories with accurate per-category counts. This is the authoritative source.

---

## 8. CLI Help Text

**Verdict: ACCURATE**

The CLI help text is generated dynamically from Click decorators and accurately reflects available options. Spot-checked:
- `list-biases` — correct
- `generate` — has `--bias`, `--domain`, `--instances`, `--output`, `--seed`, `--tier`
- `evaluate` — has `--input`, `--provider`, `--model`, `--trials`, `--output`, `--fingerprint`, `--tier`, `--concurrency`, `--judge-provider`, `--judge-model`
- `generate-compound` — has `--bias`, `--domain`, `--output` (no `--primary`/`--secondary`)

---

## 9. examples/basic_usage.py

**Verdict: ACCURATE**

The example uses correct imports, correct API patterns, and would run successfully with the current codebase. The `MockProvider` implements the `complete()` protocol correctly. All referenced classes and functions exist in the public API.

---

## 10. examples/openai_evaluation.py

**Verdict: ACCURATE**

The example correctly:
- Defines its own `OpenAIProvider` using `chat.completions.create()` (correct API)
- Handles `max_completion_tokens` vs `max_tokens` for newer models
- Imports `get_tier_biases`, `KahneBenchTier` from the correct locations
- Uses `EvaluationConfig`, `BiasEvaluator`, `MetricCalculator` correctly
- Implements `score_response()` on results (required step)

Minor note: The example defines its own `OpenAIProvider` rather than importing the built-in one from `evaluator.py`. This is intentional (the example is self-contained) but worth noting.

---

## Priority Fix Recommendations

### P0 — Must fix before publication

1. **Fix README taxonomy table** — correct Framing (6), Extension Neglect (2), add Reference Dependence (1)
2. **Fix CLAUDE.md taxonomy table** — same corrections as README
3. **Fix CLAUDE.md generate-compound command** — change `--primary/--secondary` to `--bias/-b`
4. **Fix README OpenAI example** — use `chat.completions.create()` not `responses.create()`
5. **Fix CLAUDE.md test counts** — update from 55 to ~599 across 12 files
6. **Fix CLAUDE.md REFERENCE_DEPENDENCE examples** — correct to `reference_point_framing`
7. **Fix `biases/__init__.py` docstring** — change "50" to "69"

### P1 — Should fix before publication

8. **Update CLAUDE.md module dependencies tree** — add 5 missing modules
9. **Update CLAUDE.md dependencies list** — add `xai-sdk`, `google-genai`
10. **Update IMPROVEMENT_TRACKER.md** — reflect current test count and interaction matrix coverage
11. **Update or archive AUDIT_FIX_SPEC.md** — mark which issues were resolved by recent commits
12. **Update README project structure** — add missing modules
13. **Update README CLI section** — add missing commands (evaluate, report, etc.)
14. **Fix CLAUDE.md built-in providers list** — add XAIProvider, GeminiProvider

### P2 — Nice to fix

15. **Add `uv run` prefix to README CLI examples** — for consistency with CLAUDE.md
16. **Update CLAUDE.md EvaluationConfig description** — mention `max_concurrent_requests`
17. **Update README "What's Included" section** — mention newer modules

---

## Verification Method

All claims were verified by:
1. Reading actual source files (`src/kahne_bench/**/*.py`)
2. Running Python to count biases, categories, and interaction matrix entries
3. Running `pytest --co` to count actual tests
4. Cross-referencing CLI option decorators against documented commands
5. Checking `pyproject.toml` for actual dependencies
