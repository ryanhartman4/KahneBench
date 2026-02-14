# Kahne-Bench Publication Readiness Review

**Reviewer:** quality-reviewer (automated)
**Date:** 2026-02-12
**Scope:** Full codebase review across 12 dimensions
**Verdict:** Ready for publication with minor recommendations

---

## Executive Summary

Kahne-Bench is a well-engineered cognitive bias benchmark with strong theoretical grounding, clean architecture, and comprehensive test coverage (546 tests, all passing). The codebase is publication-ready. The main risks are not code quality issues but rather transparency about methodological limitations — which the existing `docs/LIMITATIONS.md` already addresses honestly.

---

## 1. Code Quality

**Rating: Strong**

### Strengths
- Clean module organization following a clear data flow: Taxonomy → Generation → Evaluation → Metrics
- Consistent use of dataclasses with type annotations throughout
- Well-documented module-level docstrings explaining purpose and design rationale
- Pre-compiled regex patterns at module level (`_OPTION_PATTERNS`, `_ANSWER_LINE_OPTION`, etc.) for performance
- Module-level constants for `ContextSensitivityConfig` avoid dictionary recreation per method call

### Issues Found
- **`generator.py` is 4,595 lines.** This is the largest file by far and contains templates, scenario data, and generator logic all in one module. While functional, this makes navigation difficult. Consider splitting templates/scenarios into a data module for maintainability, but this is not blocking.
- **`LLMProvider` protocol duplicated in 5 files** (`evaluator.py`, `judge.py`, `quality.py`, `bloom_generator.py`, `conversation.py`). Each defines its own identical `LLMProvider` protocol. This works due to Python's structural typing but creates maintenance overhead. A shared protocol in `core.py` or a `protocols.py` module would be cleaner.
- **`biases/__init__.py` docstring says "50 cognitive biases"** but the actual taxonomy contains 69 biases. Minor doc inconsistency.
- **Pytest collection warnings** for `TestScale`, `TestResult`, and `TestCaseGenerator` classes because their names match pytest's test discovery pattern. Harmless but noisy.
- **No dead code found.** No unused imports, `# noqa` markers, or `TODO`/`FIXME` comments in source.

---

## 2. Error Handling

**Rating: Strong**

### Strengths
- **Zero bare `except:` clauses** in the entire codebase
- All `except Exception` blocks either log the error, re-raise, or produce user-facing messages
- `AnswerExtractor` gracefully returns `None` on extraction failure instead of raising
- `_safe_parse_enum` in `io.py` issues warnings on invalid enum values rather than crashing
- LLM judge fallback (`judge.py`) has structured error handling with `ValueError` for missing XML tags
- `EvaluationConfig.__post_init__` validates configuration values

### Minor Notes
- `bloom_generator.py:437` has `except Exception: continue` inside `_parse_scenarios` — this silently skips malformed XML scenario blocks. This is intentional (LLM output is unpredictable) and appropriate, but worth noting it swallows parse errors without logging.
- `compound.py:359` uses `print()` for warnings instead of `logging.warning()` in `generate_interaction_battery`. Should use the logger for consistency with the rest of the codebase.

---

## 3. Type Safety

**Rating: Good**

### Strengths
- Consistent use of modern Python type hints including `dict[K, V]`, `list[T]`, `X | None` syntax
- `Protocol` used for LLM providers enabling structural subtyping
- `pyproject.toml` configures `mypy` in strict mode (`strict = true`)
- All dataclass fields have explicit type annotations
- Return types specified on all public methods

### Issues
- **Provider `client` fields typed as `Any`** in `OpenAIProvider`, `AnthropicProvider`, `XAIProvider`, `GeminiProvider`. This is a pragmatic choice (avoids importing SDK types at module level) but weakens type checking on those objects.
- **`metadata: dict`** without value type annotation on `CognitiveBiasInstance`, `TestResult`, and `EvaluationSession`. Should be `dict[str, Any]` for explicit typing.
- **Mypy not run in CI.** Without CI, strict mypy compliance is aspirational. A local `mypy src/` run would likely surface issues with the `Any`-typed provider clients.

---

## 4. API Surface

**Rating: Excellent**

### `__init__.py` Exports
The public API is well-curated across all `__init__.py` files:

| Module | Exports | Assessment |
|--------|---------|------------|
| `kahne_bench/__init__.py` | Core types, taxonomy, engines, metrics | Clean, well-organized with section comments |
| `biases/__init__.py` | `BIAS_TAXONOMY`, lookup functions, `BIAS_INTERACTION_MATRIX` | Complete |
| `engines/__init__.py` | `TestCaseGenerator`, `BiasEvaluator` | Minimal, appropriate |
| `metrics/__init__.py` | All 6 metric classes + `MetricCalculator` + `CognitiveFingerprintReport` | Complete |
| `utils/__init__.py` | All IO and diversity functions | Complete |

The `__all__` lists are maintained consistently. No private internals leak into the public API.

---

## 5. Reproducibility

**Rating: Good**

### Clone → Install → Run Path
```bash
git clone <repo>
cd bench
uv sync
PYTHONPATH=src uv run python examples/basic_usage.py  # Works with mock provider
PYTHONPATH=src uv run pytest                           # 546 tests pass
```

### Strengths
- `uv.lock` committed for exact dependency pinning
- `TestCaseGenerator` accepts a `seed` parameter for deterministic generation
- `EvaluationConfig.temperature = 0.0` by default for deterministic API calls
- `pyproject.toml` includes `[tool.pytest.ini_options]` with `pythonpath = ["src"]`

### Issues
- **`PYTHONPATH=src` required** for all commands. This is documented in CLAUDE.md and CLI examples, but newcomers might miss it. The `pyproject.toml` already sets `[tool.pytest.ini_options] pythonpath = ["src"]` for pytest, but CLI usage still needs the prefix.
- **No `pyproject.toml` script for examples.** The CLI entry point `kahne-bench` is configured, but running examples requires manual `PYTHONPATH=src uv run python examples/...`. Consider adding a note in README or a Makefile.
- **Development dependencies duplicated.** `[project.optional-dependencies] dev` and `[dependency-groups] dev` both exist with slightly different versions. The `dependency-groups` section has `pytest>=9.0.2` while `optional-dependencies` has `pytest>=7.0.0`.

---

## 6. Security

**Rating: Excellent**

- **No hardcoded credentials.** All API keys read from environment variables via `os.getenv()`.
- **No unsafe deserialization.** `json.load()` used for all data import — no `pickle`, `eval()`, or `yaml.unsafe_load()`.
- **No injection risks.** Prompts are constructed via string formatting from controlled templates, not user input concatenation.
- **No secrets in git history.** Checked `git log` for credential patterns — clean.
- **`.gitignore` excludes** `.env`, `.venv/`, and IDE files.

### Minor Note
- `fingerprint.json`, `results.json`, `test_cases_core.json`, `test_cases_sonnet45.json` are in the repo as untracked files. These contain evaluation outputs that could include model responses. If committing these, ensure no API keys or sensitive content are embedded in model responses. The `.gitignore` already handles `fingerprint_*.json` and `results_*.json` patterns.

---

## 7. Performance

**Rating: Good**

### Strengths
- **Fully async evaluation pipeline.** `BiasEvaluator.evaluate_batch()` runs all instances concurrently via `asyncio.gather()`.
- **Semaphore-based concurrency limiting.** `max_concurrent_requests` (default 50) prevents API overload.
- **Pre-compiled regex patterns** at module level for answer extraction (avoids recompilation per call).
- **Module-level constant dictionaries** for `ContextSensitivityConfig` lookup tables.
- **O(min(m,n)) space** ROUGE-L LCS implementation in `diversity.py`.

### Potential Bottlenecks
- **`calculate_self_bleu` is O(n²)** — compares all pairs in the corpus. Mitigated by the `sample_size=100` parameter that caps computation, but worth noting for large datasets.
- **No result caching.** Each metric calculation re-iterates and re-groups results. For very large result sets, this could be slow. In practice, this is unlikely to matter for typical evaluation sizes.
- **Sync SDK wrappers** for xAI and Gemini providers use `asyncio.to_thread()`, which is correct but means these providers don't benefit from true async I/O.

---

## 8. Naming Consistency

**Rating: Excellent**

- **Bias IDs** use `snake_case` consistently (e.g., `anchoring_effect`, `gain_loss_framing`)
- **Class names** follow `PascalCase` consistently (`BiasMagnitudeScore`, `TestCaseGenerator`)
- **Module names** follow `snake_case` (`bloom_generator.py`, `answer_extractor`)
- **Metric abbreviations** consistent: BMS, BCI, BMP, HAS, RCI, CAS
- **Enum values** use `snake_case` strings (e.g., `TriggerIntensity.WEAK = "weak"`)
- **Method names** follow `verb_noun` pattern (`calculate`, `generate_instance`, `extract`)

No naming inconsistencies found across the codebase.

---

## 9. Dependencies

**Rating: Good**

### Core Dependencies (all necessary)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `openai` | >=1.0.0 | OpenAI + Fireworks providers | MIT |
| `anthropic` | >=0.18.0 | Anthropic provider | MIT |
| `xai-sdk` | >=0.1.0 | xAI Grok provider | Proprietary? |
| `google-genai` | >=0.1.0 | Google Gemini provider | Apache 2.0 |
| `pandas` | >=2.0.0 | CSV export, data analysis | BSD-3 |
| `numpy` | >=1.24.0 | `polyfit` in BMS intensity sensitivity | BSD-3 |
| `pydantic` | >=2.0.0 | Listed but not imported in source | MIT |
| `rich` | >=13.0.0 | CLI terminal output | MIT |
| `click` | >=8.0.0 | CLI framework | BSD-3 |
| `tqdm` | >=4.65.0 | Listed but not imported in source | MIT |
| `jinja2` | >=3.1.0 | Listed but not imported in source | BSD-3 |
| `pyyaml` | >=6.0.0 | Listed but not imported in source | MIT |

### Issues
- **`pydantic` is listed as a dependency but never imported.** No Pydantic models exist in the codebase — all data structures use `dataclasses`. This is unnecessary weight.
- **`tqdm` is listed but never imported.** Progress bars use `rich.progress` instead.
- **`jinja2` is listed but never imported.** Template rendering uses Python f-strings and dict-based string formatting.
- **`pyyaml` is listed but never imported.** No YAML files are read or written.
- **`xai-sdk` license unclear.** Should verify this is compatible with MIT distribution.

**Recommendation:** Remove `pydantic`, `tqdm`, `jinja2`, and `pyyaml` from `dependencies` to reduce install footprint and avoid confusion.

---

## 10. Git Hygiene

**Rating: Good**

### Tracked Files — Clean
- No `.env` files, credentials, or sensitive data in git
- `.gitignore` properly excludes `__pycache__/`, `.venv/`, IDE files, and evaluation outputs

### Untracked Files (not committed)
These files exist locally but are not tracked:
- `AUDIT_RESULTS_ANALYSIS.md` — internal audit notes
- `HANDOFF_20260209.md` — session handoff notes
- `fingerprint.json`, `results.json` — evaluation outputs
- `test_cases_core.json`, `test_cases_sonnet45.json` — generated test data
- `reports/` — review reports

**Recommendation:** The `reports/` directory content is review-specific and should remain untracked or be added to `.gitignore`. The `AUDIT_RESULTS_ANALYSIS.md` and `HANDOFF_20260209.md` are internal workflow artifacts and should not be committed.

### Commit History
Recent commits show focused, well-described changes:
- `Fix publication validity and framing-awareness issues`
- `Integrate BLOOM evaluation framework (5 modules + CLI commands)`
- `Parallelize evaluation and fix 5 CORE tier template bugs`

No evidence of force pushes, squashed sensitive data, or problematic history.

---

## 11. License

**Rating: Good**

- **MIT License** with correct copyright year (2026) and author (Ryan Hartman)
- Most dependencies are MIT or BSD-3 compatible
- `google-genai` is Apache 2.0 — compatible with MIT distribution

### Concern
- **`xai-sdk`** license needs verification. If it has a restrictive license, it could create distribution issues. Since it's an optional provider (only needed for Grok evaluations), consider making it an optional dependency: `[project.optional-dependencies] xai = ["xai-sdk>=0.1.0"]`.

---

## 12. ArXiv Readiness

### Claims the Code Can Substantiate

1. **"69 cognitive biases across 16 categories"** — Verified. `BIAS_TAXONOMY` contains exactly 69 entries across 16 `BiasCategory` values.

2. **"5 ecological domains"** — Verified. `Domain` enum has 5 values with corresponding scenario templates.

3. **"6 advanced metrics"** — Verified. All 6 metric classes (BMS, BCI, BMP, HAS, RCI, CAS) are implemented with documented formulas.

4. **"Grounded in Kahneman-Tversky dual-process theory"** — Verified. Each `BiasDefinition` includes `theoretical_basis`, `system1_mechanism`, `system2_override`, and `classic_paradigm`. The `is_kt_core` flag distinguishes 25 core K&T biases from 44 extended biases.

5. **"Susceptibility-based intensity weighting"** — The weights are well-documented with clear rationale. The code honestly acknowledges they are design decisions, not empirically calibrated values.

6. **"Template-based test generation with LLM augmentation"** — Verified. `BIAS_TEMPLATES` provides structured templates for all 15 CORE biases and many extended biases. BLOOM generator provides LLM-augmented generation.

7. **"Multi-scale testing (micro, meso, macro, meta)"** — Partially substantiated. Micro-scale is fully implemented. Meso-scale has compound testing via `CompoundTestGenerator`. Macro-scale has `MacroScaleGenerator` with decision chains. Meta-scale has debiasing prompts. However, only micro-scale has been exercised in actual evaluations.

8. **"Async evaluation with rate limiting"** — Verified. Semaphore-based concurrency control with configurable `max_concurrent_requests`.

### Claims That Would Be Overstatements

1. **"Validated benchmark"** — No human validation studies have been conducted. The framework is not validated in the psychometric sense. The `LIMITATIONS.md` correctly acknowledges this.

2. **"Human-calibrated baselines"** — The `HUMAN_BASELINES` dictionary aggregates rates from diverse published studies with different methodologies, populations, and paradigms. Comparing LLM scores to these is an approximation. The code's own comments and LIMITATIONS.md acknowledge this honestly.

3. **"69 independently testable biases"** — While 69 biases are defined, not all have dedicated template-based generation. Biases without templates in `BIAS_TEMPLATES` fall through to a generic `_generate_default_instance` method that produces less targeted test cases. The CORE tier of 15 biases has the strongest templates.

4. **"Robust answer extraction"** — The regex-based `AnswerExtractor` is sophisticated but will inevitably miss some response formats. The LLM judge fallback addresses this, but extraction reliability varies by bias type and model verbosity. The code returns `None` for failed extractions and tracks `unknown_rate` — this is the right approach.

5. **"Cross-domain consistency measurement"** — BCI measures consistency across domains, but the domain-specific scenario generation has varying quality. Some domains have richer scenario templates than others.

### Recommendations for ArXiv Paper

1. **Present as a framework, not a validated benchmark.** Emphasize the software engineering contribution (architecture, metrics, extensibility) alongside the bias evaluation methodology.

2. **Report unknown rates prominently.** The codebase tracks extraction failures via `unknown_rate` on every metric — these should appear in result tables.

3. **Distinguish CORE from EXTENDED tier.** The 15 CORE biases have the strongest template support and should be the primary results. Extended-tier results should be reported with appropriate caveats.

4. **Acknowledge the intensity weight design choices.** The `DEFAULT_INTENSITY_WEIGHTS` are well-reasoned but not empirically calibrated. Present them as a proposed weighting scheme open to refinement.

5. **Include confidence intervals.** The RCI metric already measures trial-to-trial variance — use this to report confidence intervals on all headline numbers.

---

## Summary of Recommendations

### Must-Fix (None — No Blocking Issues)

### Should-Fix (Low Effort, High Value)
1. Remove unused dependencies: `pydantic`, `tqdm`, `jinja2`, `pyyaml`
2. Fix `biases/__init__.py` docstring ("50" → "69" biases)
3. Replace `print()` with `logging.warning()` in `compound.py:359`

### Nice-to-Have (For Long-Term Maintenance)
4. Extract shared `LLMProvider` protocol into a single location
5. Split `generator.py` (4,595 lines) into templates + logic modules
6. Add `dict[str, Any]` type annotation to `metadata` fields
7. Make `xai-sdk` an optional dependency
8. Resolve duplicate dev dependency specifications in `pyproject.toml`

### Test Suite
- **546 tests, all passing** in 0.38 seconds
- 12 pytest collection warnings (harmless, caused by class naming)
- Good coverage across generators, metrics, IO, and evaluation pipeline

---

*This review was conducted as a read-only assessment. No source code was modified.*
