# Evaluation Pipeline Review

**Reviewer:** pipeline-reviewer
**Date:** 2026-02-11
**Files reviewed:** `engines/evaluator.py` (1630 lines), `engines/judge.py` (173 lines)
**Scope:** Answer extraction, scoring, judge fallback, temporal/context evaluators, providers, error handling

---

## Executive Summary

The evaluation pipeline is well-engineered overall, with thorough answer extraction, proper frame-aware scoring for gain/loss framing, and a clean LLM judge fallback path. Test coverage is strong (100+ evaluator tests). However, I found **3 bugs that will affect published benchmark results** (severity: HIGH), **2 medium-severity issues** that could cause subtle data quality problems, and **4 low-severity observations** worth noting for completeness.

---

## 1. Answer Extraction (AnswerExtractor)

### Strengths
- Priority-based extraction: explicit `Answer:` lines are checked first, preventing verbose reasoning from overriding the model's intended answer
- Negative-context filtering (`_has_negative_context`) correctly avoids extracting anchor values (e.g., "based on 6500") as the model's answer
- Confidence extraction is well-separated from numeric answer extraction via regex stripping
- Pre-compiled regex patterns at module level avoid per-call compilation overhead
- Handles markdown bold formatting (`**Answer:**`)

### Issues

#### [LOW] E-1: Option patterns limited to A-D
`evaluator.py:310-315` — All `_OPTION_PATTERNS` and `_ANSWER_LINE_OPTION` match only `[A-D]`. If any bias test presents more than 4 options, extraction will fall through to fallback (which also only catches `[A-D]` at line 526). This is currently fine since all test templates use 2-option (A/B) or 4-option formats, but documenting this constraint is recommended.

#### [LOW] E-2: Scientific notation not handled
`evaluator.py:317-321` — `_NUMERIC_PATTERNS` use `[\d,]+(?:\.\d+)?` which cannot match `1e5` or `1.5e6`. Documented in tests as known limitation (`test_evaluator.py:1156-1169`). Not critical since generated scenarios use standard numeric formats.

#### [LOW] E-3: Negative numbers not handled
Numeric extraction patterns do not match negative values (e.g., `-500`). The leading `-` is not captured. For bias tests involving losses or decreases, this could cause extraction failures that fall through to the judge. Impact is low because most anchoring/estimation tests use positive values.

#### [MEDIUM] E-4: Fallback numeric extraction returns FIRST number, not best contextual match
`evaluator.py:560-562` — The final fallback iterates all numbers with `re.finditer` and returns the **first** non-negative-context match. The comment at `test_evaluator.py:256` says "(changed from last to fix gambler_fallacy extraction)" — but returning the first number biases extraction toward numbers mentioned early in the response (often part of problem restating, not the answer). The `Answer:` priority line mitigates this for well-formatted responses, but unformatted responses may have the first number be the anchor.

**Risk:** This could inflate bias scores for anchoring tests by extracting the anchor value when no `Answer:` line is present. The negative-context filter helps, but models that restate the anchor without explicit rejection phrasing will be misscored.

---

## 2. Scoring (score_response)

### Strengths
- Frame-aware scoring via `_resolve_biased_answer()` and `_resolve_rational_answer()` correctly handles gain/loss framing — this was a P0 fix with extensive regression tests
- Numeric comparison uses 1% relative epsilon (`EPSILON = 0.01`), which is appropriate for benchmark estimation tasks
- Partial bias score calculation (position between rational and biased on numeric scale) is mathematically sound and clamped to [0, 1]
- Descriptive answers correctly fall through to LLM judge via `_is_descriptive_answer()`
- `normalize_answer()` covers a comprehensive synonym mapping with word-boundary matching for multi-char variations

### Issues

#### [LOW] S-1: Symmetric rational/biased in loss frame
When `expected_rational_response` and the frame-resolved biased answer are the same value (documented at `test_evaluator.py:1748-1752`), `score_response` always returns the rational match first `(False, 0.0)`. This is documented behavior, not a bug per se, but it means loss-frame scenarios where the instance-level `expected_rational_response` equals the loss-frame biased answer will never score as biased via regex. The judge fallback can handle this, but it's a design limitation to document in the paper.

---

## 3. LLM Judge Fallback (judge.py)

### Strengths
- Clean XML-based response parsing with explicit tag extraction
- Bias score and confidence are clamped to [0.0, 1.0]
- Missing `extracted_answer` defaults to "unknown" rather than crashing
- Judge prompt design follows best practice: asks judge to form independent assessment FIRST, then provides expected answers only as "calibration reference"
- Temperature 0.0 for deterministic scoring
- Judge is only invoked when regex scoring returns `(None, None)`, avoiding unnecessary API costs

### Issues

#### [MEDIUM] J-1: Judge prompt leaks expected answers
`judge.py:53-57` — The prompt includes:
```
Unbiased baseline: {expected_rational}
Maximum bias direction: {expected_biased}
```
While the instruction says "Use the following only to calibrate your score AFTER forming your independent assessment," a sophisticated model may simply pattern-match the response to the expected answers rather than independently assessing bias. This is a known challenge in LLM-as-judge research.

**Recommendation:** For publication, run a small ablation study comparing judge accuracy with vs. without calibration references, or move them to a second-pass prompt. Alternatively, note this as a limitation.

#### [INFO] J-2: No retry logic for judge failures
`evaluator.py:925-926` — Judge failures are logged at WARNING level and silently result in `(None, None)` scoring. This is correct behavior (graceful degradation), but there's no retry mechanism. For expensive evaluations where the judge is critical, a single transient API error means that result is permanently unscorable.

---

## 4. Rate Limiting and Async Pipeline

### Strengths
- Semaphore-based concurrency limiting (`asyncio.Semaphore(config.max_concurrent_requests)`) is the correct pattern for async API calls
- All trials for a condition run concurrently via `asyncio.gather`, with the semaphore controlling actual API parallelism
- Progress callback with `asyncio.Lock` for thread-safe counting
- Error handling wraps provider exceptions in `"ERROR: ..."` strings, preventing task cancellation from propagating to other instances

### Issues

#### [HIGH] R-1: `trial_delay_ms` is configured but never used
`evaluator.py:284` — `trial_delay_ms: int = 100` is defined in `EvaluationConfig` but is never referenced anywhere in the codebase (confirmed via grep). The `_run_trials` method launches all trials simultaneously via `asyncio.gather` with no inter-trial delay. This means:
1. The documented "delay between trials" parameter is misleading
2. For `temperature=0.0` (default), sending identical prompts simultaneously may hit provider-side deduplication or caching, which could reduce the independence of trials used for RCI (Response Consistency Index) calculation

**Impact on published results:** RCI scores could be artificially low (more consistent) if providers cache or deduplicate identical concurrent requests. This directly affects one of the 6 published metrics.

**Recommendation:** Either implement the delay (add `await asyncio.sleep(self.config.trial_delay_ms / 1000)` between trial launches) or remove the parameter and document that trials are concurrent.

---

## 5. TemporalEvaluator

### Strengths
- `evaluate_persistent()` correctly builds context chains using previous response truncated to 500 chars
- `evaluate_adaptive()` properly uses STRONG intensity for initial biased prompt, then control prompt for post-feedback
- Both methods correctly use frame-aware scoring via `_resolve_rational_answer` / `_resolve_biased_answer`

### Issues

#### [HIGH] T-1: TemporalEvaluator bypasses concurrency semaphore
`evaluator.py:1192` — `evaluate_persistent()` calls `self.provider.complete()` directly instead of `self._call_provider()`. Same at lines 1248 and 1290 in `evaluate_adaptive()`. This bypasses the semaphore-based concurrency limit.

**Impact:** If a user runs temporal evaluations on many instances concurrently (e.g., via `asyncio.gather`), all API calls will fire simultaneously without respecting `max_concurrent_requests`. This can cause rate-limit errors from providers, especially at scale. For sequential usage this is not an issue.

#### [INFO] T-2: Persistent evaluation uses same prompt for all rounds
`evaluator.py:1181` — All rounds of `evaluate_persistent()` use `instance.get_treatment(TriggerIntensity.MODERATE)`. This means the bias trigger is identical across rounds, only varying via the context prefix from the previous response. The scientific validity of testing "persistence" when the trigger is re-applied each round is debatable — it tests more of "accumulated context effect" than true persistence.

---

## 6. ContextSensitivityEvaluator

### Strengths
- Six well-chosen context configurations covering the full expertise/stakes gradient
- Expertise and stakes gradient evaluations properly isolate single variables while holding others at baseline
- Frame resolution correctly defaults to gain-frame for context conditions (no intensity token in condition string)

### Issues

#### [HIGH] C-1: Non-default intensity parameter breaks frame resolution for framing biases
`evaluator.py:1337,1469,1553` — All three context evaluation methods accept an `intensity` parameter (default: `TriggerIntensity.MODERATE`). The actual prompt uses this intensity to select the treatment:
```python
prompt = instance.apply_context_sensitivity(
    instance.get_treatment(intensity), config
)
```
But the condition string does NOT embed the intensity:
```python
condition=f"context_{config.expertise_level.value}_{config.stakes.value}"
```

When `_resolve_biased_answer(instance, condition)` processes this condition, it searches for intensity tokens ("weak", "moderate", "strong", "adversarial") in the condition string. Since none are present, it falls through to the default gain-frame biased answer.

This is **correct** for the default `intensity=MODERATE` (which IS the gain frame). But if a user calls:
```python
evaluator.evaluate_context_sensitivity(instance, model_id, intensity=TriggerIntensity.STRONG)
```
The model receives the loss-frame prompt, but scoring uses the gain-frame expected answers. This would cause all loss-frame responses to be misscored.

**Impact:** Currently, no code path calls these methods with non-default intensity. But it's a latent bug that will produce incorrect scores if anyone extends the evaluation to test context sensitivity under different frame conditions.

**Recommendation:** Embed the intensity value in the condition string, e.g.:
```python
condition=f"context_{config.expertise_level.value}_{config.stakes.value}_{intensity.value}"
```

---

## 7. Provider Protocol and Built-in Providers

### Strengths
- Clean Protocol-based design allows any LLM to be tested
- `OpenAIProvider` correctly distinguishes `max_tokens` vs `max_completion_tokens` for newer models
- `AnthropicProvider` handles empty content arrays (content filtering)
- `GeminiProvider` and `XAIProvider` correctly wrap sync SDKs with `asyncio.to_thread()`

### Issues

#### [HIGH] P-1: Three providers silently ignore the `temperature` parameter
- **AnthropicProvider** (`evaluator.py:205-208`): `temperature` is accepted as a parameter but NOT passed to `self.client.messages.create()`. The API will use its default temperature (likely 1.0).
- **GeminiProvider** (`evaluator.py:263-268`): Same issue — `temperature` not passed to `generate_content()`.
- **XAIProvider** (`evaluator.py:236-241`): Same issue — `temperature` not passed to `chat.sample()`.

Only `OpenAIProvider` correctly passes `temperature`.

**Impact on published results:** Benchmark evaluations with Anthropic, Gemini, and xAI models will NOT use `temperature=0.0` as documented and configured. They'll use each provider's API default, which is typically ~1.0. This means:
1. Results are non-deterministic, reducing reproducibility
2. RCI (Response Consistency Index) will measure provider randomness rather than systematic bias behavior
3. Cross-provider comparisons are invalid since OpenAI uses temp=0 while others use temp=1

**This is the highest-impact bug in the pipeline.** All non-OpenAI benchmark results collected so far may need to be re-run after fixing.

**Fix:** Add `temperature=temperature` to each provider's API call:
```python
# AnthropicProvider
response = await self.client.messages.create(
    model=self.model,
    max_tokens=max_tokens,
    temperature=temperature,  # ADD THIS
    messages=[{"role": "user", "content": prompt}],
)

# GeminiProvider
from google.genai import types
response = self.client.models.generate_content(
    model=self.model,
    contents=prompt,
    config=types.GenerateContentConfig(temperature=temperature),  # ADD THIS
)

# XAIProvider
response = chat.sample(temperature=temperature)  # ADD THIS (check SDK docs)
```

#### [INFO] P-2: XAIProvider creates a new chat session per call
`evaluator.py:236-239` — Each call to `_sync_complete()` creates a new `chat.create()` session and appends system + user messages. This is stateless (correct for benchmarking), but the system message "You are a helpful assistant." may influence bias expression. Other providers don't inject system messages.

**Recommendation:** For fair cross-provider comparison, either add the same system message to all providers or remove it from XAIProvider.

---

## 8. Error Handling

### Strengths
- Provider exceptions are caught and stored as `"ERROR: {str(e)}"` in `model_response`, preserving the error for debugging
- Error responses are detected before extraction (`if response.startswith("ERROR:"):`), preventing spurious answer extraction
- Judge failures are logged at WARNING level with bias ID context
- `evaluate_batch` continues processing when individual instances fail
- Test suite has dedicated `TestErrorRecovery` class covering timeouts, empty responses, exceptions, and partial batch failures
- `EvaluationConfig.__post_init__` validates `requests_per_minute >= 1`

### Issues

No significant error handling issues found. The pipeline is resilient — errors don't propagate, are logged, and result in `(None, None)` scoring which downstream metrics can handle.

---

## Summary of Findings

### HIGH severity (affects published results)

| ID | Issue | Impact |
|----|-------|--------|
| P-1 | Temperature parameter silently ignored by Anthropic, Gemini, XAI providers | Non-deterministic results, invalid cross-provider comparisons |
| R-1 | `trial_delay_ms` configured but never used; concurrent identical requests may be deduplicated | RCI metric may be artificially deflated |
| C-1 | Non-default intensity in context evaluators breaks frame resolution | Latent bug — correct for current usage but will miscore if intensity is changed |

### MEDIUM severity

| ID | Issue | Impact |
|----|-------|--------|
| E-4 | Fallback numeric extraction returns first number (often anchor) | May inflate anchoring bias scores for unformatted responses |
| J-1 | Judge prompt includes expected answers as "calibration reference" | Potential answer leakage; document as limitation |

### LOW severity / informational

| ID | Issue | Impact |
|----|-------|--------|
| E-1 | Option extraction limited to A-D | No impact currently; document constraint |
| E-2 | Scientific notation not handled | No impact; tests use standard formats |
| E-3 | Negative numbers not handled | Low impact; most tests use positive values |
| S-1 | Symmetric rational/biased in loss frame always scores rational | Documented behavior; judge handles edge case |
| T-1 | TemporalEvaluator bypasses semaphore | Only impacts concurrent temporal evaluations at scale |
| T-2 | Persistent evaluation re-applies trigger each round | Design choice to document |
| P-2 | XAIProvider adds system message; others don't | Minor cross-provider inconsistency |

### Recommended Priority

1. **P-1 (Critical):** Fix temperature passthrough for Anthropic/Gemini/XAI — must fix before any publication
2. **R-1 (High):** Implement `trial_delay_ms` or document concurrent trial design
3. **C-1 (High):** Embed intensity in context evaluator condition strings
4. **E-4 (Medium):** Consider switching fallback to last number (away from anchor) or requiring `Answer:` format in prompts
5. **J-1 (Medium):** Run ablation study or document as limitation
