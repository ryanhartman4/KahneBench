# KahneBench Audit Fix Spec

Generated: 2026-02-10
Source: Parallel audit by 3 agents (test-validator, code-reviewer, results-analyst)

## Audit Verdict: NOT PUBLISHABLE in current state. Fixable.

The framework architecture is sound and the conceptual design is strong. The issues are
in the generation/extraction/scoring layer, not in the theoretical foundations. All
critical issues below are fixable without a rewrite.

---

## P0: Showstoppers (must fix before any publication)

### 1. Answer extraction catastrophically fails on verbose responses

**Problem:** The regex extractor grabs numbers that models MENTION while rejecting them.
Example: Claude says "I decline to give a number based on 6500" and the extractor
captures "6500" as the answer. This completely inverts the data's meaning.

**Files:** `src/kahne_bench/engines/evaluator.py` (normalize_answer, extract_answer logic)

**Fix:**
- Require explicit `Answer:` marker extraction (already partially implemented for sonnet45)
- Add negative-context filtering: reject numbers preceded by "based on", "anchored to",
  "influenced by", "starting from", etc.
- Apply the improved extraction format used in sonnet45 runs to ALL providers
- Add integration tests with real verbose model responses as fixtures

**Acceptance criteria:**
- Extraction succeeds >95% on verbose responses
- Numbers mentioned in rejection context are never captured as answers
- All providers use the same extraction pipeline

---

### 2. 5 of 6 provider results have zero scored data

**Problem:** All `is_biased` and `bias_score` fields are null in results_anthropic,
results_openai, results_gemini, results_xai, results_fireworks. The fingerprint
calculator defaults everything to 0.0, producing identical meaningless fingerprints
for all 5 providers.

**Root cause:** These runs used only anchoring_effect with the old extraction pipeline.
The extraction failed, scores defaulted to None, and the fingerprint calculator
produced zeros.

**Fix:**
- Delete or archive the 5 broken results files and their fingerprints
- Re-run all providers using the sonnet45 methodology (improved prompts, Answer: marker,
  core tier = 15 biases, 5 domains, 3 trials)
- Do NOT publish results from the old runs under any circumstances

**Acceptance criteria:**
- No results file has >20% null scores
- Fingerprints show meaningful differentiation between models
- All providers tested with identical methodology

---

### 3. gain_loss_framing: loss frame is NEVER tested

**Problem:** The template defines `treatment_gain` and `treatment_loss` keys, but the
`_generate_from_template` method looks for `treatment_{intensity}` keys (e.g.,
`treatment_weak`). Since those don't exist, it falls back to dict ordering and always
picks `treatment_gain`. The loss frame template is dead code.

**Files:** `src/kahne_bench/engines/generator.py:168-197` (templates),
`generator.py:1825-1842` (template selection)

**Impact:** The benchmark only tests gain-frame risk aversion, not the actual framing
EFFECT (the reversal between gain and loss frames), which is the canonical Kahneman &
Tversky (1981) Prospect Theory result.

**Fix:**
- Split gain_loss_framing into two test conditions: `framing_gain` and `framing_loss`
- OR: modify template selection to generate BOTH gain and loss variants for each trial
- Score the framing effect as the DIFFERENCE in responses between gain and loss frames,
  not as absolute bias in either frame alone
- A model that is risk-averse in both frames is NOT exhibiting the framing effect

**Acceptance criteria:**
- Both gain and loss frames are tested
- Framing effect score reflects the reversal between frames
- Tests verify both templates are used

---

### 4. Intensity adjustment only works for anchoring

**Problem:** `_adjust_for_intensity()` (generator.py:3072-3092) only modifies
`anchor_value` for anchoring bias tests. For ALL other biases, every trigger intensity
generates the exact same prompt. This makes the BMS intensity-weighted scoring system
(WEAK=2.0x, MODERATE=1.0x, STRONG=0.67x, ADVERSARIAL=0.5x) meaningless because
there is no actual variation in stimulus strength.

**Files:** `src/kahne_bench/engines/generator.py:3072-3092`

**Fix (two options):**

Option A (recommended): Implement intensity variations for the 15 CORE biases:
- WEAK: subtle, indirect trigger language
- MODERATE: standard trigger (current templates)
- STRONG: explicit, salient trigger
- ADVERSARIAL: compound triggers, emotional pressure

Option B (minimum viable): Remove intensity weighting from BMS calculation and
document that all tests use moderate-equivalent triggers. Report BMS as unweighted
bias rate.

**Acceptance criteria:**
- Either: different intensities produce measurably different prompts for core biases
- Or: BMS calculation and documentation reflect single-intensity methodology

---

### 5. Generic fallback generates unscorable expected answers

**Problem:** For biases using the generic generation path, expected answers are
descriptive strings like "based on statistical data rather than memorable examples".
These will never match model responses via regex. Without LLM judge (off by default),
~19 of 69 biases are unscorable and produce (None, None) for every test.

**Files:** `src/kahne_bench/engines/generator.py:3276-3358`,
`src/kahne_bench/engines/evaluator.py:891-968`

**Fix:**
- For the 15 CORE biases: ensure all have specific templates with concrete expected
  answers (numeric, option, or yes/no format)
- For EXTENDED tier: either write specific templates or enable LLM judge by default
- Clearly report effective coverage: "N of 69 biases fully scorable via regex,
  M additional biases scorable with LLM judge fallback"
- Do NOT claim "69 biases" in any publication unless all 69 are actually scored

**Acceptance criteria:**
- All 15 CORE biases produce concrete, matchable expected answers
- README and any publication state actual scorable bias count
- LLM judge documentation explains when it's needed

---

## P1: High Priority (fix before publication, less urgent)

### 6. Prompts are too transparent for frontier models

**Problem:** All frontier models explicitly identify anchoring manipulation by name and
refuse to engage. The original K&T experiments used disguised anchors (e.g., spinning a
wheel of fortune). Current prompts essentially say "here's an anchor, now estimate."

**Fix:**
- Redesign anchoring prompts to embed anchors naturally (news articles, prior estimates,
  market data) without flagging them as psychological manipulations
- Test redesigned prompts on 2-3 models to verify they elicit natural responses
- Consider the variation.py module (emotional_pressure, authority_framing, etc.) for
  more ecological prompt design

---

### 7. Inconsistent default handling of unknown/failed scores

**Problem:**
- BMS: unknowns filtered out, all-unknown defaults to 0.0 (no bias)
- BMP: unknown baseline defaults to 0.5
- HAS: unknown model rate defaults to 0.5

The same measurement gap produces different metric-level conclusions depending on
which metric is reading it.

**Files:** `src/kahne_bench/metrics/core.py:158,167` (BMS), `core.py:357` (BMP),
`core.py:570` (HAS)

**Fix:**
- Standardize: all metrics should either exclude unknowns or default to the same
  sentinel value
- Recommended: exclude unknowns and report unknown rate alongside each metric
- Already partially implemented via unknown_rate tracking in CognitiveFingerprintReport

---

### 8. Silent exception suppression in LLM judge

**Problem:** `evaluator.py:801-802` has `except Exception: pass` which swallows all
errors (rate limits, auth errors, network timeouts) with zero logging.

**Fix:** Replace with `except Exception as e: logger.warning(f"LLM judge failed: {e}")`

---

### 9. TemporalEvaluator doesn't score responses

**Problem:** `evaluate_persistent()` and `evaluate_adaptive()` create results without
calling `score_response()`. All temporal evaluation data has None scores.

**Files:** `src/kahne_bench/engines/evaluator.py:979-1030`, `evaluator.py:1032-1090`

**Fix:** Call `score_response()` on temporal evaluation results.

---

### 10. present_bias treatment introduces confounding variable

**Problem:** Treatment prompt adds "uncertainty" to the future option, but the control
presents a fixed discount rate. A rational agent rejecting the uncertain future option
is NOT exhibiting present bias.

**Files:** `src/kahne_bench/engines/generator.py:400-403`

**Fix:** Remove "uncertainty" language from present_bias treatment. The treatment should
differ from control only in temporal framing, not in risk profile.

---

### 11. Expected answers for some biases are arbitrary

**Problem:** Some tests (e.g., "appropriate hiring salary") have expected answers
(e.g., 27500) that have no rational basis without additional context like company
size, role, or market.

**Fix:** Either provide sufficient context for a rational answer to exist, or switch
to relative scoring (did the answer shift toward the anchor?) rather than absolute
correctness scoring.

---

## P2: Moderate (fix for quality, not blocking)

### 12. LLM judge prompt leaks expected answers

**Files:** `src/kahne_bench/engines/judge.py:26-60`

The judge prompt includes both expected_rational and expected_biased responses,
giving it the "answer key." Consider testing judge calibration against known inputs.

### 13. RCI stability threshold doesn't account for trial count

**Files:** `src/kahne_bench/metrics/core.py:690`

`is_stable = variance < 0.025` is hardcoded. With 3 Bernoulli trials, variance is
either 0.0 or 0.333, making the threshold binary rather than graduated.

### 14. BCI defaults to perfect consistency on extraction failure

**Files:** `src/kahne_bench/metrics/core.py:273-281`

When extraction fails across all domains, BCI reports consistency_score=1.0.

### 15. Human baselines are literature-derived, not paradigm-matched

**Files:** `src/kahne_bench/metrics/core.py:414-515`

HUMAN_BASELINES use rates from diverse published studies with different methodologies.
HAS compares model performance on KahneBench prompts against these rates. Must be
flagged prominently in any publication.

### 16. Minor: re module imported inside loop

**Files:** `src/kahne_bench/engines/evaluator.py:134`

`import re` inside function body; already imported at module level (line 9).

---

## What Works Well

- Sound theoretical grounding: 69 biases with proper K&T citations and dual-process framing
- Well-designed metric system: 6 orthogonal metrics covering complementary dimensions
- Transparency features: unknown rates tracked per bias, limitations documented
- Good engineering: async evaluation, rate limiting, JSON/CSV export, reproducibility
- LLM judge fallback: properly separated as optional enhancement
- sonnet45 results show genuinely interesting patterns (high present_bias BMS=0.66,
  real framing effects, massive overconfidence calibration gap)

## Recommended Fix Order

1. Fix answer extraction (P0 #1) -- unblocks everything else
2. Fix gain_loss_framing dead code (P0 #3)
3. Add intensity variations for CORE biases OR remove intensity weighting (P0 #4)
4. Fix generic fallback / document actual coverage (P0 #5)
5. Delete old broken results (P0 #2)
6. Redesign transparent prompts (P1 #6)
7. Re-run all providers with fixed pipeline
8. Fix remaining P1 and P2 issues
9. Validate a sample of test cases against K&T literature (not yet completed)
10. Publish

## Estimated Effort

- P0 fixes: ~2-3 focused sessions (mostly generator.py and evaluator.py)
- P1 fixes: ~1-2 sessions
- Re-running all providers: ~$500 in API costs, 1 session
- Test case validation (manual): ~1 session reviewing 15 cases against literature

## Note on Test Case Validity

The test-validator agent did not complete its review before the audit ended. A manual
validation of 10-15 randomly sampled test cases against the original K&T literature
is still needed. This should be done AFTER the P0 code fixes, since some test cases
may change as part of those fixes.
