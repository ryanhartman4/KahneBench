# Benchmark Improvements (Sonnet Fingerprint v2)

This document is structured into MECE parts so multiple agents can work
in parallel with minimal overlap. Each part lists the exact files to touch
and acceptance criteria.

## MECE Parallel Work Parts (Atomic Units)

Each part is intentionally narrow. Assign one agent per part and avoid
editing files outside the listed boundaries.

### Part A: Answer Marker Extraction

Scope:
- Numeric and option extraction should prefer explicit answer markers.

Files (do not edit outside this list):
- `src/kahne_bench/engines/evaluator.py`
- `tests/test_evaluator.py`

Primary tasks:
- Prefer `Answer:` lines when extracting numeric or A/B/C choices.
- Ignore unrelated numbers (rules of thumb, example calculations).

Acceptance criteria:
- Explicit `Answer:` line is always selected when present.
- Numeric extraction success rate >95% on verbose responses.

---

### Part B: Confidence Extraction and Scoring

Scope:
- Extract confidence from `Confidence:` lines and score overconfidence
  on that value (not the trivia answer).

Files (do not edit outside this list):
- `src/kahne_bench/engines/evaluator.py`
- `tests/test_evaluator.py`

Primary tasks:
- Extract `Confidence:` as a numeric value (0-100 or 0-1).
- Ensure overconfidence scoring uses confidence only.

Acceptance criteria:
- Confidence extraction returns the stated confidence value.
- Overconfidence scores match confidence, not the trivia answer.

---

### Part C: Unknown Handling in Metrics

Scope:
- Unknown extraction should not be treated as neutral bias.

Files (do not edit outside this list):
- `src/kahne_bench/metrics/core.py`
- `tests/test_metrics.py`

Primary tasks:
- Track unknown/invalid rate per bias.
- Exclude unknowns from susceptibility ranking or report separately.

Acceptance criteria:
- Unknowns do not inflate “most resistant” lists.
- Summary exposes unknown rates.

---

### Part D: Choice-Format Prompt Contract

Scope:
- Standardize A/B/C prompts to require `Answer: A/B/C`.

Files (do not edit outside this list):
- `src/kahne_bench/engines/generator.py`
- `tests/test_generator.py`

Biases:
- conjunction_fallacy
- endowment_effect
- status_quo_bias
- confirmation_bias
- sunk_cost_fallacy
- hindsight_bias

Acceptance criteria:
- Prompts include explicit `Answer:` instruction.
- Extraction succeeds on A/B/C responses >95%.

---

### Part E: EV and Value-Model Fixes

Scope:
- Fix value-model biases where rational == biased or EV is wrong.

Files (do not edit outside this list):
- `src/kahne_bench/engines/generator.py`
- `tests/test_generator.py`

Biases:
- gain_loss_framing
- loss_aversion
- certainty_effect
- present_bias

Acceptance criteria:
- No generated instance has `expected_rational_response ==
  expected_biased_response`.
- Rational choice is grounded in a stated EV/discounting rule.

---

### Part F: Numeric-Target Redesign

Scope:
- Replace arbitrary numeric targets with defensible ones or categorical outputs.

Files (do not edit outside this list):
- `src/kahne_bench/engines/generator.py`
- `tests/test_generator.py`

Biases:
- anchoring_effect
- base_rate_neglect
- gambler_fallacy

Acceptance criteria:
- Targets are tied to a clear normative model or categorical choice.
- Prompts specify a single parseable answer format.

---

### Part G: Availability Bias Output Shape

Scope:
- Align availability bias prompt with scoring expectations.

Files (do not edit outside this list):
- `src/kahne_bench/engines/generator.py`
- `src/kahne_bench/engines/evaluator.py` (only if needed for multi-field scoring)
- `tests/test_generator.py`
- `tests/test_evaluator.py`

Acceptance criteria:
- Prompt output shape matches scoring (single target or explicit multi-field parse).
- No unresolved placeholders appear in generated prompts.

---

### Part H: Validation and Rerun Checklist

Scope:
- Add validation/tests that enforce the new contracts and update rerun steps.

Files (do not edit outside this list):
- `tests/test_generator.py`
- `tests/test_integration.py`
- `docs/BENCHMARK_IMPROVEMENTS.md`

Primary tasks:
- Fail generation if any `{var}` placeholders remain.
- Fail generation if rational == biased.
- Update rerun checklist if contracts change.

Acceptance criteria:
- Tests fail on placeholder leakage or identical rational/biased targets.
- Rerun checklist reflects updated prompt/answer contracts.

---

## Shared Context (Read-Only Reference)

Biases covered in the Sonnet v2 run:
- anchoring_effect
- availability_bias
- base_rate_neglect
- conjunction_fallacy
- gain_loss_framing
- loss_aversion
- endowment_effect
- status_quo_bias
- certainty_effect
- overconfidence_effect
- confirmation_bias
- sunk_cost_fallacy
- present_bias
- hindsight_bias
- gambler_fallacy

Source artifacts:
- `fingerprint_sonnet_v2.json`
- `results_sonnet_v2.json`
- `docs/ZERO_SCORE_BIASES.md`

Key failure modes observed:
- Numeric extraction chooses unrelated numbers.
- Unknown extractions default to 0.5 and appear as “resistant.”
- Some prompts request multi-value answers while scoring expects a single value.
- Several biases encode arbitrary biased targets or allow rational==biased.

## Rerun Checklist (After All Parts Land)

1. Regenerate cases after generator fixes.
2. Run evaluation on the same model configuration.
3. Verify that:
   - extraction success rates are high for all biases,
   - biases previously showing 0.0 due to extraction errors no longer do,
   - susceptibility rankings are not dominated by unknown scores.
