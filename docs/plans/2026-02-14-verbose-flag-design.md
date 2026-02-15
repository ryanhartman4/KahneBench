# Design: Verbose Flag for Evaluate Command

**Date:** 2026-02-14
**Status:** Approved

## Problem

The `evaluate` command runs for 10-15 minutes with minimal feedback. Users see a Rich progress bar but no detail about what's happening: which biases are being tested, API call timing, retry behavior, scoring results, or judge fallback usage.

## Decision

Add a `--verbose` flag to the `evaluate` command that enables detailed Python logging output.

## Design

### CLI Interface

- Flag: `--verbose` (long-form only, no `-v` shorthand)
- Scope: `evaluate` command only
- When set: configures `logging` at `DEBUG` level with format `HH:MM:SS module_name LEVEL message`
- When not set: logging stays at `WARNING` (current behavior unchanged)
- Progress bar: disabled when `--verbose` is set (verbose output is strictly more informative)

### Logging Calls

**`engines/evaluator.py`:**
- `INFO` per-instance start: bias_id, domain, instance number
- `DEBUG` per-condition: intensity, trial number, control/treatment/debiasing
- `DEBUG` API call completion: model, response time, token estimate
- `DEBUG` rate limit retries: attempt number, delay
- `DEBUG` answer extraction result: extracted answer, is_biased, bias_score
- `INFO` per-instance completion: total conditions tested, time elapsed

**`engines/judge.py`:**
- `DEBUG` judge fallback triggered: why regex extraction failed
- `DEBUG` judge result: extracted answer from XML

### Files Modified

| File | Change |
|------|--------|
| `cli.py` | Add `--verbose` option, configure logging, conditionally skip progress bar |
| `engines/evaluator.py` | Add `logger.info()` and `logger.debug()` calls at key points |
| `engines/judge.py` | Add `logger.debug()` calls for fallback events |

### Example Output

```
14:23:01 kahne_bench.engines.evaluator INFO Evaluating instance 1/15: anchoring_effect [INDIVIDUAL]
14:23:01 kahne_bench.engines.evaluator DEBUG  Condition: treatment_weak (trial 1/3)
14:23:02 kahne_bench.engines.evaluator DEBUG  API call: mock-model, 142 tokens, 0.8s
14:23:02 kahne_bench.engines.evaluator DEBUG  Extracted answer: 'Option A', biased=True, score=0.85
14:23:02 kahne_bench.engines.judge DEBUG  Judge fallback triggered for ambiguous response
```

### Testing

One CLI integration test with mock provider asserting `--verbose` succeeds (exit code 0). No assertion on specific log lines.

### Out of Scope

- No `--quiet` flag
- No verbosity in other commands (generate, report, etc.)
- No Rich-formatted verbose output (plain logging only)
