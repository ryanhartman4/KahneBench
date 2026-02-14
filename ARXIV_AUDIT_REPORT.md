# Kahne-Bench arXiv Readiness Audit

Date: 2026-02-12  
Auditor: Codex (GPT-5)  
Scope: Full repository audit for publication readiness, excluding `reports/` per request.

## Executive Summary

Current code quality is mixed: runtime tests are green, but publication artifacts are not consistently aligned with the current generator/evaluator behavior.

Key blockers before arXiv submission:

1. Published benchmark artifacts appear stale/inconsistent with current code.
2. Existing result artifacts show very high unknown/error rates, undermining statistical reliability.
3. Lint/type hygiene is significantly behind declared standards (`ruff` and strict `mypy` fail heavily).

Until these are addressed, reproducibility and claims strength are at risk.

## What I Ran (with `uv`)

- `uv run pytest -q`
  - Result: `607 passed, 12 warnings`.
- `uv run ruff check src tests`
  - Result: `41` violations.
- `uv run mypy src`
  - Result: `252` errors across 12 files.
- Additional artifact integrity scripts (via `uv run python`) for:
  - dataset consistency checks,
  - result unknown/error rate analysis,
  - schema drift checks,
  - generator-vs-artifact comparison.

## Findings (Ranked by Severity)

### [P0] Committed benchmark artifacts are stale/inconsistent with current generator behavior

Evidence:

- `test_cases_core.json` has `19/225` instances where `expected_rational_response == expected_biased_response`.
  - Bias breakdown: `gain_loss_framing=15`, `loss_aversion=4`.
- `test_cases_core.json` has `210/225` instances where all treatment intensities are text-identical.
- `test_cases_core.json` answer types are heavily text-based (`numeric=75`, `text=150`), unlike current generator output.
- Current generator behavior (dry run) is materially different and healthier:
  - `same_expected=0`, `same_intensity=0`, answer types: `numeric=30`, `option=165`, `yes_no=15`, `confidence=15`.
  - Command used: `PYTHONPATH=src uv run python ... TestCaseGenerator.generate_batch(...)`.

Code references:

- Frame-aware generation logic exists now in `src/kahne_bench/engines/generator.py:1882`.
- Frame map metadata population in `src/kahne_bench/engines/generator.py:1883`.
- Current gain/loss answer typing in `src/kahne_bench/engines/generator.py:2185`.

Impact:

- The benchmark files used in analysis can be non-discriminative and underpowered.
- Paper claims may reflect artifact quality issues rather than model behavior.

Recommendation:

- Regenerate and version benchmark artifacts from current code with fixed seed/config.
- Add an artifact validation gate (CI) that fails on:
  - equal rational/biased targets,
  - identical intensity prompts when intensity differentiation is expected,
  - missing required metadata (e.g., answer type).

### [P1] Result artifacts show high unknown/error rates that threaten inference validity

Evidence:

- `results_sonnet45.json`:
  - total `4725`,
  - unknown `2753` (`58.26%`),
  - provider errors `595` (`12.59%`).
- Unknown rate by answer type in `results_sonnet45.json`:
  - `numeric`: `19.8%`,
  - `choice`: `29.2%`,
  - `text`: `75.6%`.
- Several biases have near-total unknown rates (examples):
  - `sunk_cost_fallacy`: `99.7%`,
  - `hindsight_bias`: `99.4%`,
  - `endowment_effect`: `96.5%`,
  - `confirmation_bias`: `96.2%`,
  - `status_quo_bias`: `94.6%`.

Code references:

- Exceptions are converted to `"ERROR: ..."` and kept, with no retry path in `src/kahne_bench/engines/evaluator.py:866`.
- Scoring falls back to unknown when extraction/scoring fails in `src/kahne_bench/engines/evaluator.py:1113`.
- Text-answer extraction is heuristic-only in `src/kahne_bench/engines/evaluator.py:483`.

Impact:

- Aggregate metrics can be dominated by extraction failure/noise.
- Confidence in comparative conclusions is substantially reduced.

Recommendation:

- Add retry/backoff and provider-specific transient error handling.
- Separate evaluation outputs into:
  - valid-scored,
  - extraction-failed,
  - provider-error.
- Enforce a minimum valid-score threshold per bias before including it in paper tables.

### [P1] Static quality gates are not release-ready (despite strict config)

Evidence:

- `ruff`: `41` issues currently.
- `mypy --strict`: `252` errors in 12 files.
- Project declares strict typing in `pyproject.toml` (`[tool.mypy] strict = true`).

Representative code references:

- `src/kahne_bench/engines/evaluator.py:888` (`str | None` passed where `str` expected).
- `src/kahne_bench/core.py:367` (`TestResult.extracted_answer` typed as non-optional `str`).
- `src/kahne_bench/cli.py:36` (broad untyped tuple return and many untyped defs).

Impact:

- High risk of latent regressions and schema/type bugs.
- Undermines confidence in reproducibility for external reviewers.

Recommendation:

- Gate release on `ruff` + `mypy` pass for `src/` at minimum.
- Align core dataclass typing with runtime behavior (`extracted_answer` optionality).

### [P2] Calibration metric default can overstate performance when confidence is absent

Evidence:

- When no confidence is present, CAS returns a perfect calibration score (`1.0`) by construction in `src/kahne_bench/metrics/core.py:857`.
- In current artifacts, confidence is usually absent (`results_sonnet45.json` no-confidence rate: `93.76%`).

Impact:

- Calibration conclusions can be inflated or misleading when confidence reporting is sparse.

Recommendation:

- Return `None`/N/A (or explicit invalid flag) when confidence coverage is below threshold.
- Require confidence coverage reporting beside CAS in all tables.

### [P2] CLI defaults are unsafe for publication workflows

Evidence:

- Default provider is `mock` in evaluation command options (`src/kahne_bench/cli.py:377`).
- `mock` provider emits random synthetic answers (`src/kahne_bench/cli.py:43`).

Impact:

- Easy to accidentally produce non-scientific results that look valid.

Recommendation:

- Require explicit `--provider` for `evaluate`.
- If `mock` is used, watermark outputs and hard-fail when writing canonical artifact filenames.

### [P3] Artifact schema drift indicates committed results are from older code paths

Evidence:

- Current exporter includes unknown-rate fields (`src/kahne_bench/utils/io.py:422`, `src/kahne_bench/utils/io.py:429`), but committed fingerprint artifacts lack them.
- `fingerprint.json` includes only 4 biases, while `results.json` has 6 bias IDs.

Impact:

- Harder to trust provenance and compare runs across revisions.

Recommendation:

- Add a manifest file per run:
  - git commit SHA,
  - generator config,
  - evaluator config,
  - schema version.

## Warnings (Non-blocking, but should be cleaned)

- Pytest collection warnings for class names like `TestScale` and `TestResult` appearing as potential tests.
- Lint import-order and unused symbol noise in both `src/` and `tests/` reduces signal quality for real regressions.

## Pre-arXiv Release Checklist (Recommended)

1. Regenerate `test_cases_*.json`, `results_*.json`, and `fingerprint_*.json` with current code and fixed seeds.
2. Enforce artifact validation checks in CI (no equal rational/biased targets, no malformed intensities, schema checks).
3. Implement retry/backoff and classify failure modes in evaluator outputs.
4. Set publication inclusion thresholds:
   - minimum valid-score coverage per bias,
   - maximum unknown/error rates.
5. Fix `ruff` for `src/` and bring `mypy src` to zero (or define explicit temporary excludes).
6. Adjust CAS handling for missing confidence and report confidence coverage explicitly.
7. Make non-mock provider explicit in CLI for evaluation commands.
8. Add run manifest metadata (commit hash, config hash, schema version) to all exported artifacts.

## Overall Verdict

Not arXiv-ready yet for strong empirical claims.  
Code execution is functionally stable (`pytest` passes), but artifact quality/provenance and analysis reliability controls need one focused hardening pass before publication.

