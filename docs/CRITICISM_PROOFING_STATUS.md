# Kahne-Bench: Criticism-Proofing Status

This document tracks progress on making Kahne-Bench publication-ready for both academic venues and open-source release.

## Goal

Address 22 identified issues across methodological rigor, code quality, and completeness to make the benchmark "criticism-proof."

## Completed Work (6/13 items)

### 1. Python Version Compatibility ✅
- **Status**: Already correct
- **Details**: `pyproject.toml` already specifies `requires-python = ">=3.10"`
- **No changes needed**

### 2. BIAS_TEMPLATES Coverage (50% → 100%) ✅
- **File**: `src/kahne_bench/engines/generator.py`
- **Changes**: Added 34 new templates with full psychological fidelity
- **Result**: All 69 taxonomy biases now have dedicated templates
- **Templates added**:
  - Representativeness: hot_hand_fallacy, regression_neglect, stereotype_bias, prototype_heuristic
  - Availability: recency_bias, salience_bias, simulation_heuristic, primacy_bias
  - Anchoring: insufficient_adjustment, focalism, first_offer_anchoring, numeric_priming
  - Loss Aversion: disposition_effect
  - Framing: attribute_framing, reference_point_framing, risk_framing, temporal_framing, mental_accounting
  - Probability: probability_weighting, possibility_effect, denominator_neglect
  - Confirmation: belief_perseverance, myside_bias
  - Temporal: duration_neglect
  - Extension Neglect: group_attribution_bias
  - Memory: rosy_retrospection, source_confusion, misinformation_effect, memory_reconstruction_bias
  - Attention: attentional_bias, inattentional_blindness, selective_perception
  - Attribution: fundamental_attribution_error, actor_observer_bias, self_serving_bias
  - Uncertainty: ambiguity_aversion, illusion_of_validity, competence_hypothesis
  - Social: ingroup_bias, false_consensus_effect, outgroup_homogeneity_bias
- **Note**: Removed duplicate `regression_to_mean` (now `regression_neglect` to match taxonomy)

### 3. Evaluator Test Coverage ✅
- **File**: `tests/test_evaluator.py` (NEW)
- **Tests added**: 36 tests
- **Classes tested**:
  - `AnswerExtractor`: 10 tests for extraction patterns
  - `BiasEvaluator`: 9 tests for evaluation and scoring
  - `TemporalEvaluator`: 6 tests for persistent and adaptive evaluation
  - `ContextSensitivityEvaluator`: 9 tests for context sensitivity, expertise gradient, stakes gradient
  - `EvaluationConfig`: 2 tests for configuration defaults

### 4. Advanced Generator Test Coverage ✅
- **File**: `tests/test_advanced_generators.py` (NEW)
- **Tests added**: 33 tests
- **Classes tested**:
  - `NovelScenarioGenerator`: 11 tests for contamination-resistant generation
  - `MacroScaleGenerator`: 14 tests for decision chain generation
  - `DecisionNode`: 2 tests
  - `DecisionChain`: 1 test
  - `NOVEL_SCENARIO_ELEMENTS`: 5 tests

### 5. LIMITATIONS.md ✅
- **File**: `docs/LIMITATIONS.md` (NEW)
- **Content**: Transparent documentation of:
  - Validation limitations (no human validation data)
  - Human baselines from literature (not collected for this framework)
  - Intensity weighting not empirically calibrated
  - Expected answer ambiguity
  - Template-based generation constraints
  - Theoretical grounding scope (not all biases are core K&T)
  - Interaction matrix coverage (~30%)
  - Technical limitations (extraction, scoring edge cases)
  - Recommendations for users

### 6. BMS Weight Documentation ✅
- **File**: `src/kahne_bench/metrics/core.py`
- **Changes**:
  - Added `INTENSITY_WEIGHT_RATIONALE` block comment explaining susceptibility-based weighting
  - Added `DEFAULT_INTENSITY_WEIGHTS` constant with inline documentation
  - Added `DEFAULT_AGGREGATION_WEIGHTS` constant
  - Made weights configurable via `intensity_weights` and `aggregation_weights` parameters in `BiasMagnitudeScore.calculate()`
  - Enhanced class docstring with full explanation of weighting philosophy

## Test Suite Growth

| Before | After | Change |
|--------|-------|--------|
| 56 tests | 125 tests | +69 tests (+123%) |

---

## Remaining Work (7 items)

### 7. Human Baseline Improvements 🔴 HIGH PRIORITY
- **File**: `src/kahne_bench/metrics/core.py` (lines 300-390, 440)
- **Issues**:
  - Default 0.5 for unknown biases is problematic (line 440)
  - Some citations are weak/non-K&T
- **Fix needed**:
  - Change default to `None` and raise warning when used
  - Add `UNKNOWN_BASELINE_BIASES` set with explicit list
  - Add inline citations for all baselines
  - Flag non-K&T biases in HAS output

### 8. K&T Grounding Audit 🟡 MEDIUM PRIORITY
- **File**: `src/kahne_bench/biases/taxonomy.py`
- **Issue**: 23% of biases (16/69) lack direct K&T grounding
- **Fix needed**:
  - Add `is_kt_core: bool` field to `BiasDefinition` dataclass in `core.py`
  - Mark peripheral biases (primacy_bias, attention biases, etc.)
  - Update documentation to distinguish core vs extended biases

### 9. Answer Extractor Hardening 🔴 HIGH PRIORITY
- **File**: `src/kahne_bench/engines/evaluator.py` (lines 110-227)
- **Issues**:
  - "UNKNOWN" not validated as sentinel in scoring
  - Case normalization inconsistent
  - Confidence not clamped to [0, 1]
  - Numeric extraction may grab wrong number
- **Fix needed**:
  - Handle "UNKNOWN" explicitly in scoring
  - Normalize all extracted answers to consistent case
  - Clamp confidence to [0, 1] range
  - Add epsilon tolerance for numeric comparisons

### 10. Scoring Logic Fixes 🔴 HIGH PRIORITY
- **File**: `src/kahne_bench/engines/evaluator.py` (lines 423-464)
- **Issues**:
  - No numeric tolerance (epsilon)
  - Default fallback (True, 0.5) is questionable
  - Placeholder detection may miss valid answers starting with "["
- **Fix needed**:
  - Add numeric tolerance (epsilon = 0.01)
  - Change default fallback handling
  - Improve placeholder detection

### 11. Expand Interaction Matrix (29% → 60%+) 🟡 MEDIUM PRIORITY
- **File**: `src/kahne_bench/biases/taxonomy.py` (lines 974-1089)
- **Issue**: Only 20/69 biases have interaction definitions
- **Fix needed**: Add theoretically-grounded interactions for remaining biases, prioritizing:
  - All representativeness biases
  - All framing biases
  - All temporal biases
  - Memory × Overconfidence interactions

### 12. Fix Documentation Mismatches 🟢 LOW PRIORITY
- **Files**: `README.md`, `CLAUDE.md`
- **Issues**:
  - Example uses `gpt-5.2` (should be `gpt-4o`)
  - Some metric descriptions may be outdated
- **Fix needed**: Review and update all examples

### 13. Integration Tests 🟡 MEDIUM PRIORITY
- **File**: `tests/test_integration.py` (NEW)
- **Fix needed**: Add tests for full pipeline: generate → evaluate → score → metrics

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/kahne_bench/engines/generator.py` | BIAS_TEMPLATES, generators |
| `src/kahne_bench/engines/evaluator.py` | AnswerExtractor, BiasEvaluator, scoring |
| `src/kahne_bench/metrics/core.py` | 6 metrics, HUMAN_BASELINES, weights |
| `src/kahne_bench/biases/taxonomy.py` | 69 BiasDefinitions, BIAS_INTERACTION_MATRIX |
| `src/kahne_bench/core.py` | Core types (CognitiveBiasInstance, TestResult, etc.) |

---

## Quick Commands

```bash
# Run all tests
PYTHONPATH=src uv run pytest

# Run specific test file
PYTHONPATH=src uv run pytest tests/test_evaluator.py -v

# Count templates vs taxonomy biases
PYTHONPATH=src uv run python -c "
from kahne_bench.biases.taxonomy import BIAS_TAXONOMY
from kahne_bench.engines.generator import BIAS_TEMPLATES
print(f'Taxonomy: {len(BIAS_TAXONOMY)}, Templates: {len(BIAS_TEMPLATES)}')
print(f'Coverage: {len(set(BIAS_TAXONOMY.keys()) & set(BIAS_TEMPLATES.keys()))}/{len(BIAS_TAXONOMY)}')
"
```

---

## Original Issue Summary (from exploration)

22 issues were identified across 3 categories:

| Category | Critical | High | Medium | Total |
|----------|----------|------|--------|-------|
| Methodological Rigor | 2 | 4 | 3 | 9 |
| Code Quality | 1 | 5 | 3 | 9 |
| Completeness | 1 | 2 | 1 | 4 |
| **Total** | **4** | **11** | **7** | **22** |

---

*Last updated: 2025-01-06*
*Commit: c9344d2*
