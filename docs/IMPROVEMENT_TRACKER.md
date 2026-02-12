# Kahne-Bench Improvement Tracker

This document tracks improvements identified from the gap analysis comparing the implementation against the Kahne-Bench specification.

## Analysis Date: 2026-01-04

## Final State Summary

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Total Biases | 58 | 69 | COMPLETE |
| Interaction Matrix Coverage | 12% (7/58) | 100% (40 primary entries, 69 unique biases) | COMPLETE |
| Empty Categories | 2 | 0 | COMPLETE |
| Documentation Accuracy | Partial | Complete | COMPLETE |

---

## Improvement 1: Add Missing K&T Biases

**Priority:** High
**Status:** COMPLETE

### Biases Added (11 total)

| Bias | Category | K&T Foundation | Status |
|------|----------|----------------|--------|
| affect_heuristic | PROBABILITY_DISTORTION | Slovic & Peters (2006) | ADDED |
| mental_accounting | FRAMING | Thaler (1985, 1999) | ADDED |
| ingroup_bias | SOCIAL_BIAS | Tajfel & Turner (1979) | ADDED |
| fundamental_attribution_error | ATTRIBUTION_BIAS | Ross (1977) | ADDED |
| false_consensus_effect | SOCIAL_BIAS | Ross, Greene & House (1977) | ADDED |
| actor_observer_bias | ATTRIBUTION_BIAS | Jones & Nisbett (1971) | ADDED |
| self_serving_bias | ATTRIBUTION_BIAS | Miller & Ross (1975) | ADDED |
| ambiguity_aversion | UNCERTAINTY_JUDGMENT | Ellsberg (1961) | ADDED |
| illusion_of_validity | UNCERTAINTY_JUDGMENT | Kahneman & Tversky (1973) | ADDED |
| competence_hypothesis | UNCERTAINTY_JUDGMENT | Heath & Tversky (1991) | ADDED |
| outgroup_homogeneity_bias | SOCIAL_BIAS | Quattrone & Jones (1980) | ADDED |

### Files Modified
- `src/kahne_bench/biases/taxonomy.py` - Added 4 new bias lists
- `src/kahne_bench/metrics/core.py` - Added human baselines for all new biases

---

## Improvement 2: Expand Interaction Matrix

**Priority:** High
**Status:** COMPLETE

### Interactions Added

| Primary Bias | Secondary Biases Added | Cluster Type |
|--------------|------------------------|--------------|
| confirmation_bias | myside_bias, belief_perseverance, overconfidence_effect, selective_perception | Confirmation cluster |
| hindsight_bias | overconfidence_effect, illusion_of_control, memory_reconstruction_bias | Retrospective cluster |
| sunk_cost_fallacy | loss_aversion, status_quo_bias, endowment_effect | Loss-driven persistence |
| scope_insensitivity | identifiable_victim_effect, neglect_of_probability, affect_heuristic | Extension neglect |
| memory_reconstruction_bias | hindsight_bias, rosy_retrospection, self_serving_bias | Memory distortion |
| attentional_bias | salience_bias, availability_bias, selective_perception | Attention-availability |
| halo_effect | stereotype_bias, group_attribution_bias, ingroup_bias | Social judgment |
| certainty_effect | zero_risk_bias, probability_weighting, affect_heuristic | Probability distortion |
| fundamental_attribution_error | actor_observer_bias, self_serving_bias, group_attribution_bias | Attribution cluster |
| ingroup_bias | outgroup_homogeneity_bias, false_consensus_effect, halo_effect | Social bias cluster |
| ambiguity_aversion | competence_hypothesis, illusion_of_validity, affect_heuristic | Uncertainty cluster |

**Coverage:** 18 primary biases with ~60 total interaction pairs (was 7 with ~21)

---

## Improvement 3: Fix Empty Categories

**Priority:** Medium
**Status:** COMPLETE

### Categories Fixed

| Category | Before | After | Biases Added |
|----------|--------|-------|--------------|
| ATTRIBUTION_BIAS | 0 | 3 | fundamental_attribution_error, actor_observer_bias, self_serving_bias |
| UNCERTAINTY_JUDGMENT | 0 | 3 | ambiguity_aversion, illusion_of_validity, competence_hypothesis |
| SOCIAL_BIAS | 2 | 5 | ingroup_bias, false_consensus_effect, outgroup_homogeneity_bias |

---

## Improvement 4: Update Documentation

**Priority:** Medium
**Status:** COMPLETE

### Files Updated

| File | Changes |
|------|---------|
| README.md | Updated bias count 51→69, updated category table (16 categories), updated tier counts |
| CLAUDE.md | Updated bias count 58→69 throughout |
| taxonomy.py docstring | Updated bias count and category list |

---

## Change Log

| Date | Change | Files Modified |
|------|--------|----------------|
| 2026-01-04 | Created tracker | docs/IMPROVEMENT_TRACKER.md |
| 2026-01-04 | Added 11 new biases | src/kahne_bench/biases/taxonomy.py |
| 2026-01-04 | Expanded interaction matrix (7→18 pairs) | src/kahne_bench/biases/taxonomy.py |
| 2026-01-04 | Added human baselines | src/kahne_bench/metrics/core.py |
| 2026-01-04 | Updated README.md | README.md |
| 2026-01-04 | Updated CLAUDE.md | CLAUDE.md |
| 2026-01-04 | Marked all improvements complete | docs/IMPROVEMENT_TRACKER.md |

---

## Verification Checklist

After all improvements:
- [x] All tests pass (`PYTHONPATH=src uv run pytest`) - ~548 tests passed
- [x] Bias count matches documentation (69)
- [x] All categories have at least 1 bias
- [x] Interaction matrix covers 100% of biases (40 primary entries, 69 unique biases)
- [x] Human baselines added for new biases (11 added)
- [ ] BIAS_TEMPLATES added for new biases (TODO: future work)

---

## Future Work

1. **Add BIAS_TEMPLATES for new biases** - Generator templates for the 11 new biases
2. **Further expand interaction matrix** - Target 50%+ coverage
3. **Add human baselines for remaining 7 memory/attention biases** - Requires literature review
4. **Implement RAE (Rationality Alignment Environment)** - RL training infrastructure per spec
