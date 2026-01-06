# Kahne-Bench: Known Limitations

This document transparently describes the current limitations of Kahne-Bench to inform users and reviewers about areas where results should be interpreted with caution.

## 1. Validation Limitations

### 1.1 No Human Validation Data
The benchmark has not been validated against human subject data collected specifically for this framework. While the bias definitions and human baselines reference established psychological research, we have not:

- Conducted pilot studies with human participants
- Verified that test instances elicit the intended biases in humans
- Established inter-rater reliability for bias detection
- Validated that the scoring methodology produces results consistent with human expert judgments

**Implication**: Model scores should be interpreted as relative comparisons between models, not as absolute measures of human-like bias.

### 1.2 Human Baselines from Literature
The `HUMAN_BASELINES` dictionary aggregates susceptibility rates from published psychological research. Limitations include:

- **Varying methodologies**: Original studies used different experimental paradigms
- **Different populations**: Baselines often derive from WEIRD (Western, Educated, Industrialized, Rich, Democratic) samples
- **Temporal distance**: Classic studies (1970s-1990s) may not reflect current human cognition
- **Missing baselines**: Some biases (especially newly added ones) have limited or no human baseline data

**Recommendation**: Use Human Alignment Score (HAS) as a rough guide, not a definitive measure.

## 2. Methodological Limitations

### 2.1 Intensity Weighting Not Empirically Calibrated
The BMS (Bias Magnitude Score) intensity weights are design decisions, not empirically derived:

| Intensity | Weight | Rationale |
|-----------|--------|-----------|
| WEAK | 2.0x | Higher susceptibility indicates stronger bias |
| MODERATE | 1.0x | Baseline trigger |
| STRONG | 0.67x | Expected response to strong pressure |
| ADVERSARIAL | 0.5x | Expected response to extreme pressure |

These weights have not been validated through:
- Sensitivity analysis
- Calibration against human data
- Cross-validation across models

### 2.2 Expected Answer Ambiguity
Some test instances have expected answers that are:
- **Placeholder-level**: Generic descriptions rather than specific values
- **Subjectively defined**: "Rational" vs "biased" may not be objectively distinguishable
- **Context-dependent**: The "correct" answer may vary based on unstated assumptions

Affected biases are handled by returning a default score of 0.5 when expected answers cannot be evaluated.

### 2.3 Template-Based Generation Constraints
While all 69 taxonomy biases now have templates, the template-based approach has inherent limitations:

- **Fixed structure**: Templates may not capture all manifestations of a bias
- **Limited diversity**: Repeated testing may exploit template patterns
- **Domain adaptation**: Some biases don't translate equally well to all domains

## 3. Scope Limitations

### 3.1 Theoretical Grounding
Not all 69 biases have direct grounding in Kahneman-Tversky research:

| Category | Notes |
|----------|-------|
| **Core K&T** (~50 biases) | Direct lineage from prospect theory and heuristics research |
| **Extended** (~19 biases) | Related cognitive biases documented by other researchers |

Biases with weaker K&T grounding include:
- Attention biases (Simons & Chabris, 1999)
- Attribution biases (Ross, 1977)
- Some social biases (Tajfel & Turner, 1979)

### 3.2 Interaction Matrix Coverage
The `BIAS_INTERACTION_MATRIX` currently covers approximately 30% of theoretically plausible bias interactions. Many documented compound effects from the literature are not yet implemented.

### 3.3 Ecological Validity
Test scenarios, while mapped to five domains, may not fully capture:
- Real-world decision complexity
- Time pressure effects
- Emotional context
- Social influences on decision-making

## 4. Technical Limitations

### 4.1 Answer Extraction Heuristics
The `AnswerExtractor` uses pattern matching that may:
- Misidentify answers in complex responses
- Fail to extract numeric answers embedded in explanations
- Return "UNKNOWN" for valid but unconventionally formatted responses

### 4.2 Scoring Edge Cases
- **Numeric tolerance**: Partial credit scoring for numeric answers uses linear interpolation, which may not reflect psychological distance
- **Unknown responses**: Treated as 0.5 bias score, which may inflate or deflate true bias measures
- **Confidence extraction**: May miss confidence statements not matching expected patterns

### 4.3 Rate Limiting
Rate limiting is applied between instances but not between trials within an instance, which could cause API issues with strict rate limits.

## 5. Recommendations for Users

### When Publishing Results:
1. Report confidence intervals across multiple runs
2. Note which biases showed significant effects
3. Acknowledge these limitations in discussion sections
4. Consider using relative comparisons between models rather than absolute claims

### When Interpreting Results:
1. High BMS scores indicate potential bias, not confirmed bias
2. Cross-reference findings with qualitative analysis of model responses
3. Consider whether failure modes could be extraction errors vs. true bias
4. Use multiple evaluation frameworks for important assessments

## 6. Future Improvements

We are actively working to address these limitations:

- [ ] Conduct human validation studies
- [ ] Expand interaction matrix coverage to 60%+
- [ ] Add sensitivity analysis for metric weights
- [ ] Improve answer extraction robustness
- [ ] Add `is_kt_core` field to distinguish theoretical grounding levels

## References

This framework builds on foundational research in behavioral economics and cognitive psychology. Key references:

- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Tversky, A., & Kahneman, D. (1974). Judgment under Uncertainty: Heuristics and Biases. *Science*, 185(4157), 1124-1131.
- Tversky, A., & Kahneman, D. (1981). The Framing of Decisions and the Psychology of Choice. *Science*, 211(4481), 453-458.

For a complete list of bias-specific citations, see `src/kahne_bench/biases/taxonomy.py`.

---

*Last updated: 2025-01-06*
