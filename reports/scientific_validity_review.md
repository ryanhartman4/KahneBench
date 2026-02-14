# Scientific Validity Review: Kahne-Bench CORE Bias Templates

**Reviewer:** science-reviewer (automated audit)
**Date:** 2026-02-11
**Scope:** All 15 CORE biases in `KAHNE_BENCH_CORE_BIASES`
**Source files:** `src/kahne_bench/engines/generator.py` (BIAS_TEMPLATES, _get_template_variables, _adjust_for_intensity), `src/kahne_bench/biases/taxonomy.py`

---

## Summary Table

| # | Bias ID | K&T Citation | Template Fidelity | Expected Answers | Control/Treatment Isolation | Confounds | LLM Transparency | Intensity Variation | Verdict |
|---|---------|-------------|-------------------|-----------------|---------------------------|-----------|------------------|--------------------|---------|
| 1 | `anchoring_effect` | T&K (1974) | High | Defensible | Good | Low | Low | Good (numeric multiplier) | **VALID** |
| 2 | `availability_bias` | T&K (1973) | High | Defensible | Good | Low | Moderate | Good (preamble escalation) | **VALID** |
| 3 | `base_rate_neglect` | K&T (1973) | High | Defensible | Good | Low | Moderate | Good (preamble escalation) | **VALID** |
| 4 | `conjunction_fallacy` | T&K (1983) | High | Correct | Good | Low | Moderate | Good (preamble escalation) | **VALID** |
| 5 | `gain_loss_framing` | T&K (1981) | High | Defensible | Excellent | Low | Low | Excellent (frame switching) | **VALID** |
| 6 | `loss_aversion` | K&T (1979) | Moderate | Defensible | Moderate | Moderate | Moderate | Moderate | **QUESTIONABLE** |
| 7 | `endowment_effect` | Thaler (1980) / KKT (1990) | Moderate | Defensible | Good | Low | Moderate-High | Good (preamble escalation) | **QUESTIONABLE** |
| 8 | `status_quo_bias` | Samuelson & Zeckhauser (1988) | Moderate | Defensible | Good | Low | High | Good (preamble escalation) | **QUESTIONABLE** |
| 9 | `certainty_effect` | K&T (1979) | High | Correct | Good | Low | Moderate | Good (preamble escalation) | **VALID** |
| 10 | `overconfidence_effect` | Lichtenstein et al. (1982) / K&T tradition | Moderate | Defensible | Moderate | Moderate | Low | Good (preamble escalation) | **QUESTIONABLE** |
| 11 | `confirmation_bias` | Wason (1960) / K&T tradition | Moderate | Correct | Good | Low | High | Good (preamble escalation) | **QUESTIONABLE** |
| 12 | `sunk_cost_fallacy` | Arkes & Blumer (1985) / K&T tradition | High | Correct | Good | Low | Moderate | Good (numeric + preamble) | **VALID** |
| 13 | `present_bias` | Laibson (1997) / O'Donoghue & Rabin (1999) | High | Correct | Good | Low | Low | Good (preamble escalation) | **VALID** |
| 14 | `hindsight_bias` | Fischhoff (1975) / K&T tradition | High | Defensible | Good | Low | Moderate | Good (preamble escalation) | **VALID** |
| 15 | `gambler_fallacy` | T&K (1971) | High | Correct | Excellent | None | Low | Good (streak + preamble) | **VALID** |

**Summary: 10 VALID, 5 QUESTIONABLE, 0 INVALID**

---

## Detailed Per-Bias Analysis

### 1. Anchoring Effect

**Citation:** Tversky & Kahneman (1974), "Judgment under Uncertainty: Heuristics and Biases"

**Paradigm fidelity:** HIGH. The classic anchoring paradigm presents an irrelevant or semi-relevant numeric value before asking for an estimate. The template does exactly this: control asks for an estimate with a range hint, treatment introduces an anchor from a "recent {anchor_source}". The anchor is described as "may or may not be directly applicable," which is faithful to the informational anchoring paradigm (as opposed to basic anchoring with clearly irrelevant numbers).

**Expected answers:** DEFENSIBLE. Rational answer = midpoint of the provided range. Biased answer = midpoint between rational and anchor (classic insufficient adjustment). This models the well-established finding that adjustment from anchors is typically insufficient (Epley & Gilovich, 2006).

**Control/treatment isolation:** GOOD. The only difference between control and treatment is the presence of the anchor value with its source. The range hint appears in both conditions, providing a normative baseline.

**Confounds:** LOW. The anchor source descriptions are naturalistic ("industry survey," "peer-reviewed study") which avoids making the anchor obviously arbitrary.

**LLM transparency:** LOW. The treatment does not telegraph "this is an anchoring test." The anchor is framed as background information, which should elicit natural anchoring behavior.

**Intensity variation:** GOOD. Anchor values are multiplied by intensity (0.5x for WEAK, 2.0x for ADVERSARIAL), creating progressively more extreme anchors. This maps well to the research finding that more extreme anchors produce larger effects (Furnham & Boo, 2011).

**Verdict: VALID**

---

### 2. Availability Bias

**Citation:** Tversky & Kahneman (1973), "Availability: A Heuristic for Judging Frequency and Probability"

**Paradigm fidelity:** HIGH. The classic paradigm asks people to estimate frequencies where vivid/dramatic causes are overestimated relative to mundane but more common causes. The template asks for death frequency estimates, with treatment priming "highly publicized incidents" and "dramatic stories" — directly operationalizing the availability heuristic.

**Expected answers:** DEFENSIBLE. The cause pairs (heart disease vs. shark attacks, diabetes vs. plane crashes, etc.) are grounded in real CDC/NTSB data. The rational answers are approximate actual death counts; biased answers are inflated by a plausible factor (5-8x). This matches the classic Slovic, Fischhoff & Lichtenstein (1982) finding that dramatic causes are overestimated.

**Control/treatment isolation:** GOOD. Control asks for a frequency estimate with a context reference to common causes. Treatment adds vivid media framing. The target quantity is identical.

**Confounds:** LOW. The media priming is a clean manipulation. One minor concern: LLMs may have internalized actual statistics, reducing susceptibility to availability manipulation.

**LLM transparency:** MODERATE. The "highly publicized incidents" framing is somewhat obvious, but the core task (frequency estimation) is naturalistic.

**Intensity variation:** GOOD. Uses escalating preambles from "passing mention" (WEAK) to "someone close to you was personally affected" (ADVERSARIAL). This captures the availability research on personal relevance increasing availability.

**Note:** Metadata correctly marks `scoring_method: "relative"` — comparing treatment vs. control shift rather than absolute values. This is methodologically sound since the baseline frequency knowledge of different LLMs varies.

**Verdict: VALID**

---

### 3. Base Rate Neglect

**Citation:** Kahneman & Tversky (1973), "On the Psychology of Prediction"

**Paradigm fidelity:** HIGH. Faithful reproduction of the engineer/lawyer problem. Population base rates are stated (e.g., 5% engineers, 95% lawyers), then a representative description is provided. The template tests whether the model ignores base rates in favor of description fit.

**Expected answers:** DEFENSIBLE. With base rates of 5-20% for engineers, the rational answer is B (lawyer — the majority category), while the biased answer is A (engineer — matching the description "analytical, enjoys puzzles, somewhat introverted"). The description deliberately matches the engineer stereotype, creating the representativeness/base-rate tension from the original paradigm.

**Control/treatment isolation:** GOOD. Control presents base rates with NO description (pure base-rate question). Treatment adds the personality description. This cleanly isolates the effect of representative information.

**Confounds:** LOW. The description is a single canonical personality profile. One note: the description "analytical, enjoys puzzles, somewhat introverted" is somewhat mild; the original K&T descriptions were more detailed.

**LLM transparency:** MODERATE. LLMs trained on psychology texts may recognize the engineer/lawyer problem format. However, the variable base rates (5-20%) provide some novelty.

**Intensity variation:** GOOD. Preambles escalate from "brief impression" (WEAK) to "multiple independent sources" with colleague confirmation (ADVERSARIAL), increasing the representativeness pressure.

**Verdict: VALID**

---

### 4. Conjunction Fallacy

**Citation:** Tversky & Kahneman (1983), "Extensional Versus Intuitive Reasoning: The Conjunction Fallacy in Probability Judgment"

**Paradigm fidelity:** HIGH. Direct implementation of the Linda Problem. The template provides a personality description, then asks whether P(A) or P(A ∧ B) is more probable. Three scenario variants (Linda, Tom, Sarah) provide some protection against training data contamination.

**Expected answers:** CORRECT. A is always the rational answer since P(A) ≥ P(A ∧ B) is a mathematical certainty. The biased answer B requires judging the conjunction as more probable — the hallmark of the conjunction fallacy.

**Control/treatment isolation:** GOOD. Control presents the description and asks for the probability judgment. Treatment adds the explicit representativeness nudge: "This description strongly fits someone who is {specific_detail}." This increases the salience of the representative conjunction.

**Confounds:** LOW. The additional representativeness hint in treatment is a clean manipulation. Both conditions present identical options.

**LLM transparency:** MODERATE. The Linda Problem is extremely well-known. However, the variant scenarios (Tom, Sarah) provide some novelty.

**Intensity variation:** GOOD. Preambles escalate from "brief description" (WEAK) to "unmistakable picture... everyone agrees" (ADVERSARIAL), increasing the representativeness pressure.

**Concern:** LLMs are likely aware that this is a test of the conjunction fallacy, especially with the Linda variant. This may produce ceiling effects (always correct) rather than genuine bias measurement.

**Verdict: VALID** — but note contamination risk with classic Linda variant.

---

### 5. Gain/Loss Framing

**Citation:** Tversky & Kahneman (1981), "The Framing of Decisions and the Psychology of Choice" (Asian Disease Problem)

**Paradigm fidelity:** HIGH. This is arguably the best-implemented bias in the benchmark. It faithfully replicates the Asian Disease Problem structure with the critical innovation: **separate gain and loss frame templates**. The control uses a neutral EV-based frame. The WEAK/MODERATE intensities use the gain frame; STRONG/ADVERSARIAL use the loss frame.

**Expected answers:** DEFENSIBLE. The benchmark correctly identifies the K&T framing effect as a **reversal**:
- Gain frame: biased = A (risk-averse, prefer sure saving), rational = B (higher EV gamble)
- Loss frame: biased = B (risk-seeking gamble), rational = A (resist risk-seeking)

The EVs are constructed so B has slightly higher EV (35% probability × total > certain/3), making B the EV-maximizing choice in both frames.

**Control/treatment isolation:** EXCELLENT. The control presents both programs with explicit EVs in neutral language. Treatment frames the same choices using "saved" (gain) or "will die" (loss) language. This is the canonical framing manipulation.

**Confounds:** LOW. The only variable is the linguistic frame. Same numbers, same structure.

**LLM transparency:** LOW. The disease context is naturalistic and the framing is embedded in the problem description rather than explicitly signaled.

**Intensity variation:** EXCELLENT. Using gain frame for WEAK/MODERATE and loss frame for STRONG/ADVERSARIAL is a thoughtful mapping. The metadata records `frame_map` and per-frame rational/biased targets, enabling the evaluator to score each trial against the correct frame. This is sophisticated and methodologically sound.

**Verdict: VALID** — the strongest implementation in the benchmark.

---

### 6. Loss Aversion

**Citation:** Kahneman & Tversky (1979), Prospect Theory; Kahneman, Knetsch & Thaler (1990)

**Paradigm fidelity:** MODERATE. The classic loss aversion paradigm presents a mixed gamble (50/50 win X / lose Y) where the expected value is positive, testing whether loss aversion causes rejection. The template does this correctly. However, the treatment manipulation is relatively weak: it adds "Consider how you would feel about each outcome" and "money you already have that would be gone" — this is more of a framing nudge than a structural manipulation.

**Expected answers:** DEFENSIBLE. The gamble always has positive EV by construction (win amount = lose + $30-80). Rational = Accept. Biased = Reject due to loss aversion (K&T's λ ≈ 2 means losses are weighted ~2x).

**Control/treatment isolation:** MODERATE. Both conditions present the same gamble with expected value. The treatment adds emotional framing about loss. This is a valid manipulation but the EV is explicitly stated in the control, which may anchor rational behavior.

**Confounds:** MODERATE. The treatment prompt says "Consider how you would feel" — this introduces affect as a confound beyond pure loss aversion. Also, the explicit "Expected value: ${ev}" in the control is a strong debiasing cue that may produce ceiling effects.

**LLM transparency:** MODERATE. The "feel" language is somewhat transparent. However, the gamble format is naturalistic.

**Intensity variation:** MODERATE. STRONG/ADVERSARIAL increase the loss amount (1.3x, 1.5x) which changes the EV, potentially making rejection rational. Preambles escalate from "opportunity" (WEAK) to "hard-earned savings...deeply regretted" (ADVERSARIAL). The numeric adjustment is concerning because it changes the rational answer: if lose_amount increases enough, EV becomes negative and rejection IS rational.

**Issue:** At STRONG intensity (lose × 1.3), the EV calculation needs verification. If lose = $120 (max), loss becomes $156, win stays at $150-$200. EV could become negative ($150-$156)/2 = -$3, making rejection rational. This is a potential confound.

**Verdict: QUESTIONABLE** — the EV-explicit control may produce ceiling effects; intensity adjustments may change the rational answer; treatment manipulation adds affect as confound.

---

### 7. Endowment Effect

**Citation:** Thaler (1980); Kahneman, Knetsch & Thaler (1990), "Experimental Tests of the Endowment Effect"

**Paradigm fidelity:** MODERATE. The classic endowment effect paradigm shows that owners demand more for objects than buyers are willing to pay (WTA > WTP). The template approximates this: control asks "Is a market-value offer fair?" while treatment asks "Should you accept a market-value offer for YOUR item?"

However, the original KKT paradigm uses a between-subjects design comparing WTP vs. WTA for the same object. This template uses the same subject asked about fairness in two different framings, which is more of a within-subject contrast.

**Expected answers:** DEFENSIBLE. Rational = A (accept market value). Biased = B (demand more due to ownership). This captures the WTA > WTP asymmetry.

**Control/treatment isolation:** GOOD. Control is a detached fairness judgment. Treatment introduces ownership ("you've had for a while," "part of your possessions").

**Confounds:** LOW. The ownership framing is the only manipulation.

**LLM transparency:** MODERATE-HIGH. The treatment option B is labeled "No, demand more than market value because it's mine" — this is very transparent about the endowment effect. A more subtle treatment would ask for a minimum selling price rather than presenting the bias as a labeled option.

**Intensity variation:** GOOD. Preambles escalate from "recently acquired" (WEAK) to "gift from someone important, emotional attachment" (ADVERSARIAL).

**Concern:** Option B in the treatment explicitly names the bias mechanism ("because it's mine"). A frontier LLM will recognize this as irrational and almost always choose A. This may produce ceiling effects.

**Verdict: QUESTIONABLE** — transparent option labeling likely prevents genuine bias measurement in LLMs.

---

### 8. Status Quo Bias

**Citation:** Samuelson & Zeckhauser (1988), "Status Quo Bias in Decision Making"

**Paradigm fidelity:** MODERATE. The paradigm tests preference for the current option even when a superior alternative is available. The template does this: control presents two options neutrally, treatment designates Option A as the current choice with "Option B has objectively better features at the same cost."

**Expected answers:** DEFENSIBLE. Rational = B (switch to objectively better option). Biased = A (stay with status quo).

**Control/treatment isolation:** GOOD. The only difference is ownership framing and the explicit statement that B is better.

**Confounds:** LOW.

**LLM transparency:** HIGH. The treatment explicitly states "Option B has objectively better features at the same cost. Switching is free." This makes the rational answer so obvious that it's nearly impossible for an LLM to choose A unless deliberately role-playing irrationality. The original Samuelson & Zeckhauser paradigm uses subtler manipulations where the status quo advantage is not explicitly stated.

**Intensity variation:** GOOD. Preambles escalate from "short while" (WEAK) to "decade, entire professional identity" (ADVERSARIAL).

**Concern:** The treatment text stating Option B is "objectively better" eliminates any genuine decision tension. A more valid design would present options with ambiguous trade-offs where the status quo merely has the inertia advantage.

**Verdict: QUESTIONABLE** — the explicit "objectively better" language makes the rational answer too obvious for frontier LLMs, likely producing ceiling effects rather than measuring actual status quo bias.

---

### 9. Certainty Effect

**Citation:** Kahneman & Tversky (1979), Prospect Theory — Allais-type problems

**Paradigm fidelity:** HIGH. The template replicates the Allais paradox structure: a certain win vs. a probabilistic win with higher EV. Control presents both options with explicit probabilities and EVs. Treatment emphasizes certainty with "CERTAIN (guaranteed, 100% sure, no risk)."

**Expected answers:** CORRECT. By construction, EV(B) > EV(A) since gamble_amount = 1.6× certain and probability ≥ 75%, giving minimum EV ratio of 1.2×. Rational = B (higher EV). Biased = A (certainty preference).

**Control/treatment isolation:** GOOD. Control shows both options with probability labels and EVs (decision aid). Treatment presents certainty with emotional emphasis and frames B as "otherwise win nothing."

**Confounds:** LOW. The manipulation is clean: same mathematical structure, different framing emphasis.

**LLM transparency:** MODERATE. The all-caps "CERTAIN" and "no risk" language is somewhat heavy-handed, but the decision structure is naturalistic.

**Intensity variation:** GOOD. Preambles escalate from "Consider these options" (WEAK) to "ABSOLUTELY CERTAIN... many people have walked away with nothing" (ADVERSARIAL).

**Verdict: VALID**

---

### 10. Overconfidence Effect

**Citation:** Lichtenstein, Fischhoff & Phillips (1982); within K&T tradition via calibration research

**Paradigm fidelity:** MODERATE. The classic overconfidence paradigm presents general knowledge questions and measures calibration (actual accuracy vs. stated confidence). The template asks a question and requests a confidence rating. However, the treatment manipulation is simply "You have strong expertise in this area" — a role-playing instruction rather than a structural manipulation.

**Expected answers:** DEFENSIBLE. The bias is measured by the gap between stated confidence and actual accuracy, not by the answer itself. However, the variables don't encode this properly — `rational_answer` and `biased_answer` aren't well-defined for this bias because the bias manifests in the confidence score, not the answer.

**Control/treatment isolation:** MODERATE. The only difference is the expertise assertion. This is a valid manipulation (expertise increases overconfidence) but it's a relatively blunt instrument.

**Confounds:** MODERATE. The expertise framing may cause the LLM to genuinely adopt a different knowledge level, confounding overconfidence with genuine expertise effects.

**LLM transparency:** LOW. Asking for a confidence rating is naturalistic.

**Intensity variation:** GOOD. Preambles escalate from "some familiarity" (WEAK) to "THE foremost authority... reputation depends on confident answer" (ADVERSARIAL).

**Concern:** The `{question}` placeholder is not defined in `_get_template_variables` — there's no variable generation logic for the actual question content. This means the question depends entirely on what the domain scenario provides, which may not be well-calibrated for measuring overconfidence.

**Verdict: QUESTIONABLE** — the question content generation is underspecified, and the bias is measured in confidence rather than answer choice, which isn't well-supported by the current variable structure.

---

### 11. Confirmation Bias

**Citation:** Wason (1960), "On the Failure to Eliminate Hypotheses"; broader K&T tradition

**Paradigm fidelity:** MODERATE. The classic Wason paradigm uses hypothesis testing with card-selection tasks. The template simplifies this to a direct question about investigation strategy: "Which approach would you take?" with options for seeking confirming, disconfirming, or both types of evidence.

**Expected answers:** CORRECT. Rational = C (seek both types equally, corresponding to scientific method). Biased = A (focus on confirming evidence). The treatment adds "You believe that..." and "some initial evidence that supports your belief" to create commitment.

**Control/treatment isolation:** GOOD. Control is a neutral investigation. Treatment introduces prior belief and supporting evidence.

**Confounds:** LOW.

**LLM transparency:** HIGH. The options are essentially labeled descriptions of confirmation bias (A), disconfirmation (B), and balanced investigation (C). Frontier LLMs trained on reasoning tasks will almost universally choose C, as it's the textbook correct answer. The question format makes the rational answer extremely obvious.

**Intensity variation:** GOOD. Preambles escalate from "tentative hypothesis" (WEAK) to "career advancement depends on it" (ADVERSARIAL).

**Concern:** This tests whether an LLM will explicitly endorse a biased investigation strategy, not whether it would actually exhibit confirmation bias in practice. A more valid design would present evidence and measure whether the model selectively weighs supporting evidence more heavily.

**Verdict: QUESTIONABLE** — tests self-reported investigation preference rather than actual confirmation bias behavior. Likely ceiling effects with frontier LLMs.

---

### 12. Sunk Cost Fallacy

**Citation:** Arkes & Blumer (1985), "The Psychology of Sunk Costs"; within prospect theory tradition

**Paradigm fidelity:** HIGH. The template faithfully implements the sunk cost paradigm. Control presents only future costs and returns (no sunk cost information). Treatment adds a large sunk cost and frames the choice as "recover the sunk costs" vs. "cut your losses."

**Expected answers:** CORRECT. Returns < future cost by construction (returns = 0.8× future cost), so rational = B (abandon). Biased = A (continue to recover sunk costs).

**Control/treatment isolation:** GOOD. The control omits sunk cost information entirely. The treatment adds it, along with emotionally charged language ("significant investment already made," "recover the sunk costs," "cut your losses").

**Confounds:** LOW. The treatment option A is labeled "Continue the project to recover the sunk costs" — this makes the sunk cost reasoning somewhat transparent, but it's faithful to the paradigm.

**LLM transparency:** MODERATE. The "recover the sunk costs" language somewhat telegraphs the bias. However, the emotional framing ("given the significant investment already made") creates genuine decision tension.

**Intensity variation:** GOOD. Sunk cost amounts are multiplied by intensity (0.5x to 2.0x), and preambles escalate from "business decision" (WEAK) to "best years of your career, reputation on the line" (ADVERSARIAL). Both numeric and narrative manipulation.

**Verdict: VALID**

---

### 13. Present Bias

**Citation:** Laibson (1997), "Golden Eggs and Hyperbolic Discounting"; O'Donoghue & Rabin (1999)

**Paradigm fidelity:** HIGH. The template presents the classic intertemporal choice: $100 now vs. $X in Y days. The design is particularly well-done: it uses explicit temporal discounting to ground the rational choice. The future amount is calculated so that its present value exceeds $100 at a 10% annual discount rate (with a 20-40% premium), making B unambiguously rational.

**Expected answers:** CORRECT. PV(B) > A by construction. Rational = B (wait for the clearly-better-in-PV-terms future payment). Biased = A (present bias, preferring immediate gratification).

**Control/treatment isolation:** GOOD. Control provides the PV calculation and states the rational decision rule. Treatment removes the PV information and adds immediacy framing ("money today is available right now").

**Confounds:** LOW. The control provides a strong normative anchor (PV calculation), while the treatment removes it. This is a clean manipulation.

**LLM transparency:** LOW. The intertemporal choice format is naturalistic and doesn't telegraph "present bias test."

**Intensity variation:** GOOD. Preambles escalate from "Consider the following choice" (WEAK) to "Imagine holding the cash in your hand RIGHT NOW" (ADVERSARIAL).

**Verdict: VALID** — one of the stronger implementations.

---

### 14. Hindsight Bias

**Citation:** Fischhoff (1975), "Hindsight ≠ Foresight"; within K&T tradition

**Paradigm fidelity:** HIGH. The classic paradigm asks people to assess predictability of an event either before or after knowing the outcome. The template does this: control asks about predictability of a future product launch (analysts divided). Treatment reveals the outcome and asks about pre-launch predictability.

**Expected answers:** DEFENSIBLE. Rational = A (unpredictable, given that analysts were divided). Biased = B (predictable, hindsight makes the outcome seem inevitable).

**Control/treatment isolation:** GOOD. The only difference is outcome knowledge. Both conditions describe the same initial uncertainty.

**Confounds:** LOW. Clean outcome-knowledge manipulation.

**LLM transparency:** MODERATE. The "looking back" framing and "BEFORE it happened" emphasis somewhat signal the hindsight bias construct. However, the judgment task is naturalistic.

**Intensity variation:** GOOD. Preambles escalate from "Reflect on the following situation" (WEAK) to "virtually every expert now agrees the outcome was obvious" (ADVERSARIAL), increasing the hindsight pressure.

**Verdict: VALID**

---

### 15. Gambler's Fallacy

**Citation:** Tversky & Kahneman (1971), "Belief in the Law of Small Numbers"

**Paradigm fidelity:** HIGH. Faithful implementation of the classic paradigm. A fair coin has produced a streak of heads; the question asks about the next flip. Control asks about probability with no streak information. Treatment reveals the streak.

**Expected answers:** CORRECT. C (equally likely) is the uniquely rational answer — fair coin flips are independent events. The biased answer B (tails is more likely) is the classic gambler's fallacy.

**Control/treatment isolation:** EXCELLENT. Control presents a simple probability question about a fair coin with no prior history. Treatment adds the streak information. This is the cleanest possible isolation.

**Confounds:** NONE. The only variable is the streak information. The coin is explicitly stated to be fair.

**LLM transparency:** LOW. The three-option format (heads more likely / tails more likely / equally likely) is straightforward and doesn't signal the bias being tested.

**Intensity variation:** GOOD. Streak length scales with intensity (base × 0.5 to base × 2.0), and preambles escalate from "Consider this probability question" (WEAK) to "everyone is convinced the pattern MUST break" (ADVERSARIAL). The streak length adjustment is particularly well-grounded: longer streaks increase the subjective pressure to expect a reversal.

**Concern:** Most frontier LLMs will immediately recognize that fair coin flips are independent and answer C. This bias may show ceiling effects in rational responses. However, the ADVERSARIAL preamble with "statistics say this streak is almost impossibly unlikely to continue" provides genuinely misleading pressure.

**Verdict: VALID**

---

## Cross-Cutting Findings

### Strengths

1. **Mathematically grounded expected answers.** Biases like certainty_effect, present_bias, sunk_cost_fallacy, and gain_loss_framing construct the rational answer mathematically (EV comparisons, PV calculations), making the ground truth unambiguous.

2. **Gain/loss framing implementation is excellent.** The frame_map metadata, per-frame rational/biased targets, and the WEAK→gain / STRONG→loss intensity mapping is sophisticated and faithful to the K&T (1981) paradigm.

3. **Intensity preambles are well-differentiated.** Each CORE bias has WEAK, STRONG, and ADVERSARIAL preambles that escalate naturally (personal relevance, social pressure, authority pressure). This is a meaningful operationalization of trigger strength.

4. **Relative scoring for numeric biases.** Anchoring and availability correctly use `scoring_method: "relative"` to compare treatment-vs-control shift rather than absolute accuracy. This is methodologically appropriate.

5. **Domain-grounded scenarios.** The DOMAIN_SCENARIOS provide ecological validity across INDIVIDUAL, PROFESSIONAL, SOCIAL, TEMPORAL, and RISK contexts.

### Concerns

1. **Transparency/ceiling effects.** Several biases (status_quo_bias, endowment_effect, confirmation_bias) present options that explicitly label the biased choice, making it trivially easy for frontier LLMs to choose the rational option. This will likely produce very low bias scores for these biases, not because the LLMs lack the bias, but because the test is too transparent.

2. **Contamination risk.** Classic scenarios (Linda Problem, Asian Disease Problem, engineer/lawyer) are extensively represented in LLM training data. While the variants help, dedicated contamination testing (via the NovelScenarioGenerator) is advisable for the CORE tier.

3. **Loss aversion intensity confound.** The `_adjust_for_intensity` multiplier for loss_aversion increases the loss amount at STRONG/ADVERSARIAL intensity, which may change the rational answer from Accept to Reject (if the increased loss amount makes EV negative).

4. **Overconfidence question underspecification.** The overconfidence template's `{question}` placeholder has no dedicated generation logic in `_get_template_variables`, making it dependent on domain scenarios that may not be well-calibrated for calibration measurement.

5. **Confirmatory vs. behavioral measurement.** Some biases (confirmation_bias, endowment_effect) test whether the LLM will explicitly endorse a biased strategy rather than testing whether it exhibits the bias in behavior. These are conceptually different measurements.

### Recommendations for Publication

1. **High priority:** Fix the loss_aversion intensity adjustment to ensure EV remains positive at all intensity levels, or document the intended behavior.

2. **High priority:** For endowment_effect and status_quo_bias, consider reformulating options to be less transparent. Instead of "demand more because it's mine," ask for a minimum acceptable price.

3. **Medium priority:** Add dedicated question generation logic for overconfidence_effect or replace with calibration-style questions (trivia questions with known difficulty).

4. **Medium priority:** For confirmation_bias, consider supplementing with a behavioral test (present evidence and measure whether the model weighs confirming evidence more heavily).

5. **Low priority:** For contamination-sensitive biases (conjunction_fallacy Linda variant), consider using only novel scenarios from NovelScenarioGenerator for the CORE tier.

---

## Methodology Notes

This review evaluated each bias against seven criteria:
1. **Template fidelity to paradigm** — Does the template faithfully operationalize the cited research paradigm?
2. **Expected answer defensibility** — Are rational/biased answers well-grounded?
3. **Control/treatment isolation** — Does the contrast cleanly isolate the target bias?
4. **Confounding variables** — Are there extraneous influences?
5. **LLM transparency** — Would frontier LLMs see through the test?
6. **Framing measurement** (gain_loss_framing specific) — Is the reversal properly measured?
7. **Intensity variation** — Are the four levels meaningfully differentiated?

Ratings follow a three-point scale:
- **VALID:** Template faithfully operationalizes the paradigm with defensible expected answers and adequate isolation. Suitable for publication.
- **QUESTIONABLE:** Template has identifiable issues (transparency, confounds, underspecification) that may affect measurement validity. Should be addressed or acknowledged.
- **INVALID:** Template has fundamental design flaws that prevent valid measurement. Not suitable for publication without major revision.
