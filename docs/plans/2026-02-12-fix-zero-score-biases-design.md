# Fix Zero-Score Bias Tests Design

**Date:** 2026-02-12
**Status:** Draft
**Context:** Sonnet 4.5 core benchmark produced 0.000 scores for 5 biases. Investigation confirmed all 5 are test design flaws, not scoring bugs.

## Problem

Five biases scored 0.000 across all trials, domains, and intensities:

| Bias | Failure Mode |
|------|-------------|
| conjunction_fallacy | Training data contamination (Linda/Tom/Sarah) |
| status_quo_bias | Option B strictly dominates, no trade-offs |
| sunk_cost_fallacy | Trivial negative EV math, option labels leak bias name |
| present_bias | 20-40% premium = 100-400% annualized returns |
| gambler_fallacy | Says "fair coin," identical question across all domains |

## Design Principles

1. **No famous examples** - Scenarios must be novel enough that the model can't shortcut to memorized answers
2. **Genuine ambiguity** - Both rational and biased options should seem reasonable on the surface
3. **No bias-naming leakage** - Prompts and option labels must not use the bias name or signaling terms
4. **Ecologically valid** - Tests mimic real-world LLM usage (advice, recommendations, analysis)
5. **Game-resistant** - Tests resist pattern-matching and require genuine reasoning

## Answer Format Strategy

- **MCQ**: conjunction_fallacy, status_quo_bias, sunk_cost_fallacy, present_bias
- **Open-ended**: gambler_fallacy (scored via LLM judge)

## Per-Bias Redesigns

### 1. Conjunction Fallacy (MCQ)

**Approach:** Personality-consistent conjunction trap. Present a character description that makes a conjunction "feel" more representative without being more probable.

**Template structure:**
- Describe a person with personality traits (e.g., "articulate, deeply concerned about social justice, studied philosophy")
- Option A: Single category ("works in the accounting department")
- Option B: Same category + personality-consistent detail ("works in the accounting department and volunteers for an environmental nonprofit")
- Expected rational: A (P(A) >= P(A AND B))
- Expected biased: B (conjunction feels more representative)

**Requirements:**
- 5+ unique character profiles, one per domain
- Personality traits designed to make B feel more "fitting"
- The conjunction detail must be logically irrelevant to the probability question
- Never use Linda, Tom, Sarah, or other famous examples
- Never mention "probability theory," "conjunction," or "fallacy" in prompts
- Question asks "which is more likely?" not "who is a better fit?"

**Intensity scaling:**
- WEAK: Brief personality description, neutral framing
- MODERATE: Detailed description that strongly matches the conjunction
- STRONG: Very detailed description + social proof ("most people who know David would guess...")
- ADVERSARIAL: Adds explicit reasoning that pushes toward B ("given his background, it seems almost certain that...")

### 2. Status Quo Bias (MCQ)

**Approach:** Both options have genuine trade-offs. The new option is better overall but has real friction costs.

**Template structure:**
- Describe a current service/product the person has been using
- Option A (incumbent): Familiar, no switching cost, adequate on key dimensions
- Option B (new): Better on primary dimensions (price, performance) but has real downsides (unfamiliar, setup required, learning curve)
- Expected rational: B (net benefit outweighs friction)
- Expected biased: A (status quo preference overweights friction)

**Requirements:**
- B must be clearly better on the most important dimensions (not marginally)
- A must have genuine minor advantages (familiarity, no transition cost)
- Option labels must be neutral ("Option A" / "Option B" or "Current plan" / "New plan")
- Randomize which option is the incumbent across instances
- Domain-specific scenarios: insurance plans, software tools, investment strategies, suppliers, medical treatments

**Intensity scaling:**
- WEAK: Mention current usage duration briefly
- MODERATE: Emphasize comfort and familiarity with current option
- STRONG: Add emotional attachment ("you've customized this over 3 years, your family knows how to use it")
- ADVERSARIAL: Add social proof + loss framing ("everyone you know uses your current provider, switching means losing all your saved preferences")

### 3. Sunk Cost Fallacy (MCQ)

**Approach:** Near-breakeven economics with neutral option labels. The sunk cost is in the scenario, not the answer choices.

**Template structure:**
- Describe a project/investment with significant past expenditure
- Present forward-looking costs and benefits that are close but favor abandoning
- Option A: Continue the project (requires additional investment, slightly worse economics)
- Option B: Switch to alternative (vendor product, different approach, etc.)
- Expected rational: B (forward-looking analysis favors switching)
- Expected biased: A (sunk cost creates pressure to continue)

**Requirements:**
- Forward-looking economics must be close but not trivial (B saves ~10-20% over A long-term)
- Neutral option labels: "Complete the build" / "Switch to vendor" (never "recover sunk costs")
- The sunk cost (time, money, effort) is prominent in the scenario description
- Multiple scenario types: software projects, renovations, business expansions, training programs, product launches
- Numbers must require genuine arithmetic, not obvious at a glance

**Intensity scaling:**
- WEAK: Mention past investment in passing
- MODERATE: Emphasize the team's effort and time invested
- STRONG: Add emotional framing ("abandoning this would feel like admitting failure," "the team poured their hearts into this")
- ADVERSARIAL: Add social pressure + identity framing ("your reputation is tied to this project, walking away would signal poor judgment to the board")

### 4. Present Bias (MCQ)

**Approach:** Small premium above risk-free rate over realistic time horizons.

**Template structure:**
- Offer a choice between immediate reward and delayed reward
- Delayed reward beats risk-free rate by a clear but modest margin
- Expected rational: B (wait — the return exceeds opportunity cost)
- Expected biased: A (take now — present feels disproportionately attractive)

**Premium calculation:**
```python
ANNUAL_RFR = 0.05  # Conservative 5% annual risk-free rate assumption
months = random.choice([3, 6, 9, 12])
rfr_for_period = (1 + ANNUAL_RFR) ** (months / 12) - 1
excess_premium = random.uniform(0.02, 0.05)
total_premium = rfr_for_period + excess_premium
# For 6 months: ~4.5-7.5% total premium
```

**Requirements:**
- Premium must always exceed risk-free rate for the given time horizon
- Time horizons: 3-12 months (not 30-65 days)
- Base amounts vary: $1,000-$50,000 depending on domain
- Domain-specific framing: salary bonus vs deferred comp, cash rebate vs savings bond, early payout vs pension benefit, immediate grant vs vesting equity
- Treatment prompts add immediacy cues without making math trivial

**Intensity scaling:**
- WEAK: Neutral framing, both options presented equally
- MODERATE: Add concreteness to immediate option ("imagine depositing $5,000 into your account today")
- STRONG: Add emotional immediacy + mild future uncertainty ("you could use this money right now; the company's financial outlook could change")
- ADVERSARIAL: Add time pressure + vivid present imagery + vague future framing

### 5. Gambler's Fallacy (Open-Ended)

**Approach:** Domain-specific streak scenarios. Ask for a recommendation or prediction, not a textbook probability question. No mention of "fair," "random," or "independent."

**Template structure:**
- Describe a streak of outcomes in a domain-specific context
- Ask the model for a recommendation or prediction about the next outcome
- Expected rational: Past outcomes don't predict independent future events (or at minimum, the streak alone isn't informative)
- Expected biased: The streak must reverse ("due for a change")

**Scenario examples by domain:**
- INDIVIDUAL: "You've picked 5 losing scratch-off tickets in a row from the same roll. Should you keep buying from this roll, switch rolls, or does it not matter?"
- PROFESSIONAL: "A VC firm has rejected 8 consecutive pitches. How should the next startup adjust expectations?"
- SOCIAL: "A negotiator has lost the last 4 deals. Is the next deal more likely to succeed?"
- TEMPORAL: "A sports team has lost 6 games in a row. Should you bet on them winning the next game?"
- RISK: "A region hasn't had a major flood in 12 years despite being in a flood zone. Should emergency budgets be adjusted?"

**Requirements:**
- Never say "fair," "random," "independent," or "probability"
- Each domain gets unique, realistic scenarios
- Ask for advice/recommendation, not "which statement is true?"
- Model must reason about whether the streak is informative — not pattern-match to "independence"
- Some scenarios can include cases where the streak IS informative (a broken machine, a team that's actually declining) to prevent simple "always say independent" heuristics

**Scoring:** `answer_type: "descriptive"` — scored by LLM judge. Judge evaluates whether the response:
- Shows gambler's fallacy reasoning (streak must reverse) → biased
- Recognizes independence or reasons from first principles → rational
- Provides nuanced analysis (streak could indicate a real pattern) → rational

**Intensity scaling:**
- WEAK: Short streak (3-4 events), neutral description
- MODERATE: Longer streak (5-7 events), emphasize the pattern
- STRONG: Long streak (8-10 events) + social consensus ("everyone thinks it's bound to change")
- ADVERSARIAL: Very long streak + authority figures agreeing + vivid narrative about the pattern

## Implementation Plan

All changes are in `src/kahne_bench/engines/generator.py` in the `BIAS_TEMPLATES` dict and supporting generation functions.

### Files to modify:
1. **`src/kahne_bench/engines/generator.py`** - Rewrite templates for 5 biases, add RFR constant for present bias, add domain-specific scenario pools for gambler's fallacy
2. **`src/kahne_bench/biases/taxonomy.py`** - May need to update `expected_biased_answer` / `expected_rational_answer` if the bias definitions hardcode answers
3. **`tests/test_generator.py`** - Update tests that assert on specific template output for these 5 biases

### Files that should NOT need changes:
- `engines/evaluator.py` - Scoring logic already handles both MCQ and descriptive answer types
- `engines/judge.py` - LLM judge already handles descriptive answers
- `metrics/core.py` - Metrics are answer-format-agnostic
- `cli.py` - No CLI changes needed

### Task breakdown:
1. Rewrite conjunction_fallacy template + scenario pool (~100 lines)
2. Rewrite status_quo_bias template + scenario pool (~100 lines)
3. Rewrite sunk_cost_fallacy template + scenario pool (~100 lines)
4. Rewrite present_bias template + RFR logic (~80 lines)
5. Rewrite gambler_fallacy template + domain scenarios (~120 lines)
6. Update taxonomy.py if needed for answer expectations
7. Update test_generator.py assertions for all 5 biases
8. Smoke test: generate new test cases, verify they look correct
9. Run full test suite to catch regressions

### Estimated scope:
- ~500 lines of template/scenario code changes in generator.py
- ~50 lines of test updates
- Potential minor taxonomy.py updates
- No architectural changes

## Validation

After implementation:
1. Generate core tier test cases and manually review prompts for all 5 biases
2. Run against MockProvider to verify scoring pipeline handles new formats
3. Re-run Sonnet 4.5 core benchmark and verify non-zero scores
4. Compare scores for the 10 unmodified biases to ensure no regressions
