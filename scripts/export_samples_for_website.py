#!/usr/bin/env python3
"""Export sample questions from core_tests.json for the website explorer.

Reads generated test cases and picks the best representative sample per bias,
outputting JSON matching the website's SampleQuestion TypeScript type.
"""

import json
import hashlib
from pathlib import Path


def make_id(bias_id: str, domain: str, scenario: str) -> str:
    """Generate a stable unique ID for a sample."""
    raw = f"{bias_id}_{domain}_{scenario}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# Bias ID -> category mapping (must match website/src/lib/data/biases.ts)
BIAS_CATEGORIES = {
    "anchoring_effect": "anchoring",
    "availability_bias": "availability",
    "base_rate_neglect": "representativeness",
    "certainty_effect": "probability_distortion",
    "confirmation_bias": "confirmation",
    "conjunction_fallacy": "representativeness",
    "endowment_effect": "loss_aversion",
    "gain_loss_framing": "framing",
    "gambler_fallacy": "representativeness",
    "hindsight_bias": "memory_bias",
    "loss_aversion": "loss_aversion",
    "overconfidence_effect": "overconfidence",
    "present_bias": "temporal_bias",
    "status_quo_bias": "framing",
    "sunk_cost_fallacy": "loss_aversion",
}

# Human-readable names
BIAS_NAMES = {
    "anchoring_effect": "Anchoring Effect",
    "availability_bias": "Availability Bias",
    "base_rate_neglect": "Base Rate Neglect",
    "certainty_effect": "Certainty Effect",
    "confirmation_bias": "Confirmation Bias",
    "conjunction_fallacy": "Conjunction Fallacy",
    "endowment_effect": "Endowment Effect",
    "gain_loss_framing": "Gain/Loss Framing",
    "gambler_fallacy": "Gambler's Fallacy",
    "hindsight_bias": "Hindsight Bias",
    "loss_aversion": "Loss Aversion",
    "overconfidence_effect": "Overconfidence Effect",
    "present_bias": "Present Bias",
    "status_quo_bias": "Status Quo Bias",
    "sunk_cost_fallacy": "Sunk Cost Fallacy",
}


def pick_best_samples(test_cases: list[dict], per_bias: int = 2) -> list[dict]:
    """Pick the most diverse samples per bias (different scenarios)."""
    by_bias: dict[str, list[dict]] = {}
    for tc in test_cases:
        bid = tc["bias_id"]
        by_bias.setdefault(bid, []).append(tc)

    selected = []
    for bias_id, cases in by_bias.items():
        # Pick cases with the most distinct base_scenario + control_prompt combos
        seen_scenarios = set()
        for case in cases:
            scenario_key = case["metadata"].get("scenario_context", case["base_scenario"])
            if scenario_key not in seen_scenarios and len(seen_scenarios) < per_bias:
                seen_scenarios.add(scenario_key)
                selected.append(case)

    return selected


def convert_to_sample(tc: dict) -> dict:
    """Convert a Python test case to a website SampleQuestion shape."""
    bias_id = tc["bias_id"]
    return {
        "id": make_id(bias_id, tc["domain"], tc["base_scenario"]),
        "biasId": bias_id,
        "biasName": BIAS_NAMES.get(bias_id, tc["metadata"].get("bias_name", bias_id)),
        "category": BIAS_CATEGORIES.get(bias_id, "representativeness"),
        "domain": tc["domain"],
        "scale": tc.get("scale", "micro"),
        "baseScenario": tc["base_scenario"],
        "controlPrompt": tc["control_prompt"].strip(),
        "treatmentPrompts": {
            k: v.strip() for k, v in tc["treatment_prompts"].items()
        },
        "expectedRationalResponse": tc["expected_rational_response"],
        "expectedBiasedResponse": tc["expected_biased_response"],
        "debiasingPrompts": [p.strip() for p in tc.get("debiasing_prompts", [])],
        "explanation": (
            f"This tests {BIAS_NAMES.get(bias_id, bias_id)} in the context of "
            f"{tc['base_scenario']}. {tc.get('bias_trigger', '')}"
        ),
        "metadata": {
            "biasName": tc["metadata"].get("bias_name", ""),
            "theoreticalBasis": tc["metadata"].get("theoretical_basis", ""),
            "scenarioContext": tc["metadata"].get("scenario_context", tc["base_scenario"]),
        },
    }


def main():
    repo_root = Path(__file__).parent.parent
    input_path = repo_root / "core_tests.json"
    output_path = repo_root.parent / "website" / "src" / "lib" / "data" / "samples_generated.json"

    with open(input_path) as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases")

    # Pick 2 best per bias = ~30 samples
    selected = pick_best_samples(test_cases, per_bias=2)
    print(f"Selected {len(selected)} representative samples")

    samples = [convert_to_sample(tc) for tc in selected]

    # Sort by category then bias name for consistent ordering
    samples.sort(key=lambda s: (s["category"], s["biasName"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Written to {output_path}")
    print(f"Biases covered: {len(set(s['biasId'] for s in samples))}")


if __name__ == "__main__":
    main()
