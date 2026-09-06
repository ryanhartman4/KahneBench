#!/usr/bin/env python3
"""Check the README results tables against the tracked fingerprint files.

Every number in the two results tables in README.md must come from a file
in results/. This script re-derives each one and exits non-zero on any
mismatch, so CI fails if the README and the data drift apart.

Usage:
    uv run python scripts/verify_readme_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kahne_bench.metrics.core import HUMAN_BASELINES

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "results"

# README display name -> fingerprint file, in the column order of the
# per-bias table (left to right).
MODELS: dict[str, str] = {
    "Claude Opus 4.7": "fingerprint_opus47.json",
    "Claude Opus 4.8": "fingerprint_opus48.json",
    "Claude Fable 5": "fingerprint_fable5.json",
    "Claude Opus 4.6": "fingerprint_opus46.json",
    "GPT-5.5": "fingerprint_gpt55.json",
    "Claude Sonnet 4.6": "fingerprint_sonnet46.json",
    "GPT-5.4": "fingerprint_gpt54.json",
    "GPT-5.2": "fingerprint_gpt52.json",
    "Claude Sonnet 4.5": "fingerprint_sonnet45.json",
    "Grok 4.1 Fast": "fingerprint_grok.json",
    "Claude Haiku 4.5": "fingerprint_haiku45.json",
}

# Short column labels used by the per-bias table header, same order as MODELS.
BIAS_TABLE_COLUMNS = [
    "Opus 4.7",
    "Opus 4.8",
    "Fable 5",
    "Opus 4.6",
    "GPT-5.5",
    "Sonnet 4.6",
    "GPT-5.4",
    "GPT-5.2",
    "Sonnet 4.5",
    "Grok 4.1",
    "Haiku 4.5",
]

# README bias row label -> bias_id in the fingerprint.
BIASES: dict[str, str] = {
    "Base Rate Neglect": "base_rate_neglect",
    "Endowment Effect": "endowment_effect",
    "Gain-Loss Framing": "gain_loss_framing",
    "Status Quo Bias": "status_quo_bias",
    "Certainty Effect": "certainty_effect",
    "Hindsight Bias": "hindsight_bias",
    "Sunk Cost Fallacy": "sunk_cost_fallacy",
    "Overconfidence": "overconfidence_effect",
    "Present Bias": "present_bias",
    "Loss Aversion": "loss_aversion",
    "Availability Bias": "availability_bias",
    "Confirmation Bias": "confirmation_bias",
    "Anchoring Effect": "anchoring_effect",
    "Gambler's Fallacy": "gambler_fallacy",
    "Conjunction Fallacy": "conjunction_fallacy",
}
BIAS_LABELS = {bias_id: label for label, bias_id in BIASES.items()}


def markdown_table(heading: str) -> tuple[list[str], list[list[str]]]:
    """Return (header cells, body rows) of the first table under a README heading."""
    lines = README.read_text().splitlines()
    if heading not in lines:
        sys.exit(f"README heading not found: {heading!r}")
    rows: list[list[str]] = []
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("#"):
            break
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue  # separator row
        rows.append(cells)
    if not rows:
        sys.exit(f"no table found under {heading!r}")
    return rows[0], rows[1:]


def number(cell: str) -> float:
    return float(cell.strip("*").rstrip("%"))


def check_leaderboard(fingerprints: dict[str, dict]) -> list[str]:
    problems: list[str] = []
    _, rows = markdown_table("### Overall Bias Susceptibility")
    seen: set[str] = set()
    susceptibilities: list[float] = []
    for cells in rows:
        if len(cells) != 5:
            problems.append(f"leaderboard: expected 5 cells, found {len(cells)}: {cells}")
            continue
        model, _provider, pct, top_vulnerability, top_bms = cells
        if model not in fingerprints:
            problems.append(f"leaderboard: no fingerprint mapping for {model!r}")
            continue
        seen.add(model)
        fp = fingerprints[model]
        want = number(pct)
        got = fp["summary"]["overall_bias_susceptibility"] * 100
        if abs(got - want) > 0.005:
            problems.append(f"{model}: susceptibility README {want:.2f}% vs fingerprint {got:.2f}%")
        scores = {
            bias_id: fp["magnitude_scores"][bias_id]["overall_magnitude"]
            for bias_id in fp["biases_tested"]
        }
        best_score = max(scores.values())
        if abs(best_score - number(top_bms)) > 0.0005:
            problems.append(f"{model}: top BMS README {top_bms} vs fingerprint {best_score:.3f}")
        # Any bias within rounding distance of the maximum is an acceptable label.
        top_labels = {
            BIAS_LABELS[bias_id]
            for bias_id, score in scores.items()
            if abs(score - best_score) <= 0.0005
        }
        if top_vulnerability not in top_labels:
            problems.append(
                f"{model}: top vulnerability README {top_vulnerability!r} "
                f"vs fingerprint {sorted(top_labels)}"
            )
        susceptibilities.append(want)
    missing = set(fingerprints) - seen
    if missing:
        problems.append(f"leaderboard: models missing from README table: {sorted(missing)}")
    if susceptibilities != sorted(susceptibilities):
        problems.append("leaderboard: rows are not sorted by ascending susceptibility")
    return problems


def check_bias_table(fingerprints: dict[str, dict]) -> list[str]:
    problems: list[str] = []
    header, rows = markdown_table("### Per-Bias BMS Scores Across Models")
    expected_header = ["Bias", *BIAS_TABLE_COLUMNS, "Human"]
    if header != expected_header:
        return [f"bias table header is {header}, expected {expected_header}"]
    models = list(MODELS)
    for cells in rows:
        label, *values = cells
        bias_id = BIASES.get(label)
        if bias_id is None:
            problems.append(f"bias table: unknown bias label {label!r}")
            continue
        if len(values) != len(models) + 1:
            problems.append(f"{label}: expected {len(models) + 1} values, found {len(values)}")
            continue
        for model, cell in zip(models, values):
            got = fingerprints[model]["magnitude_scores"][bias_id]["overall_magnitude"]
            if abs(got - number(cell)) > 0.0005:
                problems.append(f"{label} / {model}: README {cell} vs fingerprint {got:.3f}")
        human = HUMAN_BASELINES.get(bias_id)
        if human is None or abs(human - number(values[-1])) > 0.005:
            problems.append(f"{label}: human baseline README {values[-1]} vs code {human}")
    if len(rows) != len(BIASES):
        problems.append(f"bias table: expected {len(BIASES)} rows, found {len(rows)}")
    return problems


def load_fingerprints() -> tuple[dict[str, dict], list[str]]:
    fingerprints: dict[str, dict] = {}
    problems: list[str] = []
    for name, filename in MODELS.items():
        path = RESULTS / filename
        if not path.exists():
            problems.append(f"{name}: fingerprint file missing: {path.relative_to(ROOT)}")
            continue
        fingerprints[name] = json.loads(path.read_text())
    return fingerprints, problems


def main() -> int:
    fingerprints, problems = load_fingerprints()
    if not problems:
        problems = check_leaderboard(fingerprints) + check_bias_table(fingerprints)
    for problem in problems:
        print(f"MISMATCH: {problem}")
    print(f"Checked {len(MODELS)} models across both README results tables.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
