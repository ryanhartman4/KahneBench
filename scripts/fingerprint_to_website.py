#!/usr/bin/env python3
"""Transform a KahneBench fingerprint_<m>.json into the website's camelCase
LEADERBOARD + FINGERPRINTS entries (for src/lib/data/mock-results.ts).

The website's model data is HAND-MAINTAINED, not generated from the Python
fingerprints, and a naive snake_case->camelCase conversion produces valid-but-
WRONG optional keys that TypeScript will not flag (blank charts at runtime).
The known gotchas this tool encodes:

  * drop `unknown_rate` (and `mean_bias_score`, `rci_interpretation`,
    `insufficient_confidence_data`) everywhere
  * leaderboard aggregate metrics bci/bmp/hss/rci/cas = arithmetic MEAN over
    biases of the per-bias field; bms = overall_bias_susceptibility
  * the SAME python `consistency_score` maps to `overallConsistency` in
    consistencyIndices (BCI) but `consistencyScore` in responseConsistencies (RCI)
  * `calibration_score` -> `awarenessScore`
  * treatmentScores is padded with `adversarial: 0` (the core run only does
    weak/moderate/strong, but the TS Record<TriggerIntensity, number> needs all 4)

To guard against drift, EVERY run first reproduces an existing website entry
(default: claude-opus-4-8) byte-for-byte from its fingerprint. If that
validation fails, the tool refuses to emit anything.

Usage:
    # validate the transform only (no new model):
    uv run python scripts/fingerprint_to_website.py --validate-only

    # generate entries for a newly-benchmarked model:
    python scripts/fingerprint_to_website.py results/fingerprint_fable5.json \\
        --name "Claude Fable 5" --provider Anthropic

Paste the printed LEADERBOARD block (replacing the existing array) and the
FINGERPRINTS entry (anywhere inside the FINGERPRINTS object) into
website/src/lib/data/mock-results.ts, then run `npx tsc --noEmit && npx next
build` and visually check /results (leaderboard rank + radar color). Remember
to also add a MODEL_COLORS hex entry in bias-radar-chart.tsx and bump
STATS.modelCount — those are not emitted here.
"""

import argparse
import json
import re
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEBSITE = BENCH_ROOT.parent / "website" / "src" / "lib" / "data" / "mock-results.ts"
VALIDATION_MODEL_ID = "claude-opus-4-8"
VALIDATION_FINGERPRINT = BENCH_ROOT / "results" / "fingerprint_opus48.json"
VALIDATION_NAME = "Claude Opus 4.8"


def extract_const(src: str, name: str):
    """Find `export const NAME...= <[ or {>` and balance brackets, parse as JSON.

    mock-results.ts uses quoted keys and JSON-compatible values, so the literal
    parses as JSON once trailing commas are stripped.
    """
    i = src.index("export const " + name)
    eq = src.index("=", i)
    j = min(src.index(c, eq) for c in "[{" if c in src[eq : eq + 200])
    opener = src[j]
    close = {"[": "]", "{": "}"}[opener]
    depth = in_str = esc = 0
    for k in range(j, len(src)):
        c = src[k]
        if in_str:
            if esc:
                esc = 0
            elif c == "\\":
                esc = 1
            elif c == '"':
                in_str = 0
            continue
        if c == '"':
            in_str = 1
        elif c == opener:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                blob = re.sub(r",(\s*[}\]])", r"\1", src[j : k + 1])
                return json.loads(blob)
    raise ValueError(f"unbalanced brackets for {name}")


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals)


def transform(fp: dict, model_name: str, provider: str, rank, tier: str = "core"):
    """Return (leaderboard_entry, fingerprint_entry) in the website's shape."""
    ms, ci = fp["magnitude_scores"], fp["consistency_indices"]
    mp, ha = fp["mitigation_potentials"], fp["human_alignments"]
    rc, cs = fp["response_consistencies"], fp["calibration_scores"]
    s = fp["summary"]

    def mag(b, v):
        t = dict(v["treatment_scores"])
        t.setdefault("adversarial", 0)  # pad to the full TriggerIntensity Record
        return {
            "biasId": b,
            "controlScore": v["control_score"],
            "treatmentScores": t,
            "overallMagnitude": v["overall_magnitude"],
            "intensitySensitivity": v["intensity_sensitivity"],
        }

    def con(b, v):
        return {
            "biasId": b,
            "domainScores": v["domain_scores"],
            "overallConsistency": v["consistency_score"],  # GOTCHA: -> overallConsistency
            "standardDeviation": v["standard_deviation"],
            "isSystematic": v["is_systematic"],
        }

    def mit(b, v):
        return {
            "biasId": b,
            "baselineBiasScore": v["baseline_bias_score"],
            "debiasedScores": v["debiased_scores"],
            "bestMitigationMethod": v["best_mitigation_method"],
            "mitigationEffectiveness": v["mitigation_effectiveness"],
            "requiresExplicitWarning": v["requires_explicit_warning"],
        }

    def hum(b, v):
        return {
            "biasId": b,
            "modelBiasRate": v["model_bias_rate"],
            "humanBaselineRate": v["human_baseline_rate"],
            "alignmentScore": v["alignment_score"],
            "biasDirection": v["bias_direction"],
        }

    def res(b, v):
        return {
            "biasId": b,
            "meanResponse": v["mean_response"],
            "variance": v["variance"],
            "consistencyScore": v["consistency_score"],  # GOTCHA: -> consistencyScore (diff target)
            "isStable": v["is_stable"],
            "trialCount": v["trial_count"],
        }

    def cal(b, v):
        return {
            "biasId": b,
            "meanConfidence": v["mean_confidence"],
            "actualAccuracy": v["actual_accuracy"],
            "calibrationError": v["calibration_error"],
            "awarenessScore": v["calibration_score"],  # GOTCHA: -> awarenessScore
            "overconfident": v["overconfident"],
            "metacognitiveGap": v["metacognitive_gap"],
        }

    fingerprint = {
        "modelId": fp["model_id"],
        "modelName": model_name,
        "provider": provider,
        "evaluationDate": fp["generated_at"],
        "biasesTested": fp["biases_tested"],
        "tier": tier,
        "overallBiasSusceptibility": s["overall_bias_susceptibility"],
        "mostSusceptibleBiases": s["most_susceptible_biases"],
        "mostResistantBiases": s["most_resistant_biases"],
        "humanLikeBiases": s["human_like_biases"],
        "aiSpecificBiases": s["ai_specific_biases"],
        "magnitudeScores": {b: mag(b, ms[b]) for b in ms},
        "consistencyIndices": {b: con(b, ci[b]) for b in ci},
        "mitigationPotentials": {b: mit(b, mp[b]) for b in mp},
        "humanAlignments": {b: hum(b, ha[b]) for b in ha},
        "responseConsistencies": {b: res(b, rc[b]) for b in rc},
        "calibrationScores": {b: cal(b, cs[b]) for b in cs},
    }
    leaderboard = {
        "modelId": fp["model_id"],
        "modelName": model_name,
        "provider": provider,
        "overallScore": s["overall_bias_susceptibility"],
        "tier": tier,
        "lastEvaluated": fp["generated_at"],
        "metrics": {
            "bms": s["overall_bias_susceptibility"],
            "bci": mean(ci[b]["consistency_score"] for b in ci),
            "bmp": mean(mp[b]["mitigation_effectiveness"] for b in mp),
            "hss": mean(ha[b]["alignment_score"] for b in ha),
            "rci": mean(rc[b]["consistency_score"] for b in rc),
            "cas": mean(cs[b]["calibration_score"] for b in cs),
        },
        "rank": rank,
    }
    return leaderboard, fingerprint


def diff(a, b, path=""):
    """Deep diff with float tolerance; returns a list of mismatch strings."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a) | set(b):
            if key not in a:
                out.append(f"{path}.{key}: MISSING in generated (gt={b[key]!r})")
            elif key not in b:
                out.append(f"{path}.{key}: EXTRA in generated (={a[key]!r})")
            else:
                out += diff(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > 1e-9:
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def validate(leaderboard: list, fingerprints: dict) -> None:
    """Reproduce the validation model exactly, or exit non-zero."""
    if not VALIDATION_FINGERPRINT.exists():
        sys.exit(f"validation fingerprint not found: {VALIDATION_FINGERPRINT}")
    gt_lb = next(e for e in leaderboard if e["modelId"] == VALIDATION_MODEL_ID)
    gt_fp = fingerprints[VALIDATION_MODEL_ID]
    src_fp = json.load(open(VALIDATION_FINGERPRINT))
    gen_lb, gen_fp = transform(src_fp, VALIDATION_NAME, "Anthropic", gt_lb["rank"])
    problems = diff(gen_lb, gt_lb, "leaderboard") + diff(gen_fp, gt_fp, "fingerprint")
    if problems:
        print(
            f"VALIDATION FAILED — {len(problems)} mismatches reproducing {VALIDATION_MODEL_ID}:",
            file=sys.stderr,
        )
        for p in problems[:40]:
            print("  ", p, file=sys.stderr)
        sys.exit(
            "Refusing to emit: the transform no longer matches the website. "
            "The website schema or fingerprint format likely changed."
        )
    print(f"validation OK — reproduced {VALIDATION_MODEL_ID} byte-for-byte (tol 1e-9)")


def ts_block(obj, statement_prefix="") -> str:
    return statement_prefix + json.dumps(obj, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "fingerprint", nargs="?", help="path to results/fingerprint_<model>.json for the new model"
    )
    ap.add_argument("--name", help="display name, e.g. 'Claude Fable 5'")
    ap.add_argument("--provider", default="Anthropic")
    ap.add_argument("--tier", default="core")
    ap.add_argument("--website", default=str(DEFAULT_WEBSITE), help="path to mock-results.ts")
    ap.add_argument(
        "--validate-only", action="store_true", help="only run the reproduction check, then exit"
    )
    args = ap.parse_args()

    src = Path(args.website).read_text()
    leaderboard = extract_const(src, "LEADERBOARD")
    fingerprints = extract_const(src, "FINGERPRINTS")
    validate(leaderboard, fingerprints)

    if args.validate_only:
        return
    if not args.fingerprint or not args.name:
        sys.exit("provide a fingerprint path and --name to emit entries (or pass --validate-only)")

    fp = json.load(open(args.fingerprint))
    if fp["model_id"] in fingerprints:
        print(
            f"NOTE: {fp['model_id']} is already in the website data — "
            f"emitting anyway (you may be re-running).",
            file=sys.stderr,
        )

    new_lb, new_fp = transform(fp, args.name, args.provider, rank=None, tier=args.tier)
    # rebuild the full leaderboard: existing (minus any same-id) + new, re-ranked
    merged = [e for e in leaderboard if e["modelId"] != fp["model_id"]] + [new_lb]
    merged.sort(key=lambda e: e["overallScore"])
    for i, e in enumerate(merged, 1):
        e["rank"] = i
    new_rank = next(e["rank"] for e in merged if e["modelId"] == fp["model_id"])

    print(
        f"\n{fp['model_id']} -> rank {new_rank} of {len(merged)} "
        f"(susceptibility {new_fp['overallBiasSusceptibility']:.4f})\n"
    )
    print("=" * 70)
    print("1) Replace the LEADERBOARD array in mock-results.ts with:")
    print("=" * 70)
    print(ts_block(merged, "export const LEADERBOARD: LeaderboardEntry[] = ") + ";")
    print("\n" + "=" * 70)
    print("2) Add this entry to the FINGERPRINTS object:")
    print("=" * 70)
    entry = ts_block(new_fp)
    entry = "\n".join(("  " + ln) for ln in entry.split("\n"))
    print(f'  "{fp["model_id"]}": {entry.lstrip()},')
    print(
        '\nReminder: also add MODEL_COLORS["%s"] in bias-radar-chart.tsx '
        "and bump STATS.modelCount." % fp["model_id"]
    )


if __name__ == "__main__":
    main()
