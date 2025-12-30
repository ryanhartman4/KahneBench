#!/usr/bin/env python3
"""
OpenAI Evaluation Example for Kahne-Bench.

This script demonstrates how to evaluate an OpenAI model for cognitive biases
using Kahne-Bench.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python examples/openai_evaluation.py

    # With options:
    python examples/openai_evaluation.py --model gpt-5.2 --tier extended
"""

import argparse
import asyncio
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

# Kahne-Bench imports
from kahne_bench import (
    TestCaseGenerator,
    BiasEvaluator,
    MetricCalculator,
    Domain,
    TriggerIntensity,
)
from kahne_bench.engines.evaluator import EvaluationConfig
from kahne_bench.engines.generator import get_tier_biases, KahneBenchTier
from kahne_bench.utils import (
    export_results_to_json,
    export_fingerprint_to_json,
    generate_summary_report,
)


@dataclass
class OpenAIProvider:
    """
    OpenAI LLM provider for Kahne-Bench using the modern Responses API.
    """

    client: AsyncOpenAI
    model: str = "gpt-5.2"
    instructions: str | None = None

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate completion from the model."""
        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
            instructions=self.instructions,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        return response.output_text or ""


async def run_evaluation(
    model: str,
    tier: str = "core",
    domains: list[str] | None = None,
    num_trials: int = 3,
    output_prefix: str = "evaluation",
):
    """
    Run a complete cognitive bias evaluation.

    Args:
        model: Model name (e.g., "gpt-5.2", "gpt-4.1")
        tier: Benchmark tier ("core", "extended", or "interaction")
        domains: List of domains to test (default: all)
        num_trials: Number of trials per condition
        output_prefix: Prefix for output files
    """
    # Initialize OpenAI client
    client = AsyncOpenAI()

    provider = OpenAIProvider(client=client, model=model)

    # Get biases for the selected tier
    tier_map = {
        "core": KahneBenchTier.CORE,
        "extended": KahneBenchTier.EXTENDED,
        "interaction": KahneBenchTier.INTERACTION,
    }
    bias_ids = get_tier_biases(tier_map.get(tier, KahneBenchTier.CORE))

    # Parse domains
    domain_list = [Domain.PROFESSIONAL, Domain.INDIVIDUAL]  # Default
    if domains:
        domain_list = [Domain(d) for d in domains]

    print(f"\n{'='*60}")
    print(f"KAHNE-BENCH EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Tier: {tier} ({len(bias_ids)} biases)")
    print(f"Domains: {[d.value for d in domain_list]}")
    print(f"Trials per condition: {num_trials}")
    print(f"{'='*60}\n")

    # Generate test instances
    print("Generating test instances...")
    generator = TestCaseGenerator(seed=42)
    instances = generator.generate_batch(
        bias_ids=list(bias_ids),
        domains=domain_list,
        instances_per_combination=1,
    )
    print(f"Generated {len(instances)} test instances\n")

    # Configure evaluation
    config = EvaluationConfig(
        num_trials=num_trials,
        intensities=[
            TriggerIntensity.WEAK,
            TriggerIntensity.MODERATE,
            TriggerIntensity.STRONG,
        ],
        include_control=True,
        include_debiasing=False,  # Set True for debiasing analysis
    )

    evaluator = BiasEvaluator(provider, config)

    # Run evaluation with progress tracking
    print("Running evaluation...")
    total = len(instances)

    def progress_callback(current: int, total: int):
        pct = current / total * 100
        bar = "=" * int(pct // 2) + ">" + " " * (50 - int(pct // 2))
        print(f"\r[{bar}] {current}/{total} ({pct:.1f}%)", end="", flush=True)

    session = await evaluator.evaluate_batch(
        instances=instances,
        model_id=model,
        progress_callback=progress_callback,
    )
    print(f"\n\nCompleted {len(session.results)} evaluations\n")

    # Score results
    print("Scoring responses...")
    for result in session.results:
        is_biased, score = evaluator.score_response(
            result,
            result.instance.expected_rational_response,
            result.instance.expected_biased_response,
        )
        result.is_biased = is_biased
        result.bias_score = score

    # Calculate metrics
    print("Calculating metrics...")
    calculator = MetricCalculator()
    report = calculator.calculate_all_metrics(model, session.results)

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(generate_summary_report(report))

    # Export results
    results_file = f"{output_prefix}_results.json"
    fingerprint_file = f"{output_prefix}_fingerprint.json"

    export_results_to_json(session.results, results_file)
    export_fingerprint_to_json(report, fingerprint_file)

    print(f"\nResults exported to:")
    print(f"  - {results_file}")
    print(f"  - {fingerprint_file}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM for cognitive biases using Kahne-Bench"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-5.2",
        help="Model name (default: gpt-5.2)"
    )
    parser.add_argument(
        "--tier", "-t",
        choices=["core", "extended", "interaction"],
        default="core",
        help="Benchmark tier (default: core)"
    )
    parser.add_argument(
        "--domains", "-d",
        nargs="+",
        choices=["individual", "professional", "social", "temporal", "risk"],
        help="Domains to test (default: professional, individual)"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=3,
        help="Number of trials per condition (default: 3)"
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation",
        help="Output file prefix (default: evaluation)"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return 1

    # Run evaluation
    asyncio.run(run_evaluation(
        model=args.model,
        tier=args.tier,
        domains=args.domains,
        num_trials=args.trials,
        output_prefix=args.output,
    ))

    return 0


if __name__ == "__main__":
    exit(main())
