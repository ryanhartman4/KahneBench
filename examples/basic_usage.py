#!/usr/bin/env python3
"""
Basic usage example for Kahne-Bench.

This script demonstrates how to:
1. Generate test cases for specific biases
2. Run evaluations (with mock LLM for demonstration)
3. Calculate and interpret metrics
"""

import asyncio
from dataclasses import dataclass

from kahne_bench import (
    # Core types
    CognitiveBiasInstance,
    Domain,
    TestScale,
    TriggerIntensity,
    # Taxonomy
    BIAS_TAXONOMY,
    get_bias_by_id,
    get_biases_by_category,
    BiasCategory,
    # Engines
    TestCaseGenerator,
    BiasEvaluator,
    # Metrics
    MetricCalculator,
)
from kahne_bench.core import TestResult, EvaluationSession
from kahne_bench.engines.compound import CompoundTestGenerator


# Mock LLM provider for demonstration
@dataclass
class MockProvider:
    """A mock LLM provider that returns predetermined responses."""

    bias_tendency: float = 0.6  # How often to give biased responses

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate a mock response based on prompt content."""
        import random

        # Simulate biased behavior
        if random.random() < self.bias_tendency:
            if "option a" in prompt.lower() and "option b" in prompt.lower():
                return "After careful consideration, I choose Option A. The first option presented seems more appealing. I am 85% confident in this choice."
            elif "estimate" in prompt.lower():
                # Anchored response
                import re
                numbers = re.findall(r'\d+', prompt)
                if numbers:
                    anchor = int(numbers[0])
                    return f"Based on my analysis, I estimate approximately {int(anchor * 0.9)}. I am 80% confident."
                return "I estimate approximately 5000. I am 75% confident."
            else:
                return "I would choose the safer, more certain option. The potential for loss weighs heavily in my decision. Confidence: 70%"
        else:
            return "After rational analysis considering all factors equally, I recommend the option with the highest expected value regardless of framing. Confidence: 65%"


def demonstrate_taxonomy():
    """Show the bias taxonomy structure."""
    print("\n" + "=" * 60)
    print("KAHNE-BENCH BIAS TAXONOMY")
    print("=" * 60)

    print(f"\nTotal biases: {len(BIAS_TAXONOMY)}")

    # Group by category
    for category in BiasCategory:
        biases = get_biases_by_category(category)
        if biases:
            print(f"\n{category.value.upper()} ({len(biases)} biases):")
            for bias in biases[:3]:  # Show first 3
                print(f"  - {bias.name}: {bias.description[:50]}...")
            if len(biases) > 3:
                print(f"  ... and {len(biases) - 3} more")


def demonstrate_test_generation():
    """Show test case generation."""
    print("\n" + "=" * 60)
    print("TEST CASE GENERATION")
    print("=" * 60)

    generator = TestCaseGenerator(seed=42)

    # Generate a single instance
    instance = generator.generate_instance(
        bias_id="anchoring_effect",
        domain=Domain.PROFESSIONAL,
        scale=TestScale.MICRO,
    )

    print(f"\nGenerated instance for: {instance.bias_id}")
    print(f"Domain: {instance.domain.value}")
    print(f"Scale: {instance.scale.value}")

    print("\nControl prompt (first 200 chars):")
    print(instance.control_prompt[:200] + "...")

    print("\nTreatment prompt (moderate intensity, first 200 chars):")
    print(instance.get_treatment(TriggerIntensity.MODERATE)[:200] + "...")

    # Generate a batch
    batch = generator.generate_batch(
        bias_ids=["anchoring_effect", "loss_aversion", "gain_loss_framing"],
        domains=[Domain.PROFESSIONAL, Domain.INDIVIDUAL],
        instances_per_combination=1,
    )
    print(f"\nGenerated batch of {len(batch)} test instances")

    return instance


def demonstrate_compound_generation():
    """Show compound test generation for bias interactions."""
    print("\n" + "=" * 60)
    print("COMPOUND (MESO-SCALE) TEST GENERATION")
    print("=" * 60)

    generator = CompoundTestGenerator()

    # Generate compound test
    instance = generator.generate_compound_instance(
        primary_bias="anchoring_effect",
        secondary_biases=["availability_bias", "overconfidence_effect"],
        domain=Domain.PROFESSIONAL,
        interaction_type="amplifying",
    )

    print(f"\nPrimary bias: {instance.bias_id}")
    print(f"Interacting biases: {instance.interaction_biases}")
    print(f"Scale: {instance.scale.value}")

    print("\nCompound scenario (first 300 chars):")
    print(instance.base_scenario[:300] + "...")

    return instance


async def demonstrate_evaluation():
    """Show evaluation and metrics calculation."""
    print("\n" + "=" * 60)
    print("EVALUATION AND METRICS")
    print("=" * 60)

    # Setup
    generator = TestCaseGenerator(seed=42)
    provider = MockProvider(bias_tendency=0.65)

    # Create a small test set
    instances = []
    for bias_id in ["anchoring_effect", "loss_aversion", "gain_loss_framing"]:
        instance = generator.generate_instance(bias_id, Domain.PROFESSIONAL)
        instances.append(instance)

    print(f"\nRunning evaluation on {len(instances)} instances...")

    # Create evaluator with config
    from kahne_bench.engines.evaluator import EvaluationConfig
    config = EvaluationConfig(
        num_trials=2,  # Reduced for demo
        intensities=[TriggerIntensity.MODERATE, TriggerIntensity.STRONG],
    )
    evaluator = BiasEvaluator(provider, config)

    # Run evaluation
    session = await evaluator.evaluate_batch(
        instances,
        model_id="mock-model-v1",
        progress_callback=lambda i, n: print(f"  Progress: {i}/{n}"),
    )

    print(f"\nCompleted evaluation session: {session.session_id[:8]}...")
    print(f"Total results: {len(session.results)}")

    # Score results
    for result in session.results:
        is_biased, score = evaluator.score_response(
            result,
            result.instance.expected_rational_response,
            result.instance.expected_biased_response,
        )
        result.is_biased = is_biased
        result.bias_score = score

    # Calculate metrics
    calculator = MetricCalculator()
    report = calculator.calculate_all_metrics("mock-model-v1", session.results)

    print("\n--- COGNITIVE FINGERPRINT REPORT ---")
    print(f"Model: {report.model_id}")
    print(f"Biases tested: {len(report.biases_tested)}")
    print(f"Overall bias susceptibility: {report.overall_bias_susceptibility:.2f}")

    if report.magnitude_scores:
        print("\nBias Magnitude Scores:")
        for bias_id, bms in report.magnitude_scores.items():
            print(f"  {bias_id}: {bms.overall_magnitude:.3f}")

    if report.human_alignments:
        print("\nHuman Alignment Scores:")
        for bias_id, has in report.human_alignments.items():
            print(f"  {bias_id}: {has.alignment_score:.3f} ({has.bias_direction})")

    return session


def demonstrate_debiasing():
    """Show debiasing prompt generation."""
    print("\n" + "=" * 60)
    print("DEBIASING PROMPTS")
    print("=" * 60)

    generator = TestCaseGenerator(seed=42)
    instance = generator.generate_instance(
        bias_id="anchoring_effect",
        domain=Domain.PROFESSIONAL,
        include_debiasing=True,
    )

    print(f"\nDebiasing strategies for {instance.bias_id}:")
    for i, prompt in enumerate(instance.debiasing_prompts):
        print(f"\n--- Strategy {i + 1} (first 200 chars) ---")
        print(prompt[:200] + "...")


async def main():
    """Run all demonstrations."""
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "KAHNE-BENCH DEMO" + " " * 18 + "#")
    print("#" * 60)

    # 1. Taxonomy
    demonstrate_taxonomy()

    # 2. Test generation
    demonstrate_test_generation()

    # 3. Compound tests
    demonstrate_compound_generation()

    # 4. Evaluation and metrics
    await demonstrate_evaluation()

    # 5. Debiasing
    demonstrate_debiasing()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nFor full documentation, see the docs/ directory.")
    print("To run the CLI: kahne-bench --help")


if __name__ == "__main__":
    asyncio.run(main())
