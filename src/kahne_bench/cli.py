"""
Command-line interface for Kahne-Bench.

Provides commands for generating test cases, running evaluations,
and computing metrics.
"""

import asyncio
import json
import sys

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from kahne_bench import __version__
from kahne_bench.core import Domain, TestScale
from kahne_bench.biases import BIAS_TAXONOMY, get_biases_by_category, BiasCategory
from kahne_bench.engines.generator import TestCaseGenerator
from kahne_bench.engines.compound import CompoundTestGenerator


console = Console()


def validate_positive(ctx, param, value):
    """Click callback to validate that a value is at least 1."""
    if value is not None and value < 1:
        raise click.BadParameter(f"must be at least 1, got {value}")
    return value


@click.group()
@click.version_option(version=__version__)
def main():
    """Kahne-Bench: Cognitive Bias Benchmark for LLMs

    A framework for evaluating cognitive biases in Large Language Models,
    based on the dual-process theory of Kahneman and Tversky.
    """
    pass


@main.command()
def list_biases():
    """List all 69 biases in the taxonomy."""
    table = Table(title="Kahne-Bench Bias Taxonomy")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Category", style="yellow")
    table.add_column("Description", style="white", max_width=50)

    for bias_id, bias in sorted(BIAS_TAXONOMY.items()):
        table.add_row(
            bias_id,
            bias.name,
            bias.category.value,
            bias.description[:50] + "..." if len(bias.description) > 50 else bias.description,
        )

    console.print(table)
    console.print(f"\nTotal: {len(BIAS_TAXONOMY)} biases")


@main.command()
@click.argument("category", required=False)
def list_categories(category: str | None):
    """List bias categories or biases in a specific category."""
    if category:
        try:
            cat = BiasCategory(category)
            biases = get_biases_by_category(cat)

            table = Table(title=f"Biases in {cat.value}")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Classic Paradigm", style="white", max_width=40)

            for bias in biases:
                table.add_row(
                    bias.id,
                    bias.name,
                    bias.classic_paradigm[:40] + "..." if len(bias.classic_paradigm) > 40 else bias.classic_paradigm,
                )

            console.print(table)
        except ValueError:
            console.print(f"[red]Unknown category: {category}[/red]")
            console.print("Available categories:", [c.value for c in BiasCategory])
    else:
        table = Table(title="Bias Categories")
        table.add_column("Category", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Count", style="green")

        category_descriptions = {
            BiasCategory.REPRESENTATIVENESS: "Judging by similarity to prototypes",
            BiasCategory.AVAILABILITY: "Judging by ease of mental recall",
            BiasCategory.ANCHORING: "Over-reliance on initial values",
            BiasCategory.LOSS_AVERSION: "Losses loom larger than gains",
            BiasCategory.FRAMING: "Decisions affected by presentation",
            BiasCategory.REFERENCE_DEPENDENCE: "Outcomes evaluated relative to reference points",
            BiasCategory.PROBABILITY_DISTORTION: "Misweighting probabilities",
            BiasCategory.UNCERTAINTY_JUDGMENT: "Errors in assessing uncertainty",
            BiasCategory.MEMORY_BIAS: "Systematic distortions in memory recall",
            BiasCategory.ATTENTION_BIAS: "Selective focus on certain information",
            BiasCategory.SOCIAL_BIAS: "Biases in social judgments",
            BiasCategory.ATTRIBUTION_BIAS: "Errors in explaining causes of events",
            BiasCategory.OVERCONFIDENCE: "Excessive certainty in judgments",
            BiasCategory.CONFIRMATION: "Seeking confirming evidence",
            BiasCategory.TEMPORAL_BIAS: "Biases in time perception",
            BiasCategory.EXTENSION_NEGLECT: "Ignoring scope and sample size",
        }

        for cat in BiasCategory:
            biases = get_biases_by_category(cat)
            desc = category_descriptions.get(cat, "")
            table.add_row(cat.value, desc, str(len(biases)))

        console.print(table)


@main.command()
@click.option("--bias", "-b", multiple=True, help="Bias IDs to include (default: all)")
@click.option("--domain", "-d", type=click.Choice([d.value for d in Domain]), multiple=True, help="Domains to include")
@click.option("--instances", "-n", default=3, callback=validate_positive, help="Instances per bias-domain pair")
@click.option("--output", "-o", default="test_cases.json", help="Output file path")
@click.option("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
def generate(bias: tuple, domain: tuple, instances: int, output: str, seed: int | None):
    """Generate test case instances."""
    generator = TestCaseGenerator(seed=seed)

    bias_ids = list(bias) if bias else None
    domains = [Domain(d) for d in domain] if domain else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating test cases...", total=None)

        try:
            test_instances = generator.generate_batch(
                bias_ids=bias_ids,
                domains=domains,
                instances_per_combination=instances,
            )

            progress.update(task, description=f"Generated {len(test_instances)} instances")

            generator.export_to_json(test_instances, output)
            console.print(f"[green]Saved {len(test_instances)} test cases to {output}[/green]")

        except Exception as e:
            console.print(f"[red]Error generating test cases: {e}[/red]")
            sys.exit(1)


@main.command()
@click.option("--bias", "-b", multiple=True, help="Primary bias IDs for compound tests")
@click.option("--domain", "-d", type=click.Choice([d.value for d in Domain]), default="professional")
@click.option("--output", "-o", default="compound_tests.json", help="Output file path")
def generate_compound(bias: tuple, domain: str, output: str):
    """Generate compound (meso-scale) test cases for bias interactions."""
    generator = CompoundTestGenerator()

    domain_enum = Domain(domain)

    if bias:
        instances = []
        for b in bias:
            try:
                instance = generator.generate_compound_instance(
                    primary_bias=b,
                    domain=domain_enum,
                )
                instances.append(instance)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not generate {b}: {e}[/yellow]")
    else:
        instances = generator.generate_interaction_battery(domain=domain_enum)

    console.print(f"[green]Generated {len(instances)} compound test cases[/green]")

    # Export
    data = []
    for inst in instances:
        data.append({
            "bias_id": inst.bias_id,
            "interaction_biases": inst.interaction_biases,
            "scale": inst.scale.value,
            "domain": inst.domain.value,
            "control_prompt": inst.control_prompt,
            "treatment_prompts": {k.value: v for k, v in inst.treatment_prompts.items()},
            "metadata": inst.metadata,
        })

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[green]Saved to {output}[/green]")


@main.command()
@click.argument("bias_id")
def describe(bias_id: str):
    """Show detailed information about a specific bias."""
    bias = BIAS_TAXONOMY.get(bias_id)

    if not bias:
        console.print(f"[red]Unknown bias: {bias_id}[/red]")
        console.print("Use 'kahne-bench list-biases' to see available biases.")
        sys.exit(1)

    console.print(f"\n[bold cyan]{bias.name}[/bold cyan]")
    console.print(f"[dim]ID: {bias.id}[/dim]\n")

    console.print(f"[bold]Category:[/bold] {bias.category.value}")
    console.print(f"\n[bold]Description:[/bold]\n{bias.description}")
    console.print(f"\n[bold]Theoretical Basis:[/bold]\n{bias.theoretical_basis}")
    console.print(f"\n[bold]System 1 Mechanism:[/bold]\n{bias.system1_mechanism}")
    console.print(f"\n[bold]System 2 Override:[/bold]\n{bias.system2_override}")
    console.print(f"\n[bold]Classic Paradigm:[/bold]\n{bias.classic_paradigm}")
    console.print(f"\n[bold]Trigger Template:[/bold]\n{bias.trigger_template}")


@main.command()
def info():
    """Show information about the Kahne-Bench framework."""
    console.print("\n[bold cyan]Kahne-Bench[/bold cyan]")
    console.print("A Cognitive Bias Benchmark for Large Language Models\n")

    console.print("[bold]Based on:[/bold]")
    console.print("  - Dual-Process Theory (Kahneman)")
    console.print("  - Prospect Theory (Kahneman & Tversky)")
    console.print("  - Heuristics and Biases Research Program\n")

    console.print("[bold]Coverage:[/bold]")
    console.print(f"  - {len(BIAS_TAXONOMY)} cognitive biases")
    console.print(f"  - {len(Domain)} ecological domains")
    console.print(f"  - {len(TestScale)} testing scales\n")

    console.print("[bold]Advanced Metrics:[/bold]")
    metrics = [
        ("BMS", "Bias Magnitude Score", "Quantifies bias strength"),
        ("BCI", "Bias Consistency Index", "Measures cross-domain consistency"),
        ("BMP", "Bias Mitigation Potential", "Assesses debiasing capability"),
        ("HAS", "Human Alignment Score", "Compares to human baselines"),
        ("RCI", "Response Consistency Index", "Measures trial-to-trial variance"),
        ("CAS", "Calibration Awareness Score", "Assesses metacognitive accuracy"),
    ]

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")

    for abbrev, name, desc in metrics:
        table.add_row(abbrev, name, desc)

    console.print(table)


@main.command()
@click.option("--input", "-i", "input_file", required=True, help="Input JSON file with test cases")
@click.option("--provider", "-p", type=click.Choice(["openai", "anthropic", "fireworks", "xai", "gemini", "mock"]), default="mock",
              help="LLM provider (fireworks for open-source models, xai for Grok, gemini for Google)")
@click.option("--model", "-m", default=None, help="Model name to evaluate")
@click.option("--trials", "-n", default=3, callback=validate_positive, help="Number of trials per condition")
@click.option("--output", "-o", default="results.json", help="Output file for results")
@click.option("--fingerprint", "-f", default="fingerprint.json", help="Output file for cognitive fingerprint")
@click.option("--tier", "-t", type=click.Choice(["core", "extended", "interaction"]), default="core",
              help="Benchmark tier")
def evaluate(input_file: str, provider: str, model: str | None, trials: int, output: str, fingerprint: str, tier: str):
    """Evaluate an LLM for cognitive biases.

    Run a complete bias evaluation on a model using pre-generated test cases
    or generate new ones on the fly.

    Examples:
        # Evaluate using mock provider (for testing):
        kahne-bench evaluate -i test_cases.json -p mock

        # Evaluate with OpenAI:
        kahne-bench evaluate -i test_cases.json -p openai -m gpt-4o

        # Generate and evaluate in one command:
        kahne-bench generate -o test_cases.json && kahne-bench evaluate -i test_cases.json -p openai
    """
    from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeRemainingColumn

    from kahne_bench.utils.io import import_instances_from_json, export_results_to_json, export_fingerprint_to_json
    from kahne_bench.engines.evaluator import BiasEvaluator, EvaluationConfig
    from kahne_bench.metrics import MetricCalculator

    # Load test instances
    console.print(f"[cyan]Loading test cases from {input_file}...[/cyan]")
    try:
        instances = import_instances_from_json(input_file)
        if not instances:
            console.print("[red]Error: Input file contains no test instances[/red]")
            sys.exit(1)
        console.print(f"[green]Loaded {len(instances)} test instances[/green]")
    except Exception as e:
        console.print(f"[red]Error loading test cases: {e}[/red]")
        sys.exit(1)

    # Set up provider
    if provider == "mock":
        from dataclasses import dataclass
        import random

        @dataclass
        class MockProvider:
            """Mock provider for testing the CLI without API calls."""
            model: str = "mock-model"

            async def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
                # Generate a mock response
                choices = ["Option A", "Option B", "Program A", "Program B", "Yes", "No"]
                choice = random.choice(choices)
                confidence = random.randint(50, 95)
                return f"After careful consideration, I would choose {choice}. I am {confidence}% confident in this decision."

        llm_provider = MockProvider(model=model or "mock-model")
        model_id = model or "mock-model"
        console.print("[yellow]Using mock provider (for testing only)[/yellow]")

    elif provider == "openai":
        import os
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            sys.exit(1)

        try:
            from openai import AsyncOpenAI
            from kahne_bench.engines.evaluator import OpenAIProvider

            client = AsyncOpenAI()
            model_id = model or "gpt-4o"
            llm_provider = OpenAIProvider(client=client, model=model_id)
            console.print(f"[green]Using OpenAI provider with model: {model_id}[/green]")
        except ImportError:
            console.print("[red]Error: openai package not installed. Run: pip install openai[/red]")
            sys.exit(1)

    elif provider == "anthropic":
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            console.print("[red]Error: ANTHROPIC_API_KEY environment variable not set[/red]")
            sys.exit(1)

        try:
            from anthropic import AsyncAnthropic
            from kahne_bench.engines.evaluator import AnthropicProvider

            client = AsyncAnthropic()
            model_id = model or "claude-sonnet-4-20250514"
            llm_provider = AnthropicProvider(client=client, model=model_id)
            console.print(f"[green]Using Anthropic provider with model: {model_id}[/green]")
        except ImportError:
            console.print("[red]Error: anthropic package not installed. Run: pip install anthropic[/red]")
            sys.exit(1)

    elif provider == "fireworks":
        import os
        if not os.getenv("FIREWORKS_API_KEY"):
            console.print("[red]Error: FIREWORKS_API_KEY environment variable not set[/red]")
            sys.exit(1)

        try:
            from openai import AsyncOpenAI
            from kahne_bench.engines.evaluator import OpenAIProvider

            # Fireworks uses OpenAI-compatible API
            client = AsyncOpenAI(
                api_key=os.getenv("FIREWORKS_API_KEY"),
                base_url="https://api.fireworks.ai/inference/v1"
            )
            model_id = model or "accounts/fireworks/models/llama-v3p1-70b-instruct"
            llm_provider = OpenAIProvider(client=client, model=model_id)
            console.print(f"[green]Using Fireworks provider with model: {model_id}[/green]")
        except ImportError:
            console.print("[red]Error: openai package not installed. Run: pip install openai[/red]")
            sys.exit(1)

    elif provider == "xai":
        import os
        if not os.getenv("XAI_API_KEY"):
            console.print("[red]Error: XAI_API_KEY environment variable not set[/red]")
            sys.exit(1)

        try:
            from xai_sdk import Client
            from kahne_bench.engines.evaluator import XAIProvider

            client = Client(
                api_key=os.getenv("XAI_API_KEY"),
                timeout=3600,  # Extended timeout for reasoning models
            )
            model_id = model or "grok-4-1-fast-reasoning"
            llm_provider = XAIProvider(client=client, model=model_id)
            console.print(f"[green]Using xAI provider with model: {model_id}[/green]")
        except ImportError:
            console.print("[red]Error: xai-sdk package not installed. Run: pip install xai-sdk[/red]")
            sys.exit(1)

    elif provider == "gemini":
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            console.print("[red]Error: GOOGLE_API_KEY environment variable not set[/red]")
            sys.exit(1)

        try:
            from google import genai
            from kahne_bench.engines.evaluator import GeminiProvider

            client = genai.Client()
            model_id = model or "gemini-3-pro-preview"
            llm_provider = GeminiProvider(client=client, model=model_id)
            console.print(f"[green]Using Gemini provider with model: {model_id}[/green]")
        except ImportError:
            console.print("[red]Error: google-genai package not installed. Run: pip install google-genai[/red]")
            sys.exit(1)

    # Configure evaluation
    from kahne_bench.core import TriggerIntensity
    config = EvaluationConfig(
        num_trials=trials,
        intensities=[
            TriggerIntensity.WEAK,
            TriggerIntensity.MODERATE,
            TriggerIntensity.STRONG,
        ],
        include_control=True,
        include_debiasing=True,
    )

    evaluator = BiasEvaluator(llm_provider, config)

    # Run evaluation
    console.print(f"\n[bold]Starting evaluation of {len(instances)} test instances...[/bold]")
    console.print(f"  Model: {model_id}")
    console.print(f"  Trials per condition: {trials}")
    console.print(f"  Tier: {tier}\n")

    async def run_evaluation():
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(instances))

            def progress_callback(current: int, total: int):
                progress.update(task, completed=current)

            session = await evaluator.evaluate_batch(
                instances=instances,
                model_id=model_id,
                progress_callback=progress_callback,
            )

            return session

    session = asyncio.run(run_evaluation())

    console.print(f"\n[green]Completed {len(session.results)} evaluations[/green]")

    # Calculate metrics
    console.print("[cyan]Calculating metrics...[/cyan]")
    calculator = MetricCalculator()
    report = calculator.calculate_all_metrics(model_id, session.results)

    # Export results
    export_results_to_json(session.results, output)
    export_fingerprint_to_json(report, fingerprint)

    console.print("\n[bold green]Results saved:[/bold green]")
    console.print(f"  - Results: {output}")
    console.print(f"  - Fingerprint: {fingerprint}")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Overall Bias Susceptibility: {report.overall_bias_susceptibility:.2%}")
    console.print(f"  Biases Tested: {len(report.biases_tested)}")

    if report.most_susceptible_biases:
        console.print("\n  Most Susceptible Biases:")
        for bias_id in report.most_susceptible_biases[:3]:
            if bias_id in report.magnitude_scores:
                mag = report.magnitude_scores[bias_id].overall_magnitude
                console.print(f"    - {bias_id}: {mag:.3f}")

    if report.most_resistant_biases:
        console.print("\n  Most Resistant Biases:")
        for bias_id in report.most_resistant_biases[:3]:
            if bias_id in report.magnitude_scores:
                mag = report.magnitude_scores[bias_id].overall_magnitude
                console.print(f"    - {bias_id}: {mag:.3f}")


@main.command()
@click.argument("fingerprint_file")
def report(fingerprint_file: str):
    """Generate a human-readable report from a cognitive fingerprint."""
    import json

    try:
        with open(fingerprint_file, "r") as f:
            data = json.load(f)

        console.print("\n[bold cyan]Cognitive Fingerprint Report[/bold cyan]")
        console.print(f"[dim]Model: {data.get('model_id', 'Unknown')}[/dim]")
        console.print(f"[dim]Generated: {data.get('generated_at', 'Unknown')}[/dim]\n")

        summary = data.get("summary", {})

        console.print(f"[bold]Overall Bias Susceptibility:[/bold] {summary.get('overall_bias_susceptibility', 0):.2%}\n")

        if summary.get("most_susceptible_biases"):
            console.print("[bold]Most Susceptible Biases:[/bold]")
            for bias_id in summary["most_susceptible_biases"][:5]:
                mag_data = data.get("magnitude_scores", {}).get(bias_id, {})
                mag = mag_data.get("overall_magnitude", 0)
                console.print(f"  - {bias_id}: {mag:.3f}")

        console.print()

        if summary.get("most_resistant_biases"):
            console.print("[bold]Most Resistant Biases:[/bold]")
            for bias_id in summary["most_resistant_biases"][:5]:
                mag_data = data.get("magnitude_scores", {}).get(bias_id, {})
                mag = mag_data.get("overall_magnitude", 0)
                console.print(f"  - {bias_id}: {mag:.3f}")

        console.print()

        if summary.get("human_like_biases"):
            console.print("[bold]Human-Like Biases (high alignment):[/bold]")
            for bias_id in summary["human_like_biases"][:5]:
                console.print(f"  - {bias_id}")

        if summary.get("ai_specific_biases"):
            console.print("\n[bold]AI-Specific Biases (deviates from human patterns):[/bold]")
            for bias_id in summary["ai_specific_biases"][:5]:
                console.print(f"  - {bias_id}")

    except Exception as e:
        console.print(f"[red]Error reading fingerprint file: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
