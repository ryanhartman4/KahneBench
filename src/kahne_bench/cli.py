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

PROVIDER_CHOICES = ["openai", "anthropic", "fireworks", "xai", "gemini", "mock"]


def validate_positive(ctx, param, value):
    """Click callback to validate that a value is at least 1."""
    if value is not None and value < 1:
        raise click.BadParameter(f"must be at least 1, got {value}")
    return value


def _create_provider(provider: str, model: str | None) -> tuple:
    """Create an LLM provider and resolve model_id.

    Returns (provider_instance, model_id) tuple.
    """
    import os

    if provider == "mock":
        from dataclasses import dataclass
        import random

        @dataclass
        class MockProvider:
            model: str = "mock-model"

            async def complete(
                self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
            ) -> str:
                choices = ["Option A", "Option B", "Program A", "Program B", "Yes", "No"]
                choice = random.choice(choices)
                confidence = random.randint(50, 95)
                return (
                    f"After careful consideration, I would choose {choice}. "
                    f"I am {confidence}% confident in this decision."
                )

        model_id = model or "mock-model"
        return MockProvider(model=model_id), model_id

    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            sys.exit(1)
        from openai import AsyncOpenAI
        from kahne_bench.engines.evaluator import OpenAIProvider

        client = AsyncOpenAI()
        model_id = model or "gpt-5"
        return OpenAIProvider(client=client, model=model_id), model_id

    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            console.print("[red]Error: ANTHROPIC_API_KEY environment variable not set[/red]")
            sys.exit(1)
        from anthropic import AsyncAnthropic
        from kahne_bench.engines.evaluator import AnthropicProvider

        client = AsyncAnthropic()
        model_id = model or "claude-haiku-4-5"
        return AnthropicProvider(client=client, model=model_id), model_id

    elif provider == "fireworks":
        if not os.getenv("FIREWORKS_API_KEY"):
            console.print("[red]Error: FIREWORKS_API_KEY environment variable not set[/red]")
            sys.exit(1)
        from openai import AsyncOpenAI
        from kahne_bench.engines.evaluator import OpenAIProvider

        client = AsyncOpenAI(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1",
        )
        model_id = model or "kimi-k2p5"
        return OpenAIProvider(client=client, model=model_id), model_id

    elif provider == "xai":
        if not os.getenv("XAI_API_KEY"):
            console.print("[red]Error: XAI_API_KEY environment variable not set[/red]")
            sys.exit(1)
        from xai_sdk import Client
        from kahne_bench.engines.evaluator import XAIProvider

        client = Client(api_key=os.getenv("XAI_API_KEY"), timeout=3600)
        model_id = model or "grok-4-1-fast-reasoning"
        return XAIProvider(client=client, model=model_id), model_id

    elif provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            console.print("[red]Error: GOOGLE_API_KEY environment variable not set[/red]")
            sys.exit(1)
        from google import genai
        from kahne_bench.engines.evaluator import GeminiProvider

        client = genai.Client()
        model_id = model or "gemini-3-pro-preview"
        return GeminiProvider(client=client, model=model_id), model_id

    else:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        sys.exit(1)


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
@click.option("--tier", "-t", type=click.Choice(["core", "extended"]), default=None,
              help="Benchmark tier (core=15 biases, extended=all 69)")
def generate(bias: tuple, domain: tuple, instances: int, output: str, seed: int | None, tier: str | None):
    """Generate test case instances."""
    from kahne_bench.engines.generator import KAHNE_BENCH_CORE_BIASES, KAHNE_BENCH_EXTENDED_BIASES

    generator = TestCaseGenerator(seed=seed)

    # Determine bias list: explicit biases > tier > all
    if bias:
        bias_ids = list(bias)
    elif tier == "core":
        bias_ids = KAHNE_BENCH_CORE_BIASES
        console.print(f"[cyan]Using CORE tier: {len(bias_ids)} foundational biases[/cyan]")
    elif tier == "extended":
        bias_ids = KAHNE_BENCH_EXTENDED_BIASES
        console.print(f"[cyan]Using EXTENDED tier: {len(bias_ids)} biases[/cyan]")
    else:
        bias_ids = None  # All biases

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
@click.option("--provider", "-p", type=click.Choice(PROVIDER_CHOICES), required=True,
              help="LLM provider (fireworks for open-source models, xai for Grok, gemini for Google)")
@click.option("--model", "-m", default=None, help="Model name to evaluate")
@click.option("--trials", "-n", default=3, callback=validate_positive, help="Number of trials per condition")
@click.option("--output", "-o", default="results.json", help="Output file for results")
@click.option("--fingerprint", "-f", default="fingerprint.json", help="Output file for cognitive fingerprint")
@click.option("--tier", "-t", type=click.Choice(["core", "extended", "interaction"]), default="core",
              help="Benchmark tier")
@click.option("--concurrency", "-c", default=50, callback=validate_positive,
              help="Max concurrent API requests (default: 50, increase for faster runs)")
@click.option("--judge-provider", type=click.Choice(PROVIDER_CHOICES), default="anthropic",
              help="LLM provider for judge fallback scoring (when regex extraction fails)")
@click.option("--judge-model", default="claude-haiku-4-5", help="Model for judge fallback scoring")
def evaluate(input_file: str, provider: str, model: str | None, trials: int, output: str,
             fingerprint: str, tier: str, concurrency: int,
             judge_provider: str, judge_model: str):
    """Evaluate an LLM for cognitive biases.

    Run a complete bias evaluation on a model using pre-generated test cases
    or generate new ones on the fly.

    Examples:
        # Evaluate using mock provider (for testing):
        kahne-bench evaluate -i test_cases.json -p mock

        # Evaluate with OpenAI:
        kahne-bench evaluate -i test_cases.json -p openai -m gpt-5.2

        # Evaluate with LLM judge fallback:
        kahne-bench evaluate -i test_cases.json -p openai -m gpt-5.2 --judge-provider openai --judge-model gpt-5.2
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
    llm_provider, model_id = _create_provider(provider, model)
    if provider == "mock":
        console.print(
            "[bold yellow]Warning:[/bold yellow] Mock provider produces synthetic results. "
            "Do not use for publication."
        )
    console.print(f"[green]Using {provider} provider with model: {model_id}[/green]")

    # Set up optional LLM judge
    judge = None
    if judge_provider:
        from kahne_bench.engines.judge import LLMJudge

        judge_llm, judge_model_id = _create_provider(judge_provider, judge_model)
        judge = LLMJudge(provider=judge_llm)
        console.print(f"[green]LLM judge enabled ({judge_provider}: {judge_model_id})[/green]")

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
        max_concurrent_requests=concurrency,
    )

    evaluator = BiasEvaluator(llm_provider, config, judge=judge)

    # Run evaluation
    console.print(f"\n[bold]Starting evaluation of {len(instances)} test instances...[/bold]")
    console.print(f"  Model: {model_id}")
    console.print(f"  Trials per condition: {trials}")
    console.print(f"  Concurrency: {concurrency} parallel requests")
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


@main.command("assess-quality")
@click.option("--input", "-i", "input_file", required=True, help="Input JSON file with test cases")
@click.option("--provider", "-p", type=click.Choice(PROVIDER_CHOICES), default="mock",
              help="LLM provider for quality assessment")
@click.option("--model", "-m", default=None, help="Model for quality assessment")
@click.option("--sample-rate", default=0.2, type=float, help="Fraction of instances to assess (0.0-1.0)")
@click.option("--output", "-o", default="quality_report.json", help="Output file for quality report")
def assess_quality(input_file: str, provider: str, model: str | None, sample_rate: float, output: str):
    """Assess the quality of generated test cases using LLM-as-judge.

    Evaluates test cases on realism, elicitation difficulty, and detection
    awareness. Helps identify low-quality test cases that may be too obvious
    or unrealistic.

    Examples:
        kahne-bench assess-quality -i test_cases.json -p mock

        kahne-bench assess-quality -i test_cases.json -p openai -m gpt-5.2 --sample-rate 0.3
    """
    from kahne_bench.utils.io import import_instances_from_json, export_quality_report_to_json
    from kahne_bench.engines.quality import QualityJudge

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

    llm_provider, model_id = _create_provider(provider, model)
    console.print(f"[green]Using {provider} provider ({model_id}) for quality assessment[/green]")

    quality_judge = QualityJudge(provider=llm_provider, sample_rate=sample_rate)

    console.print(f"\n[bold]Assessing quality of {len(instances)} instances "
                  f"(sampling {sample_rate:.0%})...[/bold]")

    async def run_assessment():
        return await quality_judge.assess_batch(instances)

    quality_report = asyncio.run(run_assessment())

    export_quality_report_to_json(quality_report, output)

    console.print(f"\n[bold green]Quality report saved to {output}[/bold green]")
    console.print(f"  Assessed: {quality_report.assessed_instances}/{quality_report.total_instances}")
    console.print(f"  Mean Realism: {quality_report.mean_realism:.1f}/10")
    console.print(f"  Mean Elicitation Difficulty: {quality_report.mean_elicitation_difficulty:.1f}/10")
    console.print(f"  Mean Detection Awareness: {quality_report.mean_detection_awareness:.1f}/10")
    if quality_report.low_quality_instances:
        console.print(f"  [yellow]Low quality instances: {len(quality_report.low_quality_instances)}[/yellow]")


@main.command("generate-bloom")
@click.option("--provider", "-p", type=click.Choice(PROVIDER_CHOICES), default="mock",
              help="LLM provider for scenario generation")
@click.option("--model", "-m", default=None, help="Model for scenario generation")
@click.option("--bias", "-b", multiple=True, help="Bias IDs to generate scenarios for")
@click.option("--domain", "-d", type=click.Choice([d.value for d in Domain]), multiple=True,
              help="Domains to generate for (default: professional)")
@click.option("--scenarios", "-n", default=3, callback=validate_positive,
              help="Scenarios per bias-domain pair")
@click.option("--output", "-o", default="bloom_tests.json", help="Output file")
def generate_bloom(provider: str, model: str | None, bias: tuple, domain: tuple,
                   scenarios: int, output: str):
    """Generate test cases using BLOOM-style LLM-driven pipeline.

    Uses a two-stage process: (1) deep understanding of each bias via LLM,
    then (2) diverse scenario generation with extractable answers.

    Examples:
        kahne-bench generate-bloom -p mock --bias anchoring_effect --scenarios 3

        kahne-bench generate-bloom -p openai -m gpt-5.2 --bias anchoring_effect --domain professional -n 5
    """
    from kahne_bench.engines.bloom_generator import BloomBiasGenerator
    from kahne_bench.utils.io import export_instances_to_json

    llm_provider, model_id = _create_provider(provider, model)
    console.print(f"[green]Using {provider} provider ({model_id}) for BLOOM generation[/green]")

    generator = BloomBiasGenerator(provider=llm_provider, num_scenarios=scenarios)

    bias_ids = list(bias) if bias else ["anchoring_effect"]
    domains = [Domain(d) for d in domain] if domain else [Domain.PROFESSIONAL]

    console.print("\n[bold]Generating BLOOM scenarios...[/bold]")
    console.print(f"  Biases: {', '.join(bias_ids)}")
    console.print(f"  Domains: {', '.join(d.value for d in domains)}")
    console.print(f"  Scenarios per pair: {scenarios}")

    async def run_generation():
        return await generator.generate_batch(
            bias_ids=bias_ids,
            domains=domains,
            scenarios_per_bias=scenarios,
        )

    instances = asyncio.run(run_generation())

    if not instances:
        console.print("[yellow]No instances generated (check bias IDs and provider)[/yellow]")
        sys.exit(1)

    export_instances_to_json(instances, output)
    console.print(f"\n[bold green]Generated {len(instances)} test instances -> {output}[/bold green]")


@main.command("evaluate-conversation")
@click.option("--input", "-i", "input_file", required=True, help="Input JSON file with test cases")
@click.option("--provider", "-p", "target_provider", type=click.Choice(PROVIDER_CHOICES), default="mock",
              help="LLM provider for target model")
@click.option("--model", "-m", "target_model", default=None, help="Target model to evaluate")
@click.option("--orchestrator-provider", type=click.Choice(PROVIDER_CHOICES), default=None,
              help="LLM provider for orchestrator (default: same as target)")
@click.option("--orchestrator-model", default=None, help="Model for orchestrator")
@click.option("--max-turns", default=8, callback=validate_positive, help="Max conversation turns")
@click.option("--output", "-o", default="transcripts.json", help="Output file for transcripts")
def evaluate_conversation(input_file: str, target_provider: str, target_model: str | None,
                          orchestrator_provider: str | None, orchestrator_model: str | None,
                          max_turns: int, output: str):
    """Evaluate bias persistence through multi-turn conversations.

    An orchestrator LLM plays the "user" role, probing the target model
    for cognitive bias across multiple conversation turns using strategies
    like probing, challenging, and reinforcing.

    Examples:
        kahne-bench evaluate-conversation -i test_cases.json -p mock

        kahne-bench evaluate-conversation -i test_cases.json -p openai -m gpt-5.2 \\
            --orchestrator-provider anthropic --orchestrator-model claude-opus-4-6
    """
    from kahne_bench.utils.io import import_instances_from_json, export_transcripts_to_json
    from kahne_bench.engines.conversation import ConversationalEvaluator

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

    target_llm, target_id = _create_provider(target_provider, target_model)
    console.print(f"[green]Target: {target_provider} ({target_id})[/green]")

    orch_prov = orchestrator_provider or target_provider
    orch_mod = orchestrator_model or target_model
    orchestrator_llm, orch_id = _create_provider(orch_prov, orch_mod)
    console.print(f"[green]Orchestrator: {orch_prov} ({orch_id})[/green]")

    conv_evaluator = ConversationalEvaluator(
        target_provider=target_llm,
        orchestrator_provider=orchestrator_llm,
        max_turns=max_turns,
    )

    console.print(f"\n[bold]Running conversational evaluation ({max_turns} turns max)...[/bold]")

    async def run_conversations():
        transcripts = []
        for i, instance in enumerate(instances):
            console.print(f"  [{i + 1}/{len(instances)}] {instance.bias_id}...")
            transcript = await conv_evaluator.evaluate_conversation(
                instance=instance,
                model_id=target_id,
            )
            transcripts.append(transcript)
        return transcripts

    transcripts = asyncio.run(run_conversations())

    export_transcripts_to_json(transcripts, output)

    console.print(f"\n[bold green]Saved {len(transcripts)} transcripts -> {output}[/bold green]")
    for t in transcripts[:3]:
        direction = "stable"
        if t.bias_evolution:
            if t.bias_evolution[-1] > t.bias_evolution[0] + 0.1:
                direction = "increasing"
            elif t.bias_evolution[-1] < t.bias_evolution[0] - 0.1:
                direction = "decreasing"
        console.print(f"  {t.bias_id}: persistence={t.persistence_score:.2f}, drift={direction}")


if __name__ == "__main__":
    main()
