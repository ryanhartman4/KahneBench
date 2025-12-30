"""
Command-line interface for Kahne-Bench.

Provides commands for generating test cases, running evaluations,
and computing metrics.
"""

import asyncio
import json
import sys
from pathlib import Path

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
    """List all 50 biases in the taxonomy."""
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
            BiasCategory.PROBABILITY_DISTORTION: "Misweighting probabilities",
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
@click.option("--instances", "-n", default=3, help="Instances per bias-domain pair")
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


if __name__ == "__main__":
    main()
