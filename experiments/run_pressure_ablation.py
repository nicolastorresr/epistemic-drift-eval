"""
run_pressure_ablation.py
Evaluate one model across each pressure category independently
to measure which type induces the most epistemic drift.

Usage:
    python experiments/run_pressure_ablation.py \
        --model gpt-4o \
        --provider openai \
        --turns 5 \
        --max-samples 30
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from framework.evaluator import EvaluationConfig
from framework.pipeline import EpistemicDriftPipeline
from pressure.sampler import PRESSURE_CATEGORIES
from utils.io_utils import load_all_benchmarks

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Pressure ablation study")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--output", default="data/results/ablation")
    args = parser.parse_args()

    queries = load_all_benchmarks(max_per_domain=args.max_samples)

    console.print(f"\n[bold cyan]Pressure Ablation Study — {args.model}[/bold cyan]\n")

    results = []
    for category in PRESSURE_CATEGORIES:
        console.print(f"  Running pressure: [yellow]{category}[/yellow]")

        config = EvaluationConfig(
            model=args.model,
            provider=args.provider,
            pressure_types=[category],
            max_turns=args.turns,
        )
        pipeline = EpistemicDriftPipeline(
            config=config,
            output_dir=f"{args.output}/{category}",
        )
        summary = pipeline.run(benchmark=queries)
        results.append((category, summary))

    # Results table
    table = Table(title=f"Ablation Results — {args.model}")
    table.add_column("Pressure Type", style="cyan")
    table.add_column("CCR ↓", justify="right", style="red")
    table.add_column("ESS ↑", justify="right", style="green")
    table.add_column("FRR ↑", justify="right")
    table.add_column("CD ↓", justify="right")
    table.add_column("Reversals", justify="right")

    results.sort(key=lambda x: x[1].ccr, reverse=True)
    for category, s in results:
        table.add_row(
            category,
            f"{s.ccr:.4f}",
            f"{s.ess:.4f}",
            f"{s.frr:.4f}",
            f"{s.cd:.4f}",
            str(s.n_reversals),
        )

    console.print(table)


if __name__ == "__main__":
    main()
