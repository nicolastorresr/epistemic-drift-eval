"""
run_full_eval.py
Run epistemic drift evaluation across all models and benchmark domains
defined in a YAML config file.

Usage:
    python experiments/run_full_eval.py --config configs/default.yaml
    python experiments/run_full_eval.py --config configs/default.yaml --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from rich.console import Console
from rich.table import Table

from framework.pipeline import pipeline_from_yaml
from utils.io_utils import load_all_benchmarks

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run full epistemic drift evaluation")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Load config and print plan without running")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per domain")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    benchmark_cfg = cfg.get("benchmark", {})
    domains = benchmark_cfg.get("domains", None)
    max_per_domain = args.max_samples or benchmark_cfg.get("max_samples_per_domain", None)

    benchmark_dir = "data/benchmarks"
    queries = load_all_benchmarks(
        benchmark_dir=benchmark_dir,
        domains=domains,
        max_per_domain=max_per_domain,
    )

    console.print(f"\n[bold cyan]Epistemic Drift Evaluation[/bold cyan]")
    console.print(f"  Config   : {args.config}")
    console.print(f"  Queries  : {len(queries)}")
    console.print(f"  Domains  : {domains or 'all'}")
    console.print(f"  Models   : {[m['name'] for m in cfg.get('models', [])]}\n")

    if args.dry_run:
        console.print("[yellow]Dry-run mode — exiting without evaluation.[/yellow]")
        return

    pipelines = pipeline_from_yaml(args.config)
    all_summaries = []

    for pipeline in pipelines:
        console.print(f"\n[bold]Running: {pipeline.config.model}[/bold]")
        summary = pipeline.run(
            benchmark=queries if isinstance(queries[0], dict) else benchmark_dir,
            max_samples=max_per_domain,
        )
        all_summaries.append(summary)
        console.print(str(summary))

    # Print comparison table
    if len(all_summaries) > 1:
        table = Table(title="Model Comparison Summary")
        table.add_column("Model", style="cyan")
        table.add_column("ESS ↑", justify="right")
        table.add_column("CCR ↓", justify="right")
        table.add_column("FRR ↑", justify="right")
        table.add_column("CD ↓", justify="right")
        table.add_column("Reversals", justify="right")

        for s in all_summaries:
            table.add_row(
                s.model_name,
                f"{s.ess:.4f}",
                f"{s.ccr:.4f}",
                f"{s.frr:.4f}",
                f"{s.cd:.4f}",
                str(s.n_reversals),
            )
        console.print(table)


if __name__ == "__main__":
    main()
