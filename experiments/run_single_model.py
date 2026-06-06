"""
run_single_model.py
Quick evaluation of a single model with configurable options.

Usage:
    python experiments/run_single_model.py --model gpt-4o --provider openai
    python experiments/run_single_model.py \
        --model claude-3-5-sonnet-20241022 \
        --provider anthropic \
        --pressure authority majority \
        --turns 3 \
        --domain science \
        --max-samples 20
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from framework.evaluator import EvaluationConfig
from framework.pipeline import EpistemicDriftPipeline
from utils.io_utils import load_all_benchmarks

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single model for epistemic drift")
    parser.add_argument("--model", required=True, help="Model name (e.g. gpt-4o)")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "huggingface"])
    parser.add_argument("--pressure", nargs="+",
                        default=["authority", "majority", "emotional",
                                 "iterative_contradiction", "social_validation", "confidence_manipulation"],
                        help="Pressure types to apply")
    parser.add_argument("--turns", type=int, default=5, help="Max pressure turns")
    parser.add_argument("--domain", nargs="+", default=None, help="Benchmark domain(s) to use")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per domain")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="data/results", help="Output directory")
    parser.add_argument("--stop-on-reversal", action="store_true")
    args = parser.parse_args()

    config = EvaluationConfig(
        model=args.model,
        provider=args.provider,
        pressure_types=args.pressure,
        max_turns=args.turns,
        temperature=args.temperature,
        stop_on_reversal=args.stop_on_reversal,
    )

    queries = load_all_benchmarks(
        benchmark_dir="data/benchmarks",
        domains=args.domain,
        max_per_domain=args.max_samples,
    )

    console.print(f"\n[bold cyan]Single Model Evaluation[/bold cyan]")
    console.print(f"  Model    : {args.model}")
    console.print(f"  Provider : {args.provider}")
    console.print(f"  Pressure : {args.pressure}")
    console.print(f"  Turns    : {args.turns}")
    console.print(f"  Queries  : {len(queries)}\n")

    pipeline = EpistemicDriftPipeline(config=config, output_dir=args.output)
    summary = pipeline.run(benchmark=queries)

    console.print(str(summary))


if __name__ == "__main__":
    main()
