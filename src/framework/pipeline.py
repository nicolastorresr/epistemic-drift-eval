"""
pipeline.py
End-to-end pipeline: loads config, benchmark data, runs evaluations,
saves results, and returns aggregated metrics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jsonlines
import yaml
from tqdm import tqdm

from framework.conversation import ConversationTrajectory
from framework.evaluator import EpistemicDriftEvaluator, EvaluationConfig
from metrics.aggregator import MetricsAggregator, EvaluationSummary
from pressure.sampler import PressureSampler
from evaluation.factual_checker import FactualChecker
from evaluation.confidence_extractor import ConfidenceExtractor
from utils.api_client import build_client
from utils.io_utils import load_benchmark, save_trajectory

logger = logging.getLogger(__name__)


class EpistemicDriftPipeline:
    """
    High-level pipeline for running epistemic drift evaluations.

    Usage:
        config = EvaluationConfig(model="gpt-4o", ...)
        pipeline = EpistemicDriftPipeline(config)
        summary = pipeline.run(benchmark="data/benchmarks/science.jsonl")
        print(summary)
    """

    def __init__(self, config: EvaluationConfig, output_dir: str = "data/results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = build_client(
            model=config.model,
            provider=config.provider,
        )
        self.pressure_sampler = PressureSampler.from_yaml("configs/pressure.yaml")
        self.factual_checker = FactualChecker(
            method=config.factual_method,
            semantic_threshold=config.semantic_threshold,
        )
        self.confidence_extractor = ConfidenceExtractor()
        self.evaluator = EpistemicDriftEvaluator(
            config=config,
            client=self.client,
            pressure_sampler=self.pressure_sampler,
            factual_checker=self.factual_checker,
            confidence_extractor=self.confidence_extractor,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        benchmark: str | list[str],
        max_samples: Optional[int] = None,
    ) -> EvaluationSummary:
        """
        Run the full evaluation pipeline.

        Args:
            benchmark: Path(s) to JSONL benchmark file(s).
            max_samples: Optional cap on number of queries evaluated.

        Returns:
            EvaluationSummary with all metrics.
        """
        queries = self._load_queries(benchmark, max_samples)
        trajectories: list[ConversationTrajectory] = []

        output_path = self.output_dir / f"{self.config.model.replace('/', '_')}_{int(time.time())}.jsonl"

        logger.info(
            "Starting evaluation | model=%s | queries=%d | pressure_turns=%d",
            self.config.model, len(queries), self.config.max_turns
        )

        with jsonlines.open(output_path, mode="w") as writer:
            for query in tqdm(queries, desc=f"Evaluating {self.config.model}"):
                try:
                    trajectory = self.evaluator.evaluate_query(query)
                    if trajectory is not None:
                        trajectories.append(trajectory)
                        writer.write(trajectory.to_dict())
                except Exception as exc:
                    logger.error("Error on query %s: %s", query.get("id"), exc, exc_info=True)
                    continue

        logger.info("Evaluation complete. Saved to %s", output_path)

        aggregator = MetricsAggregator(trajectories)
        summary = aggregator.compute()
        self._save_summary(summary, output_path)
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_queries(
        self, benchmark: str | list[str], max_samples: Optional[int]
    ) -> list[dict]:
        if isinstance(benchmark, str):
            benchmark = [benchmark]

        queries: list[dict] = []
        for path in benchmark:
            queries.extend(load_benchmark(path))

        if max_samples:
            queries = queries[:max_samples]
        return queries

    def _save_summary(self, summary: EvaluationSummary, results_path: Path) -> None:
        summary_path = results_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info("Summary saved to %s", summary_path)


# ------------------------------------------------------------------
# Config loader utility
# ------------------------------------------------------------------

def pipeline_from_yaml(config_path: str) -> list[EpistemicDriftPipeline]:
    """Instantiate one pipeline per model defined in a YAML config file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pipelines = []
    for model_cfg in cfg.get("models", []):
        eval_config = EvaluationConfig(
            model=model_cfg["name"],
            provider=model_cfg.get("provider", "openai"),
            temperature=model_cfg.get("temperature", 0.7),
            top_p=model_cfg.get("top_p", 0.9),
            max_tokens=model_cfg.get("max_tokens", 256),
            pressure_types=cfg.get("pressure", {}).get("types", [
                "authority", "majority", "emotional",
                "iterative_contradiction", "social_validation", "confidence_manipulation"
            ]),
            max_turns=cfg.get("pressure", {}).get("max_turns", 5),
            stop_on_reversal=cfg.get("pressure", {}).get("stop_on_reversal", False),
            factual_method=cfg.get("evaluation", {}).get("factual_method", "hybrid"),
            semantic_threshold=cfg.get("evaluation", {}).get("semantic_threshold", 0.85),
            confidence_extraction=cfg.get("evaluation", {}).get("confidence_extraction", "prompted"),
        )
        pipeline = EpistemicDriftPipeline(
            config=eval_config,
            output_dir=cfg.get("experiment", {}).get("output_dir", "data/results"),
        )
        pipelines.append(pipeline)

    return pipelines
