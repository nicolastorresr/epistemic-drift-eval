"""
sampler.py
Loads pressure prompts from config and samples sequences for evaluation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import yaml


PRESSURE_CATEGORIES = [
    "authority",
    "majority",
    "emotional",
    "iterative_contradiction",
    "social_validation",
    "confidence_manipulation",
]


class PressureSampler:
    """
    Manages pressure prompt templates and samples sequences for evaluation.

    Attributes:
        prompts: dict mapping category → intensity → list of prompt strings
    """

    def __init__(self, prompts: dict, seed: Optional[int] = 42):
        self.prompts = prompts
        self.rng = random.Random(seed)

    @classmethod
    def from_yaml(cls, path: str, seed: int = 42) -> "PressureSampler":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(prompts=cfg.get("pressure_prompts", {}), seed=seed)

    def sample_sequence(
        self,
        pressure_types: list[str],
        max_turns: int,
        intensity_levels: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Sample a sequence of pressure prompts for a conversation trajectory.

        Returns a list of dicts with keys: text, category, intensity.
        """
        if intensity_levels is None:
            intensity_levels = ["low", "medium", "high"]

        sequence = []
        for turn_idx in range(max_turns):
            # Cycle through pressure types, escalating intensity
            category = pressure_types[turn_idx % len(pressure_types)]
            intensity = intensity_levels[min(turn_idx, len(intensity_levels) - 1)]

            prompt_text = self._sample_one(category, intensity)
            sequence.append({
                "text": prompt_text,
                "category": category,
                "intensity": intensity,
            })
        return sequence

    def sample_single(
        self, category: str, intensity: str = "medium"
    ) -> dict:
        """Sample a single pressure prompt."""
        text = self._sample_one(category, intensity)
        return {"text": text, "category": category, "intensity": intensity}

    def _sample_one(self, category: str, intensity: str) -> str:
        """Sample one prompt from a given category and intensity level."""
        cat_prompts = self.prompts.get(category, {})
        level_prompts = cat_prompts.get(intensity, [])

        if not level_prompts:
            # Fall back to medium if requested intensity not found
            level_prompts = cat_prompts.get("medium", [])
        if not level_prompts:
            # Final fallback: any prompt from the category
            all_prompts = []
            for lvl_prompts in cat_prompts.values():
                if isinstance(lvl_prompts, list):
                    all_prompts.extend(lvl_prompts)
            level_prompts = all_prompts or [f"That answer seems incorrect. ({category})"]

        return self.rng.choice(level_prompts)

    def list_categories(self) -> list[str]:
        return list(self.prompts.keys())

    def get_all_prompts(self, category: str) -> list[str]:
        """Return all prompts for a category across all intensities."""
        result = []
        for level_prompts in self.prompts.get(category, {}).values():
            if isinstance(level_prompts, list):
                result.extend(level_prompts)
        return result
