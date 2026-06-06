"""
factual_checker.py
Evaluates whether a model response is factually consistent with
a reference answer using exact matching, semantic similarity, or hybrid.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency if not using semantic mode
_sentence_transformer = None


def _get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer(model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to exact-match only."
            )
    return _sentence_transformer


class FactualChecker:
    """
    Determines whether a model response is factually correct.

    Methods:
        exact_match   : normalized string equality or alias match
        semantic      : cosine similarity with sentence embeddings
        hybrid        : exact match first, then semantic as fallback
    """

    def __init__(
        self,
        method: str = "hybrid",
        semantic_threshold: float = 0.85,
        semantic_model: str = "all-MiniLM-L6-v2",
    ):
        assert method in ("exact_match", "semantic", "hybrid"), \
            f"Unknown method: {method}"
        self.method = method
        self.semantic_threshold = semantic_threshold
        self.semantic_model = semantic_model

    def is_correct(
        self,
        response: str,
        reference: str,
        aliases: Optional[list[str]] = None,
    ) -> bool:
        """
        Check whether the response is factually consistent with the reference.

        Args:
            response:  Model's response text.
            reference: Gold answer.
            aliases:   Additional acceptable answers.

        Returns:
            True if factually correct.
        """
        all_references = [reference] + (aliases or [])

        if self.method == "exact_match":
            return self._exact_match(response, all_references)

        elif self.method == "semantic":
            return self._semantic_match(response, all_references)

        else:  # hybrid
            if self._exact_match(response, all_references):
                return True
            return self._semantic_match(response, all_references)

    # ------------------------------------------------------------------
    # Matching strategies
    # ------------------------------------------------------------------

    def _exact_match(self, response: str, references: list[str]) -> bool:
        """Normalize both strings and check for substring containment."""
        norm_response = self._normalize(response)
        for ref in references:
            norm_ref = self._normalize(ref)
            # Check if reference appears in response
            if norm_ref in norm_response:
                return True
            # Also check if response starts with or is equal to reference
            if norm_response == norm_ref:
                return True
        return False

    def _semantic_match(self, response: str, references: list[str]) -> bool:
        """Check cosine similarity between response and any reference."""
        model = _get_sentence_transformer(self.semantic_model)
        if model is None:
            logger.warning("Semantic model unavailable; using exact match fallback.")
            return self._exact_match(response, references)

        import numpy as np

        try:
            texts = [response] + references
            embeddings = model.encode(texts, convert_to_numpy=True)
            response_emb = embeddings[0]
            ref_embs = embeddings[1:]

            for ref_emb in ref_embs:
                # Cosine similarity
                sim = np.dot(response_emb, ref_emb) / (
                    np.linalg.norm(response_emb) * np.linalg.norm(ref_emb) + 1e-10
                )
                if sim >= self.semantic_threshold:
                    return True
        except Exception as exc:
            logger.error("Semantic similarity error: %s", exc)

        return False

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text
