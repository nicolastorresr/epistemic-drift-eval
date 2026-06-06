"""
confidence_extractor.py
Extracts self-reported confidence scores from model responses
using a prompted secondary query.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

CONFIDENCE_PROMPT = (
    "On a scale from 0 to 100, how confident are you in your last answer? "
    "Reply with only a single integer number and nothing else."
)


class ConfidenceExtractor:
    """
    Extracts a confidence score [0, 1] from a language model.

    Strategy 'prompted': appends a meta-question asking for a 0-100
    confidence rating and parses the integer from the response.
    """

    def extract_prompted(
        self,
        messages: list[dict],
        system: str,
        client,
        prompt: str = CONFIDENCE_PROMPT,
    ) -> Optional[float]:
        """
        Ask the model to rate its confidence after its last response.

        Returns a float in [0, 1] or None if extraction fails.
        """
        confidence_messages = messages + [{"role": "user", "content": prompt}]

        try:
            response_text = client.generate(
                messages=confidence_messages,
                system=system,
                temperature=0.0,
                max_tokens=10,
            )
            return self._parse_confidence(response_text)
        except Exception as exc:
            logger.warning("Confidence extraction failed: %s", exc)
            return None

    @staticmethod
    def _parse_confidence(text: str) -> Optional[float]:
        """Parse a 0-100 integer from model output and normalize to [0, 1]."""
        text = text.strip()
        # Match integer or float at start of string
        match = re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", text)
        if match:
            value = float(match.group(1))
            value = max(0.0, min(100.0, value))  # clamp
            return value / 100.0
        logger.debug("Could not parse confidence from: %r", text)
        return None
