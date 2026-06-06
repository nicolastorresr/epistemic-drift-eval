"""
api_client.py
Unified LLM client interface with support for OpenAI and Anthropic APIs.
Uses tenacity for automatic retry with exponential backoff.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------

class BaseModelClient(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> str:
        """Generate a response given a message history."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# ------------------------------------------------------------------
# OpenAI client
# ------------------------------------------------------------------

class OpenAIClient(BaseModelClient):

    def __init__(self, model: str, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required: pip install openai")

        self._model = model
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def generate(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> str:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


# ------------------------------------------------------------------
# Anthropic client
# ------------------------------------------------------------------

class AnthropicClient(BaseModelClient):

    def __init__(self, model: str, api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required: pip install anthropic")

        self._model = model
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def generate(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> str:
        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text.strip()


# ------------------------------------------------------------------
# HuggingFace local client
# ------------------------------------------------------------------

class HuggingFaceClient(BaseModelClient):
    """Runs a local HuggingFace model via the transformers pipeline."""

    def __init__(self, model: str, device: str = "auto"):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("transformers package is required: pip install transformers")

        self._model = model
        self._pipe = hf_pipeline(
            "text-generation",
            model=model,
            device_map=device,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> str:
        prompt_messages = []
        if system:
            prompt_messages.append({"role": "system", "content": system})
        prompt_messages.extend(messages)

        output = self._pipe(
            prompt_messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        return output[0]["generated_text"][-1]["content"].strip()


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def build_client(model: str, provider: str) -> BaseModelClient:
    """Instantiate the appropriate client based on provider name."""
    provider = provider.lower()

    if provider == "openai":
        return OpenAIClient(model=model)
    elif provider == "anthropic":
        return AnthropicClient(model=model)
    elif provider in ("huggingface", "hf", "local"):
        return HuggingFaceClient(model=model)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            "Supported: openai, anthropic, huggingface"
        )
