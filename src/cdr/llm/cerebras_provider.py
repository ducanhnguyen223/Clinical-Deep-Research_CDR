"""
Cerebras LLM Provider

Integration with Cerebras Inference API via OpenAI-compatible endpoint.

Base URL: https://api.cerebras.ai/v1
Uses standard OpenAI SDK with modified base_url.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from cdr.core.exceptions import LLMError, LLMProviderError, LLMRateLimitError
from cdr.llm.base import BaseLLMProvider, LLMResponse, Message, StreamChunk

try:
    import openai
    from openai import AsyncOpenAI, OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Retry configuration for rate limits
MAX_RETRIES = 10
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds


# Cerebras Shared Inference model presets.
CEREBRAS_MODELS = {
    "default": "qwen-3.8-27b",
    "fast": "qwen-3.8-27b",
    "reasoning": "gpt-oss-120b",
    "qwen": "qwen-3.8-27b",
    "large": "gpt-oss-120b",
}


class CerebrasProvider(BaseLLMProvider):
    """Cerebras LLM provider via the OpenAI-compatible API.

    Cerebras provides extremely fast inference on their custom LPU chips.
    OpenAI-compatible API endpoint for easy integration.

    API Key: Get from https://cloud.cerebras.ai/
    """

    CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

    def __init__(
        self,
        model: str = CEREBRAS_MODELS["default"],
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize Cerebras provider.

        Args:
            model: Cerebras Shared Inference model ID.
            api_key: Cerebras API key. Falls back to CEREBRAS_API_KEY env var.
            timeout: Request timeout.
            max_retries: Maximum retries on transient errors.
        """
        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                provider="cerebras",
                message="OpenAI package not installed. Run: pip install openai",
            )

        # Get API key from env if not provided
        api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise LLMProviderError(
                provider="cerebras",
                message="Cerebras API key required. Set CEREBRAS_API_KEY env var.",
            )

        super().__init__(model, api_key, self.CEREBRAS_BASE_URL, timeout, max_retries)

        # Initialize OpenAI-compatible clients with Cerebras base URL
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.CEREBRAS_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.CEREBRAS_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return "cerebras"

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove unsupported kwargs for Cerebras
        kwargs.pop("response_format", None)
        kwargs.pop("model", None)
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        # Cerebras doesn't support these
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)
        kwargs.pop("logit_bias", None)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[m.to_dict() for m in normalized],
                    temperature=temperature,
                    max_completion_tokens=max_tokens or 4096,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider="cerebras",
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                )
            except openai.RateLimitError as e:
                last_error = e
                error_msg = str(e)

                # Extract wait time if available
                import re

                wait_match = re.search(r"try again in (\d+)", error_msg)
                if wait_match:
                    wait_seconds = int(wait_match.group(1))
                    print(f"[Cerebras] Rate limit hit. Waiting {wait_seconds + 5} seconds...")
                    time.sleep(wait_seconds + 5)
                else:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                    print(
                        f"[Cerebras] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Cerebras API error: {e}") from e

        raise LLMRateLimitError(provider="cerebras") from last_error

    async def acomplete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion with automatic retry for rate limits."""
        normalized = self._normalize_messages(messages)

        # Remove unsupported kwargs
        kwargs.pop("response_format", None)
        kwargs.pop("model", None)
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)
        kwargs.pop("logit_bias", None)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._async_client.chat.completions.create(
                    model=self._model,
                    messages=[m.to_dict() for m in normalized],
                    temperature=temperature,
                    max_completion_tokens=max_tokens or 4096,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider="cerebras",
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                )
            except openai.RateLimitError as e:
                last_error = e
                error_msg = str(e)

                import re

                wait_match = re.search(r"try again in (\d+)", error_msg)
                if wait_match:
                    wait_seconds = int(wait_match.group(1))
                    print(f"[Cerebras] Rate limit hit. Waiting {wait_seconds + 5} seconds...")
                    await asyncio.sleep(wait_seconds + 5)
                else:
                    delay = min(BASE_DELAY * (2**attempt) + random.uniform(0, 1), MAX_DELAY)
                    print(
                        f"[Cerebras] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
            except openai.APIError as e:
                raise LLMError(f"Cerebras API error: {e}") from e

        raise LLMRateLimitError(provider="cerebras") from last_error

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Streaming completion."""
        normalized = self._normalize_messages(messages)
        kwargs.pop("response_format", None)
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in normalized],
                temperature=temperature,
                max_completion_tokens=max_tokens or 4096,
                stream=True,
                **kwargs,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
        except openai.RateLimitError as e:
            raise LLMRateLimitError(provider="cerebras") from e
        except openai.APIError as e:
            raise LLMError(f"Cerebras API error: {e}") from e

    async def astream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming completion."""
        normalized = self._normalize_messages(messages)
        kwargs.pop("response_format", None)
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)

        try:
            stream = await self._async_client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in normalized],
                temperature=temperature,
                max_completion_tokens=max_tokens or 4096,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
        except openai.RateLimitError as e:
            raise LLMRateLimitError(provider="cerebras") from e
        except openai.APIError as e:
            raise LLMError(f"Cerebras API error: {e}") from e
