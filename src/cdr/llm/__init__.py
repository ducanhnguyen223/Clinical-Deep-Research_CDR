"""
CDR LLM Abstraction Layer

Unified interface for multiple LLM providers.

Providers can be hosted services or local OpenAI-compatible servers. Availability,
pricing, and quotas depend on the provider, model, and account.
"""

from cdr.llm.base import (
    BaseLLMProvider,
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    build_messages,
)
from cdr.llm.factory import (
    create_provider,
    create_provider_with_fallback,
    get_anthropic,
    get_cerebras,
    get_cloudflare,
    get_default_provider,
    get_gemini,
    get_groq,
    get_huggingface,
    get_openai,
    get_openrouter,
)

__all__ = [
    # Base
    "BaseLLMProvider",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "StreamChunk",
    "build_messages",
    # Factory
    "create_provider",
    "create_provider_with_fallback",
    "get_default_provider",
    # Hosted providers
    "get_gemini",
    "get_cerebras",
    "get_cloudflare",
    "get_openrouter",
    "get_groq",
    "get_huggingface",
    "get_openai",
    "get_anthropic",
]
