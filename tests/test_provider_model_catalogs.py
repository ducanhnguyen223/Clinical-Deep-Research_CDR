"""Pinned model IDs reviewed against official provider catalogs on 2026-10-13."""

from inspect import signature
from pathlib import Path

from cdr.config import LLMSettings
from cdr.llm.cerebras_provider import CEREBRAS_MODELS, CerebrasProvider
from cdr.llm.cloudflare_provider import CLOUDFLARE_MODELS, CloudflareProvider
from cdr.llm.gemini_provider import GEMINI_MODELS, GeminiProvider
from cdr.llm.groq_provider import GROQ_MODELS, GroqProvider
from cdr.llm.huggingface_provider import RECOMMENDED_MODELS, HuggingFaceProvider
from cdr.llm.openai_provider import OpenAIProvider
from cdr.llm.openrouter_provider import OPENROUTER_MODELS, OpenRouterProvider


def test_recommended_model_ids_match_the_reviewed_provider_catalogs():
    assert GEMINI_MODELS == {
        "default": "gemini-3.8-flash",
        "fast": "gemini-3.5-flash-lite",
        "reasoning": "gemini-3.8-flash",
        "large": "gemini-3.8-flash",
    }
    assert CEREBRAS_MODELS == {
        "default": "gpt-oss-120b",
        "fast": "gpt-oss-120b",
        "reasoning": "qwen-3.8-27b",
        "large": "gpt-oss-120b",
    }
    assert GROQ_MODELS == {
        "default": "openai/gpt-oss-20b",
        "fast": "openai/gpt-oss-20b",
        "large": "openai/gpt-oss-120b",
        "reasoning": "openai/gpt-oss-120b",
        "medical": "openai/gpt-oss-120b",
    }
    assert OPENROUTER_MODELS == {
        "default": "meta-llama/llama-3.1-8b-instruct",
        "fast": "meta-llama/llama-3.1-8b-instruct",
        "large": "meta-llama/llama-3.1-70b-instruct",
        "reasoning": "deepseek/deepseek-r1",
        "premium": "anthropic/claude-sonnet-4.6",
        "gpt": "openai/gpt-4o-mini",
    }
    assert CLOUDFLARE_MODELS == {
        "default": "@cf/meta/llama-3.2-3b-instruct",
        "fast": "@cf/meta/llama-3.1-8b-instruct-fast",
        "large": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "reasoning": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "qwen": "@cf/qwen/qwen3-30b-a3b-fp8",
    }
    assert RECOMMENDED_MODELS == {
        "default": "Qwen/Qwen2.5-72B-Instruct",
        "fast": "google/gemma-3-27b-it",
        "large": "meta-llama/Llama-3.3-70B-Instruct",
        "medical": "Qwen/Qwen2.5-72B-Instruct",
        "coding": "meta-llama/Llama-3.3-70B-Instruct",
        "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "qwq": "Qwen/QwQ-32B",
        "llama70b": "meta-llama/Llama-3.3-70B-Instruct",
        "qwen72b": "Qwen/Qwen2.5-72B-Instruct",
        "gemma": "google/gemma-3-27b-it",
    }


def test_settings_and_provider_constructors_share_the_documented_defaults():
    settings = LLMSettings(_env_file=None)

    assert settings.gemini_model == GEMINI_MODELS["default"]
    assert settings.groq_model == GROQ_MODELS["default"]
    assert settings.hf_model == RECOMMENDED_MODELS["default"]
    assert settings.openai_model == "gpt-4o"

    assert signature(GeminiProvider).parameters["model"].default == GEMINI_MODELS["default"]
    assert signature(CerebrasProvider).parameters["model"].default == CEREBRAS_MODELS["default"]
    assert signature(GroqProvider).parameters["model"].default == GROQ_MODELS["default"]
    assert signature(CloudflareProvider).parameters["model"].default == CLOUDFLARE_MODELS["default"]
    assert signature(OpenRouterProvider).parameters["model"].default == OPENROUTER_MODELS["default"]
    assert (
        signature(HuggingFaceProvider).parameters["model"].default == RECOMMENDED_MODELS["default"]
    )
    assert signature(OpenAIProvider).parameters["model"].default == settings.openai_model


def test_provider_guide_lists_each_audited_default():
    guide = (Path(__file__).parents[1] / "docs" / "providers.md").read_text()
    defaults = (
        GEMINI_MODELS["default"],
        GROQ_MODELS["default"],
        CEREBRAS_MODELS["default"],
        OPENROUTER_MODELS["default"],
        CLOUDFLARE_MODELS["default"],
        RECOMMENDED_MODELS["default"],
    )

    for model_id in defaults:
        assert f"`{model_id}`" in guide
