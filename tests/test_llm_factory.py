"""Tests for provider selection in cdr.llm.factory."""

from importlib import import_module
from inspect import signature

import pytest

from cdr.config import LLMSettings, reset_settings
from cdr.core.exceptions import ConfigurationError
from cdr.llm.cerebras_provider import CEREBRAS_MODELS, CerebrasProvider
from cdr.llm.cloudflare_provider import CLOUDFLARE_MODELS, CloudflareProvider
from cdr.llm.factory import create_provider
from cdr.llm.gemini_provider import GEMINI_MODELS, GeminiProvider
from cdr.llm.groq_provider import GROQ_MODELS, GroqProvider
from cdr.llm.huggingface_provider import RECOMMENDED_MODELS, HuggingFaceProvider
from cdr.llm.openai_provider import OpenAIProvider
from cdr.llm.openrouter_provider import OPENROUTER_MODELS, OpenRouterProvider

PROVIDER_ENV_VARS = [
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "CEREBRAS_API_KEY",
    "OPENROUTER_API_KEY",
    "CLOUDFLARE_API_KEY",
    "CLOUDFLARE_ACCOUNT_ID",
    "HF_TOKEN",
    "HF_ENDPOINT_URL",
    "HF_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "LLM_DEFAULT_PROVIDER",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Isolate each test from the developer's real .env and environment."""
    monkeypatch.chdir(tmp_path)  # settings read .env from the working directory
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_settings()
    yield
    reset_settings()


def test_openai_without_key_or_base_url_is_a_config_error():
    with pytest.raises(ConfigurationError, match="OPENAI_BASE_URL"):
        create_provider("openai")


def test_openai_base_url_alone_is_enough_for_local_servers(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_MODEL", "llama3.1:8b")
    reset_settings()

    provider = create_provider("openai")

    assert provider.model == "llama3.1:8b"
    assert str(provider._client.base_url).startswith("http://localhost:11434/v1")


def test_explicit_key_is_kept_with_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    reset_settings()

    provider = create_provider("openai")

    assert provider._client.api_key == "sk-test"


def test_provider_defaults_match_config_and_model_maps():
    mapped_providers = [
        (GeminiProvider, GEMINI_MODELS),
        (CerebrasProvider, CEREBRAS_MODELS),
        (CloudflareProvider, CLOUDFLARE_MODELS),
        (GroqProvider, GROQ_MODELS),
        (HuggingFaceProvider, RECOMMENDED_MODELS),
        (OpenRouterProvider, OPENROUTER_MODELS),
    ]
    for provider, models in mapped_providers:
        assert signature(provider).parameters["model"].default == models["default"]

    config_defaults = {name: field.default for name, field in LLMSettings.model_fields.items()}
    assert config_defaults["gemini_model"] == GEMINI_MODELS["default"]
    assert config_defaults["groq_model"] == GROQ_MODELS["default"]
    assert config_defaults["hf_model"] == RECOMMENDED_MODELS["default"]
    assert config_defaults["openai_model"] == signature(OpenAIProvider).parameters["model"].default


@pytest.mark.parametrize(
    ("name", "key_env", "module_name", "class_name", "expected_model", "extra_env"),
    [
        (
            "gemini",
            "GEMINI_API_KEY",
            "gemini_provider",
            "GeminiProvider",
            GEMINI_MODELS["default"],
            {},
        ),
        (
            "cerebras",
            "CEREBRAS_API_KEY",
            "cerebras_provider",
            "CerebrasProvider",
            CEREBRAS_MODELS["default"],
            {},
        ),
        (
            "cloudflare",
            "CLOUDFLARE_API_KEY",
            "cloudflare_provider",
            "CloudflareProvider",
            CLOUDFLARE_MODELS["default"],
            {"CLOUDFLARE_ACCOUNT_ID": "test-account"},
        ),
        ("groq", "GROQ_API_KEY", "groq_provider", "GroqProvider", GROQ_MODELS["default"], {}),
        (
            "huggingface",
            "HF_TOKEN",
            "huggingface_provider",
            "HuggingFaceProvider",
            RECOMMENDED_MODELS["default"],
            {},
        ),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "openrouter_provider",
            "OpenRouterProvider",
            OPENROUTER_MODELS["default"],
            {},
        ),
    ],
)
def test_factory_uses_each_provider_default(
    monkeypatch, name, key_env, module_name, class_name, expected_model, extra_env
):
    class ModelCapture:
        def __init__(self, model, **kwargs):
            self.model = model

    monkeypatch.setenv(key_env, "test-key")
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)
    module = import_module(f"cdr.llm.{module_name}")
    monkeypatch.setattr(module, class_name, ModelCapture)
    reset_settings()

    provider = create_provider(name)

    assert provider.model == expected_model


@pytest.mark.parametrize(
    ("name", "key_env", "model_env", "module_name", "class_name"),
    [
        ("gemini", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini_provider", "GeminiProvider"),
        ("huggingface", "HF_TOKEN", "HF_MODEL", "huggingface_provider", "HuggingFaceProvider"),
    ],
)
def test_factory_uses_configured_model(
    monkeypatch, name, key_env, model_env, module_name, class_name
):
    class ModelCapture:
        def __init__(self, model, **kwargs):
            self.model = model

    configured_model = f"configured-{name}-model"
    monkeypatch.setenv(key_env, "test-key")
    monkeypatch.setenv(model_env, configured_model)
    module = import_module(f"cdr.llm.{module_name}")
    monkeypatch.setattr(module, class_name, ModelCapture)
    reset_settings()

    provider = create_provider(name)

    assert provider.model == configured_model
