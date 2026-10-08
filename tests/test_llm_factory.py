"""Tests for provider selection in cdr.llm.factory."""

from inspect import signature

import pytest

from cdr.config import LLMSettings, reset_settings
from cdr.core.exceptions import ConfigurationError
from cdr.llm.base import Message
from cdr.llm.cerebras_provider import CEREBRAS_MODELS, CerebrasProvider
from cdr.llm.cloudflare_provider import CLOUDFLARE_MODELS, CloudflareProvider
from cdr.llm.factory import create_provider
from cdr.llm.gemini_provider import GEMINI_MODELS, GeminiProvider
from cdr.llm.groq_provider import GROQ_MODELS, GroqProvider
from cdr.llm.huggingface_provider import RECOMMENDED_MODELS
from cdr.llm.openrouter_provider import OPENROUTER_MODELS, OpenRouterProvider

PROVIDER_ENV_VARS = [
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
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


def test_provider_constructor_defaults_match_model_maps():
    for provider, models in (
        (CerebrasProvider, CEREBRAS_MODELS),
        (CloudflareProvider, CLOUDFLARE_MODELS),
        (GeminiProvider, GEMINI_MODELS),
        (GroqProvider, GROQ_MODELS),
        (OpenRouterProvider, OPENROUTER_MODELS),
    ):
        assert signature(provider).parameters["model"].default == models["default"]

    assert LLMSettings.model_fields["groq_model"].default == GROQ_MODELS["default"]
    assert LLMSettings.model_fields["hf_model"].default == RECOMMENDED_MODELS["default"]
    assert LLMSettings.model_fields["gemini_model"].default == GEMINI_MODELS["default"]
    assert set(CEREBRAS_MODELS.values()) <= {"qwen-3.8-27b", "gpt-oss-120b"}
    assert set(GROQ_MODELS.values()) <= {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
    assert CLOUDFLARE_MODELS["fast"] == "@cf/openai/gpt-oss-20b"
    assert GEMINI_MODELS["fast"] == "gemini-3.5-flash-lite"
    assert OPENROUTER_MODELS["premium"] == "anthropic/claude-sonnet-5.5"


@pytest.mark.parametrize(
    ("model", "supports_sampling_parameters"),
    [
        ("gemini-3.8-flash", False),
        ("gemini-3.5-flash-lite", False),
        ("gemini-3.1-pro-preview", False),
        ("gemini-2.5-flash", True),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_gemini_request_args_match_model_generation_api(
    model, supports_sampling_parameters, stream
):
    provider = GeminiProvider(model=model, api_key="test-key")

    args = provider._request_args(
        [Message(role="user", content="hello")],
        temperature=0.4,
        max_tokens=256,
        kwargs={"top_p": 0.8, "top_k": 20, "candidate_count": 2, "reasoning_effort": "low"},
        stream=stream,
    )

    assert args["model"] == model
    assert args["messages"] == [{"role": "user", "content": "hello"}]
    assert args["max_tokens"] == 256
    assert args["reasoning_effort"] == "low"
    assert args.get("stream", False) is stream

    sampling_parameters = {"temperature", "top_p", "top_k", "candidate_count"}
    if supports_sampling_parameters:
        assert args["temperature"] == 0.4
        assert {"top_p", "top_k", "candidate_count"} <= args.keys()
    else:
        assert not sampling_parameters.intersection(args)
