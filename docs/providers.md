# LLM providers

CDR runs on any of eight hosted providers, or on a local model. You need **one**. Set its key in
`.env` and, if you like, pick it with `LLM_DEFAULT_PROVIDER`. CDR tries that provider first and
falls back through any others that are configured.

```bash
make check-env    # tells you whether a usable key is present
```

## Which one should I use?

| Situation | Use |
|---|---|
| Want a hosted provider | Choose one whose current model availability, billing, and limits fit your account. |
| Want low-latency inference | Compare **Groq** and **Cerebras** against your workload and current account limits. |
| No quota, no key, full control | **A local model** through Ollama, vLLM or LM Studio (below). |
| Need a particular model | **OpenAI**, **Anthropic**, or **OpenRouter** expose different model catalogs. |

A full pipeline run makes dozens of LLM calls. Check the selected provider's current model
catalog, billing, and rate limits before running a large workload; these vary by model and
account and can change over time.

## Setup

| Provider | `.env` | Get a key | Default model | Official model catalog |
|---|---|---|---|---|
| Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | https://aistudio.google.com/apikey | `gemini-3.8-flash` | [Gemini API](https://ai.google.dev/gemini-api/docs/models) |
| Groq | `GROQ_API_KEY` | https://console.groq.com/keys | `openai/gpt-oss-20b` | [GroqCloud](https://console.groq.com/docs/models) |
| Cerebras | `CEREBRAS_API_KEY` | https://cloud.cerebras.ai | `qwen-3.8-27b` | [Cerebras Inference](https://inference-docs.cerebras.ai/models) |
| OpenRouter | `OPENROUTER_API_KEY` | https://openrouter.ai/keys | `meta-llama/llama-3.1-8b-instruct` | [OpenRouter](https://openrouter.ai/models) |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` | Cloudflare dashboard → AI | `@cf/openai/gpt-oss-20b` | [Workers AI](https://developers.cloudflare.com/workers-ai/models/) |
| Hugging Face | `HF_TOKEN` (optional `HF_ENDPOINT_URL`) | https://huggingface.co/settings/tokens | `Qwen/Qwen2.5-72B-Instruct` | [Inference Providers](https://huggingface.co/inference/models) |
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys | `gpt-4o` | [OpenAI API](https://developers.openai.com/api/docs/models) |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com | Deferred to SDK migration [#119](https://github.com/BlueRingsLabs/Clinical-Deep-Research_CDR/issues/119) | [Claude models](https://platform.claude.com/docs/en/models/overview) |

The defaults updated in this audit were checked against current provider catalogs on 2026-10-08.
The table lists starting defaults, not a guarantee that a model is enabled for every account.
Availability and account limits can change; check the linked catalog and your account before use.
On 2026-10-08, the former Cloudflare default `@cf/meta/llama-3.1-8b-instruct-fast` redirected to a
"Model not available" page. The replacement `@cf/openai/gpt-oss-20b` is listed in the
[current Workers AI catalog](https://developers.cloudflare.com/workers-ai/models/gpt-oss-20b/)
and supports OpenAI-compatible chat completions.
The Anthropic default is intentionally not changed in issue #77; it is coupled to the SDK migration in #119.

Defaults live in `src/cdr/llm/*_provider.py` and `src/cdr/config.py`. Providers deprecate
models regularly. If a default starts returning 404s, that's a
[good first issue](https://github.com/BlueRingsLabs/Clinical-Deep-Research_CDR/issues/new?template=bug_report.yml).

## Local models (no key, no quota)

The `openai` provider can talk to any OpenAI-compatible server. With [Ollama](https://ollama.com):

```bash
ollama pull llama3.1:8b
```

```dotenv
LLM_DEFAULT_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.1:8b
```

vLLM (`http://localhost:8000/v1`, so run CDR's API on another port), LM Studio
(`http://localhost:1234/v1`) and `llama.cpp`'s server work the same way. No `OPENAI_API_KEY` is
needed when `OPENAI_BASE_URL` is set.

A warning about small models: 8B models get through the pipeline but produce thinner claims and
fail more verification checks. The bundled online runs were made with 8B models on purpose, to
show the floor rather than the ceiling.

## Known failure modes

| Symptom | Meaning | Fix |
|---|---|---|
| HTTP 402 | Out of credits (Hugging Face, OpenRouter) | Add credits or switch provider |
| HTTP 429 / rate-limit message | Provider or model rate limit | Wait, reduce request volume, or switch provider |
| HTTP 404 on the model | Provider retired the model | Set the model env var to a current one, and open an issue |
| Run finishes as `unpublishable` | The evidence didn't pass the gates | Working as intended. Read `status_reason` in the report |

The fallback factory skips providers that have no key configured. The Hugging Face provider
retries only transient errors (429, 5xx) and fails fast on a 402.

## Adding a provider

1. Create `src/cdr/llm/<name>_provider.py` subclassing `BaseLLMProvider` (see `base.py`).
2. Register it in `src/cdr/llm/factory.py` (`create_provider` and the fallback order) and add its
   settings to `src/cdr/config.py`.
3. Add its env vars to `.env.example` and a row to the table above.
4. Add tests with the HTTP layer mocked. Tests must never hit the network.

If the provider speaks the OpenAI API, you may not need any of this. Try `OPENAI_BASE_URL` first.
