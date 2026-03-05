# LLM Providers Guidelines (v0.3.0)

## Scope
This folder contains provider adapters that expose a consistent runtime-facing interface for cloud and local backends.

## Required Layout
- `LLMProviders/<provider_name>/__init__.py`
- `LLMProviders/<provider_name>/provider.py`
- Shared contracts in `LLMProviders/base.py`
- Provider folder names must be lowercase.

## Required Adapter Behavior
- Provide model listing for runtime selection.
- Support non-streaming and streaming completions used by `core/llm_client.py`.
- Normalize provider output into runtime-compatible assistant text.
- Handle tool-calling metadata when backend supports tools.

## Error Handling and Stability
- Raise explicit provider errors with actionable messages.
- Do not print from provider internals.
- Keep request/response normalization isolated.
- Keep env/config fallback deterministic.

## Validation
```bash
uv run --with pytest pytest tests/test_llm_client_streaming.py tests/test_openai_compatible_provider_stream.py -q
```
