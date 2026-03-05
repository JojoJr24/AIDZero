# LLM Providers Guidelines (v0.2.0)

## Scope
This folder contains provider adapters that expose a consistent API for model interaction across cloud and local backends.

## Required Folder Layout
- `LLMProviders/<provider_name>/__init__.py`
- `LLMProviders/<provider_name>/provider.py`
- Optional helper modules in the same subfolder.
- Shared contracts belong in `LLMProviders/base.py`.

Use lowercase provider folder names (example: `google_gemini`, `ollama`, `lmstudio`).

## Required Public API
Each provider class must implement methods defined in `LLMProviders.base.LLMProvider`:
- `list_models`, `list_model_names`
- `generate_content`, `stream_generate_content`
- `generate_text`, `stream_generate_text`
- `chat`, `stream_chat`
- `count_tokens`
- `embed_content`
- `supports_tool_calling`
- `extract_tool_calls`
- `build_tool_result_message`

If a backend does not support a feature natively, provide the closest equivalent or raise a clear `ProviderError`.

## Tool Calling Contract
Providers must accept tool metadata in chat/content calls:
- OpenAI-compatible providers: `tools` and optional `tool_choice`.
- Gemini: `tools` and either `tool_config` or `tool_choice`.
- Claude: `tools` and optional `tool_choice`.

Providers must preserve multi-turn tool loops by accepting/normalizing:
- Assistant tool call messages.
- Tool result messages.

`extract_tool_calls(response)` must return normalized OpenAI-style tool calls.
`build_tool_result_message(...)` must return a normalized tool message appendable to chat history.

## Behavior Rules
- Keep source code, comments, and docs in English.
- Do not print from provider internals.
- Raise typed exceptions inheriting from `ProviderError`.
- Keep request/response normalization isolated in helper functions.
- Support deterministic configuration through explicit args and env fallback.

## Testing
- Add tests under `tests/` for each provider module.
- Cover non-streaming, streaming, model listing, and error paths.
- Validate malformed payload handling and network/API failure behavior.
