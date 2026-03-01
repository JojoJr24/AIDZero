# LLM Providers Guidelines

## Scope
This folder contains provider adapters that expose a consistent API for model interaction, regardless of backend (cloud or local). Every new provider must follow the same contract and behavior style.

## Required Folder Layout
- `LLMProviders/<provider_name>/__init__.py`
- `LLMProviders/<provider_name>/provider.py`
- Optional helper modules in the same subfolder.
- Shared contracts belong in `LLMProviders/base.py`.

Use lowercase provider folder names (example: `google_gemini`, `ollama`, `lm_studio`).

## Required Public API
Each provider class must implement the methods defined in `LLMProviders.base.LLMProvider`:
- `list_models`, `list_model_names`
- `generate_content`, `stream_generate_content`
- `generate_text`, `stream_generate_text`
- `chat`, `stream_chat`
- `count_tokens`
- `embed_content`
- `supports_tool_calling`
- `extract_tool_calls`
- `build_tool_result_message`

If a backend does not support one feature natively, provide the closest equivalent or raise a clear `ProviderError` with a direct explanation.

## Tool Calling Contract (Mandatory)
- Providers must accept tool metadata in chat/content calls:
  - OpenAI-compatible providers: `tools` and optional `tool_choice`.
  - Gemini: accept `tools` and either `tool_config` or `tool_choice` (translated to Gemini `functionCallingConfig`).
  - Claude: accept `tools` and optional `tool_choice`.
- Providers must preserve multi-turn tool loops by accepting/normalizing:
  - assistant tool call messages (`tool_calls`, provider-native tool call blocks)
  - tool result messages (`role=tool` or provider-native equivalents)
- `extract_tool_calls(response)` must return normalized OpenAI-style tool calls:
  - `{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}`
- `build_tool_result_message(...)` must return a normalized tool message that can be appended directly to chat history.

## Behavior Rules
- Keep all source code, comments, and docs in English.
- Do not print from provider internals. Return structured data or raise typed exceptions.
- Use provider-specific error classes that inherit from `ProviderError`.
- Preserve raw provider payloads when possible to avoid data loss.
- Keep request/response normalization isolated in helper functions, not mixed into transport code.
- Support deterministic configuration through explicit constructor args and environment variable fallback.

## Testing and Validation
- Add tests under `tests/` for each provider module.
- Cover non-streaming, streaming, model listing, and error paths.
- Validate malformed payload handling and network/API failure behavior.
