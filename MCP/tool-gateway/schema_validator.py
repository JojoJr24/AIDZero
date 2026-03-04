"""JSON-schema validation for tool call arguments."""

from __future__ import annotations

from hashlib import sha1
import json
from typing import Any

from jsonschema import Draft202012Validator


class SchemaValidator:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, Draft202012Validator]] = {}

    def validate(self, tool_id: str, schema: dict[str, Any] | None, args: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = _clone_args(args)
        if not isinstance(schema, dict):
            return payload

        validator = self._get_validator(tool_id, schema)
        errors = sorted(validator.iter_errors(payload), key=lambda error: str(error.path))
        if errors:
            detail = "; ".join(_format_error(error) for error in errors[:5])
            raise ValueError(detail or "Arguments failed schema validation.")
        return payload

    def _get_validator(self, tool_id: str, schema: dict[str, Any]) -> Draft202012Validator:
        fingerprint = _schema_fingerprint(schema)
        cached = self._cache.get(tool_id)
        if cached and cached[0] == fingerprint:
            return cached[1]

        cloned = _clone_schema(schema, tool_id)
        validator = Draft202012Validator(cloned)
        self._cache[tool_id] = (fingerprint, validator)
        return validator


def _clone_args(args: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    return json.loads(json.dumps(args))


def _clone_schema(schema: dict[str, Any], tool_id: str) -> dict[str, Any]:
    cloned = json.loads(json.dumps(schema))
    if "$id" not in cloned:
        cloned["$id"] = f"tool://{tool_id}"
    return cloned


def _schema_fingerprint(schema: dict[str, Any]) -> str:
    raw = json.dumps(schema, sort_keys=True, ensure_ascii=True)
    return sha1(raw.encode("utf-8")).hexdigest()


def _format_error(error) -> str:
    if error.path:
        path = "/" + "/".join(str(token) for token in error.path)
    else:
        path = error.schema_path[0] if error.schema_path else "schema"
    return f"{path}: {error.message}"
