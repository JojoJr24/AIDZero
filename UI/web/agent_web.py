#!/usr/bin/env python3
"""Web interface for the AgentCreator core."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.models import AgentPlan
from agent.prompt_history import PromptHistoryStore
from agent.provider_registry import ProviderRegistry
from agent.service import AgentCreator
from agent.ui_display import to_ui_label, to_ui_model_label


@dataclass(frozen=True)
class WebDefaults:
    provider_name: str
    model: str
    request: str
    dry_run: bool
    overwrite: bool


@dataclass(frozen=True)
class WebRuntime:
    repo_root: Path
    defaults: WebDefaults


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_str(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        return value.strip()
    return default


def _model_candidates(provider_registry: ProviderRegistry, provider_name: str) -> tuple[list[str], str | None]:
    default_model = provider_registry.default_model(provider_name)
    try:
        listed_models = provider_registry.try_list_models(provider_name)
    except Exception as error:  # noqa: BLE001
        return [default_model], str(error)

    unique_models: list[str] = []
    for model_name in listed_models:
        if model_name not in unique_models:
            unique_models.append(model_name)
    if default_model not in unique_models:
        unique_models.insert(0, default_model)
    return unique_models, None


def _to_option_items(
    values: list[str],
    *,
    label_fn: Callable[[str], str] = to_ui_label,
) -> list[dict[str, str]]:
    return [{"id": value, "label": label_fn(value)} for value in values]


def _build_options_payload(
    *,
    provider_registry: ProviderRegistry,
    default_provider: str,
    default_model: str,
    default_request: str,
    dry_run: bool,
    overwrite: bool,
    prompt_history: list[str],
) -> dict[str, Any]:
    provider_names = provider_registry.names()
    selected_provider = default_provider if provider_registry.has(default_provider) else provider_names[0]
    models, model_error = _model_candidates(provider_registry, selected_provider)
    selected_model = default_model if default_model in models else models[0]
    return {
        "providers": _to_option_items(provider_names),
        "models": _to_option_items(models, label_fn=to_ui_model_label),
        "selected_provider": selected_provider,
        "selected_model": selected_model,
        "default_request": default_request,
        "dry_run": dry_run,
        "overwrite": overwrite,
        "prompt_history": prompt_history,
        "model_error": model_error,
    }


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def _plan_from_payload(payload: Any) -> AgentPlan:
    if not isinstance(payload, dict):
        raise ValueError("Field 'plan' must be an object.")

    agent_name = _coerce_str(payload.get("agent_name"))
    project_folder = _coerce_str(payload.get("project_folder"))
    goal = _coerce_str(payload.get("goal"))
    summary = _coerce_str(payload.get("summary"))
    if not agent_name:
        raise ValueError("Plan field 'agent_name' is required.")
    if not project_folder:
        raise ValueError("Plan field 'project_folder' is required.")
    if not goal:
        raise ValueError("Plan field 'goal' is required.")
    if not summary:
        raise ValueError("Plan field 'summary' is required.")

    return AgentPlan(
        agent_name=agent_name,
        project_folder=project_folder,
        goal=goal,
        summary=summary,
        required_llm_providers=_normalize_string_list(payload.get("required_llm_providers")),
        required_skills=_normalize_string_list(payload.get("required_skills")),
        required_tools=_normalize_string_list(payload.get("required_tools")),
        required_mcp=_normalize_string_list(payload.get("required_mcp")),
        required_ui=_normalize_string_list(payload.get("required_ui")),
        folder_blueprint=_normalize_string_list(payload.get("folder_blueprint")),
        implementation_steps=_normalize_string_list(payload.get("implementation_steps")),
        warnings=_normalize_string_list(payload.get("warnings")),
        raw_response=_coerce_str(payload.get("raw_response")),
    )


def _build_scaffold_payload(scaffold_result: Any) -> dict[str, Any]:
    return {
        "destination": str(scaffold_result.destination),
        "created_directories": [str(path) for path in scaffold_result.created_directories],
        "copied_items": [str(path) for path in scaffold_result.copied_items],
        "entrypoint_file": str(scaffold_result.entrypoint_file) if scaffold_result.entrypoint_file else None,
        "runtime_config_file": (
            str(scaffold_result.runtime_config_file) if scaffold_result.runtime_config_file else None
        ),
        "metadata_file": str(scaffold_result.metadata_file) if scaffold_result.metadata_file else None,
        "process_log_file": str(scaffold_result.process_log_file) if scaffold_result.process_log_file else None,
    }


def _run_initial_plan_request(
    *,
    repo_root: Path,
    provider_name: str,
    model: str,
    user_request: str,
) -> dict[str, Any]:
    registry = ProviderRegistry(repo_root)
    prompt_history = PromptHistoryStore(repo_root)
    if not registry.has(provider_name):
        raise ValueError(f"Unknown provider '{provider_name}'.")

    resolved_model = model.strip() or registry.default_model(provider_name)
    prompt_history_items = prompt_history.add_prompt(user_request)
    provider = registry.create(provider_name)
    creator = AgentCreator(provider=provider, model=resolved_model, repo_root=repo_root)
    llm_invocations: list[dict[str, Any]] = []
    planning_result = creator.describe_requirements(
        user_request=user_request,
        ui_name="web",
        invocation_tracer=lambda payload: llm_invocations.append(payload),
    )
    return {
        "mode": "plan",
        "provider": provider_name,
        "model": resolved_model,
        "request": user_request,
        "plan": planning_result.plan.to_dict(),
        "prompt_history": prompt_history_items,
        "llm_invocations": llm_invocations,
    }


def _run_plan_revision_request(
    *,
    repo_root: Path,
    provider_name: str,
    model: str,
    user_request: str,
    current_plan_payload: Any,
    plan_change_request: str,
) -> dict[str, Any]:
    registry = ProviderRegistry(repo_root)
    prompt_history = PromptHistoryStore(repo_root)
    if not registry.has(provider_name):
        raise ValueError(f"Unknown provider '{provider_name}'.")

    normalized_change_request = plan_change_request.strip()
    if not normalized_change_request:
        raise ValueError("Field 'plan_change_request' is required.")

    resolved_model = model.strip() or registry.default_model(provider_name)
    current_plan = _plan_from_payload(current_plan_payload)
    provider = registry.create(provider_name)
    creator = AgentCreator(provider=provider, model=resolved_model, repo_root=repo_root)
    llm_invocations: list[dict[str, Any]] = []
    planning_result = creator.revise_requirements(
        user_request=user_request,
        current_plan=current_plan,
        plan_change_request=normalized_change_request,
        ui_name="web",
        invocation_tracer=lambda payload: llm_invocations.append(payload),
    )

    return {
        "mode": "plan_revision",
        "provider": provider_name,
        "model": resolved_model,
        "request": user_request,
        "plan": planning_result.plan.to_dict(),
        "prompt_history": prompt_history.list_prompts(),
        "llm_invocations": llm_invocations,
    }


def _run_scaffold_from_plan(
    *,
    repo_root: Path,
    provider_name: str,
    model: str,
    user_request: str,
    plan_payload: Any,
    overwrite: bool,
) -> dict[str, Any]:
    registry = ProviderRegistry(repo_root)
    prompt_history = PromptHistoryStore(repo_root)
    if not registry.has(provider_name):
        raise ValueError(f"Unknown provider '{provider_name}'.")

    resolved_model = model.strip() or registry.default_model(provider_name)
    prompt_history_items = prompt_history.add_prompt(user_request)
    plan = _plan_from_payload(plan_payload)

    provider = registry.create(provider_name)
    creator = AgentCreator(provider=provider, model=resolved_model, repo_root=repo_root)
    scaffold_result = creator.create_agent_project_from_plan(
        user_request=user_request,
        plan=plan,
        catalog=creator.scan_components(),
        overwrite=overwrite,
    )
    return {
        "mode": "scaffold_from_plan",
        "provider": provider_name,
        "model": resolved_model,
        "request": user_request,
        "plan": plan.to_dict(),
        "prompt_history": prompt_history_items,
        "scaffold": _build_scaffold_payload(scaffold_result),
    }


def _render_index_html() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>AIDZero Web UI</title>
  <style>
    :root {
      --bg-a: #0f172a;
      --bg-b: #1e293b;
      --card: rgba(255, 255, 255, 0.12);
      --card-border: rgba(255, 255, 255, 0.26);
      --text: #f8fafc;
      --muted: #dbeafe;
      --accent: #fb7185;
      --accent-strong: #e11d48;
      --ok: #86efac;
      --warn: #fbbf24;
      --error: #fda4af;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: \"Avenir Next\", \"Trebuchet MS\", \"Segoe UI\", sans-serif;
      background:
        radial-gradient(circle at 15% 15%, rgba(251, 113, 133, 0.18), transparent 35%),
        radial-gradient(circle at 85% 0%, rgba(56, 189, 248, 0.18), transparent 35%),
        linear-gradient(145deg, var(--bg-a), var(--bg-b));
      padding: 24px;
      animation: fadeIn 0.45s ease-out;
    }
    .layout {
      width: min(1040px, 100%);
      margin: 0 auto;
      display: grid;
      gap: 18px;
      grid-template-columns: 1fr;
    }
    .card {
      backdrop-filter: blur(8px);
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.35);
    }
    h1 {
      margin: 0 0 6px;
      letter-spacing: 0.03em;
      font-size: clamp(1.5rem, 2vw, 2rem);
    }
    p { margin: 0; color: var(--muted); }
    .grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .field {
      display: grid;
      gap: 6px;
    }
    .field.full { grid-column: 1 / -1; }
    label { font-size: 0.9rem; color: var(--muted); }
    select, textarea {
      width: 100%;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: rgba(15, 23, 42, 0.4);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 0.98rem;
    }
    textarea {
      min-height: 100px;
      resize: vertical;
      line-height: 1.45;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-top: 4px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 9px 14px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      color: #fff;
      background: linear-gradient(125deg, var(--accent), var(--accent-strong));
      box-shadow: 0 8px 16px rgba(225, 29, 72, 0.35);
    }
    button.secondary {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.2);
      box-shadow: none;
    }
    button:active { transform: translateY(1px); }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.9rem;
      color: var(--muted);
    }
    #status {
      margin: 0;
      font-size: 0.92rem;
      color: var(--muted);
    }
    #status.ok { color: var(--ok); }
    #status.warn { color: var(--warn); }
    #status.error { color: var(--error); }
    pre {
      margin: 0;
      border-radius: 10px;
      padding: 12px;
      background: rgba(2, 6, 23, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.18);
      overflow: auto;
      font-family: \"Iosevka\", \"Consolas\", monospace;
      font-size: 0.86rem;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .note {
      font-size: 0.82rem;
      color: var(--muted);
      opacity: 0.9;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 760px) {
      body { padding: 14px; }
      .grid { grid-template-columns: 1fr; }
      .card { padding: 14px; }
    }
  </style>
</head>
<body>
  <main class=\"layout\">
    <section class=\"card\">
      <h1>AIDZero Agent Creator</h1>
      <p>Build a plan, request changes, and scaffold only after the plan is approved.</p>
    </section>

    <section class=\"card\">
      <div class=\"grid\">
        <div class=\"field\">
          <label for=\"provider\">Provider</label>
          <select id=\"provider\"></select>
        </div>
        <div class=\"field\">
          <label for=\"model\">Model</label>
          <select id=\"model\"></select>
        </div>
        <div class=\"field full\">
          <label for=\"request\">Request</label>
          <textarea id=\"request\" placeholder=\"Build an agent for daily KPI reports...\"></textarea>
        </div>
        <div class=\"field full\">
          <label for=\"historyPrompt\">Prompt History</label>
          <select id=\"historyPrompt\"></select>
        </div>
        <div class=\"field full\">
          <label for=\"planChangeRequest\">Plan Change Request</label>
          <textarea id=\"planChangeRequest\" placeholder=\"Ask for plan changes here before scaffolding...\"></textarea>
        </div>
      </div>
      <div class=\"actions\">
        <label class=\"toggle\"><input id=\"dryRun\" type=\"checkbox\" /> Dry run (plan only)</label>
        <label class=\"toggle\"><input id=\"overwrite\" type=\"checkbox\" /> Overwrite non-empty destination</label>
        <button id=\"generatePlanBtn\">Generate Plan</button>
        <button id=\"revisePlanBtn\" class=\"secondary\" type=\"button\" disabled>Request Plan Changes</button>
        <button id=\"scaffoldBtn\" class=\"secondary\" type=\"button\" disabled>Scaffold Approved Plan</button>
        <button id=\"useHistoryBtn\" class=\"secondary\" type=\"button\">Use History Prompt</button>
        <button id=\"refreshBtn\" class=\"secondary\" type=\"button\">Refresh Models</button>
      </div>
      <p id=\"status\">Loading options...</p>
      <p class=\"note\">Step 1 is interactive: generate a plan, iterate revisions, then continue to step 2.</p>
    </section>

    <section class=\"card\">
      <h2 style=\"margin:0 0 10px; font-size:1.05rem; color:var(--muted);\">Result</h2>
      <pre id=\"result\">{}</pre>
    </section>

    <section class=\"card\">
      <h2 style=\"margin:0 0 10px; font-size:1.05rem; color:var(--muted);\">LLM Invocations</h2>
      <pre id=\"llmInvocations\">[]</pre>
    </section>

  </main>

  <script>
    const providerEl = document.getElementById("provider");
    const modelEl = document.getElementById("model");
    const requestEl = document.getElementById("request");
    const planChangeRequestEl = document.getElementById("planChangeRequest");
    const historyPromptEl = document.getElementById("historyPrompt");
    const dryRunEl = document.getElementById("dryRun");
    const overwriteEl = document.getElementById("overwrite");
    const generatePlanBtn = document.getElementById("generatePlanBtn");
    const revisePlanBtn = document.getElementById("revisePlanBtn");
    const scaffoldBtn = document.getElementById("scaffoldBtn");
    const useHistoryBtn = document.getElementById("useHistoryBtn");
    const refreshBtn = document.getElementById("refreshBtn");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");
    const llmInvocationsEl = document.getElementById("llmInvocations");
    let currentPlanContext = null;

    function setStatus(message, cls = "") {
      statusEl.className = cls;
      statusEl.textContent = message;
    }

    function truncateText(value, maxLen = 320) {
      const text = String(value || "").trim();
      if (text.length <= maxLen) return text;
      return `${text.slice(0, maxLen - 3)}...`;
    }

    function compactPlan(plan) {
      if (!plan || typeof plan !== "object") return plan;
      return {
        agent_name: plan.agent_name,
        project_folder: plan.project_folder,
        goal: plan.goal,
        summary: truncateText(plan.summary, 420),
        required_llm_providers: Array.isArray(plan.required_llm_providers) ? plan.required_llm_providers : [],
        required_skills: Array.isArray(plan.required_skills) ? plan.required_skills : [],
        required_tools: Array.isArray(plan.required_tools) ? plan.required_tools : [],
        required_mcp: Array.isArray(plan.required_mcp) ? plan.required_mcp : [],
        required_ui: Array.isArray(plan.required_ui) ? plan.required_ui : [],
        folder_blueprint_count: Array.isArray(plan.folder_blueprint) ? plan.folder_blueprint.length : 0,
        implementation_steps_count: Array.isArray(plan.implementation_steps) ? plan.implementation_steps.length : 0,
        warnings_count: Array.isArray(plan.warnings) ? plan.warnings.length : 0
      };
    }

    function formatResultForDisplay(payload) {
      if (!payload || typeof payload !== "object") return payload;

      const base = {
        mode: payload.mode,
        provider: payload.provider,
        model: payload.model,
        request: payload.request,
        plan: compactPlan(payload.plan)
      };

      if (payload.mode === "plan" || payload.mode === "plan_revision" || payload.mode === "dry_run") {
        return {
          ...base,
          llm_invocations_count: Array.isArray(payload.llm_invocations) ? payload.llm_invocations.length : 0,
          prompt_history_count: Array.isArray(payload.prompt_history) ? payload.prompt_history.length : 0
        };
      }

      if (payload.mode === "scaffold" || payload.mode === "scaffold_from_dry_run" || payload.mode === "scaffold_from_plan") {
        return {
          ...base,
          scaffold: payload.scaffold || null,
          prompt_history_count: Array.isArray(payload.prompt_history) ? payload.prompt_history.length : 0
        };
      }

      return payload;
    }

    function setResult(payload) {
      resultEl.textContent = JSON.stringify(formatResultForDisplay(payload), null, 2);
    }

    function setInvocations(payload) {
      if (!llmInvocationsEl) return;
      llmInvocationsEl.textContent = JSON.stringify(payload, null, 2);
    }

    function setCurrentPlanContext(payload) {
      if (!payload || typeof payload !== "object" || !payload.plan) {
        currentPlanContext = null;
        updateActionButtons();
        return;
      }
      currentPlanContext = {
        provider: String(payload.provider || "").trim(),
        model: String(payload.model || "").trim(),
        request: String(payload.request || "").trim(),
        plan: payload.plan
      };
      updateActionButtons();
    }

    function clearCurrentPlanContext() {
      currentPlanContext = null;
      updateActionButtons();
    }

    function updateActionButtons() {
      const hasPlan = Boolean(currentPlanContext && currentPlanContext.plan);
      revisePlanBtn.disabled = !hasPlan;
      scaffoldBtn.disabled = !hasPlan || dryRunEl.checked;
    }

    function invalidatePlanIfInputsChanged() {
      if (!currentPlanContext) return;
      const request = requestEl.value.trim();
      if (
        providerEl.value !== currentPlanContext.provider ||
        modelEl.value !== currentPlanContext.model ||
        request !== currentPlanContext.request
      ) {
        clearCurrentPlanContext();
        setStatus("Plan context changed. Generate a new plan first.", "warn");
      }
    }

    function refillSelect(selectEl, items, selectedValue) {
      selectEl.innerHTML = "";
      for (const item of items) {
        const value = typeof item === "string" ? item : item.id;
        const label = typeof item === "string" ? item : (item.label || item.id);
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        if (value === selectedValue) option.selected = true;
        selectEl.appendChild(option);
      }
      if (!selectEl.value && items.length > 0) {
        const first = items[0];
        selectEl.value = typeof first === "string" ? first : first.id;
      }
    }

    function refillPromptHistory(items) {
      historyPromptEl.innerHTML = "";
      if (!Array.isArray(items) || items.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No prompt history yet";
        historyPromptEl.appendChild(option);
        historyPromptEl.disabled = true;
        useHistoryBtn.disabled = true;
        return;
      }

      historyPromptEl.disabled = false;
      useHistoryBtn.disabled = false;
      for (let index = 0; index < items.length; index += 1) {
        const prompt = String(items[index]);
        const option = document.createElement("option");
        option.value = prompt;
        option.textContent = `${index + 1}. ${prompt}`;
        if (index === 0) option.selected = true;
        historyPromptEl.appendChild(option);
      }
    }

    async function fetchOptions() {
      const response = await fetch("/api/options");
      if (!response.ok) {
        throw new Error("Could not load options.");
      }
      return response.json();
    }

    async function refreshModels() {
      const provider = providerEl.value;
      if (!provider) return;
      setStatus("Refreshing models...", "warn");
      const response = await fetch(`/api/models?provider=${encodeURIComponent(provider)}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Could not fetch models.");
      }
      refillSelect(modelEl, payload.models, payload.selected_model);
      invalidatePlanIfInputsChanged();
      if (payload.model_error) {
        setStatus(`Models loaded with fallback: ${payload.model_error}`, "warn");
      } else {
        setStatus("Models refreshed.", "ok");
      }
    }

    async function submitPlan() {
      const request = requestEl.value.trim();
      if (!request) {
        setStatus("Request is required.", "error");
        return;
      }

      generatePlanBtn.disabled = true;
      revisePlanBtn.disabled = true;
      scaffoldBtn.disabled = true;
      setStatus("Generating plan...", "warn");
      setInvocations({ info: "Waiting for LLM response..." });
      try {
        const response = await fetch("/api/plan", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider: providerEl.value,
            model: modelEl.value,
            request
          })
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Plan generation failed.");
        }
        setResult(payload);
        setInvocations(payload.llm_invocations || []);
        if (Array.isArray(payload.prompt_history)) {
          refillPromptHistory(payload.prompt_history);
        }
        setCurrentPlanContext(payload);
        if (dryRunEl.checked) {
          setStatus("Plan generated. Dry-run is enabled, scaffolding is disabled.", "ok");
        } else {
          setStatus("Plan generated. Request changes or scaffold when ready.", "ok");
        }
      } catch (error) {
        setStatus(String(error), "error");
        setInvocations({ error: String(error) });
      } finally {
        generatePlanBtn.disabled = false;
        updateActionButtons();
      }
    }

    async function submitPlanRevision() {
      if (!currentPlanContext || !currentPlanContext.plan) {
        setStatus("Generate a plan before requesting changes.", "warn");
        return;
      }
      const planChangeRequest = planChangeRequestEl.value.trim();
      if (!planChangeRequest) {
        setStatus("Plan change request is required.", "error");
        return;
      }

      generatePlanBtn.disabled = true;
      revisePlanBtn.disabled = true;
      scaffoldBtn.disabled = true;
      setStatus("Revising plan...", "warn");
      setInvocations({ info: "Waiting for LLM plan revision..." });
      try {
        const response = await fetch("/api/plan/revise", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider: currentPlanContext.provider,
            model: currentPlanContext.model,
            request: currentPlanContext.request,
            current_plan: currentPlanContext.plan,
            plan_change_request: planChangeRequest
          })
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Plan revision failed.");
        }
        setResult(payload);
        setInvocations(payload.llm_invocations || []);
        if (Array.isArray(payload.prompt_history)) {
          refillPromptHistory(payload.prompt_history);
        }
        setCurrentPlanContext(payload);
        if (dryRunEl.checked) {
          setStatus("Plan revised. Dry-run is enabled, scaffolding is disabled.", "ok");
        } else {
          setStatus("Plan revised. Continue iterating or scaffold.", "ok");
        }
      } catch (error) {
        setStatus(String(error), "error");
        setInvocations({ error: String(error) });
      } finally {
        generatePlanBtn.disabled = false;
        updateActionButtons();
      }
    }

    async function submitScaffold() {
      if (dryRunEl.checked) {
        setStatus("Disable dry-run to scaffold.", "warn");
        return;
      }
      if (!currentPlanContext || !currentPlanContext.plan) {
        setStatus("Generate and approve a plan before scaffolding.", "warn");
        return;
      }

      generatePlanBtn.disabled = true;
      revisePlanBtn.disabled = true;
      scaffoldBtn.disabled = true;
      setStatus("Scaffolding approved plan...", "warn");
      try {
        const response = await fetch("/api/scaffold", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider: currentPlanContext.provider,
            model: currentPlanContext.model,
            request: currentPlanContext.request,
            plan: currentPlanContext.plan,
            overwrite: overwriteEl.checked
          })
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Scaffolding failed.");
        }
        setResult(payload);
        setInvocations({ info: "No LLM call: scaffolding executed from approved plan." });
        if (Array.isArray(payload.prompt_history)) {
          refillPromptHistory(payload.prompt_history);
        }
        clearCurrentPlanContext();
        setStatus("Project scaffolded successfully.", "ok");
      } catch (error) {
        setStatus(String(error), "error");
        setInvocations({ error: String(error) });
      } finally {
        generatePlanBtn.disabled = false;
        updateActionButtons();
      }
    }

    function applySelectedHistoryPrompt() {
      if (historyPromptEl.disabled) return;
      const selectedPrompt = historyPromptEl.value.trim();
      if (!selectedPrompt) return;
      requestEl.value = selectedPrompt;
      invalidatePlanIfInputsChanged();
      setStatus("Prompt loaded from history.", "ok");
    }

    async function boot() {
      try {
        const options = await fetchOptions();
        refillSelect(providerEl, options.providers, options.selected_provider);
        refillSelect(modelEl, options.models, options.selected_model);
        refillPromptHistory(options.prompt_history || []);
        requestEl.value = options.default_request || "";
        dryRunEl.checked = Boolean(options.dry_run);
        overwriteEl.checked = Boolean(options.overwrite);
        planChangeRequestEl.value = "";
        clearCurrentPlanContext();
        setResult({ info: "Ready." });
        setInvocations([]);
        if (options.model_error) {
          setStatus(`Ready (model list fallback): ${options.model_error}`, "warn");
        } else {
          setStatus("Ready.", "ok");
        }
      } catch (error) {
        setStatus(String(error), "error");
      }
    }

    providerEl.addEventListener("change", () => {
      invalidatePlanIfInputsChanged();
      void refreshModels();
    });
    modelEl.addEventListener("change", () => { invalidatePlanIfInputsChanged(); });
    requestEl.addEventListener("input", () => { invalidatePlanIfInputsChanged(); });
    dryRunEl.addEventListener("change", () => { updateActionButtons(); });
    refreshBtn.addEventListener("click", () => { void refreshModels(); });
    generatePlanBtn.addEventListener("click", () => { void submitPlan(); });
    revisePlanBtn.addEventListener("click", () => { void submitPlanRevision(); });
    scaffoldBtn.addEventListener("click", () => { void submitScaffold(); });
    useHistoryBtn.addEventListener("click", () => { applySelectedHistoryPrompt(); });
    void boot();
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status.value)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _text_response(
    handler: BaseHTTPRequestHandler,
    *,
    body: str,
    content_type: str,
    status: HTTPStatus = HTTPStatus.OK,
) -> None:
    data = body.encode("utf-8")
    handler.send_response(status.value)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = handler.headers.get("Content-Length", "0").strip()
    if not content_length.isdigit():
        raise ValueError("Invalid Content-Length header.")
    length = int(content_length)
    if length <= 0:
        raise ValueError("Request body is required.")

    raw_body = handler.rfile.read(length)
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Body must be valid JSON.") from error

    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object.")
    return payload


def _handler_factory(runtime: WebRuntime) -> type[BaseHTTPRequestHandler]:
    class AgentCreatorWebHandler(BaseHTTPRequestHandler):
        server_version = "AIDZeroWebUI/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                _text_response(
                    self,
                    body=_render_index_html(),
                    content_type="text/html; charset=utf-8",
                )
                return
            if parsed.path == "/api/options":
                self._handle_options()
                return
            if parsed.path == "/api/models":
                self._handle_models(parsed.query)
                return
            _json_response(
                self,
                {"error": f"Unknown path '{escape(parsed.path)}'."},
                status=HTTPStatus.NOT_FOUND,
            )

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/api/run", "/api/plan"}:
                self._handle_plan()
                return
            if parsed.path == "/api/plan/revise":
                self._handle_plan_revision()
                return
            if parsed.path in {"/api/scaffold-from-dry-run", "/api/scaffold"}:
                self._handle_scaffold_from_plan()
                return
            _json_response(
                self,
                {"error": f"Unknown path '{escape(parsed.path)}'."},
                status=HTTPStatus.NOT_FOUND,
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            sys.stdout.write(f"[webui] {format % args}\n")

        def _handle_options(self) -> None:
            registry = ProviderRegistry(runtime.repo_root)
            prompt_history = PromptHistoryStore(runtime.repo_root)
            payload = _build_options_payload(
                provider_registry=registry,
                default_provider=runtime.defaults.provider_name,
                default_model=runtime.defaults.model,
                default_request=runtime.defaults.request,
                dry_run=runtime.defaults.dry_run,
                overwrite=runtime.defaults.overwrite,
                prompt_history=prompt_history.list_prompts(),
            )
            _json_response(self, payload)

        def _handle_models(self, query: str) -> None:
            params = parse_qs(query, keep_blank_values=False)
            raw_provider = params.get("provider", [""])[0]
            provider_name = raw_provider.strip()
            if not provider_name:
                _json_response(
                    self,
                    {"error": "Query parameter 'provider' is required."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            registry = ProviderRegistry(runtime.repo_root)
            if not registry.has(provider_name):
                _json_response(
                    self,
                    {"error": f"Unknown provider '{provider_name}'."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            models, model_error = _model_candidates(registry, provider_name)
            _json_response(
                self,
                {
                    "provider": provider_name,
                    "models": _to_option_items(models, label_fn=to_ui_model_label),
                    "selected_model": models[0],
                    "model_error": model_error,
                },
            )

        def _handle_plan(self) -> None:
            try:
                payload = _read_json_body(self)
                user_request = _coerce_str(payload.get("request"))
                provider_name = _coerce_str(payload.get("provider"), default=runtime.defaults.provider_name)
                model_name = _coerce_str(payload.get("model"), default=runtime.defaults.model)
                if not user_request:
                    raise ValueError("Field 'request' is required.")

                result = _run_initial_plan_request(
                    repo_root=runtime.repo_root,
                    provider_name=provider_name,
                    model=model_name,
                    user_request=user_request,
                )
            except ValueError as error:
                _json_response(self, {"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as error:  # noqa: BLE001
                _json_response(self, {"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            _json_response(self, result)

        def _handle_plan_revision(self) -> None:
            try:
                payload = _read_json_body(self)
                user_request = _coerce_str(payload.get("request"))
                provider_name = _coerce_str(payload.get("provider"), default=runtime.defaults.provider_name)
                model_name = _coerce_str(payload.get("model"), default=runtime.defaults.model)
                current_plan_payload = payload.get("current_plan")
                plan_change_request = _coerce_str(payload.get("plan_change_request"))
                if not user_request:
                    raise ValueError("Field 'request' is required.")

                result = _run_plan_revision_request(
                    repo_root=runtime.repo_root,
                    provider_name=provider_name,
                    model=model_name,
                    user_request=user_request,
                    current_plan_payload=current_plan_payload,
                    plan_change_request=plan_change_request,
                )
            except ValueError as error:
                _json_response(self, {"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as error:  # noqa: BLE001
                _json_response(self, {"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            _json_response(self, result)

        def _handle_scaffold_from_plan(self) -> None:
            try:
                payload = _read_json_body(self)
                user_request = _coerce_str(payload.get("request"))
                provider_name = _coerce_str(payload.get("provider"), default=runtime.defaults.provider_name)
                model_name = _coerce_str(payload.get("model"), default=runtime.defaults.model)
                overwrite = _coerce_bool(payload.get("overwrite"), default=runtime.defaults.overwrite)
                plan_payload = payload.get("plan")
                if not user_request:
                    raise ValueError("Field 'request' is required.")

                result = _run_scaffold_from_plan(
                    repo_root=runtime.repo_root,
                    provider_name=provider_name,
                    model=model_name,
                    user_request=user_request,
                    plan_payload=plan_payload,
                    overwrite=overwrite,
                )
            except ValueError as error:
                _json_response(self, {"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as error:  # noqa: BLE001
                _json_response(self, {"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            _json_response(self, result)

    return AgentCreatorWebHandler


def run_web_agent(
    *,
    provider_name: str,
    model: str,
    user_request: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    yes: bool = False,  # kept for parity with terminal function signature
    repo_root: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8787,
) -> int:
    del yes
    root = (repo_root or Path.cwd()).resolve()
    registry = ProviderRegistry(root)
    providers = registry.names()
    if not providers:
        print("error> no providers available.")
        return 2

    selected_provider = provider_name.strip() if provider_name.strip() else providers[0]
    if not registry.has(selected_provider):
        selected_provider = providers[0]

    selected_model = model.strip() or registry.default_model(selected_provider)
    runtime = WebRuntime(
        repo_root=root,
        defaults=WebDefaults(
            provider_name=selected_provider,
            model=selected_model,
            request=(user_request or "").strip(),
            dry_run=dry_run,
            overwrite=overwrite,
        ),
    )

    handler_class = _handler_factory(runtime)
    try:
        server = ThreadingHTTPServer((host, port), handler_class)
    except OSError as error:
        print(f"error> could not start web server on {host}:{port}: {error}")
        return 1

    print(f"Web UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web UI.")
    finally:
        server.server_close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AIDZero web interface.")
    parser.add_argument("--provider", default="AID-google_gemini", help="Provider name.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model name.")
    parser.add_argument("--request", default="", help="Optional prefilled user request.")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry-run by default in UI.")
    parser.add_argument("--overwrite", action="store_true", help="Enable overwrite by default in UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server.")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind the server.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_web_agent(
        provider_name=args.provider,
        model=args.model,
        user_request=args.request,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        repo_root=REPO_ROOT,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
