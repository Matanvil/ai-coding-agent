# Local Model Integration Design

**Date:** 2026-03-30
**Status:** Approved
**Scope:** Add Qwen3-Coder-30B-A3B as local-first LLM with Claude fallback

---

## Background

Jarvis is local-first by philosophy. The ai-coding-agent currently calls Claude (Haiku by default) for all reasoning. Embeddings are already local via Ollama (`nomic-embed-text`). This spec covers making the reasoning model local-first too.

**Model chosen:** `qwen3-coder:30b` (Qwen3-Coder-30B-A3B-Instruct, Q4_K_M ~17GB)
- MoE: 30.5B total params, 3.3B active at inference
- 256K context natively
- Explicitly designed for agentic coding and tool calling
- Runs well on M1 Max 64GB (~40–55 t/s at Q4)
- Served via Ollama's OpenAI-compatible endpoint

**Motivation:** Cost reduction, privacy, and product alignment with Jarvis's local-first goal.

---

## Empirical Findings

Validated via the Jarvis benchmark script (`feat/local-model-benchmark`) and direct curl testing (2026-03-30):

- Tool call format: **OpenAI `tool_calls`** (proper structured response) with ≤5 tools
- Tool call format: **XML fallback** (`<function=name><parameter=key>value</parameter></function>`) with 6+ tools — a known Ollama chat template rendering bug
- Our agent has 3 tools — safely under the threshold, consistently gets OpenAI format
- Tool selection accuracy with our 3-tool schema: correct on all tested cases
- Latency: ~600ms–1800ms p50 on M1 Max (comparable to the 7b models tested previously)

**Critical constraint:** If we ever add a 4th–6th tool, re-test format consistency. At 6+ tools, the XML parser becomes a primary path and an Ollama template workaround may be needed (e.g. embedding tools directly in the system prompt).

---

## Architecture

### New Components

**`src/ollama_client.py`** — `OllamaClient(BaseLLMClient)`

Wraps Ollama's `/v1/chat/completions` endpoint. Responsibilities:
- Translate Anthropic tool schema (`input_schema`) → OpenAI format (`parameters`) once at init
- Run the ReAct loop, accumulating `current_messages` in **Anthropic-compatible format** throughout
- Parse each response via `_parse_tool_call()` (see below)
- Raise `ToolCallParseError(partial=current_messages)` when a response cannot be parsed as either a tool call or a final answer

**`src/hybrid_client.py`** — `HybridClient(BaseLLMClient)`

Owns both an `OllamaClient` and a `ClaudeClient`. Responsibilities:
- Try `OllamaClient.respond()` first
- On `ToolCallParseError` or any connection/HTTP error: log the handoff, extract `partial` messages from the exception, pass them to `ClaudeClient.respond()` so Claude continues with full context
- Accept `force_claude: bool` flag — when `True`, skip Ollama entirely for that call

### Modified Components

**`src/llm.py`**
- Extract `BaseLLMClient` abstract base with `respond()` signature
- Add `ToolCallParseError(partial: list)` exception
- `ClaudeClient` inherits from `BaseLLMClient`, otherwise unchanged

**`src/config.py`**
- Add `local_model: str` field (default `""`)
- Empty string = local disabled, use Claude directly (zero behavior change)

**`agent.py`**
- `build_shared()`: instantiate `HybridClient` when `config.local_model` is set, `ClaudeClient` otherwise
- Accept `--model claude` CLI flag → sets `force_claude=True` on `HybridClient` for the session

### Unchanged

All agents (`AgentLoop`, `Planner`, `Reviewer`, `Executor`) — they receive `llm` and call `llm.respond()`. No knowledge of which provider runs.

---

## Data Flow

### Normal path (local model configured, healthy)

```
HybridClient.respond(messages, tool_handler)
  → OllamaClient.respond()
      current_messages = list(messages)           # Anthropic format
      loop:
        POST /v1/chat/completions → Ollama
        _parse_tool_call(response)
        append [assistant tool_use + user tool_result] to current_messages
      return final answer string
```

### Fallback path

```
OllamaClient raises ToolCallParseError(partial=current_messages)
  OR connection/HTTP error

HybridClient:
  log " → local model failed after N turns, handing off to Claude"
  ClaudeClient.respond(messages=partial, tool_handler)
  # Claude sees all tool results gathered so far, continues without repeated work
```

If Ollama fails on turn 1 (no history yet), Claude gets the original messages — no worse than today.

### Local model not configured

`HybridClient` is not instantiated. `ClaudeClient` is used directly. Identical to current behavior.

---

## OllamaClient Internals

### Tool schema translation

```python
def _to_openai_tools(anthropic_tools: list) -> list:
    return [{"type": "function", "function": {
        "name": t["name"],
        "description": t["description"],
        "parameters": t["input_schema"],
    }} for t in anthropic_tools]
```

### Response parser

```python
def _parse_tool_call(message: dict) -> dict | None:
    # 1. Standard OpenAI tool_calls (primary path, ≤5 tools)
    if message.get("tool_calls"):
        tc = message["tool_calls"][0]
        return {
            "id": tc["id"],
            "name": tc["function"]["name"],
            "input": json.loads(tc["function"]["arguments"]),
        }
    # 2. Qwen3 XML format (defensive, triggers at 6+ tools)
    content = (message.get("content") or "").strip()
    if "<function=" in content:
        return _parse_xml_tool_call(content)
    # 3. Unparseable as tool call
    return None
```

`_parse_xml_tool_call` uses regex to extract function name and parameter key/value pairs from the `<function=name><parameter=key>value</parameter></function>` format.

When `_parse_tool_call` returns `None`:
- `finish_reason == "stop"` → treat as final answer, return content text
- Otherwise → raise `ToolCallParseError(partial=current_messages)`

---

## Configuration

`config.json`:
```json
{
  "local_model": "qwen3-coder:30b",
  "ollama_url": "http://localhost:11434"
}
```

`ollama_url` already exists. `local_model` is new. Empty string disables local model.

---

## Fallback Triggers

| Trigger | Behavior |
|---|---|
| `ToolCallParseError` from OllamaClient | Fall back to Claude with partial history |
| Connection error / HTTP error from Ollama | Fall back to Claude with original messages |
| `--model claude` CLI flag | Skip Ollama entirely for the session |

---

## What This Does Not Change

- Embedding model (stays Ollama `nomic-embed-text`)
- All agent classes (`AgentLoop`, `Planner`, `Reviewer`, `Executor`)
- `ClaudeClient` internals
- Plan storage, tool implementations, narration
- Default behavior when `local_model` is not set in config
