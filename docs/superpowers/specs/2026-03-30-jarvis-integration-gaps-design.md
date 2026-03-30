# Jarvis Integration Gaps — Design Spec

**Date:** 2026-03-30
**Goal:** Close the gaps that prevent Jarvis from integrating with the AI Coding Agent programmatically, with full narration support.

---

## Problem Statement

The AI Coding Agent works as a CLI REPL but is not yet usable as a Python library by an external orchestrator (Jarvis). Three blockers exist:

1. `Planner` and `Reviewer` have no progress callbacks — their ReAct loops are silent to the caller
2. `Executor` blocks on `input()` — cannot be driven programmatically
3. No structured narration utility — callers must format raw event data themselves

`AgentLoop` already supports `on_tool_call`; this work extends that pattern consistently across all agents and unifies it into a single `on_event` callback.

---

## Design

### 1. Unified `on_event` Callback

Every agent method that performs async work gains an optional `on_event` parameter:

```python
on_event: Optional[Callable[[str, dict], None]] = None
```

**Signature:** `on_event(event_type: str, data: dict) -> None`

The existing `on_tool_call` parameter on `AgentLoop.ask()` and `ClaudeClient.respond()` is renamed to `on_event`. Tool calls become one event type among several.

#### Standardized Event Types

| Event | Fired by | Data keys |
|---|---|---|
| `tool_call` | All agents (in ReAct loop) | `tool: str`, `input: dict` |
| `planning_started` | `Planner.plan()`, `Planner.revise()` | `task: str` |
| `planning_complete` | `Planner.plan()`, `Planner.revise()` | `edit_count: int` |
| `review_started` | `Reviewer.review()` | _(empty)_ |
| `review_complete` | `Reviewer.review()` | `issue_count: int`, `critical_count: int`, `suggest_fix_plan: bool` |
| `edit_presented` | `Executor.execute()` | `index: int`, `total: int`, `file: str`, `description: str` |
| `edit_applied` | `Executor.execute()` | `file: str` |
| `edit_skipped` | `Executor.execute()` | `file: str` |
| `edit_revised` | `Executor.execute()` | `file: str` |
| `execution_complete` | `Executor.execute()` | `applied: int`, `skipped: int` |

All `on_event` parameters are optional and default to `None`. When `None`, no events are fired and existing behavior is unchanged. Unknown event types are ignored by `narrate_event()`.

#### Firing Order (Planner example)

```
on_event("planning_started", {"task": "add retry logic"})
  → on_event("tool_call", {"tool": "search_codebase", "input": {...}})
  → on_event("tool_call", {"tool": "read_file", "input": {...}})
on_event("planning_complete", {"edit_count": 3})
```

#### Propagation Through `ClaudeClient`

`Planner` and `Reviewer` bypass `ClaudeClient.respond()` and run their own ReAct loops directly via `self.llm.client.messages.create()`. The `on_event` callback is threaded into these loops manually (same pattern as `ClaudeClient.respond()` already uses for `AgentLoop`). The `submit_plan` and `submit_review` tool calls are not fired as `tool_call` events — they are internal termination signals, not observable actions.

---

### 2. Executor Approval Callback

#### `ApprovalDecision` Dataclass

Added to `src/plan_store.py` alongside existing dataclasses:

```python
@dataclass
class ApprovalDecision:
    action: str   # "apply" | "skip" | "quit" | "revise"
    feedback: str = ""  # only used when action == "revise"
```

#### Updated `Executor.execute()` Signature

```python
def execute(self, plan: Plan, approval_fn=None, on_event=None) -> Plan:
```

- `approval_fn(edit: FileEdit) -> ApprovalDecision` — called once per edit instead of `input()`
- When `approval_fn` is `None`, falls back to existing `input()` behavior — CLI is unchanged
- When `action == "revise"`, executor calls `_revise_edit(edit, decision.feedback)`, fires `edit_revised`, then calls `approval_fn` again on the revised edit — the revision loop stays inside `execute()`
- `_show_diff()` and the summary print are only called when `approval_fn` is `None` (CLI mode). In programmatic mode, `edit_presented` and `execution_complete` events carry that information instead.

#### Jarvis Usage Examples

```python
# Auto-approve all edits
plan = executor.execute(plan, approval_fn=lambda edit: ApprovalDecision("apply"))

# Relay to user via Jarvis UI
def ask_user(edit):
    jarvis.show(f"Proposed edit to {edit.file}: {edit.description}")
    choice = jarvis.prompt("[apply/skip/revise/quit]")
    if choice == "revise":
        feedback = jarvis.prompt("Feedback:")
        return ApprovalDecision("revise", feedback)
    return ApprovalDecision(choice)

plan = executor.execute(plan, approval_fn=ask_user, on_event=jarvis.on_event)
```

---

### 3. `src/narration.py`

New module with a single public function:

```python
def narrate_event(event_type: str, data: dict) -> str:
    """Return a human-readable string for any on_event payload. Returns '' for unknown types."""
```

Hardcoded string templates — no LLM involved. Fast, free, predictable. Provides a useful default; Jarvis is free to ignore it and build richer LLM-narrated commentary using the raw `event_type` and `data` from `on_event`.

| Input | Output |
|---|---|
| `tool_call`, `{"tool": "search_codebase", "input": {"query": "retry logic"}}` | `Searching: "retry logic"` |
| `tool_call`, `{"tool": "trace_flow", "input": {"entry_point": "embed"}}` | `Tracing: embed` |
| `tool_call`, `{"tool": "read_file", "input": {"path": "src/embedder.py"}}` | `Reading: src/embedder.py` |
| `planning_started`, `{"task": "add retry"}` | `Planning: "add retry"` |
| `planning_complete`, `{"edit_count": 3}` | `Plan ready: 3 edit(s) proposed` |
| `review_started`, `{}` | `Reviewing changes...` |
| `review_complete`, `{"issue_count": 2, "critical_count": 1, "suggest_fix_plan": True}` | `Review done: 2 issue(s), 1 critical` |
| `edit_presented`, `{"index": 2, "total": 3, "file": "src/foo.py", "description": "add retry"}` | `Edit 2/3 — src/foo.py: add retry` |
| `edit_applied`, `{"file": "src/foo.py"}` | `Applied: src/foo.py` |
| `edit_skipped`, `{"file": "src/foo.py"}` | `Skipped: src/foo.py` |
| `edit_revised`, `{"file": "src/foo.py"}` | `Revised: src/foo.py` |
| `execution_complete`, `{"applied": 2, "skipped": 1}` | `Done: 2 applied, 1 skipped` |

Wiring in one line:

```python
agent.ask(question, on_event=lambda t, d: print(narrate_event(t, d)))
```

---

## Files Changed

| File | Change type | Summary |
|---|---|---|
| `src/plan_store.py` | Edit | Add `ApprovalDecision` dataclass |
| `src/llm.py` | Edit | Rename `on_tool_call` → `on_event`; fire `{"tool": ..., "input": ...}` |
| `src/agent_loop.py` | Edit | Rename `on_tool_call` → `on_event`; pass through to `llm.respond()` |
| `src/planner.py` | Edit | Add `on_event` to `plan()`, `revise()`, `_run()`; fire `planning_started`, `tool_call`, `planning_complete` |
| `src/reviewer.py` | Edit | Add `on_event` to `review()`; fire `review_started`, `tool_call`, `review_complete` |
| `src/executor.py` | Edit | Add `approval_fn`, `on_event` to `execute()`; fire all edit/execution events; guard `_show_diff()` behind CLI mode |
| `src/narration.py` | New | `narrate_event(event_type, data) -> str` |
| `agent.py` | Edit | Update call sites: `on_tool_call=` → `on_event=`; pass lambda that fires `narrate_event()` |
| `tests/test_llm.py` | Edit | Rename `on_tool_call` → `on_event` in existing tests |
| `tests/test_agent_loop.py` | Edit | Rename `on_tool_call` → `on_event` in existing tests |
| `tests/test_planner.py` | Edit | Add `on_event` callback assertion tests |
| `tests/test_reviewer.py` | Edit | Add `on_event` callback assertion tests |
| `tests/test_executor.py` | Edit | Replace `patch("builtins.input")` with `approval_fn`; add `on_event` tests |
| `tests/test_narration.py` | New | One test per event type |

No changes to `src/indexer.py`, `src/store.py`, `src/embedder.py`, `src/tools.py`, `src/models.py`.

---

## Non-Goals

- Async/await versions of agent methods (Jarvis uses `asyncio.to_thread`)
- A `CodingAgentClient` facade class (Jarvis builds its own thin wrapper)
- LLM-narrated events (Jarvis can layer this on top using raw `on_event` data)
- HTTP/RPC interface (import as Python library only)
