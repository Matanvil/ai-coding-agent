# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_agent_loop.py

# Run a single test
pytest tests/test_agent_loop.py::test_function_name

# Run the agent
python agent.py [repo_name]
```

No build or lint tooling is configured.

## Architecture

This is a Python CLI that implements an AI coding assistant with RAG (retrieval-augmented generation) over local codebases. It uses local Ollama embeddings and Claude as the reasoning model.

### Core Pipeline

**Indexing:** `indexer.py` chunks files from a repo (naive: 50-line splits, or semantic: Python AST boundaries), embeds via `embedder.py` (Ollama `nomic-embed-text`), and stores in ChromaDB (`store.py`).

**Retrieval:** `tools.py` implements three tools: `search_codebase` (embed query → similarity search), `trace_flow` (keyword search + semantic expansion), and `read_file` (safe file reads with path validation).

**Agents:** All three agents use the same ReAct loop pattern — query LLM → receive tool calls or final answer → execute tools → loop until `end_turn`:
- `agent_loop.py` (`AgentLoop`) — Q&A over indexed repos; auto-summarizes history at 10 turns
- `planner.py` (`Planner`) — Generates lists of `FileEdit` objects (file, old_code, new_code); saves as JSON plans
- `reviewer.py` (`Reviewer`) — Analyzes `git diff HEAD` and outputs categorized issues (critical/important/suggestions)

**Execution:** `executor.py` (`Executor`) reads a saved plan, shows diffs, awaits user approval, then applies edits by exact string replacement.

**Narration:** `narration.py` provides `narrate_event(event_type, data) -> str` — converts structured `on_event` callback payloads into human-readable strings for display or logging.

### Key Relationships

```
agent.py (REPL)
├── AgentLoop → ClaudeClient + tools.py
├── Planner → ClaudeClient + tools.py → plan_store.py
├── Executor → plan_store.py → writes files
└── Reviewer → ClaudeClient + tools.py + git diff
```

`ClaudeClient` in `llm.py` defines the tool schemas and drives the ReAct loop. All agents inject their own `_tool_handler` to dispatch tool calls to the appropriate implementations.

All agent methods that do LLM work accept an optional `on_event(event_type: str, data: dict) -> None` callback. Standard event types: `tool_call`, `planning_started`, `planning_complete`, `review_started`, `review_complete`, `edit_presented`, `edit_applied`, `edit_skipped`, `edit_revised`, `execution_complete`. `Executor.execute()` also accepts `approval_fn(edit: FileEdit) -> ApprovalDecision` to replace stdin approval for programmatic use.

### Configuration

`config.json` is the runtime state file — it stores active repo, repo paths/index timestamps, model names, Ollama URL, and API key. Do not commit API keys; they go in `.env` or `config.json` (gitignored).

### Plan Storage

Plans are JSON files in `plans/` with status tracking (`pending` → `in_progress` → `completed`). Each edit stores exact `old_code` strings that must match the file content for replacement to succeed.
