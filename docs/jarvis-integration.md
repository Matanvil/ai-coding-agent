# AI Coding Agent — Jarvis Integration Guide

## Overview

The AI Coding Agent is a Python-based, locally-running RAG system that answers questions about codebases, plans code edits, executes those edits, and reviews code changes. This document describes how it works internally and how to integrate it programmatically — including live narration support for use inside an orchestration loop like Jarvis.

All integration gaps have been resolved. The agent is fully usable as a Python library with no CLI required.

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  agent.py (CLI REPL)                                        │
│    Commands: ask / plan / execute / review / index / use    │
└────────────────┬────────────────────────────────────────────┘
                 │
     ┌───────────┼────────────────┬──────────────┐
     ▼           ▼                ▼              ▼
AgentLoop     Planner         Reviewer       Executor
(Q&A)       (edit plans)   (review diffs)  (apply edits)
     │           │                │
     └─────┬─────┘                │
           ▼                      │
      ClaudeClient ◄──────────────┘
      (ReAct loop)
           │
     ┌─────┴──────────────┐
     ▼                    ▼
OllamaEmbedder       VectorStore
(nomic-embed-text)   (ChromaDB)
```

### ReAct Loop (All Agents)

Every agent — Q&A, Planner, Reviewer — runs the same loop:

```
query LLM (Claude Haiku) with tool definitions
  ↓
receive response: tool_use OR end_turn
  ↓ if tool_use
  execute tool (search_codebase / read_file / trace_flow / submit_*)
  append result to messages
  loop again
  ↓ if end_turn
  return final text answer
```

Tools available per agent:

| Agent      | Tools                                              |
|------------|----------------------------------------------------|
| AgentLoop  | search_codebase, trace_flow, read_file             |
| Planner    | search_codebase, read_file, submit_plan            |
| Reviewer   | search_codebase, read_file, submit_review          |
| Executor   | (no LLM loop; uses Planner for inline revisions)   |

---

### Indexing Pipeline

Before any agent can answer questions, a repo must be indexed:

```python
index_repo(repo_path, embedder, store)
```

1. Walk the repo tree (skips `.git`, `node_modules`, `__pycache__`, etc.)
2. Chunk each file — either naive (50-line windows) or semantic (AST boundaries for Python)
3. Embed each chunk via Ollama (`nomic-embed-text`, 768-dim)
4. Store chunk text + embedding in ChromaDB (cosine distance, HNSW index)

ChromaDB persists to `.chroma/` relative to the working directory. Each repo gets its own named collection.

---

### Data Models

**`Plan`** — output of Planner, input to Executor:
```python
@dataclass
class Plan:
    task: str
    repo: str
    created_at: str           # "2026-03-30 14:23"
    status: str               # "pending" | "in_progress" | "completed"
    edits: List[FileEdit]
```

**`FileEdit`** — one file modification:
```python
@dataclass
class FileEdit:
    file: str          # relative path
    description: str   # human-readable intent
    old_code: str      # exact string to find and replace
    new_code: str      # replacement content
    status: str        # "pending" | "applied" | "rejected"
```

**`ApprovalDecision`** — programmatic executor input:
```python
@dataclass
class ApprovalDecision:
    action: str   # "apply" | "skip" | "quit" | "revise"
    feedback: str = ""  # only used when action == "revise"
```

**`ReviewResult`** — output of Reviewer:
```python
@dataclass
class ReviewResult:
    summary: str
    issues: list[ReviewIssue]
    suggest_fix_plan: bool

@dataclass
class ReviewIssue:
    category: str          # "critical" | "important" | "suggestion"
    description: str
    file: str              # empty string if not file-specific
    recommendation: str
```

---

## Programmatic Integration

The entire system is importable as a Python library. No CLI required.

### Minimal Setup

```python
from src.config import load_config
from src.llm import ClaudeClient
from src.embedder import OllamaEmbedder
from src.store import VectorStore
from src.agent_loop import AgentLoop
from src.planner import Planner
from src.reviewer import Reviewer
from src.executor import Executor
from src.narration import narrate_event
from src.plan_store import ApprovalDecision
import subprocess

config = load_config("config.json")

# Shared components (create once per process)
llm = ClaudeClient(model=config.model, api_key=config.api_key)
embedder = OllamaEmbedder(model=config.embedding_model, base_url=config.ollama_url)

# Per-repo components
repo = config.repos["my-repo"]
store = VectorStore(chroma_path=config.chroma_path, collection_name="my-repo")
agent = AgentLoop(llm=llm, embedder=embedder, store=store, repo_root=repo["path"])
planner = Planner(llm=llm, embedder=embedder, store=store, repo_root=repo["path"])
reviewer = Reviewer(llm=llm, embedder=embedder, store=store, repo_root=repo["path"])
executor = Executor(llm=llm, repo_root=repo["path"], plans_dir="plans")
```

---

## The `on_event` Callback

All agent methods accept an optional `on_event(event_type: str, data: dict) -> None` callback. It fires at every meaningful step — tool calls, lifecycle transitions, and edit decisions.

### Standard Event Types

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

`submit_plan` and `submit_review` are internal termination signals — they do not fire `tool_call` events.

### `narrate_event()` — built-in formatter

`src/narration.py` provides a utility that converts any event into a human-readable string:

```python
from src.narration import narrate_event

narrate_event("tool_call", {"tool": "search_codebase", "input": {"query": "retry logic"}})
# → 'Searching: "retry logic"'

narrate_event("planning_complete", {"edit_count": 3})
# → "Plan ready: 3 edits proposed"

narrate_event("review_complete", {"issue_count": 2, "critical_count": 1, "suggest_fix_plan": True})
# → "Review done: 2 issues, 1 critical"

narrate_event("edit_presented", {"index": 2, "total": 3, "file": "src/foo.py", "description": "add retry"})
# → "Edit 2/3 — src/foo.py: add retry"

narrate_event("some_unknown_event", {})
# → ""  (empty string — safe to ignore)
```

Jarvis can use `narrate_event()` as a default formatter, or ignore it entirely and build richer LLM-narrated commentary from the raw event data.

---

## Q&A

```python
def on_event(event_type, data):
    msg = narrate_event(event_type, data)
    if msg:
        print(f" → {msg}")

answer = agent.ask("How does the embedder handle timeouts?", on_event=on_event)
print(answer)
```

Example output:
```
 → Searching: "embedder timeout handling"
 → Reading: src/embedder.py
The embedder raises EmbedderError on connection timeout...
```

---

## Planning

```python
def on_event(event_type, data):
    msg = narrate_event(event_type, data)
    if msg:
        print(f" → {msg}")

plan = planner.plan(
    task="Add retry logic to OllamaEmbedder.embed()",
    repo="my-repo",
    on_event=on_event,
)
for edit in plan.edits:
    print(f"  {edit.file}: {edit.description}")
```

Example output:
```
 → Planning: "Add retry logic to OllamaEmbedder.embed()"
 → Searching: "OllamaEmbedder embed method"
 → Reading: src/embedder.py
 → Plan ready: 2 edits proposed
  src/embedder.py: add retry loop with backoff
  src/embedder.py: add max_retries parameter
```

`planner.revise(plan, feedback, on_event=on_event)` follows the same pattern.

---

## Execution

`Executor.execute()` accepts an optional `approval_fn(edit: FileEdit) -> ApprovalDecision`. When provided, `input()` is never called and the executor is fully programmatic. When omitted, CLI behavior is preserved unchanged.

**Auto-approve all edits:**
```python
result = executor.execute(
    plan,
    approval_fn=lambda edit: ApprovalDecision("apply"),
    on_event=on_event,
)
```

**Relay approval to user via Jarvis UI:**
```python
def ask_user(edit: FileEdit) -> ApprovalDecision:
    jarvis.show(f"Proposed: {edit.file} — {edit.description}")
    jarvis.show_diff(edit.old_code, edit.new_code)
    choice = jarvis.prompt("[apply/skip/revise/quit]")
    if choice == "revise":
        feedback = jarvis.prompt("Feedback:")
        return ApprovalDecision("revise", feedback)
    return ApprovalDecision(choice)

result = executor.execute(plan, approval_fn=ask_user, on_event=on_event)
```

When `action == "revise"`, the executor calls its internal `_revise_edit()` with the feedback, fires `edit_revised`, then calls `approval_fn` again on the updated edit. Jarvis does not need to implement the revision loop.

---

## Review

```python
diff = subprocess.run(
    ["git", "diff", "HEAD"],
    cwd=repo["path"], capture_output=True, text=True
).stdout

result = reviewer.review(diff=diff, context="Added retry logic", on_event=on_event)

print(result.summary)
for issue in result.issues:
    print(f"[{issue.category}] {issue.file}: {issue.description}")
    print(f"  → {issue.recommendation}")
if result.suggest_fix_plan:
    print("Reviewer suggests a follow-up fix plan.")
```

---

## Full Jarvis Loop

```python
from src.config import load_config
from src.llm import ClaudeClient
from src.embedder import OllamaEmbedder
from src.store import VectorStore
from src.agent_loop import AgentLoop
from src.planner import Planner
from src.reviewer import Reviewer
from src.executor import Executor
from src.narration import narrate_event
from src.plan_store import ApprovalDecision
import subprocess


def make_on_event(narrator):
    def on_event(event_type, data):
        msg = narrate_event(event_type, data)
        if msg:
            narrator(msg)
    return on_event


def jarvis_coding_task(task: str, repo_name: str, narrator, approval_fn=None):
    config = load_config("config.json")
    repo = config.repos[repo_name]
    on_event = make_on_event(narrator)

    llm = ClaudeClient(model=config.model, api_key=config.api_key)
    embedder = OllamaEmbedder(model=config.embedding_model, base_url=config.ollama_url)
    store = VectorStore(chroma_path=config.chroma_path, collection_name=repo_name)
    root = repo["path"]

    agent = AgentLoop(llm, embedder, store, root)
    planner = Planner(llm, embedder, store, root)
    reviewer = Reviewer(llm, embedder, store, root)
    executor = Executor(llm, root, plans_dir="plans")

    # Step 1: Understand context
    context = agent.ask(f"Summarise the area relevant to: {task}", on_event=on_event)

    # Step 2: Plan
    plan = planner.plan(task, repo_name, on_event=on_event)
    for edit in plan.edits:
        narrator(f"  [{edit.file}] {edit.description}")

    # Step 3: Execute
    auto_approve = approval_fn or (lambda e: ApprovalDecision("apply"))
    result = executor.execute(plan, approval_fn=auto_approve, on_event=on_event)

    # Step 4: Review
    diff = subprocess.run(
        ["git", "diff", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout
    if diff:
        review = reviewer.review(diff, context=task, on_event=on_event)
        if review.suggest_fix_plan:
            fix = planner.plan(f"Fix review issues: {review.summary}", repo_name, on_event=on_event)
            executor.execute(fix, approval_fn=auto_approve, on_event=on_event)
```

---

## Async Usage

All operations are synchronous. Use `asyncio.to_thread` from Jarvis's side — no changes needed:

```python
import asyncio

answer = await asyncio.to_thread(agent.ask, question, on_event=on_event)
plan = await asyncio.to_thread(planner.plan, task, repo_name, on_event=on_event)
```
