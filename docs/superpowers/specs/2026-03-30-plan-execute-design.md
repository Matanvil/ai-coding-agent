# Plan / Execute Design Spec

**Date:** 2026-03-30
**Status:** Approved

---

## Overview

Add autonomous code-editing capability to the agent via two cooperating components: a **Planner** that reads the codebase and produces a structured list of file edits, and an **Executor** that walks through those edits one at a time, presenting a diff and waiting for approval before writing anything to disk. Plans are persisted in a `plans/` directory so execution can resume across restarts.

The design mirrors how Claude Code CLI works: the agent figures out exactly what needs to change, you approve each edit, and it applies them. No code is written without explicit approval.

Future extension points: a reviewer/verifier agent that runs tests after each applied edit (not in scope here).

---

## Architecture

```
plan "add a cache layer"
  → Planner uses search + read_file tools to understand codebase
  → Calls submit_plan tool with structured list of FileEdit objects
  → Plan saved to plans/jarvis-2026-03-30-14:23.json (status: "pending")
  → Summary printed in terminal

plan revise "<feedback>"
  → Re-runs Planner with original task + current plan + feedback
  → Saves revised plan (replaces current pending plan)
  → Updated summary printed

execute
  → Executor loads most recent pending/in_progress plan for active repo
  → For each edit: shows before/after diff, prompts [a]pply/[s]kip/[r]evise/[q]uit
  → [r]evise: re-runs LLM on single edit with feedback, shows updated diff
  → Saves progress after each step
  → On completion: plan status → "completed", summary printed
```

---

## File Structure

```
src/
  plan_store.py   ← Plan + FileEdit dataclasses, load/save/list JSON
  planner.py      ← Planner class: ReAct agent with submit_plan tool
  executor.py     ← Executor class: diff display, approval loop, file writes
agent.py          ← Add plan/execute/plans commands
plans/            ← Persistent plan files (plans/<repo>-<YYYY-MM-DD-HH:MM>.json)
tests/
  test_plan_store.py
  test_planner.py
  test_executor.py
```

---

## Data Model — `src/plan_store.py`

```python
@dataclass
class FileEdit:
    file: str          # path relative to repo root
    description: str   # human-readable description of this edit
    old_code: str      # exact string to replace ("" = new file)
    new_code: str      # replacement string
    status: str        # "pending" | "applied" | "rejected"

@dataclass
class Plan:
    task: str          # original user request
    repo: str          # active repo name at creation time
    created_at: str    # "2026-03-30 14:23"
    status: str        # "pending" | "in_progress" | "completed"
    edits: List[FileEdit]
```

**Filename:** `plans/<repo>-<YYYY-MM-DD-HH:MM>.json`

**Functions:**
- `save_plan(plan, plans_dir)` → writes JSON to correct path
- `load_plan(path)` → returns `Plan`
- `list_plans(repo, plans_dir)` → returns all plans for repo sorted newest-first
- `get_active_plan(repo, plans_dir)` → returns most recent `pending` or `in_progress` plan, or `None`

---

## Planner — `src/planner.py`

A ReAct-style agent with three tools. The `submit_plan` tool is the structured output mechanism — the LLM calls it when it's done reasoning, ending the loop and returning a validated `Plan`.

**Tools:**
- `search_codebase(query)` — same as existing agent
- `read_file(path)` — same as existing agent
- `submit_plan(task, edits)` — structured output: `edits` is a list of `{file, description, old_code, new_code}`

**System prompt:** Expert coding assistant. Read the codebase carefully. Produce minimal, targeted edits. Each edit must have an exact `old_code` string that exists verbatim in the file. Prefer multiple small edits over one large one.

**Interface:**
```python
class Planner:
    def __init__(self, llm, embedder, store, repo_root):
        ...

    def plan(self, task: str) -> Plan:
        """Run the planner agent. Returns a Plan or raises PlannerError."""

    def revise(self, plan: Plan, feedback: str) -> Plan:
        """Re-run planner with existing plan + feedback. Returns revised Plan."""
```

**`revise` context:** passes original task + current plan edits (as readable text) + feedback to the planner. The planner may keep, modify, add, or remove edits.

**Failure:** If `submit_plan` is never called within max iterations, raises `PlannerError("Planner could not produce a plan. Try a more specific task.")`.

---

## Executor — `src/executor.py`

Loads the active plan and walks through each `FileEdit` sequentially. Applies edits by string replacement. Saves plan state after every user decision.

**Diff display format:**
```
Edit 1/3 — src/embedder.py
Add cache dict to __init__

--- before ---
def __init__(self, model, base_url):
    self.model = model
    self.base_url = base_url

+++ after +++
def __init__(self, model, base_url):
    self.model = model
    self.base_url = base_url
    self._cache: dict = {}

[a]pply / [s]kip / [r]evise / [q]uit:
```

**Approval options:**
- `a` — write edit to disk, mark `status: "applied"`, save plan, advance
- `s` — mark `status: "rejected"`, save plan, advance
- `r` — prompt for feedback, re-run LLM on this single edit, show updated diff, loop
- `q` — save plan as `"in_progress"`, exit executor (resumable)

**Apply logic:** `file_content.replace(old_code, new_code, 1)`. If `old_code == ""` — create new file. If `old_code` not found in file — skip with warning, mark `status: "rejected"`.

**Completion:** When all edits are processed (applied or rejected), set plan `status: "completed"`. Print:
```
Done. 2 applied, 1 skipped.
```

**`revise` single edit:** Re-runs LLM with the original edit description + `old_code` context + user feedback. Returns a revised `FileEdit`. Does not touch other edits in the plan.

**Interface:**
```python
class Executor:
    def __init__(self, llm, repo_root, plans_dir):
        ...

    def execute(self, plan: Plan) -> Plan:
        """Walk through plan edits interactively. Returns updated plan."""
```

---

## REPL Commands — `agent.py`

```
plan <task>           Create a new plan for the given task
plan revise <feedback>  Revise current pending plan with feedback
execute               Execute current pending/in_progress plan
plans                 List all plans for the active repo
plan clear            Discard current pending/in_progress plan
```

**`plan <task>`:**
- Errors if no active repo
- If a pending/in_progress plan exists: `"A plan already exists for <repo>. Overwrite? [y/n]"`
- On success, prints plan summary:
  ```
  Plan created: 3 edits for "add a cache layer"
    1. src/embedder.py — Add cache dict to __init__
    2. src/embedder.py — Check cache before Ollama call
    3. src/embedder.py — Store result in cache after call
  Run 'execute' to apply.
  ```

**`plan revise <feedback>`:**
- Errors if no pending plan
- Re-runs planner with feedback, overwrites the existing plan file (same filename, same timestamp — revisions are not versioned separately), shows updated summary

**`execute`:**
- Errors if no pending/in_progress plan for active repo
- If plan repo ≠ active repo: `"Plan was created for '<other>'. Switch to it with 'use <other>' first."`
- Runs executor interactively

**`plans`:**
```
Plans for jarvis:
  [completed]    2026-03-30 14:23  "add a cache layer"           (3 edits)
  [pending]      2026-03-30 15:45  "extract retry logic"         (2 edits)
```

**`plan clear`:**
- Deletes current pending/in_progress plan file after confirming: `"Discard current plan? [y/n]"`

---

## Error Handling

| Situation | Message |
|---|---|
| `plan` / `execute` with no active repo | `"No active repo. Type 'use <repo>' to select one."` |
| `execute` with no pending plan | `"No pending plan for <repo>. Run: plan <task>"` |
| `execute` when plan repo ≠ active repo | `"Plan was created for '<other>'. Switch to it with 'use <other>' first."` |
| `plan` when pending plan exists | `"A plan already exists for <repo>. Overwrite? [y/n]"` |
| `plan revise` with no pending plan | `"No pending plan to revise. Run: plan <task>"` |
| Planner never calls `submit_plan` | `"Planner could not produce a plan. Try a more specific task."` |
| `old_code` not found in file during apply | `"Warning: could not apply edit to <file> — code not found. Skipping."` |

---

## Testing

**`test_plan_store.py`:**
- Save and reload a plan round-trip
- `list_plans` returns plans sorted newest-first
- `get_active_plan` returns most recent pending/in_progress, ignores completed
- `get_active_plan` returns `None` when none exist

**`test_planner.py`:**
- Mock LLM that calls `submit_plan` on first iteration → returns valid `Plan`
- Mock LLM that never calls `submit_plan` → raises `PlannerError`
- `revise` passes existing plan edits + feedback in context

**`test_executor.py`:**
- Mock file I/O + user input `"a"` → edit applied, status `"applied"`
- Mock user input `"s"` → status `"rejected"`
- Mock user input `"q"` → plan status `"in_progress"`, loop exits
- `old_code` not found → warning printed, status `"rejected"`, advances
- All edits processed → plan status `"completed"`

No automated tests for diff display or `revise` prompt input — manual verification only.
