# Plan / Execute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Planner agent that reads the codebase and proposes structured file edits, and an Executor that presents each edit as a diff and applies it on approval.

**Architecture:** Four sequential tasks. Task 1 builds the data model and persistence layer. Task 2 implements the Planner (ReAct agent with a `submit_plan` structured-output tool). Task 3 implements the Executor (approval loop with file writes). Task 4 wires all three into the REPL with `plan`, `execute`, `plans` commands.

**Tech Stack:** Python, Anthropic SDK (already present), dataclasses, pathlib for file I/O.

---

## File Structure

```
src/
  plan_store.py   ← NEW: Plan + FileEdit dataclasses, save/load/list/delete JSON
  planner.py      ← NEW: Planner class with plan() and revise() — ReAct + submit_plan tool
  executor.py     ← NEW: Executor class with execute() — diff display, approval loop, file writes
agent.py          ← MODIFY: add plan/execute/plans commands to REPL
plans/            ← already created (empty dir)
tests/
  test_plan_store.py  ← NEW
  test_planner.py     ← NEW
  test_executor.py    ← NEW
```

---

## Task 1: Plan Store — Data Model and Persistence

**Files:**
- Create: `src/plan_store.py`
- Create: `tests/test_plan_store.py`

---

- [ ] **Step 1: Write 4 failing tests — create `tests/test_plan_store.py`**

```python
import pytest
from src.plan_store import FileEdit, Plan, save_plan, load_plan, list_plans, get_active_plan, delete_plan, plan_filepath


def _make_plan(task="add cache", repo="myrepo", created_at="2026-03-30 14:23", status="pending"):
    return Plan(
        task=task,
        repo=repo,
        created_at=created_at,
        status=status,
        edits=[
            FileEdit(file="src/foo.py", description="change a", old_code="a = 1", new_code="a = 42", status="pending")
        ],
    )


def test_save_and_load_round_trip(tmp_path):
    plan = _make_plan()
    save_plan(plan, str(tmp_path))
    path = plan_filepath(plan, str(tmp_path))
    loaded = load_plan(str(path))
    assert loaded.task == "add cache"
    assert loaded.repo == "myrepo"
    assert loaded.status == "pending"
    assert len(loaded.edits) == 1
    assert loaded.edits[0].file == "src/foo.py"
    assert loaded.edits[0].old_code == "a = 1"
    assert loaded.edits[0].status == "pending"


def test_list_plans_sorted_newest_first(tmp_path):
    for ts in ["2026-03-30 14:00", "2026-03-30 15:00", "2026-03-30 13:00"]:
        save_plan(_make_plan(created_at=ts), str(tmp_path))
    plans = list_plans("myrepo", str(tmp_path))
    assert [p.created_at for p in plans] == [
        "2026-03-30 15:00",
        "2026-03-30 14:00",
        "2026-03-30 13:00",
    ]


def test_get_active_plan_ignores_completed(tmp_path):
    save_plan(_make_plan(created_at="2026-03-30 14:00", status="completed"), str(tmp_path))
    save_plan(_make_plan(created_at="2026-03-30 15:00", status="pending"), str(tmp_path))
    active = get_active_plan("myrepo", str(tmp_path))
    assert active is not None
    assert active.created_at == "2026-03-30 15:00"


def test_get_active_plan_returns_none_when_all_completed(tmp_path):
    save_plan(_make_plan(status="completed"), str(tmp_path))
    assert get_active_plan("myrepo", str(tmp_path)) is None


def test_get_active_plan_returns_none_when_empty(tmp_path):
    assert get_active_plan("myrepo", str(tmp_path)) is None


def test_delete_plan_removes_file(tmp_path):
    plan = _make_plan()
    save_plan(plan, str(tmp_path))
    path = plan_filepath(plan, str(tmp_path))
    assert path.exists()
    delete_plan(plan, str(tmp_path))
    assert not path.exists()
```

- [ ] **Step 2: Run tests to confirm they FAIL**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest tests/test_plan_store.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `plan_store` does not exist yet.

- [ ] **Step 3: Create `src/plan_store.py`**

```python
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


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


def plan_filepath(plan: Plan, plans_dir: str) -> Path:
    """Return the filesystem path for this plan's JSON file."""
    safe_ts = plan.created_at.replace(":", "-").replace(" ", "_")
    return Path(plans_dir) / f"{plan.repo}-{safe_ts}.json"


def save_plan(plan: Plan, plans_dir: str) -> Path:
    """Write plan to disk. Creates plans_dir if needed. Returns the file path."""
    Path(plans_dir).mkdir(parents=True, exist_ok=True)
    path = plan_filepath(plan, plans_dir)
    data = {
        "task": plan.task,
        "repo": plan.repo,
        "created_at": plan.created_at,
        "status": plan.status,
        "edits": [asdict(e) for e in plan.edits],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_plan(path: str) -> Plan:
    """Load a Plan from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    edits = [FileEdit(**e) for e in data["edits"]]
    return Plan(
        task=data["task"],
        repo=data["repo"],
        created_at=data["created_at"],
        status=data["status"],
        edits=edits,
    )


def list_plans(repo: str, plans_dir: str) -> List[Plan]:
    """Return all plans for a repo sorted newest-first."""
    plans_path = Path(plans_dir)
    if not plans_path.exists():
        return []
    plans = []
    for f in plans_path.glob(f"{repo}-*.json"):
        try:
            plans.append(load_plan(str(f)))
        except Exception:
            continue
    return sorted(plans, key=lambda p: p.created_at, reverse=True)


def get_active_plan(repo: str, plans_dir: str) -> Optional[Plan]:
    """Return the most recent pending or in_progress plan for a repo, or None."""
    for plan in list_plans(repo, plans_dir):
        if plan.status in ("pending", "in_progress"):
            return plan
    return None


def delete_plan(plan: Plan, plans_dir: str) -> None:
    """Delete a plan's JSON file from disk."""
    plan_filepath(plan, plans_dir).unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests to confirm they PASS**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest tests/test_plan_store.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && git add src/plan_store.py tests/test_plan_store.py && git commit -m "feat: plan store — Plan/FileEdit dataclasses, save/load/list/delete"
```

---

## Task 2: Planner — ReAct Agent with Structured Output

**Files:**
- Create: `src/planner.py`
- Create: `tests/test_planner.py`

---

- [ ] **Step 1: Write 3 failing tests — create `tests/test_planner.py`**

```python
import pytest
from unittest.mock import MagicMock
from src.planner import Planner, PlannerError
from src.plan_store import Plan, FileEdit


def _make_submit_plan_response(edits):
    """Mock Anthropic response that calls submit_plan."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "submit_plan"
    block.id = "tool_1"
    block.input = {"edits": edits}
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


def _make_end_turn_response():
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = []
    return response


def test_plan_returns_plan_when_submit_plan_called():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_plan_response([
        {"file": "src/foo.py", "description": "add x", "old_code": "a = 1", "new_code": "a = 42"},
    ])
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    store = MagicMock()
    store.search.return_value = []
    store.keyword_search.return_value = []

    planner = Planner(llm=llm, embedder=embedder, store=store, repo_root="/repo")
    plan = planner.plan("add a cache layer", repo="myrepo")

    assert isinstance(plan, Plan)
    assert plan.repo == "myrepo"
    assert plan.task == "add a cache layer"
    assert plan.status == "pending"
    assert len(plan.edits) == 1
    assert plan.edits[0].file == "src/foo.py"
    assert plan.edits[0].status == "pending"


def test_plan_raises_planner_error_when_submit_plan_not_called():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_end_turn_response()

    planner = Planner(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")

    with pytest.raises(PlannerError):
        planner.plan("add a cache layer", repo="myrepo")


def test_revise_includes_original_task_and_feedback_in_message():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_plan_response([
        {"file": "src/foo.py", "description": "revised", "old_code": "x", "new_code": "y"},
    ])
    original_plan = Plan(
        task="add cache",
        repo="myrepo",
        created_at="2026-03-30 14:00",
        status="pending",
        edits=[FileEdit(file="src/foo.py", description="old", old_code="old", new_code="new", status="pending")],
    )

    planner = Planner(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    revised = planner.revise(original_plan, "use a dict instead")

    call_args = llm.client.messages.create.call_args
    messages = call_args.kwargs["messages"]
    assert "add cache" in messages[0]["content"]
    assert "use a dict instead" in messages[0]["content"]
    assert revised.task == "add cache"   # original task preserved
    assert revised.repo == "myrepo"
```

- [ ] **Step 2: Run tests to confirm they FAIL**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest tests/test_planner.py -v
```

Expected: `ModuleNotFoundError` — `planner` does not exist yet.

- [ ] **Step 3: Create `src/planner.py`**

```python
from datetime import datetime
from src.plan_store import Plan, FileEdit
from src.tools import search_codebase, read_file
from src.agent_loop import format_chunks

PLANNER_SYSTEM_PROMPT = """You are an expert coding assistant tasked with planning code changes.

Your job:
1. Search the codebase to understand the relevant code
2. Read specific files to get the exact current content
3. When you have a complete understanding, call submit_plan with a list of targeted edits

Rules for edits:
- Each edit's old_code must be an EXACT string copied verbatim from the file
- Prefer multiple small edits over one large replacement
- Only include files that actually need to change
- Be minimal — do not change what does not need to change"""

PLANNER_TOOL_DEFINITIONS = [
    {
        "name": "search_codebase",
        "description": "Search the indexed codebase for relevant code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the full contents of a file in the repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path from repo root"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "submit_plan",
        "description": "Submit the completed plan of file edits. Call this when you are ready.",
        "input_schema": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "List of file edits",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "description": {"type": "string"},
                            "old_code": {"type": "string"},
                            "new_code": {"type": "string"},
                        },
                        "required": ["file", "description", "old_code", "new_code"],
                    },
                }
            },
            "required": ["edits"],
        },
    },
]


class PlannerError(Exception):
    pass


class Planner:
    def __init__(self, llm, embedder, store, repo_root: str, max_iterations: int = 15):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.repo_root = repo_root
        self.max_iterations = max_iterations

    def _tool_handler(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "search_codebase":
            chunks = search_codebase(tool_input["query"], self.embedder, self.store)
            return format_chunks(chunks)
        if tool_name == "read_file":
            try:
                return read_file(tool_input["path"], self.repo_root)
            except (ValueError, FileNotFoundError) as e:
                return f"Error: {e}"
        return f"Unknown tool: {tool_name}"

    def _run(self, messages: list, task: str) -> Plan:
        """Run the planner ReAct loop. Returns Plan when submit_plan is called."""
        current_messages = list(messages)

        for _ in range(self.max_iterations):
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=4096,
                system=PLANNER_SYSTEM_PROMPT,
                tools=PLANNER_TOOL_DEFINITIONS,
                messages=current_messages,
            )

            if response.stop_reason == "tool_use":
                current_messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if block.name == "submit_plan":
                        edits = [
                            FileEdit(
                                file=e["file"],
                                description=e["description"],
                                old_code=e["old_code"],
                                new_code=e["new_code"],
                                status="pending",
                            )
                            for e in block.input["edits"]
                        ]
                        return Plan(
                            task=task,
                            repo="",  # set by caller
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
                            status="pending",
                            edits=edits,
                        )
                    result = self._tool_handler(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                if tool_results:
                    current_messages.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                break

        raise PlannerError("Planner could not produce a plan. Try a more specific task.")

    def plan(self, task: str, repo: str) -> Plan:
        """Run the planner agent. Returns a Plan or raises PlannerError."""
        messages = [{"role": "user", "content": task}]
        plan = self._run(messages, task=task)
        plan.repo = repo
        return plan

    def revise(self, plan: Plan, feedback: str) -> Plan:
        """Re-run planner with existing plan + feedback. Returns revised Plan."""
        edits_text = "\n".join(
            f"  {i + 1}. {e.file} — {e.description}"
            for i, e in enumerate(plan.edits)
        )
        message = (
            f"Original task: {plan.task}\n\n"
            f"Current plan:\n{edits_text}\n\n"
            f"Feedback: {feedback}\n\n"
            f"Please revise the plan based on this feedback."
        )
        revised = self._run([{"role": "user", "content": message}], task=plan.task)
        revised.repo = plan.repo
        revised.task = plan.task  # always preserve original task
        return revised
```

- [ ] **Step 4: Run tests to confirm they PASS**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest tests/test_planner.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && git add src/planner.py tests/test_planner.py && git commit -m "feat: Planner — ReAct agent with submit_plan structured output"
```

---

## Task 3: Executor — Diff Display, Approval Loop, File Writes

**Files:**
- Create: `src/executor.py`
- Create: `tests/test_executor.py`

---

- [ ] **Step 1: Write 5 failing tests — create `tests/test_executor.py`**

```python
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.executor import Executor
from src.plan_store import Plan, FileEdit


def _make_plan(*edits):
    return Plan(
        task="test task",
        repo="myrepo",
        created_at="2026-03-30 14:00",
        status="pending",
        edits=list(edits),
    )


def _edit(file="src/foo.py", old_code="a = 1", new_code="a = 42"):
    return FileEdit(file=file, description="change a", old_code=old_code, new_code=new_code, status="pending")


def test_apply_edit_writes_file_and_marks_applied(tmp_path):
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="a"):
        result = executor.execute(plan)

    assert result.edits[0].status == "applied"
    assert result.status == "completed"
    assert "a = 42" in target.read_text(encoding="utf-8")


def test_skip_edit_marks_rejected(tmp_path):
    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="s"):
        result = executor.execute(plan)

    assert result.edits[0].status == "rejected"
    assert result.status == "completed"


def test_quit_saves_in_progress(tmp_path):
    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="q"):
        result = executor.execute(plan)

    assert result.status == "in_progress"
    assert result.edits[0].status == "pending"  # not yet processed


def test_old_code_not_found_warns_and_marks_rejected(tmp_path, capsys):
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("completely different content\n", encoding="utf-8")

    plan = _make_plan(_edit(old_code="not_in_file"))
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="a"):
        result = executor.execute(plan)

    assert result.edits[0].status == "rejected"
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "code not found" in captured.out


def test_all_edits_processed_marks_completed(tmp_path):
    plan = _make_plan(_edit(file="a.py"), _edit(file="b.py"))
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="s"):
        result = executor.execute(plan)

    assert result.status == "completed"
    assert all(e.status == "rejected" for e in result.edits)
```

- [ ] **Step 2: Run tests to confirm they FAIL**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest tests/test_executor.py -v
```

Expected: `ModuleNotFoundError` — `executor` does not exist yet.

- [ ] **Step 3: Create `src/executor.py`**

```python
from pathlib import Path
from src.plan_store import Plan, FileEdit, save_plan

REVISION_SYSTEM_PROMPT = """You are an expert coding assistant revising a single file edit based on feedback.
Call submit_plan with one revised edit. The edit's old_code must be an exact string from the file."""

REVISION_TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file in the repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path from repo root"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "submit_plan",
        "description": "Submit the single revised edit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "description": {"type": "string"},
                            "old_code": {"type": "string"},
                            "new_code": {"type": "string"},
                        },
                        "required": ["file", "description", "old_code", "new_code"],
                    },
                }
            },
            "required": ["edits"],
        },
    },
]


class Executor:
    def __init__(self, llm, repo_root: str, plans_dir: str):
        self.llm = llm
        self.repo_root = repo_root
        self.plans_dir = plans_dir

    def _show_diff(self, edit: FileEdit, index: int, total: int) -> None:
        print(f"\nEdit {index}/{total} — {edit.file}")
        print(edit.description)
        print()
        print("--- before ---")
        print(edit.old_code if edit.old_code else "(new file)")
        print()
        print("+++ after +++")
        print(edit.new_code)
        print()

    def _apply_edit(self, edit: FileEdit) -> bool:
        """Write edit to disk. Returns True on success, False if old_code not found."""
        repo = Path(self.repo_root).resolve()
        target = (repo / edit.file).resolve()
        try:
            target.relative_to(repo)
        except ValueError:
            print(f"Warning: could not apply edit to {edit.file} — path outside repo. Skipping.")
            return False

        if edit.old_code == "":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edit.new_code, encoding="utf-8")
            return True

        if not target.exists():
            print(f"Warning: could not apply edit to {edit.file} — file not found. Skipping.")
            return False

        content = target.read_text(encoding="utf-8")
        if edit.old_code not in content:
            print(f"Warning: could not apply edit to {edit.file} — code not found. Skipping.")
            return False

        target.write_text(content.replace(edit.old_code, edit.new_code, 1), encoding="utf-8")
        return True

    def _revise_edit(self, edit: FileEdit, feedback: str) -> FileEdit:
        """Ask LLM to revise a single edit. Returns revised FileEdit (fallback: original)."""
        from src.tools import read_file as _read_file

        message = (
            f"File: {edit.file}\n"
            f"Description: {edit.description}\n\n"
            f"Current old_code:\n{edit.old_code}\n\n"
            f"Current new_code:\n{edit.new_code}\n\n"
            f"Feedback: {feedback}\n\n"
            f"Call submit_plan with the revised edit."
        )
        messages = [{"role": "user", "content": message}]

        for _ in range(5):
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=2048,
                system=REVISION_SYSTEM_PROMPT,
                tools=REVISION_TOOL_DEFINITIONS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                break

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "submit_plan":
                    edits_data = block.input.get("edits", [])
                    if edits_data:
                        e = edits_data[0]
                        return FileEdit(
                            file=e.get("file", edit.file),
                            description=e.get("description", edit.description),
                            old_code=e.get("old_code", edit.old_code),
                            new_code=e.get("new_code", edit.new_code),
                            status="pending",
                        )
                elif block.name == "read_file":
                    try:
                        result = _read_file(block.input["path"], self.repo_root)
                    except Exception as e:
                        result = f"Error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        return edit  # fallback: return original unchanged

    def execute(self, plan: Plan) -> Plan:
        """Walk through plan edits interactively. Returns updated plan."""
        plan.status = "in_progress"
        save_plan(plan, self.plans_dir)

        total = len(plan.edits)

        for i, edit in enumerate(plan.edits):
            if edit.status != "pending":
                continue  # resume: skip already-processed edits

            self._show_diff(edit, i + 1, total)

            while True:
                choice = input("[a]pply / [s]kip / [r]evise / [q]uit: ").strip().lower()

                if choice == "a":
                    success = self._apply_edit(edit)
                    edit.status = "applied" if success else "rejected"
                    save_plan(plan, self.plans_dir)
                    break
                elif choice == "s":
                    edit.status = "rejected"
                    save_plan(plan, self.plans_dir)
                    break
                elif choice == "r":
                    feedback = input("Feedback: ").strip()
                    if feedback:
                        revised = self._revise_edit(edit, feedback)
                        edit.file = revised.file
                        edit.description = revised.description
                        edit.old_code = revised.old_code
                        edit.new_code = revised.new_code
                    self._show_diff(edit, i + 1, total)
                elif choice == "q":
                    plan.status = "in_progress"
                    save_plan(plan, self.plans_dir)
                    return plan
                else:
                    print("Please enter a, s, r, or q.")

        plan.status = "completed"
        save_plan(plan, self.plans_dir)

        applied = sum(1 for e in plan.edits if e.status == "applied")
        skipped = sum(1 for e in plan.edits if e.status == "rejected")
        print(f"\nDone. {applied} applied, {skipped} skipped.")

        return plan
```

- [ ] **Step 4: Run tests to confirm they PASS**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest tests/test_executor.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && git add src/executor.py tests/test_executor.py && git commit -m "feat: Executor — diff display, approval loop, file writes"
```

---

## Task 4: agent.py — REPL Commands

**Files:**
- Modify: `agent.py`

No automated tests — verify manually after writing.

---

- [ ] **Step 1: Read current `agent.py`**

Read `/Users/matanvilensky/dev/ai-coding-agent/agent.py` in full before editing.

- [ ] **Step 2: Add imports and `PLANS_DIR` constant**

At the top of `agent.py`, after the existing imports, add:

```python
from src.plan_store import get_active_plan, list_plans, save_plan, delete_plan, plan_filepath
from src.planner import Planner, PlannerError
from src.executor import Executor
```

After the existing constants (`BANNER`, `HELP_TEXT`, `DIVIDER`, `_PROMPT_STYLE`), add:

```python
PLANS_DIR = str(Path(__file__).parent / "plans")
```

- [ ] **Step 3: Update `HELP_TEXT`**

Replace the existing `HELP_TEXT` with:

```python
HELP_TEXT = """
Commands:
  ask <question>              Ask a question about the codebase
  trace <symbol>              Trace where a function/class is defined and used
  index [--repo <path>]       Index or re-index a repository
  use <repo>                  Switch to an indexed repo
  repos                       List all indexed repos
  plan <task>                 Create a plan to edit the codebase
  plan revise <feedback>      Revise the current pending plan
  plan clear                  Discard the current pending plan
  execute                     Execute the current pending plan
  plans                       List all plans for the active repo
  clear                       Clear conversation history
  help                        Show this help
  exit                        Quit

You can also type a question directly without 'ask'.
"""
```

- [ ] **Step 4: Add helper functions before `main()`**

Add these functions after `handle_question` and before `main()`:

```python
def _print_plan_summary(plan) -> None:
    count = len(plan.edits)
    print(f"\nPlan created: {count} edit{'s' if count != 1 else ''} for \"{plan.task}\"")
    for i, edit in enumerate(plan.edits, 1):
        print(f"  {i}. {edit.file} — {edit.description}")
    print("\nRun 'execute' to apply.")


def run_plan(task: str, config, embedder, llm, store) -> None:
    if not config.active_repo or not store:
        print("No active repo. Type 'use <repo>' to select one.")
        return
    active = get_active_plan(config.active_repo, PLANS_DIR)
    if active:
        confirm = input(f"A plan already exists for {config.active_repo}. Overwrite? [y/n]: ").strip().lower()
        if confirm != "y":
            return
    repo_path = config.repos[config.active_repo]["path"]
    planner = Planner(llm=llm, embedder=embedder, store=store, repo_root=repo_path)
    print("Planning...")
    try:
        plan = planner.plan(task=task, repo=config.active_repo)
    except PlannerError as e:
        print(f"Error: {e}")
        return
    save_plan(plan, PLANS_DIR)
    _print_plan_summary(plan)


def run_plan_revise(feedback: str, config, embedder, llm, store) -> None:
    if not config.active_repo or not store:
        print("No active repo. Type 'use <repo>' to select one.")
        return
    active = get_active_plan(config.active_repo, PLANS_DIR)
    if not active:
        print("No pending plan to revise. Run: plan <task>")
        return
    repo_path = config.repos[config.active_repo]["path"]
    planner = Planner(llm=llm, embedder=embedder, store=store, repo_root=repo_path)
    print("Revising plan...")
    try:
        revised = planner.revise(active, feedback)
    except PlannerError as e:
        print(f"Error: {e}")
        return
    revised.created_at = active.created_at  # overwrite same file
    save_plan(revised, PLANS_DIR)
    _print_plan_summary(revised)


def run_plan_clear(config) -> None:
    if not config.active_repo:
        print("No active repo. Type 'use <repo>' to select one.")
        return
    active = get_active_plan(config.active_repo, PLANS_DIR)
    if not active:
        print(f"No pending plan for {config.active_repo}.")
        return
    confirm = input("Discard current plan? [y/n]: ").strip().lower()
    if confirm != "y":
        return
    delete_plan(active, PLANS_DIR)
    print("Plan discarded.")


def run_execute(config, llm) -> None:
    if not config.active_repo:
        print("No active repo. Type 'use <repo>' to select one.")
        return
    active = get_active_plan(config.active_repo, PLANS_DIR)
    if not active:
        print(f"No pending plan for {config.active_repo}. Run: plan <task>")
        return
    if active.repo != config.active_repo:
        print(f"Plan was created for '{active.repo}'. Switch to it with 'use {active.repo}' first.")
        return
    repo_path = config.repos[config.active_repo]["path"]
    executor = Executor(llm=llm, repo_root=repo_path, plans_dir=PLANS_DIR)
    executor.execute(active)


def run_plans_list(config) -> None:
    if not config.active_repo:
        print("No active repo. Type 'use <repo>' to select one.")
        return
    all_plans = list_plans(config.active_repo, PLANS_DIR)
    if not all_plans:
        print(f"No plans for {config.active_repo}.")
        return
    print(f"Plans for {config.active_repo}:")
    for p in all_plans:
        count = len(p.edits)
        print(f"  [{p.status:<12}]  {p.created_at}  \"{p.task}\"  ({count} edit{'s' if count != 1 else ''})")
```

- [ ] **Step 5: Add commands to the REPL loop**

In `main()`, inside the `while True` REPL loop, add the new command handlers after the existing `elif command == "repos":` block and before `elif command == "ask":`:

```python
        elif command == "plan":
            if not rest:
                print("Usage: plan <task>  |  plan revise <feedback>  |  plan clear")
            elif rest.startswith("revise "):
                run_plan_revise(rest[len("revise "):], config, embedder, llm, store)
            elif rest == "clear":
                run_plan_clear(config)
            else:
                run_plan(rest, config, embedder, llm, store)
        elif command == "execute":
            run_execute(config, llm)
        elif command == "plans":
            run_plans_list(config)
```

- [ ] **Step 6: Verify imports are clean**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && python -c "import agent; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Run full test suite**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && git add agent.py && git commit -m "feat: plan/execute REPL commands — plan, execute, plans"
```

- [ ] **Step 9: Manual smoke test**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && python agent.py
```

With an active repo indexed:
```
> plan add a docstring to the OllamaEmbedder class
# Expected: Planning... then plan summary with edits

> plans
# Expected: [pending] entry listed

> execute
# Expected: Edit 1/N diff display with [a]pply / [s]kip / [r]evise / [q]uit prompt

> plan revise "only touch the embed method, not __init__"
# Expected: Revising plan... then updated summary

> plan clear
# Expected: "Discard current plan? [y/n]:" → plan removed

> execute   (with no pending plan)
# Expected: "No pending plan for <repo>. Run: plan <task>"
```
