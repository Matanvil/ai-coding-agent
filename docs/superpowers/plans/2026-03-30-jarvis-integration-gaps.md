# Jarvis Integration Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `on_event` callbacks to all agents, replace `Executor.execute()` stdin with an `approval_fn` callback, and create a `narrate_event()` utility so Jarvis can orchestrate the AI Coding Agent programmatically with full narration.

**Architecture:** Eight tasks in dependency order — new additions first (no breakage), then rename `on_tool_call` → `on_event` in `llm.py` / `agent_loop.py`, then extend Planner / Reviewer / Executor, then update `agent.py` call sites. All tasks follow TDD: write failing test → verify fail → implement → verify pass → commit.

**Tech Stack:** Python, pytest, unittest.mock, Anthropic SDK (already installed)

---

### Task 1: Add `ApprovalDecision` to `src/plan_store.py`

**Files:**
- Modify: `src/plan_store.py`
- Modify: `tests/test_plan_store.py`

- [ ] **Step 1: Write the failing test**

First update the existing import at line 2 of `tests/test_plan_store.py`:

```python
from src.plan_store import FileEdit, Plan, save_plan, load_plan, list_plans, get_active_plan, delete_plan, plan_filepath, ApprovalDecision
```

Then add to the bottom of `tests/test_plan_store.py`:

```python
def test_approval_decision_defaults():
    d = ApprovalDecision("apply")
    assert d.action == "apply"
    assert d.feedback == ""


def test_approval_decision_with_feedback():
    d = ApprovalDecision("revise", "use 99 instead")
    assert d.action == "revise"
    assert d.feedback == "use 99 instead"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_plan_store.py::test_approval_decision_defaults -v
```

Expected: `ImportError: cannot import name 'ApprovalDecision'`

- [ ] **Step 3: Add `ApprovalDecision` to `src/plan_store.py`**

Insert after the `Plan` dataclass (after line 22, before `def plan_filepath`):

```python
@dataclass
class ApprovalDecision:
    action: str   # "apply" | "skip" | "quit" | "revise"
    feedback: str = ""  # only used when action == "revise"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_plan_store.py -v
```

Expected: all pass, including the two new tests.

- [ ] **Step 5: Commit**

```bash
git add src/plan_store.py tests/test_plan_store.py
git commit -m "feat: add ApprovalDecision dataclass to plan_store"
```

---

### Task 2: Create `src/narration.py` and `tests/test_narration.py`

**Files:**
- Create: `src/narration.py`
- Create: `tests/test_narration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_narration.py`:

```python
from src.narration import narrate_event


def test_tool_call_search():
    result = narrate_event("tool_call", {"tool": "search_codebase", "input": {"query": "retry logic"}})
    assert result == 'Searching: "retry logic"'


def test_tool_call_trace():
    result = narrate_event("tool_call", {"tool": "trace_flow", "input": {"entry_point": "embed"}})
    assert result == "Tracing: embed"


def test_tool_call_read():
    result = narrate_event("tool_call", {"tool": "read_file", "input": {"path": "src/embedder.py"}})
    assert result == "Reading: src/embedder.py"


def test_tool_call_unknown_tool():
    result = narrate_event("tool_call", {"tool": "mystery_tool", "input": {}})
    assert result == "Using tool: mystery_tool"


def test_planning_started():
    result = narrate_event("planning_started", {"task": "add retry"})
    assert result == 'Planning: "add retry"'


def test_planning_complete_singular():
    result = narrate_event("planning_complete", {"edit_count": 1})
    assert result == "Plan ready: 1 edit proposed"


def test_planning_complete_plural():
    result = narrate_event("planning_complete", {"edit_count": 3})
    assert result == "Plan ready: 3 edits proposed"


def test_review_started():
    result = narrate_event("review_started", {})
    assert result == "Reviewing changes..."


def test_review_complete_no_criticals():
    result = narrate_event("review_complete", {"issue_count": 2, "critical_count": 0, "suggest_fix_plan": False})
    assert result == "Review done: 2 issues"


def test_review_complete_with_criticals():
    result = narrate_event("review_complete", {"issue_count": 3, "critical_count": 1, "suggest_fix_plan": True})
    assert result == "Review done: 3 issues, 1 critical"


def test_review_complete_singular_issue():
    result = narrate_event("review_complete", {"issue_count": 1, "critical_count": 0, "suggest_fix_plan": False})
    assert result == "Review done: 1 issue"


def test_edit_presented():
    result = narrate_event("edit_presented", {"index": 2, "total": 3, "file": "src/foo.py", "description": "add retry"})
    assert result == "Edit 2/3 — src/foo.py: add retry"


def test_edit_applied():
    result = narrate_event("edit_applied", {"file": "src/foo.py"})
    assert result == "Applied: src/foo.py"


def test_edit_skipped():
    result = narrate_event("edit_skipped", {"file": "src/foo.py"})
    assert result == "Skipped: src/foo.py"


def test_edit_revised():
    result = narrate_event("edit_revised", {"file": "src/foo.py"})
    assert result == "Revised: src/foo.py"


def test_execution_complete():
    result = narrate_event("execution_complete", {"applied": 2, "skipped": 1})
    assert result == "Done: 2 applied, 1 skipped"


def test_unknown_event_returns_empty_string():
    result = narrate_event("some_future_event", {"anything": True})
    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_narration.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.narration'`

- [ ] **Step 3: Create `src/narration.py`**

```python
def narrate_event(event_type: str, data: dict) -> str:
    """Return a human-readable string for any on_event payload. Returns '' for unknown types."""
    if event_type == "tool_call":
        tool = data.get("tool", "")
        inp = data.get("input", {})
        if tool == "search_codebase":
            return f'Searching: "{inp.get("query", "")}"'
        if tool == "trace_flow":
            return f"Tracing: {inp.get('entry_point', '')}"
        if tool == "read_file":
            return f"Reading: {inp.get('path', '')}"
        return f"Using tool: {tool}"
    if event_type == "planning_started":
        return f'Planning: "{data.get("task", "")}"'
    if event_type == "planning_complete":
        count = data.get("edit_count", 0)
        return f"Plan ready: {count} edit{'s' if count != 1 else ''} proposed"
    if event_type == "review_started":
        return "Reviewing changes..."
    if event_type == "review_complete":
        issue_count = data.get("issue_count", 0)
        critical_count = data.get("critical_count", 0)
        label = "issue" if issue_count == 1 else "issues"
        base = f"Review done: {issue_count} {label}"
        if critical_count:
            return f"{base}, {critical_count} critical"
        return base
    if event_type == "edit_presented":
        index = data.get("index", 0)
        total = data.get("total", 0)
        return f"Edit {index}/{total} — {data.get('file', '')}: {data.get('description', '')}"
    if event_type == "edit_applied":
        return f"Applied: {data.get('file', '')}"
    if event_type == "edit_skipped":
        return f"Skipped: {data.get('file', '')}"
    if event_type == "edit_revised":
        return f"Revised: {data.get('file', '')}"
    if event_type == "execution_complete":
        return f"Done: {data.get('applied', 0)} applied, {data.get('skipped', 0)} skipped"
    return ""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_narration.py -v
```

Expected: all 17 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/narration.py tests/test_narration.py
git commit -m "feat: add narrate_event utility for on_event callback formatting"
```

---

### Task 3: Rename `on_tool_call` → `on_event` in `src/llm.py`

**Files:**
- Modify: `src/llm.py`
- Modify: `tests/test_llm.py`

The event payload changes from `(tool_name, tool_input)` to `("tool_call", {"tool": name, "input": inp})`.

- [ ] **Step 1: Update the existing callback test in `tests/test_llm.py`**

Replace the existing `test_on_tool_call_callback_is_invoked` function (lines 60–74) with:

```python
def test_on_event_callback_is_invoked():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    client.client.messages.create.side_effect = [
        make_tool_use_response("read_file", {"path": "auth.py"}, "id2"),
        make_end_turn_response("auth.py contains..."),
    ]

    events = []
    client.respond(
        messages=[{"role": "user", "content": "show auth.py"}],
        tool_handler=lambda name, inp: "content",
        on_event=lambda event_type, data: events.append((event_type, data)),
    )
    assert len(events) == 1
    assert events[0] == ("tool_call", {"tool": "read_file", "input": {"path": "auth.py"}})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_llm.py::test_on_event_callback_is_invoked -v
```

Expected: `TypeError: respond() got an unexpected keyword argument 'on_event'`

- [ ] **Step 3: Update `src/llm.py`**

Replace the `respond` method signature and the callback invocation (lines 66–108). The full updated method:

```python
def respond(
    self,
    messages: List[Dict[str, Any]],
    tool_handler: Callable[[str, Dict], str],
    on_event: Optional[Callable[[str, Dict], None]] = None,
    max_iterations: int = 10,
) -> str:
    """Run the ReAct loop until a final answer is produced."""
    current_messages = list(messages)

    for _ in range(max_iterations):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=current_messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        if response.stop_reason == "tool_use":
            current_messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if on_event:
                        on_event("tool_call", {"tool": block.name, "input": block.input})
                    result = tool_handler(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            current_messages.append({"role": "user", "content": tool_results})

    return "Maximum iterations reached without a final answer."
```

- [ ] **Step 4: Run all llm tests**

```bash
pytest tests/test_llm.py -v
```

Expected: all 5 tests pass (the old `test_on_tool_call_callback_is_invoked` is now replaced by `test_on_event_callback_is_invoked`).

- [ ] **Step 5: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat: rename on_tool_call to on_event in ClaudeClient.respond"
```

---

### Task 4: Rename `on_tool_call` → `on_event` in `src/agent_loop.py`

**Files:**
- Modify: `src/agent_loop.py`
- Modify: `tests/test_agent_loop.py`

- [ ] **Step 1: Update the existing callback test in `tests/test_agent_loop.py`**

Replace `test_ask_passes_on_tool_call_callback` (lines 50–56) with:

```python
def test_ask_passes_on_event_callback():
    agent = make_agent()
    agent.llm.respond.return_value = "answer"
    callback = MagicMock()
    agent.ask("question", on_event=callback)
    call_kwargs = agent.llm.respond.call_args.kwargs
    assert call_kwargs["on_event"] is callback
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_agent_loop.py::test_ask_passes_on_event_callback -v
```

Expected: `TypeError: ask() got an unexpected keyword argument 'on_event'`

- [ ] **Step 3: Update `src/agent_loop.py`**

Replace the `ask` method (lines 75–84):

```python
def ask(self, question: str, on_event: Optional[Callable] = None) -> str:
    self.history.append({"role": "user", "content": question})
    answer = self.llm.respond(
        messages=list(self.history),
        tool_handler=self._tool_handler,
        on_event=on_event,
    )
    self.history.append({"role": "assistant", "content": answer})
    self._summarize_and_truncate_history()
    return answer
```

- [ ] **Step 4: Run all agent_loop tests**

```bash
pytest tests/test_agent_loop.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run full test suite to catch any regressions**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agent_loop.py tests/test_agent_loop.py
git commit -m "feat: rename on_tool_call to on_event in AgentLoop.ask"
```

---

### Task 5: Add `on_event` to `src/planner.py`

**Files:**
- Modify: `src/planner.py`
- Modify: `tests/test_planner.py`

Events fired: `planning_started` (in `plan()`/`revise()` before loop), `tool_call` (per non-submit-plan tool block in `_run()`), `planning_complete` (in `plan()`/`revise()` after loop).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_planner.py`:

```python
def test_plan_fires_planning_started_and_complete_events():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_plan_response([
        {"file": "src/foo.py", "description": "add x", "old_code": "a = 1", "new_code": "a = 42"},
        {"file": "src/bar.py", "description": "add y", "old_code": "b = 1", "new_code": "b = 42"},
    ])

    events = []
    planner = Planner(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    planner.plan("add cache", repo="myrepo", on_event=lambda t, d: events.append((t, d)))

    assert events[0] == ("planning_started", {"task": "add cache"})
    assert events[-1] == ("planning_complete", {"edit_count": 2})


def test_plan_fires_tool_call_events_for_search_and_read():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"

    search_block = MagicMock()
    search_block.type = "tool_use"
    search_block.name = "search_codebase"
    search_block.id = "t1"
    search_block.input = {"query": "cache"}

    submit_block = MagicMock()
    submit_block.type = "tool_use"
    submit_block.name = "submit_plan"
    submit_block.id = "t2"
    submit_block.input = {"edits": [
        {"file": "src/foo.py", "description": "add x", "old_code": "a = 1", "new_code": "a = 42"},
    ]}

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [search_block, submit_block]
    llm.client.messages.create.return_value = response

    store = MagicMock()
    store.search.return_value = []
    store.keyword_search.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768

    events = []
    planner = Planner(llm=llm, embedder=embedder, store=store, repo_root="/repo")
    planner.plan("add cache", repo="myrepo", on_event=lambda t, d: events.append((t, d)))

    tool_call_events = [e for e in events if e[0] == "tool_call"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0] == ("tool_call", {"tool": "search_codebase", "input": {"query": "cache"}})


def test_plan_does_not_fire_tool_call_for_submit_plan():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_plan_response([
        {"file": "src/foo.py", "description": "add x", "old_code": "a = 1", "new_code": "a = 42"},
    ])

    events = []
    planner = Planner(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    planner.plan("add cache", repo="myrepo", on_event=lambda t, d: events.append((t, d)))

    tool_names = [e[1].get("tool") for e in events if e[0] == "tool_call"]
    assert "submit_plan" not in tool_names


def test_on_event_none_does_not_crash():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_plan_response([
        {"file": "src/foo.py", "description": "add x", "old_code": "a = 1", "new_code": "a = 42"},
    ])
    planner = Planner(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    # Should not raise even though on_event is not passed
    plan = planner.plan("add cache", repo="myrepo")
    assert isinstance(plan, Plan)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_planner.py::test_plan_fires_planning_started_and_complete_events -v
```

Expected: `TypeError: plan() got an unexpected keyword argument 'on_event'`

- [ ] **Step 3: Update `src/planner.py`**

Replace `_run`, `plan`, and `revise` with:

```python
def _run(self, messages: list, task: str, on_event=None) -> Plan:
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
            plan_to_return = None
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "submit_plan":
                    plan_to_return = Plan(
                        task=task,
                        repo="",  # set by caller
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
                        status="pending",
                        edits=[
                            FileEdit(
                                file=e["file"],
                                description=e["description"],
                                old_code=e["old_code"],
                                new_code=e["new_code"],
                                status="pending",
                            )
                            for e in block.input["edits"]
                        ],
                    )
                else:
                    if on_event:
                        on_event("tool_call", {"tool": block.name, "input": block.input})
                    result = self._tool_handler(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if tool_results:
                current_messages.append({"role": "user", "content": tool_results})

            if plan_to_return is not None:
                return plan_to_return

        else:
            break

    raise PlannerError("Planner could not produce a plan. Try a more specific task.")

def plan(self, task: str, repo: str, on_event=None) -> Plan:
    """Run the planner agent. Returns a Plan or raises PlannerError."""
    if on_event:
        on_event("planning_started", {"task": task})
    messages = [{"role": "user", "content": task}]
    plan = self._run(messages, task=task, on_event=on_event)
    plan.repo = repo
    if on_event:
        on_event("planning_complete", {"edit_count": len(plan.edits)})
    return plan

def revise(self, plan: Plan, feedback: str, on_event=None) -> Plan:
    """Re-run planner with existing plan + feedback. Returns revised Plan."""
    if on_event:
        on_event("planning_started", {"task": plan.task})
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
    revised = self._run([{"role": "user", "content": message}], task=plan.task, on_event=on_event)
    revised.repo = plan.repo
    revised.task = plan.task  # always preserve original task
    if on_event:
        on_event("planning_complete", {"edit_count": len(revised.edits)})
    return revised
```

- [ ] **Step 4: Run all planner tests**

```bash
pytest tests/test_planner.py -v
```

Expected: all tests pass (existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/planner.py tests/test_planner.py
git commit -m "feat: add on_event callbacks to Planner.plan and Planner.revise"
```

---

### Task 6: Add `on_event` to `src/reviewer.py`

**Files:**
- Modify: `src/reviewer.py`
- Modify: `tests/test_reviewer.py`

Events fired: `review_started` (before loop), `tool_call` (per non-submit-review tool block), `review_complete` (before returning result).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_reviewer.py`:

```python
def test_review_fires_review_started_and_complete_events():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_review_response(
        summary="Looks good.",
        issues=[{
            "category": "critical",
            "description": "Missing check",
            "file": "src/foo.py",
            "recommendation": "Add check",
        }],
        suggest_fix_plan=True,
    )

    events = []
    reviewer = Reviewer(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    reviewer.review(diff="+ x = 1", context="", on_event=lambda t, d: events.append((t, d)))

    assert events[0] == ("review_started", {})
    complete = [e for e in events if e[0] == "review_complete"]
    assert len(complete) == 1
    assert complete[0][1]["issue_count"] == 1
    assert complete[0][1]["critical_count"] == 1
    assert complete[0][1]["suggest_fix_plan"] is True


def test_review_fires_tool_call_events_not_submit_review():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"

    search_block = MagicMock()
    search_block.type = "tool_use"
    search_block.name = "search_codebase"
    search_block.id = "t0"
    search_block.input = {"query": "error handling"}

    submit_block = MagicMock()
    submit_block.type = "tool_use"
    submit_block.name = "submit_review"
    submit_block.id = "t1"
    submit_block.input = {"summary": "OK", "issues": [], "suggest_fix_plan": False}

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [search_block, submit_block]
    llm.client.messages.create.return_value = response

    store = MagicMock()
    store.search.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768

    events = []
    reviewer = Reviewer(llm=llm, embedder=embedder, store=store, repo_root="/repo")
    reviewer.review(diff="+ x", context="", on_event=lambda t, d: events.append((t, d)))

    tool_calls = [e for e in events if e[0] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0][1]["tool"] == "search_codebase"
    # submit_review must NOT appear as a tool_call event
    assert all(e[1].get("tool") != "submit_review" for e in tool_calls)


def test_review_on_event_none_does_not_crash():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_review_response(
        summary="Clean.", issues=[], suggest_fix_plan=False
    )
    reviewer = Reviewer(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    result = reviewer.review(diff="+ x", context="")
    assert result.summary == "Clean."
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_reviewer.py::test_review_fires_review_started_and_complete_events -v
```

Expected: `TypeError: review() got an unexpected keyword argument 'on_event'`

- [ ] **Step 3: Update `src/reviewer.py`**

Replace the `review` method (lines 144–202) with:

```python
def review(self, diff: str, context: str, on_event=None) -> ReviewResult:
    """Run the reviewer ReAct loop. Returns ReviewResult when submit_review is called."""
    if on_event:
        on_event("review_started", {})
    message = f"Git diff:\n{diff}\n\nContext: {context}\n\nReview the changes above."
    messages = [{"role": "user", "content": message}]

    for _ in range(self.max_iterations):
        response = self.llm.client.messages.create(
            model=self.llm.model,
            max_tokens=4096,
            system=REVIEWER_SYSTEM_PROMPT,
            tools=REVIEWER_TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            review_result = None

            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "submit_review":
                    data = block.input
                    review_result = ReviewResult(
                        summary=data["summary"],
                        issues=[
                            ReviewIssue(
                                category=i["category"],
                                description=i["description"],
                                file=i.get("file", ""),
                                recommendation=i["recommendation"],
                            )
                            for i in data.get("issues", [])
                        ],
                        suggest_fix_plan=data.get("suggest_fix_plan", False),
                    )
                else:
                    if on_event:
                        on_event("tool_call", {"tool": block.name, "input": block.input})
                    result = self._tool_handler(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if review_result is not None:
                if on_event:
                    critical_count = sum(1 for i in review_result.issues if i.category == "critical")
                    on_event("review_complete", {
                        "issue_count": len(review_result.issues),
                        "critical_count": critical_count,
                        "suggest_fix_plan": review_result.suggest_fix_plan,
                    })
                return review_result
        else:
            break

    raise ReviewerError("Reviewer could not produce a review. Try providing more context.")
```

- [ ] **Step 4: Run all reviewer tests**

```bash
pytest tests/test_reviewer.py -v
```

Expected: all tests pass (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/reviewer.py tests/test_reviewer.py
git commit -m "feat: add on_event callbacks to Reviewer.review"
```

---

### Task 7: Add `approval_fn` and `on_event` to `src/executor.py`

**Files:**
- Modify: `src/executor.py`
- Modify: `tests/test_executor.py`

When `approval_fn=None` (default): existing `input()` behavior is preserved exactly — no behavior change for CLI users. When `approval_fn` is provided: `input()` is not called; `_show_diff()` and the final summary print are suppressed; events are fired instead.

- [ ] **Step 1: Write the failing tests**

First update the existing import at line 3 of `tests/test_executor.py`:

```python
from src.plan_store import Plan, FileEdit, ApprovalDecision
```

Then add to the bottom of `tests/test_executor.py`:

```python
def test_approval_fn_apply_marks_applied(tmp_path):
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    result = executor.execute(plan, approval_fn=lambda e: ApprovalDecision("apply"))

    assert result.edits[0].status == "applied"
    assert result.status == "completed"
    assert "a = 42" in target.read_text(encoding="utf-8")


def test_approval_fn_skip_marks_rejected(tmp_path):
    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    result = executor.execute(plan, approval_fn=lambda e: ApprovalDecision("skip"))

    assert result.edits[0].status == "rejected"
    assert result.status == "completed"


def test_approval_fn_quit_returns_in_progress(tmp_path):
    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    result = executor.execute(plan, approval_fn=lambda e: ApprovalDecision("quit"))

    assert result.status == "in_progress"
    assert result.edits[0].status == "pending"


def test_on_event_fires_edit_presented_applied_and_complete(tmp_path):
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    events = []
    result = executor.execute(
        plan,
        approval_fn=lambda e: ApprovalDecision("apply"),
        on_event=lambda t, d: events.append((t, d)),
    )

    event_types = [e[0] for e in events]
    assert "edit_presented" in event_types
    assert "edit_applied" in event_types
    assert "execution_complete" in event_types

    complete = next(e[1] for e in events if e[0] == "execution_complete")
    assert complete == {"applied": 1, "skipped": 0}


def test_on_event_fires_edit_skipped(tmp_path):
    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    events = []
    executor.execute(
        plan,
        approval_fn=lambda e: ApprovalDecision("skip"),
        on_event=lambda t, d: events.append((t, d)),
    )

    event_types = [e[0] for e in events]
    assert "edit_skipped" in event_types
    assert "edit_applied" not in event_types


def test_approval_fn_revise_then_apply(tmp_path):
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    submit_block = MagicMock()
    submit_block.type = "tool_use"
    submit_block.name = "submit_plan"
    submit_block.id = "tool_1"
    submit_block.input = {"edits": [
        {"file": "src/foo.py", "description": "revised", "old_code": "a = 1", "new_code": "a = 99"},
    ]}
    revision_response = MagicMock()
    revision_response.stop_reason = "tool_use"
    revision_response.content = [submit_block]

    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = revision_response

    plan = _make_plan(_edit())
    executor = Executor(llm=llm, repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    calls = iter([ApprovalDecision("revise", "use 99"), ApprovalDecision("apply")])
    result = executor.execute(plan, approval_fn=lambda e: next(calls))

    assert result.edits[0].status == "applied"
    assert "a = 99" in target.read_text(encoding="utf-8")


def test_on_event_fires_edit_revised(tmp_path):
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    submit_block = MagicMock()
    submit_block.type = "tool_use"
    submit_block.name = "submit_plan"
    submit_block.id = "tool_1"
    submit_block.input = {"edits": [
        {"file": "src/foo.py", "description": "revised", "old_code": "a = 1", "new_code": "a = 99"},
    ]}
    revision_response = MagicMock()
    revision_response.stop_reason = "tool_use"
    revision_response.content = [submit_block]

    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = revision_response

    plan = _make_plan(_edit())
    executor = Executor(llm=llm, repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    events = []
    calls = iter([ApprovalDecision("revise", "use 99"), ApprovalDecision("apply")])
    executor.execute(
        plan,
        approval_fn=lambda e: next(calls),
        on_event=lambda t, d: events.append((t, d)),
    )

    event_types = [e[0] for e in events]
    assert "edit_revised" in event_types
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_executor.py::test_approval_fn_apply_marks_applied -v
```

Expected: `TypeError: execute() got an unexpected keyword argument 'approval_fn'`

- [ ] **Step 3: Update `src/executor.py`**

First, update the import at the top of the file:

```python
from src.plan_store import Plan, FileEdit, save_plan, ApprovalDecision
```

Then replace the `execute` method (lines 152–200) with:

```python
def execute(self, plan: Plan, approval_fn=None, on_event=None) -> Plan:
    """Walk through plan edits. Uses approval_fn if provided, else falls back to input()."""
    plan.status = "in_progress"
    save_plan(plan, self.plans_dir)

    total = len(plan.edits)

    for i, edit in enumerate(plan.edits):
        if edit.status != "pending":
            continue  # resume: skip already-processed edits

        if approval_fn is None:
            self._show_diff(edit, i + 1, total)
        elif on_event:
            on_event("edit_presented", {
                "index": i + 1,
                "total": total,
                "file": edit.file,
                "description": edit.description,
            })

        while True:
            if approval_fn is None:
                raw = input("[a]pply / [s]kip / [r]evise / [q]uit: ").strip().lower()
                if raw == "r":
                    feedback = input("Feedback: ").strip()
                    decision = ApprovalDecision("revise", feedback)
                elif raw == "a":
                    decision = ApprovalDecision("apply")
                elif raw == "s":
                    decision = ApprovalDecision("skip")
                elif raw == "q":
                    decision = ApprovalDecision("quit")
                else:
                    print("Please enter a, s, r, or q.")
                    continue
            else:
                decision = approval_fn(edit)

            if decision.action == "apply":
                success = self._apply_edit(edit)
                edit.status = "applied" if success else "rejected"
                save_plan(plan, self.plans_dir)
                if on_event:
                    on_event("edit_applied" if success else "edit_skipped", {"file": edit.file})
                break
            elif decision.action == "skip":
                edit.status = "rejected"
                save_plan(plan, self.plans_dir)
                if on_event:
                    on_event("edit_skipped", {"file": edit.file})
                break
            elif decision.action == "revise":
                if decision.feedback:
                    revised = self._revise_edit(edit, decision.feedback)
                    edit.file = revised.file
                    edit.description = revised.description
                    edit.old_code = revised.old_code
                    edit.new_code = revised.new_code
                if on_event:
                    on_event("edit_revised", {"file": edit.file})
                elif approval_fn is None:
                    self._show_diff(edit, i + 1, total)
            elif decision.action == "quit":
                plan.status = "in_progress"
                save_plan(plan, self.plans_dir)
                return plan

    plan.status = "completed"
    save_plan(plan, self.plans_dir)

    applied = sum(1 for e in plan.edits if e.status == "applied")
    skipped = sum(1 for e in plan.edits if e.status == "rejected")

    if approval_fn is None:
        print(f"\nDone. {applied} applied, {skipped} skipped.")
    if on_event:
        on_event("execution_complete", {"applied": applied, "skipped": skipped})

    return plan
```

- [ ] **Step 4: Run all executor tests**

```bash
pytest tests/test_executor.py -v
```

Expected: all tests pass — existing CLI-mode tests (using `patch("builtins.input")`) plus 8 new programmatic tests.

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/executor.py tests/test_executor.py
git commit -m "feat: add approval_fn and on_event callbacks to Executor.execute"
```

---

### Task 8: Update `agent.py` call sites

**Files:**
- Modify: `agent.py`

Wire `on_event` + `narrate_event` into `handle_question`, `run_plan`, `run_plan_revise`, and `run_review`. The executor in `run_execute` stays unchanged (CLI mode — no `approval_fn`, no `on_event`).

- [ ] **Step 1: Add `narration` import to `agent.py`**

Add to the imports block at the top (after the existing `src.*` imports):

```python
from src.narration import narrate_event
```

- [ ] **Step 2: Update `handle_question` in `agent.py`**

Replace lines 196–204 (the `on_tool_call` definition and `agent.ask` call):

```python
    def on_event(event_type, data):
        msg = narrate_event(event_type, data)
        if msg:
            print(f" → {msg}")

    answer = agent.ask(question, on_event=on_event)
```

- [ ] **Step 3: Update `run_plan` in `agent.py`**

Replace the `print("Planning...")` and `planner.plan(...)` call (lines 248–250):

```python
    def on_event(event_type, data):
        msg = narrate_event(event_type, data)
        if msg:
            print(f" → {msg}")

    try:
        plan = planner.plan(task=task, repo=config.active_repo, on_event=on_event)
```

- [ ] **Step 4: Update `run_plan_revise` in `agent.py`**

Replace the `print("Revising plan...")` and `planner.revise(...)` call (lines 268–271):

```python
    def on_event(event_type, data):
        msg = narrate_event(event_type, data)
        if msg:
            print(f" → {msg}")

    try:
        revised = planner.revise(active, feedback, on_event=on_event)
```

- [ ] **Step 5: Update `run_review` in `agent.py`**

Replace the `print("Reviewing...")` and `reviewer.review(...)` call (lines 323–325):

```python
    def on_event(event_type, data):
        msg = narrate_event(event_type, data)
        if msg:
            print(f" → {msg}")

    reviewer = Reviewer(llm=llm, embedder=embedder, store=store, repo_root=repo_path)
    try:
        review = reviewer.review(diff=diff, context=context, on_event=on_event)
```

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add agent.py
git commit -m "feat: wire on_event and narrate_event into agent.py CLI handlers"
```
