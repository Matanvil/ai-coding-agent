# Code Review Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `Reviewer` ReAct agent that reviews git diffs against coding standards, outputting categorized issues and optionally triggering a fix plan.

**Architecture:** `src/reviewer.py` contains `Reviewer` (mirroring `Planner`), `ReviewIssue`/`ReviewResult` dataclasses, and `ReviewerError`. `agent.py` gets a `review [context]` REPL command plus auto-trigger at the end of `run_execute`.

**Tech Stack:** Python, Anthropic SDK (`llm.client.messages.create`), subprocess (git diff), pytest + unittest.mock

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/reviewer.py` | Create | `Reviewer` class, dataclasses, system prompt, tool definitions |
| `agent.py` | Modify | `run_review()`, `_print_review()`, `review` command, post-execute hook |
| `tests/test_reviewer.py` | Create | Unit tests for `Reviewer.review()` |

---

### Task 1: Create `src/reviewer.py` skeleton — dataclasses, constants, tool definitions

**Files:**
- Create: `src/reviewer.py`

- [ ] **Step 1: Write the file**

```python
from dataclasses import dataclass
from src.tools import search_codebase, read_file
from src.agent_loop import format_chunks

REVIEWER_SYSTEM_PROMPT = """You are a Senior Code Reviewer with expertise in software architecture, design patterns, and best practices. Your role is to review completed project steps against original plans and ensure code quality standards are met.

When reviewing completed work, you will:

1. Plan Alignment Analysis:
   - Compare the implementation against the original planning document or step description
   - Identify any deviations from the planned approach, architecture, or requirements
   - Assess whether deviations are justified improvements or problematic departures
   - Verify that all planned functionality has been implemented

2. Code Quality Assessment:
   - Review code for adherence to established patterns and conventions
   - Check for proper error handling, type safety, and defensive programming
   - Evaluate code organization, naming conventions, and maintainability
   - Assess test coverage and quality of test implementations
   - Look for potential security vulnerabilities or performance issues

3. Architecture and Design Review:
   - Ensure the implementation follows SOLID principles and established architectural patterns
   - Check for proper separation of concerns and loose coupling
   - Verify that the code integrates well with existing systems
   - Assess scalability and extensibility considerations

4. Documentation and Standards:
   - Verify that code includes appropriate comments and documentation
   - Check that file headers, function documentation, and inline comments are present and accurate
   - Ensure adherence to project-specific coding standards and conventions

5. Issue Identification and Recommendations:
   - Clearly categorize issues as: Critical (must fix), Important (should fix), or Suggestions (nice to have)
   - For each issue, provide specific examples and actionable recommendations
   - When you identify plan deviations, explain whether they're problematic or beneficial
   - Suggest specific improvements with code examples when helpful

6. Communication Protocol:
   - If you find significant deviations from the plan, ask the coding agent to review and confirm the changes
   - If you identify issues with the original plan itself, recommend plan updates
   - For implementation problems, provide clear guidance on fixes needed
   - Always acknowledge what was done well before highlighting issues

When you have gathered enough context, call submit_review with your structured findings."""

REVIEWER_TOOL_DEFINITIONS = [
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
        "name": "submit_review",
        "description": "Submit the completed review with categorized issues. Call this when you have gathered enough context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Overall summary of the review"},
                "issues": {
                    "type": "array",
                    "description": "List of issues found",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["critical", "important", "suggestion"],
                            },
                            "description": {"type": "string"},
                            "file": {"type": "string", "description": "Relevant file path, or empty string"},
                            "recommendation": {"type": "string"},
                        },
                        "required": ["category", "description", "file", "recommendation"],
                    },
                },
                "suggest_fix_plan": {
                    "type": "boolean",
                    "description": "True if critical or important issues warrant a fix plan",
                },
            },
            "required": ["summary", "issues", "suggest_fix_plan"],
        },
    },
]


@dataclass
class ReviewIssue:
    category: str       # "critical" | "important" | "suggestion"
    description: str
    file: str           # empty string if not file-specific
    recommendation: str


@dataclass
class ReviewResult:
    summary: str
    issues: list
    suggest_fix_plan: bool


class ReviewerError(Exception):
    pass


class Reviewer:
    def __init__(self, llm, embedder, store, repo_root: str, max_iterations: int = 15):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.repo_root = repo_root
        self.max_iterations = max_iterations

    def _tool_handler(self, tool_name: str, tool_input: dict) -> str:
        raise NotImplementedError

    def review(self, diff: str, context: str) -> ReviewResult:
        raise NotImplementedError
```

- [ ] **Step 2: Verify the file imports cleanly**

Run: `python -c "from src.reviewer import Reviewer, ReviewerError, ReviewResult, ReviewIssue"`

Expected: no output (no import errors)

- [ ] **Step 3: Commit**

```bash
git add src/reviewer.py
git commit -m "feat: reviewer skeleton — dataclasses, constants, tool definitions"
```

---

### Task 2: Write failing tests for `Reviewer.review()`

**Files:**
- Create: `tests/test_reviewer.py`

- [ ] **Step 1: Write the test file**

```python
import pytest
from unittest.mock import MagicMock
from src.reviewer import Reviewer, ReviewerError, ReviewResult, ReviewIssue


def _make_submit_review_response(summary, issues, suggest_fix_plan):
    block = MagicMock()
    block.type = "tool_use"
    block.name = "submit_review"
    block.id = "tool_1"
    block.input = {
        "summary": summary,
        "issues": issues,
        "suggest_fix_plan": suggest_fix_plan,
    }
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


def _make_end_turn_response():
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = []
    return response


def test_review_returns_result_when_submit_review_called():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_review_response(
        summary="Overall looks good.",
        issues=[{
            "category": "important",
            "description": "Missing error handling",
            "file": "src/foo.py",
            "recommendation": "Wrap in try/except",
        }],
        suggest_fix_plan=True,
    )

    reviewer = Reviewer(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    result = reviewer.review(diff="- a = 1\n+ a = 42", context="updated constant")

    assert isinstance(result, ReviewResult)
    assert result.summary == "Overall looks good."
    assert len(result.issues) == 1
    assert isinstance(result.issues[0], ReviewIssue)
    assert result.issues[0].category == "important"
    assert result.issues[0].file == "src/foo.py"
    assert result.issues[0].recommendation == "Wrap in try/except"
    assert result.suggest_fix_plan is True


def test_review_raises_reviewer_error_when_submit_review_not_called():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_end_turn_response()

    reviewer = Reviewer(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")

    with pytest.raises(ReviewerError):
        reviewer.review(diff="some diff", context="")


def test_review_includes_diff_and_context_in_initial_message():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_review_response(
        summary="OK", issues=[], suggest_fix_plan=False
    )

    reviewer = Reviewer(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    reviewer.review(diff="+ new line", context="added feature X")

    call_args = llm.client.messages.create.call_args
    messages = call_args.kwargs["messages"]
    assert "+ new line" in messages[0]["content"]
    assert "added feature X" in messages[0]["content"]


def test_review_no_issues_returns_empty_list_and_no_fix_plan():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = _make_submit_review_response(
        summary="Clean code.", issues=[], suggest_fix_plan=False
    )

    reviewer = Reviewer(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    result = reviewer.review(diff="+ x = 1", context="")

    assert result.issues == []
    assert result.suggest_fix_plan is False


def test_review_raises_reviewer_error_on_unexpected_stop_reason():
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    response = MagicMock()
    response.stop_reason = "max_tokens"
    response.content = []
    llm.client.messages.create.return_value = response

    reviewer = Reviewer(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")

    with pytest.raises(ReviewerError):
        reviewer.review(diff="some diff", context="")
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_reviewer.py -v`

Expected: all 5 tests FAIL with `NotImplementedError`

- [ ] **Step 3: Commit**

```bash
git add tests/test_reviewer.py
git commit -m "test: reviewer — failing tests for Reviewer.review()"
```

---

### Task 3: Implement `Reviewer._tool_handler()` and `.review()`

**Files:**
- Modify: `src/reviewer.py`

- [ ] **Step 1: Replace the stub methods with full implementations**

Replace the two `raise NotImplementedError` stubs in `src/reviewer.py`:

```python
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

    def review(self, diff: str, context: str) -> ReviewResult:
        """Run the reviewer ReAct loop. Returns ReviewResult when submit_review is called."""
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
                        result = self._tool_handler(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

                if review_result is not None:
                    return review_result
            else:
                break

        raise ReviewerError("Reviewer could not produce a review. Try providing more context.")
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_reviewer.py -v`

Expected: all 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/reviewer.py
git commit -m "feat: implement Reviewer ReAct loop with submit_review tool"
```

---

### Task 4: Integrate into `agent.py` — `run_review`, `_print_review`, `review` command, post-execute hook

**Files:**
- Modify: `agent.py`

- [ ] **Step 1: Add import for `subprocess` and `Reviewer`/`ReviewerError`/`ReviewResult`**

At the top of `agent.py`, add `import subprocess` to the stdlib imports block (after `import sys`):

```python
import subprocess
```

Also add to the `src.*` imports block (after the `from src.executor import Executor` line):

```python
from src.reviewer import Reviewer, ReviewerError, ReviewResult
```

- [ ] **Step 2: Add `_print_review` helper after `_print_plan_summary`**

Add the following function directly after `_print_plan_summary` in `agent.py`:

```python
def _print_review(result: ReviewResult) -> None:
    print(f"\n{DIVIDER}")
    print("Code Review")
    print(DIVIDER)
    print(result.summary)
    for category, label in [("critical", "CRITICAL"), ("important", "IMPORTANT"), ("suggestion", "SUGGESTIONS")]:
        issues = [i for i in result.issues if i.category == category]
        if not issues:
            continue
        print(f"\n{label}")
        for issue in issues:
            prefix = f"  \u2022 {issue.file} \u2014 " if issue.file else "  \u2022 "
            print(f"{prefix}{issue.description}")
            print(f"    \u2192 {issue.recommendation}")
    if not result.issues:
        print("\nNo issues found.")
    print(DIVIDER)
```

- [ ] **Step 3: Add `run_review` function after `run_plans_list`**

Add the following function directly after `run_plans_list` in `agent.py`:

```python
def run_review(context: str, config, embedder, llm, store) -> None:
    if not config.active_repo or not store:
        print("No active repo. Type 'use <repo>' to select one.")
        return

    repo_path = config.repos[config.active_repo]["path"]

    proc = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"Error running git diff: {proc.stderr}")
        return

    diff = proc.stdout

    if not diff and not context:
        print("No uncommitted changes to review.")
        return

    reviewer = Reviewer(llm=llm, embedder=embedder, store=store, repo_root=repo_path)
    print("Reviewing...")
    try:
        review = reviewer.review(diff=diff, context=context)
    except ReviewerError as e:
        print(f"Error: {e}")
        return

    _print_review(review)

    if review.issues:
        choice = input("Generate a fix plan for the above issues? [y/n]: ").strip().lower()
        if choice == "y":
            fix_desc = f"Fix code review issues: {review.summary}"
            run_plan(fix_desc, config, embedder, llm, store)
```

- [ ] **Step 4: Update `run_execute` signature and add post-execute auto-trigger**

Change the signature of `run_execute` from:

```python
def run_execute(config, llm) -> None:
```

to:

```python
def run_execute(config, llm, embedder, store) -> None:
```

And replace:

```python
    executor = Executor(llm=llm, repo_root=repo_path, plans_dir=PLANS_DIR)
    executor.execute(active)
```

with:

```python
    executor = Executor(llm=llm, repo_root=repo_path, plans_dir=PLANS_DIR)
    plan = executor.execute(active)
    if plan.status == "completed":
        run_review(active.task, config, embedder, llm, store)
```

- [ ] **Step 5: Update the `execute` call site in `main()` to pass `embedder` and `store`**

Change:

```python
        elif command == "execute":
            run_execute(config, llm)
```

to:

```python
        elif command == "execute":
            run_execute(config, llm, embedder, store)
```

- [ ] **Step 6: Add `review` command to the REPL loop in `main()`**

Add the following `elif` block after the `elif command == "plans":` block:

```python
        elif command == "review":
            if not agent:
                print("No active repo. Type 'use <repo>' to select one.")
            else:
                run_review(rest, config, embedder, llm, store)
```

- [ ] **Step 7: Add `review` to `HELP_TEXT`**

In the `HELP_TEXT` string, add a line after the `execute` line:

```
  review [context]            Review current git diff; optional context about the changes
```

- [ ] **Step 8: Run the full test suite**

Run: `pytest -v`

Expected: all existing tests pass, all 5 new reviewer tests pass

- [ ] **Step 9: Commit**

```bash
git add agent.py
git commit -m "feat: review command — manual and post-execute code review agent"
```
