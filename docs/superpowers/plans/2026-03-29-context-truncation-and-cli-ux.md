# Context Truncation + CLI UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hard history-drop with LLM-based summarization, and upgrade the CLI with command history, styled input, and visual dividers via `prompt_toolkit`.

**Architecture:** Two independent tasks. Task 1 modifies `AgentLoop` in `src/agent_loop.py` — the summarization calls Claude directly via `self.llm.client.messages.create` (bypassing the ReAct loop). Task 2 is a surface change to `agent.py` only — swaps `input()` for `prompt_toolkit.PromptSession`.

**Tech Stack:** Python, Anthropic SDK (already present), `prompt_toolkit>=3.0.0` (new dependency).

---

## File Structure

```
src/
  agent_loop.py        ← Replace _truncate_history with _summarize_and_truncate_history
tests/
  test_agent_loop.py   ← Add 2 new tests; existing tests need no changes
agent.py               ← Swap input() for PromptSession; add DIVIDER
requirements.txt       ← Add prompt_toolkit>=3.0.0
```

---

## Task 1: Summarization-Based Context Truncation

**Files:**
- Modify: `src/agent_loop.py`
- Modify: `tests/test_agent_loop.py`

---

- [ ] **Step 1: Write the two failing tests — add to bottom of `tests/test_agent_loop.py`**

```python
def test_summarize_and_truncate_inserts_summary():
    # max_history_turns=2 → max_messages=4
    # After 3 asks: 6 messages > 4 → summarization triggers
    agent = AgentLoop(
        llm=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
        repo_root="/repo",
        max_history_turns=2,
    )
    mock_msg_response = MagicMock()
    mock_msg_response.content = [MagicMock(text="- Asked about auth\n- Found validate_token in auth.py")]
    agent.llm.client.messages.create.return_value = mock_msg_response
    agent.llm.respond.return_value = "answer"

    agent.ask("q0")
    agent.ask("q1")
    agent.ask("q2")  # triggers summarization

    assert agent.history[0]["role"] == "user"
    assert "[Earlier conversation summary:" in agent.history[0]["content"]
    assert agent.history[1] == {
        "role": "assistant",
        "content": "Understood, I have context from our earlier conversation.",
    }


def test_summarize_and_truncate_falls_back_on_error():
    # When the summarization API call fails, keep the recent half (no crash, no growth)
    agent = AgentLoop(
        llm=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
        repo_root="/repo",
        max_history_turns=2,
    )
    agent.llm.client.messages.create.side_effect = Exception("API error")
    agent.llm.respond.return_value = "answer"

    agent.ask("q0")
    agent.ask("q1")
    agent.ask("q2")  # triggers fallback

    # No summary messages — fallback keeps recent half
    assert all(
        "[Earlier conversation summary:" not in m.get("content", "")
        for m in agent.history
    )
    # History is capped, not unbounded
    assert len(agent.history) <= 4
```

- [ ] **Step 2: Run the new tests to confirm they FAIL**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest tests/test_agent_loop.py::test_summarize_and_truncate_inserts_summary tests/test_agent_loop.py::test_summarize_and_truncate_falls_back_on_error -v
```

Expected: `AttributeError` or `AssertionError` — `_summarize_and_truncate_history` does not exist yet.

- [ ] **Step 3: Update `src/agent_loop.py`**

Read the current file first. Then make two changes:

**Add the constant and new method — replace the existing `_truncate_history` method:**

```python
SUMMARIZATION_SYSTEM_PROMPT = (
    "Summarize the following conversation between a developer and a coding assistant. "
    "Focus on: what was asked, what files/functions were found, and key technical findings. "
    "Be concise — bullet points preferred. This summary will be used as context in future turns."
)
```

Add this constant at module level, directly above the `AgentLoop` class definition.

Then replace `_truncate_history` with `_summarize_and_truncate_history`:

```python
    def _summarize_and_truncate_history(self) -> None:
        """Summarize the older half of history when it exceeds the limit.
        Falls back to dropping the old half if summarization fails."""
        max_messages = self.max_history_turns * 2
        if len(self.history) <= max_messages:
            return

        mid = len(self.history) // 2
        old_messages = self.history[:mid]
        recent_messages = self.history[mid:]

        try:
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=512,
                system=SUMMARIZATION_SYSTEM_PROMPT,
                messages=old_messages,
            )
            summary = response.content[0].text
            self.history = [
                {"role": "user", "content": f"[Earlier conversation summary: {summary}]"},
                {"role": "assistant", "content": "Understood, I have context from our earlier conversation."},
            ] + recent_messages
        except Exception:
            # Fallback: discard old half, keep recent
            self.history = recent_messages
```

Also update the `ask()` method — replace the `_truncate_history()` call:

```python
    def ask(self, question: str, on_tool_call: Optional[Callable] = None) -> str:
        self.history.append({"role": "user", "content": question})
        answer = self.llm.respond(
            messages=list(self.history),
            tool_handler=self._tool_handler,
            on_tool_call=on_tool_call,
        )
        self.history.append({"role": "assistant", "content": answer})
        self._summarize_and_truncate_history()
        return answer
```

- [ ] **Step 4: Run all agent loop tests to confirm they PASS**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest tests/test_agent_loop.py -v
```

Expected: all 12 tests PASS (10 existing + 2 new).

Note: `test_history_is_capped_at_max_history_turns` and `test_history_not_truncated_when_under_limit` need no changes — they still pass because the MagicMock llm auto-returns a valid MagicMock from `client.messages.create`, and the length invariant is unchanged.

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest -v
```

Expected: all 56 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && git add src/agent_loop.py tests/test_agent_loop.py && git commit -m "feat: summarize old history turns instead of dropping them"
```

---

## Task 2: CLI UX — prompt_toolkit

**Files:**
- Modify: `requirements.txt`
- Modify: `agent.py`

No automated tests — manual verification only.

---

- [ ] **Step 1: Add `prompt_toolkit` to `requirements.txt`**

The file currently contains:
```
chromadb>=0.5.0,<1.0.0
anthropic>=0.40.0
requests>=2.31.0
pytest>=7.4.0
pytest-mock>=3.12.0
```

Add `prompt_toolkit>=3.0.0` as a new line. Result:
```
chromadb>=0.5.0,<1.0.0
anthropic>=0.40.0
requests>=2.31.0
pytest>=7.4.0
pytest-mock>=3.12.0
prompt_toolkit>=3.0.0
```

- [ ] **Step 2: Install it**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pip install prompt_toolkit>=3.0.0
```

Expected: `Successfully installed prompt-toolkit-X.X.X`

- [ ] **Step 3: Update `agent.py`**

Read `agent.py` first. Then make the following changes:

**Add imports at the top** (after existing imports):

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
```

**Add constants** (after `HELP_TEXT`):

```python
DIVIDER = "─" * 48

_PROMPT_STYLE = Style.from_dict({"": "bg:#1a1a2e #ffffff"})
```

**Add session builder** (after `build_components` function):

```python
def build_session() -> PromptSession:
    history_file = Path.home() / ".ai-agent-history"
    return PromptSession(
        history=FileHistory(str(history_file)),
        style=_PROMPT_STYLE,
    )
```

**Update `handle_question`** — add the divider after the answer:

Replace:
```python
    answer = agent.ask(question, on_tool_call=on_tool_call)
    print(f"\n{answer}\n")
```

With:
```python
    answer = agent.ask(question, on_tool_call=on_tool_call)
    print(f"\n{answer}\n")
    print(DIVIDER)
```

**Update `main`** — replace `input()` with `PromptSession`:

Replace in `main()`:
```python
    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
```

With:
```python
    session = build_session()
    while True:
        try:
            user_input = session.prompt("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
```

- [ ] **Step 4: Verify it imports cleanly**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && python -c "import agent; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && pytest -v
```

Expected: all 56 tests PASS.

- [ ] **Step 6: Manual smoke test**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && source .venv/bin/activate && python agent.py
```

Verify:
- Prompt appears with dark background styling on the input area
- Up arrow shows previous commands
- After an answer, a `────────────────────────────────────────────────` divider appears
- Type `exit` to quit, then re-run — previous commands still accessible via up arrow

- [ ] **Step 7: Commit**

```bash
cd /Users/matanvilensky/dev/ai-coding-agent && git add agent.py requirements.txt && git commit -m "feat: prompt_toolkit CLI — history, styled input, dividers"
```
