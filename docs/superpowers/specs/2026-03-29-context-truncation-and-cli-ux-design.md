# Context Truncation + CLI UX — Design Spec

**Date:** 2026-03-29
**Status:** Approved

---

## Overview

Two improvements to the AI coding agent:

1. **Smarter context truncation** — replace hard message-drop with LLM-based summarization so the agent retains awareness of earlier conversation turns rather than losing them entirely.
2. **CLI UX polish** — add command history (up arrow), styled command input, and visual dividers between exchanges using `prompt_toolkit`.

---

## Part 1: Context Truncation — Summarization

### Current behavior

`AgentLoop._truncate_history()` drops the oldest messages when history exceeds `max_history_turns * 2`. Information is lost.

### Clarification: tool calls are already excluded

`self.history` stores only clean `{"role": "user", "content": question}` and `{"role": "assistant", "content": answer}` pairs. Tool calls and their results are handled entirely inside `ClaudeClient.respond()` and never written to history. No change needed here.

### New behavior: summarization

When history exceeds `max_history_turns * 2` messages:

1. Split history into two halves: **old** (first half) and **recent** (second half)
2. Call Claude with the old messages + summarization prompt
3. Replace the old messages with two synthetic messages:
   - `{"role": "user", "content": "[Earlier conversation summary: <summary>]"}`
   - `{"role": "assistant", "content": "Understood, I have context from our earlier conversation."}`
4. Keep the recent half intact
5. Resulting history = 2 summary messages + recent half — always fits within the limit

**Summarization prompt:**
```
Summarize the following conversation between a developer and a coding assistant.
Focus on: what was asked, what files/functions were found, and key technical findings.
Be concise — bullet points preferred. This summary will be used as context in future turns.
```

**Trigger:** identical to current — `len(self.history) > max_history_turns * 2`.

**Fallback:** if the summarization LLM call fails for any reason, fall back silently to the existing drop behavior (keep recent half, discard old half).

### Where it lives

- `AgentLoop._summarize_and_truncate_history()` replaces `_truncate_history()`
- Uses `self.llm.client` directly (single non-tool call, no ReAct loop needed)
- Called at the end of `ask()` exactly where `_truncate_history()` is called today

### Files changed

- `src/agent_loop.py` — replace `_truncate_history` with `_summarize_and_truncate_history`
- `tests/test_agent_loop.py` — update truncation tests to cover new behavior + fallback

---

## Part 2: CLI UX

### Dependencies

Add to `requirements.txt`:
```
prompt_toolkit>=3.0.0
```

### Command history

Replace `input("> ")` with `prompt_toolkit.PromptSession`. History persists to `~/.ai-agent-history` between sessions. Up arrow navigates previous commands across restarts.

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

session = PromptSession(history=FileHistory(Path.home() / ".ai-agent-history"))
user_input = session.prompt("> ")
```

### Styled command input

The prompt renders with a styled input area — dark background, white text — matching the Claude Code CLI aesthetic. Implemented via `prompt_toolkit`'s `style` parameter on `PromptSession`.

```python
from prompt_toolkit.styles import Style

style = Style.from_dict({"": "bg:#1a1a2e #ffffff"})
session = PromptSession(history=..., style=style)
```

### Visual divider

After each answer prints, a full-width divider separates it from the next prompt:

```python
DIVIDER = "─" * 48

# in handle_question(), after printing the answer:
print(f"\n{answer}\n")
print(DIVIDER)
```

### Files changed

- `agent.py` — swap `input()` for `PromptSession`, add `DIVIDER` constant, print divider after each answer
- `requirements.txt` — add `prompt_toolkit>=3.0.0`

No changes to any `src/` module.

---

## Testing

**Context truncation:**
- Existing truncation tests updated to mock the summarization call
- New test: verify summary messages are inserted correctly
- New test: verify fallback to drop behavior when summarization fails

**CLI UX:**
- No automated tests — `prompt_toolkit` integration is a thin wrapper, manual verification sufficient

---

## Build order

1. Context truncation (logic change in `src/`, has tests)
2. CLI UX (surface change in `agent.py`, no tests)
