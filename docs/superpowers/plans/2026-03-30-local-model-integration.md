# Local Model Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Qwen3-Coder-30B via Ollama as the local-first reasoning model, with automatic fallback to Claude on failure, and a `--model claude` CLI override.

**Architecture:** Extract `BaseLLMClient` from `src/llm.py`, implement `OllamaClient` that drives the ReAct loop using Ollama's OpenAI-compatible API while accumulating history in Anthropic format, and `HybridClient` that tries Ollama first and falls back to Claude with partial history on failure. All agent classes remain untouched.

**Tech Stack:** Python, `requests` (already in requirements), Ollama `/v1/chat/completions` endpoint, `re` (stdlib) for XML tool call parsing.

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `src/llm.py` | Add `BaseLLMClient` ABC and `ToolCallParseError`; `ClaudeClient` inherits from base |
| Create | `src/ollama_client.py` | `OllamaClient`: Ollama API calls, tool schema translation, dual-format parser, ReAct loop |
| Create | `src/hybrid_client.py` | `HybridClient`: try Ollama, fall back to Claude with partial history |
| Modify | `src/config.py` | Add `local_model: str` field (default `""`) |
| Modify | `src/narration.py` | Add `model_fallback` event narration |
| Modify | `agent.py` | Use `HybridClient` when `config.local_model` set; add `--model claude` CLI flag |
| Modify | `tests/test_llm.py` | Assert `ClaudeClient` is a `BaseLLMClient` instance |
| Create | `tests/test_ollama_client.py` | Unit tests for parser, schema translation, ReAct loop |
| Create | `tests/test_hybrid_client.py` | Unit tests for fallback paths and `force_claude` |
| Modify | `tests/test_config.py` | Assert `local_model` loads with default and explicit value |

---

## Task 1: BaseLLMClient + ToolCallParseError

**Files:**
- Modify: `src/llm.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_llm.py`:

```python
from src.llm import BaseLLMClient, ClaudeClient, ToolCallParseError


def test_claude_client_is_base_llm_client():
    client = ClaudeClient(api_key="test")
    assert isinstance(client, BaseLLMClient)


def test_tool_call_parse_error_carries_partial():
    partial = [{"role": "user", "content": "hi"}]
    err = ToolCallParseError("bad response", partial=partial)
    assert err.partial == partial
    assert str(err) == "bad response"


def test_tool_call_parse_error_defaults_partial_to_empty():
    err = ToolCallParseError("bad response")
    assert err.partial == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_llm.py::test_claude_client_is_base_llm_client tests/test_llm.py::test_tool_call_parse_error_carries_partial tests/test_llm.py::test_tool_call_parse_error_defaults_partial_to_empty -v
```

Expected: FAIL with `ImportError: cannot import name 'BaseLLMClient'`

- [ ] **Step 3: Add BaseLLMClient and ToolCallParseError to src/llm.py**

At the top of `src/llm.py`, before the existing imports, add the ABC import. Then add the base class and exception before `ClaudeClient`. Finally update `ClaudeClient` to inherit from it.

Replace the top of `src/llm.py`:

```python
import os
from abc import ABC, abstractmethod
import anthropic
from typing import List, Dict, Any, Callable, Optional
```

Add after the `SYSTEM_PROMPT` constant and before `ClaudeClient`:

```python
class ToolCallParseError(Exception):
    """Raised by OllamaClient when a response cannot be parsed as a tool call or final answer."""
    def __init__(self, message: str, partial: list = None):
        super().__init__(message)
        self.partial = partial if partial is not None else []


class BaseLLMClient(ABC):
    @abstractmethod
    def respond(
        self,
        messages: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict], str],
        on_event: Optional[Callable[[str, Dict], None]] = None,
        max_iterations: int = 10,
    ) -> str:
        ...
```

Change `ClaudeClient` class line:

```python
class ClaudeClient(BaseLLMClient):
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat: add BaseLLMClient ABC and ToolCallParseError to llm.py"
```

---

## Task 2: OllamaClient — Parser and Schema Translation

**Files:**
- Create: `src/ollama_client.py`
- Create: `tests/test_ollama_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ollama_client.py`:

```python
import json
import pytest
from src.ollama_client import _to_openai_tools, _parse_tool_call, _to_ollama_messages
from src.llm import TOOL_DEFINITIONS


def test_to_openai_tools_translates_schema():
    anthropic_tools = [
        {
            "name": "search_codebase",
            "description": "Search the codebase",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    result = _to_openai_tools(anthropic_tools)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    func = result[0]["function"]
    assert func["name"] == "search_codebase"
    assert func["description"] == "Search the codebase"
    assert func["parameters"] == anthropic_tools[0]["input_schema"]


def test_to_openai_tools_handles_all_tool_definitions():
    result = _to_openai_tools(TOOL_DEFINITIONS)
    assert len(result) == len(TOOL_DEFINITIONS)
    for tool in result:
        assert "parameters" in tool["function"]
        assert "input_schema" not in tool["function"]


def test_parse_tool_call_openai_format():
    message = {
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "search_codebase",
                    "arguments": '{"query": "auth flow"}',
                },
            }
        ]
    }
    result = _parse_tool_call(message)
    assert result["name"] == "search_codebase"
    assert result["input"] == {"query": "auth flow"}
    assert result["id"] == "call_abc"


def test_parse_tool_call_xml_format():
    content = (
        "<function=read_file>\n"
        "<parameter=path>\nsrc/main.py\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    message = {"content": content, "tool_calls": None}
    result = _parse_tool_call(message)
    assert result["name"] == "read_file"
    assert result["input"] == {"path": "src/main.py"}


def test_parse_tool_call_xml_multi_param():
    content = (
        "<function=write_file>\n"
        "<parameter=path>/tmp/out.txt</parameter>\n"
        "<parameter=content>hello world</parameter>\n"
        "</function>"
    )
    message = {"content": content, "tool_calls": None}
    result = _parse_tool_call(message)
    assert result["name"] == "write_file"
    assert result["input"]["path"] == "/tmp/out.txt"
    assert result["input"]["content"] == "hello world"


def test_parse_tool_call_returns_none_for_plain_text():
    message = {"content": "Here is my answer.", "tool_calls": None}
    result = _parse_tool_call(message)
    assert result is None


def test_parse_tool_call_prefers_tool_calls_over_xml():
    message = {
        "tool_calls": [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
            }
        ],
        "content": "<function=search_codebase><parameter=query>something</parameter></function>",
    }
    result = _parse_tool_call(message)
    assert result["name"] == "read_file"


def test_to_ollama_messages_passes_through_text():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    result = _to_ollama_messages(messages)
    assert result == messages


def test_to_ollama_messages_converts_tool_use():
    messages = [
        {"role": "user", "content": "search for auth"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "id1", "name": "search_codebase", "input": {"query": "auth"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "id1", "content": "auth.py:1"}],
        },
    ]
    result = _to_ollama_messages(messages)
    assert result[0] == {"role": "user", "content": "search for auth"}
    assert result[1]["role"] == "assistant"
    assert result[1]["tool_calls"][0]["id"] == "id1"
    assert result[1]["tool_calls"][0]["function"]["name"] == "search_codebase"
    assert json.loads(result[1]["tool_calls"][0]["function"]["arguments"]) == {"query": "auth"}
    assert result[2]["role"] == "tool"
    assert result[2]["tool_call_id"] == "id1"
    assert result[2]["content"] == "auth.py:1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ollama_client.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.ollama_client'`

- [ ] **Step 3: Implement the parser and schema translation**

Create `src/ollama_client.py`:

```python
import json
import re
from typing import Any, Dict, List, Optional, Callable

from src.llm import BaseLLMClient, ToolCallParseError, TOOL_DEFINITIONS, SYSTEM_PROMPT


def _to_openai_tools(anthropic_tools: List[Dict]) -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_tools
    ]


def _parse_xml_tool_call(content: str) -> Optional[Dict]:
    name_match = re.search(r"<function=(\w+)>", content)
    if not name_match:
        return None
    name = name_match.group(1)
    params = {}
    for m in re.finditer(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", content, re.DOTALL):
        params[m.group(1)] = m.group(2).strip()
    return {"id": "xml_0", "name": name, "input": params}


def _parse_tool_call(message: Dict) -> Optional[Dict]:
    tool_calls = message.get("tool_calls")
    if tool_calls:
        tc = tool_calls[0]
        return {
            "id": tc.get("id", "call_0"),
            "name": tc["function"]["name"],
            "input": json.loads(tc["function"]["arguments"]),
        }
    content = (message.get("content") or "").strip()
    if "<function=" in content:
        return _parse_xml_tool_call(content)
    return None


def _to_ollama_messages(messages: List[Dict]) -> List[Dict]:
    """Convert Anthropic-format message history to OpenAI/Ollama format."""
    result = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            result.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
                tool_calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b["input"]),
                        },
                    }
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
                result.append({"role": "assistant", "tool_calls": tool_calls})
            elif any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": b["content"],
                        })
            else:
                text = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
                result.append({"role": msg["role"], "content": text})
        else:
            result.append(msg)
    return result


class OllamaClient(BaseLLMClient):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._openai_tools = _to_openai_tools(TOOL_DEFINITIONS)

    def respond(
        self,
        messages: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict], str],
        on_event: Optional[Callable[[str, Dict], None]] = None,
        max_iterations: int = 10,
    ) -> str:
        # Placeholder — ReAct loop added in Task 3
        raise NotImplementedError
```

- [ ] **Step 4: Run the parser/schema tests to verify they pass**

```bash
pytest tests/test_ollama_client.py -v
```

Expected: All parser and schema tests PASS. The `OllamaClient` instantiation test will pass too (no test calls `respond()` yet).

- [ ] **Step 5: Commit**

```bash
git add src/ollama_client.py tests/test_ollama_client.py
git commit -m "feat: add OllamaClient tool schema translation and dual-format response parser"
```

---

## Task 3: OllamaClient — ReAct Loop

**Files:**
- Modify: `src/ollama_client.py`
- Modify: `tests/test_ollama_client.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ollama_client.py`:

```python
from unittest.mock import MagicMock, patch
from src.ollama_client import OllamaClient
from src.llm import ToolCallParseError


def _ollama_tool_response(tool_name: str, args: dict, call_id: str = "call_1"):
    return MagicMock(
        status_code=200,
        json=MagicMock(return_value={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }]
        }),
    )


def _ollama_text_response(text: str):
    return MagicMock(
        status_code=200,
        json=MagicMock(return_value={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }]
        }),
    )


def _ollama_unparseable_response():
    return MagicMock(
        status_code=200,
        json=MagicMock(return_value={
            "choices": [{
                "message": {"role": "assistant", "content": "", "tool_calls": None},
                "finish_reason": "length",
            }]
        }),
    )


@patch("src.ollama_client.requests.post")
def test_ollama_respond_returns_final_answer(mock_post):
    mock_post.return_value = _ollama_text_response("Auth is in auth.py")
    client = OllamaClient(model="qwen3-coder:30b")
    result = client.respond(
        messages=[{"role": "user", "content": "where is auth?"}],
        tool_handler=lambda name, inp: "",
    )
    assert result == "Auth is in auth.py"


@patch("src.ollama_client.requests.post")
def test_ollama_respond_calls_tool_then_returns_answer(mock_post):
    mock_post.side_effect = [
        _ollama_tool_response("search_codebase", {"query": "auth"}, "id1"),
        _ollama_text_response("Auth is in auth.py"),
    ]
    client = OllamaClient(model="qwen3-coder:30b")
    tool_calls = []

    def handler(name, inp):
        tool_calls.append((name, inp))
        return "auth.py:1 - def authenticate()"

    result = client.respond(
        messages=[{"role": "user", "content": "where is auth?"}],
        tool_handler=handler,
    )
    assert result == "Auth is in auth.py"
    assert tool_calls == [("search_codebase", {"query": "auth"})]


@patch("src.ollama_client.requests.post")
def test_ollama_respond_accumulates_history_in_anthropic_format(mock_post):
    mock_post.side_effect = [
        _ollama_tool_response("search_codebase", {"query": "auth"}, "id1"),
        _ollama_text_response("done"),
    ]
    client = OllamaClient(model="qwen3-coder:30b")
    client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "result",
    )
    # Second call to Ollama should include tool_use + tool_result in Anthropic format
    # converted to OpenAI format for the API
    second_call_body = mock_post.call_args_list[1].kwargs["json"]
    msgs = second_call_body["messages"]
    # msgs[-2] is the assistant tool_calls message (OpenAI format)
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-2]["tool_calls"][0]["function"]["name"] == "search_codebase"
    # msgs[-1] is the tool result
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["content"] == "result"


@patch("src.ollama_client.requests.post")
def test_ollama_respond_raises_tool_call_parse_error_with_partial(mock_post):
    mock_post.side_effect = [
        _ollama_tool_response("search_codebase", {"query": "auth"}, "id1"),
        _ollama_unparseable_response(),
    ]
    client = OllamaClient(model="qwen3-coder:30b")
    with pytest.raises(ToolCallParseError) as exc_info:
        client.respond(
            messages=[{"role": "user", "content": "q"}],
            tool_handler=lambda name, inp: "result",
        )
    # partial should include the original message + the tool exchange
    assert len(exc_info.value.partial) == 3


@patch("src.ollama_client.requests.post")
def test_ollama_respond_fires_on_event(mock_post):
    mock_post.side_effect = [
        _ollama_tool_response("read_file", {"path": "src/llm.py"}, "id2"),
        _ollama_text_response("content"),
    ]
    client = OllamaClient(model="qwen3-coder:30b")
    events = []
    client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "data",
        on_event=lambda t, d: events.append((t, d)),
    )
    assert events == [("tool_call", {"tool": "read_file", "input": {"path": "src/llm.py"}})]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ollama_client.py::test_ollama_respond_returns_final_answer -v
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement the ReAct loop in OllamaClient**

Replace the `respond` method stub in `src/ollama_client.py`:

```python
import requests

# (add this import at the top of the file alongside the existing imports)
```

Replace the `respond` method:

```python
def respond(
    self,
    messages: List[Dict[str, Any]],
    tool_handler: Callable[[str, Dict], str],
    on_event: Optional[Callable[[str, Dict], None]] = None,
    max_iterations: int = 10,
) -> str:
    current_messages = list(messages)

    for _ in range(max_iterations):
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "messages": _to_ollama_messages(current_messages),
                    "tools": self._openai_tools,
                },
                timeout=120,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ToolCallParseError(f"Ollama request failed: {e}", partial=current_messages)

        body = resp.json()
        choice = body["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")

        tool_call = _parse_tool_call(message)

        if tool_call is None:
            if finish_reason == "stop":
                return (message.get("content") or "").strip()
            raise ToolCallParseError(
                f"Unparseable response (finish_reason={finish_reason!r})",
                partial=current_messages,
            )

        if on_event:
            on_event("tool_call", {"tool": tool_call["name"], "input": tool_call["input"]})

        result = tool_handler(tool_call["name"], tool_call["input"])

        current_messages.append({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["name"],
                "input": tool_call["input"],
            }],
        })
        current_messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": result,
            }],
        })

    return "Maximum iterations reached without a final answer."
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ollama_client.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ollama_client.py tests/test_ollama_client.py
git commit -m "feat: implement OllamaClient ReAct loop with Anthropic-format history accumulation"
```

---

## Task 4: HybridClient

**Files:**
- Create: `src/hybrid_client.py`
- Create: `tests/test_hybrid_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hybrid_client.py`:

```python
import pytest
from unittest.mock import MagicMock
from src.hybrid_client import HybridClient
from src.llm import ToolCallParseError


def make_ollama(return_value=None, side_effect=None):
    m = MagicMock()
    if side_effect:
        m.respond.side_effect = side_effect
    else:
        m.respond.return_value = return_value or "ollama answer"
    return m


def make_claude(return_value="claude answer"):
    m = MagicMock()
    m.respond.return_value = return_value
    return m


def test_hybrid_uses_ollama_when_healthy():
    ollama = make_ollama(return_value="local answer")
    claude = make_claude()
    client = HybridClient(ollama=ollama, claude=claude)

    result = client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "",
    )
    assert result == "local answer"
    ollama.respond.assert_called_once()
    claude.respond.assert_not_called()


def test_hybrid_falls_back_on_tool_call_parse_error():
    partial = [{"role": "user", "content": "q"}, {"role": "assistant", "content": []}]
    ollama = make_ollama(side_effect=ToolCallParseError("bad", partial=partial))
    claude = make_claude(return_value="claude saved it")
    client = HybridClient(ollama=ollama, claude=claude)

    result = client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "",
    )
    assert result == "claude saved it"
    # Claude receives the partial history, not the original messages
    call_messages = claude.respond.call_args.kwargs["messages"]
    assert call_messages == partial


def test_hybrid_falls_back_on_connection_error():
    ollama = make_ollama(side_effect=Exception("connection refused"))
    claude = make_claude(return_value="claude fallback")
    original_messages = [{"role": "user", "content": "q"}]
    client = HybridClient(ollama=ollama, claude=claude)

    result = client.respond(
        messages=original_messages,
        tool_handler=lambda name, inp: "",
    )
    assert result == "claude fallback"
    # On generic exception, Claude gets original messages
    call_messages = claude.respond.call_args.kwargs["messages"]
    assert call_messages == original_messages


def test_hybrid_force_claude_skips_ollama():
    ollama = make_ollama()
    claude = make_claude(return_value="forced claude")
    client = HybridClient(ollama=ollama, claude=claude)
    client.force_claude = True

    result = client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "",
    )
    assert result == "forced claude"
    ollama.respond.assert_not_called()
    claude.respond.assert_called_once()


def test_hybrid_force_claude_defaults_to_false():
    client = HybridClient(ollama=make_ollama(), claude=make_claude())
    assert client.force_claude is False


def test_hybrid_fires_fallback_event():
    partial = [{"role": "user", "content": "q"}]
    ollama = make_ollama(side_effect=ToolCallParseError("bad", partial=partial))
    claude = make_claude()
    client = HybridClient(ollama=ollama, claude=claude)

    events = []
    client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "",
        on_event=lambda t, d: events.append((t, d)),
    )
    assert any(t == "model_fallback" for t, d in events)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hybrid_client.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.hybrid_client'`

- [ ] **Step 3: Implement HybridClient**

Create `src/hybrid_client.py`:

```python
from typing import Any, Callable, Dict, List, Optional

from src.llm import BaseLLMClient, ToolCallParseError


class HybridClient(BaseLLMClient):
    """Tries OllamaClient first; falls back to ClaudeClient on failure.

    Set force_claude = True to skip Ollama for the session (--model claude flag).
    """

    def __init__(self, ollama: BaseLLMClient, claude: BaseLLMClient):
        self.ollama = ollama
        self.claude = claude
        self.force_claude: bool = False

    def respond(
        self,
        messages: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict], str],
        on_event: Optional[Callable[[str, Dict], None]] = None,
        max_iterations: int = 10,
    ) -> str:
        if self.force_claude:
            return self.claude.respond(
                messages=messages,
                tool_handler=tool_handler,
                on_event=on_event,
                max_iterations=max_iterations,
            )

        try:
            return self.ollama.respond(
                messages=messages,
                tool_handler=tool_handler,
                on_event=on_event,
                max_iterations=max_iterations,
            )
        except ToolCallParseError as e:
            if on_event:
                on_event("model_fallback", {"reason": str(e), "turns": len(e.partial)})
            return self.claude.respond(
                messages=e.partial if e.partial else messages,
                tool_handler=tool_handler,
                on_event=on_event,
                max_iterations=max_iterations,
            )
        except Exception as e:
            if on_event:
                on_event("model_fallback", {"reason": str(e), "turns": 0})
            return self.claude.respond(
                messages=messages,
                tool_handler=tool_handler,
                on_event=on_event,
                max_iterations=max_iterations,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hybrid_client.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hybrid_client.py tests/test_hybrid_client.py
git commit -m "feat: add HybridClient with Ollama-first and Claude fallback"
```

---

## Task 5: Config — local_model Field

**Files:**
- Modify: `src/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_config_local_model_defaults_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.CONFIG_PATH", tmp_path / "config.json")
    config = load_config()
    assert config.local_model == ""


def test_config_local_model_loads_from_file(tmp_path, monkeypatch):
    import json
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"local_model": "qwen3-coder:30b"}))
    monkeypatch.setattr("src.config.CONFIG_PATH", config_path)
    config = load_config()
    assert config.local_model == "qwen3-coder:30b"


def test_config_local_model_persists_on_save(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.CONFIG_PATH", tmp_path / "config.json")
    config = load_config()
    config.local_model = "qwen3-coder:30b"
    save_config(config)
    reloaded = load_config()
    assert reloaded.local_model == "qwen3-coder:30b"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL (either `AttributeError` on `config.local_model` or test function not found).

- [ ] **Step 3: Add local_model to Config**

In `src/config.py`, add the field to the `Config` dataclass:

```python
@dataclass
class Config:
    active_repo: str
    repos: dict
    model: str
    embedding_model: str
    ollama_url: str
    chroma_path: str
    max_results: int
    api_key: str
    local_model: str      # add this line
```

In `load_config`:

```python
return Config(
    active_repo=data.get("active_repo", ""),
    repos=data.get("repos", {}),
    model=data.get("model", "claude-haiku-4-5-20251001"),
    embedding_model=data.get("embedding_model", "nomic-embed-text"),
    ollama_url=data.get("ollama_url", "http://localhost:11434"),
    chroma_path=data.get("chroma_path", ".chroma"),
    max_results=data.get("max_results", 5),
    api_key=data.get("api_key", ""),
    local_model=data.get("local_model", ""),    # add this line
)
```

In `save_config`, add to the dict:

```python
"local_model": config.local_model,
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add local_model field to Config"
```

---

## Task 6: Narration — model_fallback Event

**Files:**
- Modify: `src/narration.py`
- Modify: `tests/test_narration.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_narration.py`:

```python
def test_narrate_model_fallback_with_turns():
    result = narrate_event("model_fallback", {"reason": "bad response", "turns": 3})
    assert "Claude" in result
    assert "3" in result


def test_narrate_model_fallback_cold():
    result = narrate_event("model_fallback", {"reason": "connection refused", "turns": 0})
    assert "Claude" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_narration.py -v
```

Expected: FAIL — `narrate_event("model_fallback", ...)` returns `""`.

- [ ] **Step 3: Add model_fallback to narration.py**

In `src/narration.py`, add before the final `return ""`:

```python
if event_type == "model_fallback":
    turns = data.get("turns", 0)
    if turns:
        return f"Local model failed after {turns} turn{'s' if turns != 1 else ''} — handing off to Claude"
    return "Local model unavailable — falling back to Claude"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_narration.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/narration.py tests/test_narration.py
git commit -m "feat: add model_fallback event narration"
```

---

## Task 7: Wire agent.py

**Files:**
- Modify: `agent.py`

No new tests — this is REPL glue code. Verified manually in the smoke test (Task 8).

- [ ] **Step 1: Update imports at top of agent.py**

Add to the existing imports:

```python
from src.ollama_client import OllamaClient
from src.hybrid_client import HybridClient
```

- [ ] **Step 2: Add parse_model_flag function**

Add after `parse_cli_args()`:

```python
def parse_model_flag() -> bool:
    """Return True if --model claude is in sys.argv."""
    args = sys.argv[1:]
    try:
        idx = args.index("--model")
        return idx + 1 < len(args) and args[idx + 1] == "claude"
    except ValueError:
        return False
```

- [ ] **Step 3: Update build_shared to accept force_claude and wire HybridClient**

Replace the existing `build_shared`:

```python
def build_shared(config, force_claude: bool = False):
    """Create embedder and LLM — shared across all repos."""
    embedder = OllamaEmbedder(model=config.embedding_model, base_url=config.ollama_url)
    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    claude = ClaudeClient(model=config.model, api_key=api_key)

    if config.local_model:
        ollama = OllamaClient(model=config.local_model, base_url=config.ollama_url)
        llm = HybridClient(ollama=ollama, claude=claude)
        llm.force_claude = force_claude
    else:
        llm = claude

    return embedder, llm
```

- [ ] **Step 4: Call parse_model_flag in main()**

In `main()`, replace:

```python
embedder, llm = build_shared(config)
```

with:

```python
force_claude = parse_model_flag()
embedder, llm = build_shared(config, force_claude=force_claude)
```

- [ ] **Step 5: Run the full test suite to ensure nothing broke**

```bash
pytest
```

Expected: All existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add agent.py
git commit -m "feat: wire HybridClient into agent.py with --model claude flag"
```

---

## Task 8: Smoke Test

Manual verification — no automated test.

- [ ] **Step 1: Add local_model to config.json**

```json
"local_model": "qwen3-coder:30b"
```

- [ ] **Step 2: Start the agent and ask a question**

```bash
python agent.py
```

At the prompt, with an indexed repo active:
```
> where is the Planner class defined?
```

Expected: The `→ Searching: "..."` line appears (tool call fired), followed by an answer. No errors.

- [ ] **Step 3: Test the force_claude flag**

```bash
python agent.py --model claude
```

Ask the same question. Expected: Same answer quality, goes through Claude (no difference visible to user, but you can temporarily add a print in `build_shared` to confirm).

- [ ] **Step 4: Test fallback narration**

Temporarily stop Ollama (`pkill ollama` or similar), then ask a question. Expected: `→ Local model unavailable — falling back to Claude` is printed, and Claude answers normally. Start Ollama again when done.
