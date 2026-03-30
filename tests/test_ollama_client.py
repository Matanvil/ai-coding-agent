import json
import pytest
from unittest.mock import MagicMock, patch
from src.ollama_client import _to_openai_tools, _parse_tool_call, _to_ollama_messages, OllamaClient
from src.llm import TOOL_DEFINITIONS, ToolCallParseError


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
    second_call_body = mock_post.call_args_list[1].kwargs["json"]
    msgs = second_call_body["messages"]
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-2]["tool_calls"][0]["function"]["name"] == "search_codebase"
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


@patch("src.ollama_client.requests.post")
def test_ollama_respond_raises_on_connection_error(mock_post):
    import requests as req
    mock_post.side_effect = req.RequestException("connection refused")
    client = OllamaClient(model="qwen3-coder:30b")
    with pytest.raises(ToolCallParseError) as exc_info:
        client.respond(
            messages=[{"role": "user", "content": "q"}],
            tool_handler=lambda name, inp: "",
        )
    assert "Ollama request failed" in str(exc_info.value)
