from unittest.mock import MagicMock
from src.llm import BaseLLMClient, ClaudeClient, ToolCallParseError


def make_end_turn_response(text: str):
    response = MagicMock()
    response.stop_reason = "end_turn"
    block = MagicMock()
    block.text = text
    response.content = [block]
    return response


def make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "id123"):
    response = MagicMock()
    response.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    response.content = [block]
    return response


def test_respond_returns_final_answer():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    client.client.messages.create.return_value = make_end_turn_response("Auth is in auth.py")

    result = client.respond(
        messages=[{"role": "user", "content": "how does auth work?"}],
        tool_handler=lambda name, inp: "",
    )
    assert result == "Auth is in auth.py"


def test_respond_calls_tool_then_returns_answer():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    client.client.messages.create.side_effect = [
        make_tool_use_response("search_codebase", {"query": "auth"}, "id1"),
        make_end_turn_response("Auth is handled in auth.py"),
    ]

    tool_calls = []
    def tool_handler(name, inp):
        tool_calls.append((name, inp))
        return "auth.py:1 - def authenticate()..."

    result = client.respond(
        messages=[{"role": "user", "content": "how does auth work?"}],
        tool_handler=tool_handler,
    )
    assert result == "Auth is handled in auth.py"
    assert len(tool_calls) == 1
    assert tool_calls[0] == ("search_codebase", {"query": "auth"})


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


def test_respond_sends_tool_result_as_user_message():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    client.client.messages.create.side_effect = [
        make_tool_use_response("search_codebase", {"query": "auth"}, "id3"),
        make_end_turn_response("done"),
    ]

    client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "result text",
    )

    second_call_messages = client.client.messages.create.call_args_list[1].kwargs["messages"]
    # Last message in second call should be the tool result from user
    last_message = second_call_messages[-1]
    assert last_message["role"] == "user"
    assert last_message["content"][0]["type"] == "tool_result"
    assert last_message["content"][0]["tool_use_id"] == "id3"
    assert last_message["content"][0]["content"] == "result text"


def test_respond_returns_fallback_after_max_iterations():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    # Always return tool_use — never terminates naturally
    client.client.messages.create.return_value = make_tool_use_response(
        "search_codebase", {"query": "x"}, "id99"
    )

    result = client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "result",
        max_iterations=3,
    )
    assert "Maximum iterations" in result
    assert client.client.messages.create.call_count == 3


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
