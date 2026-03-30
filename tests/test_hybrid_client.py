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
    fallback_events = [(t, d) for t, d in events if t == "model_fallback"]
    assert len(fallback_events) == 1
    assert fallback_events[0][1]["kind"] == "parse_error"


def test_hybrid_falls_back_to_original_messages_when_partial_empty():
    ollama = make_ollama(side_effect=ToolCallParseError("bad", partial=[]))
    claude = make_claude(return_value="claude answer")
    original_messages = [{"role": "user", "content": "q"}]
    client = HybridClient(ollama=ollama, claude=claude)

    client.respond(
        messages=original_messages,
        tool_handler=lambda name, inp: "",
    )
    call_messages = claude.respond.call_args.kwargs["messages"]
    assert call_messages == original_messages
