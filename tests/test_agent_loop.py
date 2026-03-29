import chromadb
from unittest.mock import MagicMock
from src.agent_loop import AgentLoop
from src.store import VectorStore
from src.models import Chunk


def make_agent():
    llm = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = [0.0] * 768
    store = MagicMock()
    return AgentLoop(llm=llm, embedder=embedder, store=store, repo_root="/repo")


def test_ask_returns_answer():
    agent = make_agent()
    agent.llm.respond.return_value = "Auth is in auth.py"
    result = agent.ask("How does auth work?")
    assert result == "Auth is in auth.py"


def test_ask_adds_user_and_assistant_to_history():
    agent = make_agent()
    agent.llm.respond.return_value = "Auth is in auth.py"
    agent.ask("How does auth work?")
    assert len(agent.history) == 2
    assert agent.history[0] == {"role": "user", "content": "How does auth work?"}
    assert agent.history[1] == {"role": "assistant", "content": "Auth is in auth.py"}


def test_ask_passes_full_history_to_llm():
    agent = make_agent()
    agent.llm.respond.return_value = "answer"
    agent.ask("first question")
    agent.ask("follow up")
    # Second call must include 3 messages: user1, assistant1, user2
    second_call_messages = agent.llm.respond.call_args_list[1].kwargs["messages"]
    assert len(second_call_messages) == 3


def test_clear_history_empties_history():
    agent = make_agent()
    agent.llm.respond.return_value = "answer"
    agent.ask("a question")
    agent.clear_history()
    assert agent.history == []


def test_ask_passes_on_tool_call_callback():
    agent = make_agent()
    agent.llm.respond.return_value = "answer"
    callback = MagicMock()
    agent.ask("question", on_tool_call=callback)
    call_kwargs = agent.llm.respond.call_args.kwargs
    assert call_kwargs["on_tool_call"] is callback


def test_tool_handler_dispatches_search_codebase(tmp_path):
    # Use a real store to test tool dispatch end-to-end
    client = chromadb.EphemeralClient()
    store = VectorStore(_client=client)
    store.add(
        [Chunk("def foo(): pass", "foo.py", 1, 0.0, "block")],
        [[0.1] * 768],
    )
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    llm = MagicMock()

    agent = AgentLoop(llm=llm, embedder=embedder, store=store, repo_root=str(tmp_path))
    result = agent._tool_handler("search_codebase", {"query": "foo function"})
    assert "foo.py" in result


def test_tool_handler_dispatches_read_file(tmp_path):
    f = tmp_path / "auth.py"
    f.write_text("def authenticate(): pass")
    agent = make_agent()
    agent.repo_root = str(tmp_path)
    result = agent._tool_handler("read_file", {"path": "auth.py"})
    assert "authenticate" in result


def test_tool_handler_returns_error_for_unknown_tool():
    agent = make_agent()
    result = agent._tool_handler("nonexistent_tool", {})
    assert "Unknown tool" in result


def test_history_is_capped_at_max_history_turns():
    agent = AgentLoop(
        llm=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
        repo_root="/repo",
        max_history_turns=3,
    )
    agent.llm.respond.return_value = "answer"

    for i in range(10):
        agent.ask(f"question {i}")

    # Exactly max_history_turns * 2 messages (3 exchanges = 6 messages)
    assert len(agent.history) == 3 * 2


def test_history_not_truncated_when_under_limit():
    agent = AgentLoop(
        llm=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
        repo_root="/repo",
        max_history_turns=10,
    )
    agent.llm.respond.return_value = "answer"

    agent.ask("q1")
    agent.ask("q2")

    assert len(agent.history) == 4  # 2 questions + 2 answers, under limit
