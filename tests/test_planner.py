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


def test_unexpected_stop_reason_raises_planner_error():
    # stop_reason "max_tokens" should break the loop and raise PlannerError
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    response = MagicMock()
    response.stop_reason = "max_tokens"
    response.content = []
    llm.client.messages.create.return_value = response

    planner = Planner(llm=llm, embedder=MagicMock(), store=MagicMock(), repo_root="/repo")
    with pytest.raises(PlannerError):
        planner.plan("add a cache layer", repo="myrepo")


def test_mixed_tool_response_returns_plan_and_calls_other_tools():
    # When submit_plan appears alongside other tool calls, plan is still returned
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"

    search_block = MagicMock()
    search_block.type = "tool_use"
    search_block.name = "search_codebase"
    search_block.id = "tool_search"
    search_block.input = {"query": "cache"}

    submit_block = MagicMock()
    submit_block.type = "tool_use"
    submit_block.name = "submit_plan"
    submit_block.id = "tool_submit"
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

    planner = Planner(llm=llm, embedder=embedder, store=store, repo_root="/repo")
    plan = planner.plan("add a cache layer", repo="myrepo")

    assert isinstance(plan, Plan)
    assert len(plan.edits) == 1
