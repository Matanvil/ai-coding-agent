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


def test_review_with_mixed_tool_response_returns_result():
    """When submit_review and a regular tool appear in the same response, returns ReviewResult."""
    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"

    search_block = MagicMock()
    search_block.type = "tool_use"
    search_block.name = "search_codebase"
    search_block.id = "tool_0"
    search_block.input = {"query": "error handling patterns"}

    submit_block = MagicMock()
    submit_block.type = "tool_use"
    submit_block.name = "submit_review"
    submit_block.id = "tool_1"
    submit_block.input = {
        "summary": "Mixed response summary.",
        "issues": [],
        "suggest_fix_plan": False,
    }

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [search_block, submit_block]

    llm.client.messages.create.return_value = response

    embedder = MagicMock()
    store = MagicMock()
    store.search.return_value = []

    reviewer = Reviewer(llm=llm, embedder=embedder, store=store, repo_root="/repo")
    result = reviewer.review(diff="+ new code", context="")

    assert isinstance(result, ReviewResult)
    assert result.summary == "Mixed response summary."
    # Verify _tool_handler was called for the search_codebase block
    assert store.search.called


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
