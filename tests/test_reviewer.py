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
