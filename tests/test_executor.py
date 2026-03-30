import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.executor import Executor
from src.plan_store import Plan, FileEdit


def _make_plan(*edits):
    return Plan(
        task="test task",
        repo="myrepo",
        created_at="2026-03-30 14:00",
        status="pending",
        edits=list(edits),
    )


def _edit(file="src/foo.py", old_code="a = 1", new_code="a = 42"):
    return FileEdit(file=file, description="change a", old_code=old_code, new_code=new_code, status="pending")


def test_apply_edit_writes_file_and_marks_applied(tmp_path):
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="a"):
        result = executor.execute(plan)

    assert result.edits[0].status == "applied"
    assert result.status == "completed"
    assert "a = 42" in target.read_text(encoding="utf-8")


def test_skip_edit_marks_rejected(tmp_path):
    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="s"):
        result = executor.execute(plan)

    assert result.edits[0].status == "rejected"
    assert result.status == "completed"


def test_quit_saves_in_progress(tmp_path):
    plan = _make_plan(_edit())
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="q"):
        result = executor.execute(plan)

    assert result.status == "in_progress"
    assert result.edits[0].status == "pending"


def test_old_code_not_found_warns_and_marks_rejected(tmp_path, capsys):
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("completely different content\n", encoding="utf-8")

    plan = _make_plan(_edit(old_code="not_in_file"))
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="a"):
        result = executor.execute(plan)

    assert result.edits[0].status == "rejected"
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "code not found" in captured.out


def test_all_edits_processed_marks_completed(tmp_path):
    plan = _make_plan(_edit(file="a.py"), _edit(file="b.py"))
    executor = Executor(llm=MagicMock(), repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    with patch("builtins.input", return_value="s"):
        result = executor.execute(plan)

    assert result.status == "completed"
    assert all(e.status == "rejected" for e in result.edits)


def test_revise_then_apply_uses_revised_edit(tmp_path):
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    plan = _make_plan(_edit())

    # LLM returns a revised edit via submit_plan
    submit_block = MagicMock()
    submit_block.type = "tool_use"
    submit_block.name = "submit_plan"
    submit_block.id = "tool_1"
    submit_block.input = {"edits": [
        {"file": "src/foo.py", "description": "revised change", "old_code": "a = 1", "new_code": "a = 99"},
    ]}
    revision_response = MagicMock()
    revision_response.stop_reason = "tool_use"
    revision_response.content = [submit_block]

    llm = MagicMock()
    llm.model = "claude-haiku-4-5-20251001"
    llm.client.messages.create.return_value = revision_response

    executor = Executor(llm=llm, repo_root=str(tmp_path), plans_dir=str(tmp_path / "plans"))

    # Simulate: r → feedback → a (apply the revised edit)
    inputs = iter(["r", "use 99 instead", "a"])
    with patch("builtins.input", side_effect=inputs):
        result = executor.execute(plan)

    assert result.edits[0].status == "applied"
    assert "a = 99" in target.read_text(encoding="utf-8")
