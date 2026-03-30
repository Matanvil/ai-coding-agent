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
