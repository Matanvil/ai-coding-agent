import pytest
from src.plan_store import FileEdit, Plan, save_plan, load_plan, list_plans, get_active_plan, delete_plan, plan_filepath, ApprovalDecision


def _make_plan(task="add cache", repo="myrepo", created_at="2026-03-30 14:23", status="pending"):
    return Plan(
        task=task,
        repo=repo,
        created_at=created_at,
        status=status,
        edits=[
            FileEdit(file="src/foo.py", description="change a", old_code="a = 1", new_code="a = 42", status="pending")
        ],
    )


def test_save_and_load_round_trip(tmp_path):
    plan = _make_plan()
    save_plan(plan, str(tmp_path))
    path = plan_filepath(plan, str(tmp_path))
    loaded = load_plan(str(path))
    assert loaded.task == "add cache"
    assert loaded.repo == "myrepo"
    assert loaded.status == "pending"
    assert len(loaded.edits) == 1
    assert loaded.edits[0].file == "src/foo.py"
    assert loaded.edits[0].old_code == "a = 1"
    assert loaded.edits[0].status == "pending"


def test_list_plans_sorted_newest_first(tmp_path):
    for ts in ["2026-03-30 14:00", "2026-03-30 15:00", "2026-03-30 13:00"]:
        save_plan(_make_plan(created_at=ts), str(tmp_path))
    plans = list_plans("myrepo", str(tmp_path))
    assert [p.created_at for p in plans] == [
        "2026-03-30 15:00",
        "2026-03-30 14:00",
        "2026-03-30 13:00",
    ]


def test_get_active_plan_ignores_completed(tmp_path):
    save_plan(_make_plan(created_at="2026-03-30 14:00", status="completed"), str(tmp_path))
    save_plan(_make_plan(created_at="2026-03-30 15:00", status="pending"), str(tmp_path))
    active = get_active_plan("myrepo", str(tmp_path))
    assert active is not None
    assert active.created_at == "2026-03-30 15:00"


def test_get_active_plan_returns_none_when_all_completed(tmp_path):
    save_plan(_make_plan(status="completed"), str(tmp_path))
    assert get_active_plan("myrepo", str(tmp_path)) is None


def test_get_active_plan_returns_none_when_empty(tmp_path):
    assert get_active_plan("myrepo", str(tmp_path)) is None


def test_delete_plan_removes_file(tmp_path):
    plan = _make_plan()
    save_plan(plan, str(tmp_path))
    path = plan_filepath(plan, str(tmp_path))
    assert path.exists()
    delete_plan(plan, str(tmp_path))
    assert not path.exists()


def test_approval_decision_defaults():
    d = ApprovalDecision("apply")
    assert d.action == "apply"
    assert d.feedback == ""


def test_approval_decision_with_feedback():
    d = ApprovalDecision("revise", "use 99 instead")
    assert d.action == "revise"
    assert d.feedback == "use 99 instead"
