import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class FileEdit:
    file: str          # path relative to repo root
    description: str   # human-readable description of this edit
    old_code: str      # exact string to replace ("" = new file)
    new_code: str      # replacement string
    status: str        # "pending" | "applied" | "rejected"


@dataclass
class Plan:
    task: str          # original user request
    repo: str          # active repo name at creation time
    created_at: str    # "2026-03-30 14:23"
    status: str        # "pending" | "in_progress" | "completed"
    edits: List[FileEdit]


def plan_filepath(plan: Plan, plans_dir: str) -> Path:
    """Return the filesystem path for this plan's JSON file."""
    safe_ts = plan.created_at.replace(":", "-").replace(" ", "_")
    return Path(plans_dir) / f"{plan.repo}-{safe_ts}.json"


def save_plan(plan: Plan, plans_dir: str) -> Path:
    """Write plan to disk. Creates plans_dir if needed. Returns the file path."""
    Path(plans_dir).mkdir(parents=True, exist_ok=True)
    path = plan_filepath(plan, plans_dir)
    data = {
        "task": plan.task,
        "repo": plan.repo,
        "created_at": plan.created_at,
        "status": plan.status,
        "edits": [asdict(e) for e in plan.edits],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_plan(path: str) -> Plan:
    """Load a Plan from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    edits = [FileEdit(**e) for e in data["edits"]]
    return Plan(
        task=data["task"],
        repo=data["repo"],
        created_at=data["created_at"],
        status=data["status"],
        edits=edits,
    )


def list_plans(repo: str, plans_dir: str) -> List[Plan]:
    """Return all plans for a repo sorted newest-first."""
    plans_path = Path(plans_dir)
    if not plans_path.exists():
        return []
    plans = []
    for f in plans_path.glob(f"{repo}-*.json"):
        try:
            plans.append(load_plan(str(f)))
        except Exception:
            continue
    return sorted(plans, key=lambda p: p.created_at, reverse=True)


def get_active_plan(repo: str, plans_dir: str) -> Optional[Plan]:
    """Return the most recent pending or in_progress plan for a repo, or None."""
    for plan in list_plans(repo, plans_dir):
        if plan.status in ("pending", "in_progress"):
            return plan
    return None


def delete_plan(plan: Plan, plans_dir: str) -> None:
    """Delete a plan's JSON file from disk."""
    plan_filepath(plan, plans_dir).unlink(missing_ok=True)
