from pathlib import Path
from src.plan_store import Plan, FileEdit, save_plan

REVISION_SYSTEM_PROMPT = """You are an expert coding assistant revising a single file edit based on feedback.
Call submit_plan with one revised edit. The edit's old_code must be an exact string from the file."""

REVISION_TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file in the repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path from repo root"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "submit_plan",
        "description": "Submit the single revised edit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "description": {"type": "string"},
                            "old_code": {"type": "string"},
                            "new_code": {"type": "string"},
                        },
                        "required": ["file", "description", "old_code", "new_code"],
                    },
                }
            },
            "required": ["edits"],
        },
    },
]


class Executor:
    def __init__(self, llm, repo_root: str, plans_dir: str):
        self.llm = llm
        self.repo_root = repo_root
        self.plans_dir = plans_dir

    def _show_diff(self, edit: FileEdit, index: int, total: int) -> None:
        print(f"\nEdit {index}/{total} — {edit.file}")
        print(edit.description)
        print()
        print("--- before ---")
        print(edit.old_code if edit.old_code else "(new file)")
        print()
        print("+++ after +++")
        print(edit.new_code)
        print()

    def _apply_edit(self, edit: FileEdit) -> bool:
        """Write edit to disk. Returns True on success, False if old_code not found."""
        repo = Path(self.repo_root).resolve()
        target = (repo / edit.file).resolve()
        try:
            target.relative_to(repo)
        except ValueError:
            print(f"Warning: could not apply edit to {edit.file} — path outside repo. Skipping.")
            return False

        if edit.old_code == "":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edit.new_code, encoding="utf-8")
            return True

        if not target.exists():
            print(f"Warning: could not apply edit to {edit.file} — file not found. Skipping.")
            return False

        content = target.read_text(encoding="utf-8")
        if edit.old_code not in content:
            print(f"Warning: could not apply edit to {edit.file} — code not found. Skipping.")
            return False

        target.write_text(content.replace(edit.old_code, edit.new_code, 1), encoding="utf-8")
        return True

    def _revise_edit(self, edit: FileEdit, feedback: str) -> FileEdit:
        """Ask LLM to revise a single edit. Returns revised FileEdit (fallback: original)."""
        from src.tools import read_file as _read_file

        message = (
            f"File: {edit.file}\n"
            f"Description: {edit.description}\n\n"
            f"Current old_code:\n{edit.old_code}\n\n"
            f"Current new_code:\n{edit.new_code}\n\n"
            f"Feedback: {feedback}\n\n"
            f"Call submit_plan with the revised edit."
        )
        messages = [{"role": "user", "content": message}]

        for _ in range(5):
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=2048,
                system=REVISION_SYSTEM_PROMPT,
                tools=REVISION_TOOL_DEFINITIONS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                break

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            plan_edit = None
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "submit_plan":
                    edits_data = block.input.get("edits", [])
                    if edits_data:
                        e = edits_data[0]
                        plan_edit = FileEdit(
                            file=e.get("file", edit.file),
                            description=e.get("description", edit.description),
                            old_code=e.get("old_code", edit.old_code),
                            new_code=e.get("new_code", edit.new_code),
                            status="pending",
                        )
                elif block.name == "read_file":
                    try:
                        result = _read_file(block.input["path"], self.repo_root)
                    except Exception as e:
                        result = f"Error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if plan_edit is not None:
                return plan_edit

        print("Revision failed. Showing original edit.")
        return edit  # fallback: return original unchanged

    def execute(self, plan: Plan) -> Plan:
        """Walk through plan edits interactively. Returns updated plan."""
        plan.status = "in_progress"
        save_plan(plan, self.plans_dir)

        total = len(plan.edits)

        for i, edit in enumerate(plan.edits):
            if edit.status != "pending":
                continue  # resume: skip already-processed edits

            self._show_diff(edit, i + 1, total)

            while True:
                choice = input("[a]pply / [s]kip / [r]evise / [q]uit: ").strip().lower()

                if choice == "a":
                    success = self._apply_edit(edit)
                    edit.status = "applied" if success else "rejected"
                    save_plan(plan, self.plans_dir)
                    break
                elif choice == "s":
                    edit.status = "rejected"
                    save_plan(plan, self.plans_dir)
                    break
                elif choice == "r":
                    feedback = input("Feedback: ").strip()
                    if feedback:
                        revised = self._revise_edit(edit, feedback)
                        edit.file = revised.file
                        edit.description = revised.description
                        edit.old_code = revised.old_code
                        edit.new_code = revised.new_code
                    self._show_diff(edit, i + 1, total)
                elif choice == "q":
                    plan.status = "in_progress"
                    save_plan(plan, self.plans_dir)
                    return plan
                else:
                    print("Please enter a, s, r, or q.")

        plan.status = "completed"
        save_plan(plan, self.plans_dir)

        applied = sum(1 for e in plan.edits if e.status == "applied")
        skipped = sum(1 for e in plan.edits if e.status == "rejected")
        print(f"\nDone. {applied} applied, {skipped} skipped.")

        return plan
