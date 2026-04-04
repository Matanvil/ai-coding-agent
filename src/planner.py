from datetime import datetime
from src.plan_store import Plan, FileEdit
from src.tools import search_codebase, read_file
from src.agent_loop import format_chunks

PLANNER_SYSTEM_PROMPT = """You are an expert coding assistant tasked with planning code changes.

Your job:
1. Search the codebase to understand the relevant code
2. Read specific files to get the exact current content
3. When you have a complete understanding, call submit_plan with a list of targeted edits

Rules for edits:
- Each edit's old_code must be an EXACT string copied verbatim from the file
- Prefer multiple small edits over one large replacement
- Only include files that actually need to change
- Be minimal — do not change what does not need to change"""

PLANNER_TOOL_DEFINITIONS = [
    {
        "name": "search_codebase",
        "description": "Search the indexed codebase for relevant code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
            },
            "required": ["query"],
        },
    },
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
        "description": "Submit the completed plan of file edits. Call this when you are ready.",
        "input_schema": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "List of file edits",
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


class PlannerError(Exception):
    pass


class _PlanSubmitted(Exception):
    """Sentinel raised by the tool handler when submit_plan is called."""
    def __init__(self, plan: Plan):
        self.plan = plan


class Planner:
    def __init__(self, llm, embedder, store, repo_root: str, max_iterations: int = 15):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.repo_root = repo_root
        self.max_iterations = max_iterations

    def _tool_handler(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "search_codebase":
            chunks = search_codebase(tool_input["query"], self.embedder, self.store)
            return format_chunks(chunks)
        if tool_name == "read_file":
            try:
                return read_file(tool_input["path"], self.repo_root)
            except (ValueError, FileNotFoundError) as e:
                return f"Error: {e}"
        return f"Unknown tool: {tool_name}"

    def _run(self, messages: list, task: str, on_event=None) -> Plan:
        """Run the planner ReAct loop using llm.respond(). Returns Plan when submit_plan is called."""
        def combined_handler(tool_name: str, tool_input: dict) -> str:
            if tool_name == "submit_plan":
                plan = Plan(
                    task=task,
                    repo="",  # set by caller
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    status="pending",
                    edits=[
                        FileEdit(
                            file=e["file"],
                            description=e["description"],
                            old_code=e["old_code"],
                            new_code=e["new_code"],
                            status="pending",
                        )
                        for e in tool_input["edits"]
                    ],
                )
                raise _PlanSubmitted(plan)
            return self._tool_handler(tool_name, tool_input)

        try:
            self.llm.respond(
                messages=messages,
                tool_handler=combined_handler,
                on_event=on_event,
                max_iterations=self.max_iterations,
                system=PLANNER_SYSTEM_PROMPT,
                tools=PLANNER_TOOL_DEFINITIONS,
            )
        except _PlanSubmitted as e:
            return e.plan

        raise PlannerError("Planner could not produce a plan. Try a more specific task.")

    def plan(self, task: str, repo: str, on_event=None) -> Plan:
        """Run the planner agent. Returns a Plan or raises PlannerError."""
        if on_event:
            on_event("planning_started", {"task": task})
        messages = [{"role": "user", "content": task}]
        plan = self._run(messages, task=task, on_event=on_event)
        plan.repo = repo
        if on_event:
            on_event("planning_complete", {"edit_count": len(plan.edits)})
        return plan

    def revise(self, plan: Plan, feedback: str, on_event=None) -> Plan:
        """Re-run planner with existing plan + feedback. Returns revised Plan."""
        if on_event:
            on_event("planning_started", {"task": plan.task})
        edits_text = "\n".join(
            f"  {i + 1}. {e.file} — {e.description}"
            for i, e in enumerate(plan.edits)
        )
        message = (
            f"Original task: {plan.task}\n\n"
            f"Current plan:\n{edits_text}\n\n"
            f"Feedback: {feedback}\n\n"
            f"Please revise the plan based on this feedback."
        )
        revised = self._run([{"role": "user", "content": message}], task=plan.task, on_event=on_event)
        revised.repo = plan.repo
        revised.task = plan.task  # always preserve original task
        if on_event:
            on_event("planning_complete", {"edit_count": len(revised.edits)})
        return revised
