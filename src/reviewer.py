from dataclasses import dataclass
from typing import Literal
from src.tools import search_codebase, read_file
from src.agent_loop import format_chunks

REVIEWER_SYSTEM_PROMPT = """You are a Senior Code Reviewer with expertise in software architecture, design patterns, and best practices. Your role is to review completed project steps against original plans and ensure code quality standards are met.

When reviewing completed work, you will:

1. Plan Alignment Analysis:
   - Compare the implementation against the original planning document or step description
   - Identify any deviations from the planned approach, architecture, or requirements
   - Assess whether deviations are justified improvements or problematic departures
   - Verify that all planned functionality has been implemented

2. Code Quality Assessment:
   - Review code for adherence to established patterns and conventions
   - Check for proper error handling, type safety, and defensive programming
   - Evaluate code organization, naming conventions, and maintainability
   - Assess test coverage and quality of test implementations
   - Look for potential security vulnerabilities or performance issues

3. Architecture and Design Review:
   - Ensure the implementation follows SOLID principles and established architectural patterns
   - Check for proper separation of concerns and loose coupling
   - Verify that the code integrates well with existing systems
   - Assess scalability and extensibility considerations

4. Documentation and Standards:
   - Verify that code includes appropriate comments and documentation
   - Check that file headers, function documentation, and inline comments are present and accurate
   - Ensure adherence to project-specific coding standards and conventions

5. Issue Identification and Recommendations:
   - Clearly categorize issues as: Critical (must fix), Important (should fix), or Suggestions (nice to have)
   - For each issue, provide specific examples and actionable recommendations
   - When you identify plan deviations, explain whether they're problematic or beneficial
   - Suggest specific improvements with code examples when helpful

6. Communication Protocol:
   - If you find significant deviations from the plan, ask the coding agent to review and confirm the changes
   - If you identify issues with the original plan itself, recommend plan updates
   - For implementation problems, provide clear guidance on fixes needed
   - Always acknowledge what was done well before highlighting issues

When you have gathered enough context, call submit_review with your structured findings."""

REVIEWER_TOOL_DEFINITIONS = [
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
        "name": "submit_review",
        "description": "Submit the completed review with categorized issues. Call this when you have gathered enough context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Overall summary of the review"},
                "issues": {
                    "type": "array",
                    "description": "List of issues found",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["critical", "important", "suggestion"],
                            },
                            "description": {"type": "string"},
                            "file": {"type": "string", "description": "Relevant file path, or empty string"},
                            "recommendation": {"type": "string"},
                        },
                        "required": ["category", "description", "file", "recommendation"],
                    },
                },
                "suggest_fix_plan": {
                    "type": "boolean",
                    "description": "True if critical or important issues warrant a fix plan",
                },
            },
            "required": ["summary", "issues", "suggest_fix_plan"],
        },
    },
]


@dataclass
class ReviewIssue:
    category: Literal["critical", "important", "suggestion"]
    description: str
    file: str           # empty string if not file-specific
    recommendation: str


@dataclass
class ReviewResult:
    summary: str
    issues: list[ReviewIssue]
    suggest_fix_plan: bool


class ReviewerError(Exception):
    pass


class _ReviewSubmitted(Exception):
    """Sentinel raised by the tool handler when submit_review is called."""
    def __init__(self, result: ReviewResult):
        self.result = result


class Reviewer:
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

    def review(self, diff: str, context: str, on_event=None) -> ReviewResult:
        """Run the reviewer ReAct loop using llm.respond(). Returns ReviewResult when submit_review is called."""
        if on_event:
            on_event("review_started", {})
        message = f"Git diff:\n{diff}\n\nContext: {context}\n\nReview the changes above."
        messages = [{"role": "user", "content": message}]

        def combined_handler(tool_name: str, tool_input: dict) -> str:
            if tool_name == "submit_review":
                data = tool_input
                result = ReviewResult(
                    summary=data["summary"],
                    issues=[
                        ReviewIssue(
                            category=i["category"],
                            description=i["description"],
                            file=i.get("file", ""),
                            recommendation=i["recommendation"],
                        )
                        for i in data.get("issues", [])
                    ],
                    suggest_fix_plan=data.get("suggest_fix_plan", False),
                )
                raise _ReviewSubmitted(result)
            if on_event:
                on_event("tool_call", {"tool": tool_name, "input": tool_input})
            return self._tool_handler(tool_name, tool_input)

        try:
            self.llm.respond(
                messages=messages,
                tool_handler=combined_handler,
                on_event=on_event,
                max_iterations=self.max_iterations,
                system=REVIEWER_SYSTEM_PROMPT,
                tools=REVIEWER_TOOL_DEFINITIONS,
            )
        except _ReviewSubmitted as e:
            if on_event:
                critical_count = sum(1 for i in e.result.issues if i.category == "critical")
                on_event("review_complete", {
                    "issue_count": len(e.result.issues),
                    "critical_count": critical_count,
                    "suggest_fix_plan": e.result.suggest_fix_plan,
                })
            return e.result

        raise ReviewerError("Reviewer could not produce a review. Try providing more context.")
