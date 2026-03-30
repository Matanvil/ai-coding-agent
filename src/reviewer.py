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

    def review(self, diff: str, context: str) -> ReviewResult:
        """Run the reviewer ReAct loop. Returns ReviewResult when submit_review is called."""
        message = f"Git diff:\n{diff}\n\nContext: {context}\n\nReview the changes above."
        messages = [{"role": "user", "content": message}]

        for _ in range(self.max_iterations):
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=4096,
                system=REVIEWER_SYSTEM_PROMPT,
                tools=REVIEWER_TOOL_DEFINITIONS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                review_result = None

                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    # submit_review terminates the loop. Any preceding tool results
                    # in this same response batch are appended to messages (below),
                    # but since we return immediately after, the messages list is
                    # discarded. This is intentional — do not "fix" by skipping the
                    # tool_results append.
                    if block.name == "submit_review":
                        data = block.input
                        review_result = ReviewResult(
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
                    else:
                        result = self._tool_handler(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

                if review_result is not None:
                    return review_result
            else:
                break

        raise ReviewerError("Reviewer could not produce a review. Try providing more context.")
