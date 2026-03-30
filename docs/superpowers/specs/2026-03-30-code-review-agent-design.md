# Code Review Agent Design

**Date:** 2026-03-30

## Overview

A code review agent that reviews git diffs against coding standards and plan alignment. Triggered manually via a `review` command or automatically after `execute` completes. Outputs structured feedback (Critical/Important/Suggestions) and optionally generates a fix plan.

## Architecture & Components

### `src/reviewer.py` — `Reviewer` class

A ReAct agent loop, parallel to the existing `Planner`. Runs until it calls `submit_review` with a structured result.

**Tools available to the reviewer:**
- `search_codebase` — search the indexed codebase for relevant context
- `read_file` — read a full file from the repo
- `submit_review` — structured output tool; terminates the loop

**System prompt:** Senior Code Reviewer persona (see prompt below). Reviews for plan alignment, code quality, architecture, documentation, and produces categorized issues.

### `ReviewResult` dataclass (in `src/models.py` or inline in `reviewer.py`)

```python
@dataclass
class ReviewIssue:
    category: str          # "critical" | "important" | "suggestion"
    description: str
    file: str              # optional, empty string if not file-specific
    recommendation: str

@dataclass
class ReviewResult:
    summary: str
    issues: list[ReviewIssue]
    suggest_fix_plan: bool
```

### `agent.py` integration points

1. `run_review(context, config, embedder, llm, store)` — handles the `review [context]` command
2. End of `run_execute()` — auto-triggers review after executor completes, using `plan.task` as context

## Data Flow

### Manual trigger

```
review [optional context]
  → git diff HEAD (active repo path)
  → Reviewer ReAct loop
      tools: search_codebase, read_file
      submit_review → ReviewResult
  → print structured report
  → if issues found: "Generate a fix plan? [y/n]"
      → y: run_plan(fix description) using existing Planner
```

### Post-execution trigger

```
execute → Executor.execute(plan) completes
  → auto: git diff HEAD + plan.task as context
  → same Reviewer loop → ReviewResult
  → print report
  → if issues found: "Generate a fix plan? [y/n]"
```

### Reviewer initial message

```
Git diff:
<diff output>

Context: <plan.task or user-provided string>

Review the changes above.
```

If the diff is empty and no user context is provided, skip the review and print: `No uncommitted changes to review.`

### `submit_review` tool schema

```json
{
  "summary": "string",
  "issues": [
    {
      "category": "critical|important|suggestion",
      "description": "string",
      "file": "string (optional)",
      "recommendation": "string"
    }
  ],
  "suggest_fix_plan": true|false
}
```

## CLI Integration & UX

### New command

```
review [context]     Review current git diff; optional context about the changes
```

Added to `HELP_TEXT` in `agent.py`.

### Output format

```
────────────────────────────────────────────────
Code Review
────────────────────────────────────────────────
<summary paragraph>

CRITICAL
  • src/executor.py — description of issue
    → Recommendation

IMPORTANT
  • src/planner.py — description of issue
    → Recommendation

SUGGESTIONS
  • src/agent_loop.py — description of issue
────────────────────────────────────────────────
Generate a fix plan for the above issues? [y/n]:
```

Sections with no issues are omitted. If there are no issues at all, the prompt is skipped and a clean message is printed instead (e.g., `No issues found.`).

### Error handling

- No active repo → print error, same as other commands
- Empty git diff + no context → `No uncommitted changes to review.`
- Reviewer loop exceeds max iterations or fails → print error, do not crash
- `git diff` subprocess fails → print error with stderr output

## Reviewer System Prompt

```
You are a Senior Code Reviewer with expertise in software architecture, design patterns, and best practices. Your role is to review completed project steps against original plans and ensure code quality standards are met.

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

When you have gathered enough context, call submit_review with your structured findings.
```

## Files Changed

| File | Change |
|------|--------|
| `src/reviewer.py` | New — `Reviewer` class, `ReviewResult`, `ReviewIssue` |
| `agent.py` | Add `review` command, `run_review()`, auto-trigger in `run_execute()` |
