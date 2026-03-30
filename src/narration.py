def narrate_event(event_type: str, data: dict) -> str:
    """Return a human-readable string for any on_event payload. Returns '' for unknown types."""
    if event_type == "tool_call":
        tool = data.get("tool", "")
        inp = data.get("input", {})
        if tool == "search_codebase":
            return f'Searching: "{inp.get("query", "")}"'
        if tool == "trace_flow":
            return f"Tracing: {inp.get('entry_point', '')}"
        if tool == "read_file":
            return f"Reading: {inp.get('path', '')}"
        return f"Using tool: {tool}"
    if event_type == "planning_started":
        return f'Planning: "{data.get("task", "")}"'
    if event_type == "planning_complete":
        count = data.get("edit_count", 0)
        return f"Plan ready: {count} edit{'s' if count != 1 else ''} proposed"
    if event_type == "review_started":
        return "Reviewing changes..."
    if event_type == "review_complete":
        issue_count = data.get("issue_count", 0)
        critical_count = data.get("critical_count", 0)
        label = "issue" if issue_count == 1 else "issues"
        base = f"Review done: {issue_count} {label}"
        if critical_count:
            return f"{base}, {critical_count} critical"
        return base
    if event_type == "edit_presented":
        index = data.get("index", 0)
        total = data.get("total", 0)
        return f"Edit {index}/{total} — {data.get('file', '')}: {data.get('description', '')}"
    if event_type == "edit_applied":
        return f"Applied: {data.get('file', '')}"
    if event_type == "edit_skipped":
        return f"Skipped: {data.get('file', '')}"
    if event_type == "edit_revised":
        return f"Revised: {data.get('file', '')}"
    if event_type == "execution_complete":
        return f"Done: {data.get('applied', 0)} applied, {data.get('skipped', 0)} skipped"
    if event_type == "model_fallback":
        turns = data.get("turns", 0)
        kind = data.get("kind", "")
        if kind == "parse_error" and turns:
            return f"Local model failed after {turns} turn{'s' if turns != 1 else ''} — handing off to Claude"
        if kind == "connection_error":
            return "Local model unavailable — falling back to Claude"
        return "Falling back to Claude"
    return ""
