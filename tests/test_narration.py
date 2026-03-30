from src.narration import narrate_event


def test_tool_call_search():
    result = narrate_event("tool_call", {"tool": "search_codebase", "input": {"query": "retry logic"}})
    assert result == 'Searching: "retry logic"'


def test_tool_call_trace():
    result = narrate_event("tool_call", {"tool": "trace_flow", "input": {"entry_point": "embed"}})
    assert result == "Tracing: embed"


def test_tool_call_read():
    result = narrate_event("tool_call", {"tool": "read_file", "input": {"path": "src/embedder.py"}})
    assert result == "Reading: src/embedder.py"


def test_tool_call_unknown_tool():
    result = narrate_event("tool_call", {"tool": "mystery_tool", "input": {}})
    assert result == "Using tool: mystery_tool"


def test_planning_started():
    result = narrate_event("planning_started", {"task": "add retry"})
    assert result == 'Planning: "add retry"'


def test_planning_complete_singular():
    result = narrate_event("planning_complete", {"edit_count": 1})
    assert result == "Plan ready: 1 edit proposed"


def test_planning_complete_plural():
    result = narrate_event("planning_complete", {"edit_count": 3})
    assert result == "Plan ready: 3 edits proposed"


def test_review_started():
    result = narrate_event("review_started", {})
    assert result == "Reviewing changes..."


def test_review_complete_no_criticals():
    result = narrate_event("review_complete", {"issue_count": 2, "critical_count": 0, "suggest_fix_plan": False})
    assert result == "Review done: 2 issues"


def test_review_complete_with_criticals():
    result = narrate_event("review_complete", {"issue_count": 3, "critical_count": 1, "suggest_fix_plan": True})
    assert result == "Review done: 3 issues, 1 critical"


def test_review_complete_singular_issue():
    result = narrate_event("review_complete", {"issue_count": 1, "critical_count": 0, "suggest_fix_plan": False})
    assert result == "Review done: 1 issue"


def test_edit_presented():
    result = narrate_event("edit_presented", {"index": 2, "total": 3, "file": "src/foo.py", "description": "add retry"})
    assert result == "Edit 2/3 — src/foo.py: add retry"


def test_edit_applied():
    result = narrate_event("edit_applied", {"file": "src/foo.py"})
    assert result == "Applied: src/foo.py"


def test_edit_skipped():
    result = narrate_event("edit_skipped", {"file": "src/foo.py"})
    assert result == "Skipped: src/foo.py"


def test_edit_revised():
    result = narrate_event("edit_revised", {"file": "src/foo.py"})
    assert result == "Revised: src/foo.py"


def test_execution_complete():
    result = narrate_event("execution_complete", {"applied": 2, "skipped": 1})
    assert result == "Done: 2 applied, 1 skipped"


def test_unknown_event_returns_empty_string():
    result = narrate_event("some_future_event", {"anything": True})
    assert result == ""


def test_narrate_model_fallback_parse_error_with_turns():
    result = narrate_event("model_fallback", {"kind": "parse_error", "reason": "bad", "turns": 3})
    assert "Claude" in result
    assert "3" in result


def test_narrate_model_fallback_connection_error():
    result = narrate_event("model_fallback", {"kind": "connection_error", "reason": "refused", "turns": 0})
    assert "Claude" in result


def test_narrate_model_fallback_returns_empty_for_unknown():
    # model_fallback with no known kind should still return something useful
    result = narrate_event("model_fallback", {"kind": "unknown", "reason": "x", "turns": 0})
    assert "Claude" in result
