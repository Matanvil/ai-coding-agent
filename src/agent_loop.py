from typing import List, Dict, Any, Callable, Optional
from src.models import Chunk
from src.tools import search_codebase, trace_flow, read_file


def format_chunks(chunks: List[Chunk]) -> str:
    if not chunks:
        return "No relevant code found."
    parts = []
    for c in chunks:
        header = f"[{c.file}:{c.start_line}]"
        if c.score > 0:
            header += f" (score: {c.score:.2f})"
        parts.append(f"{header}\n```\n{c.text}\n```")
    return "\n\n".join(parts)


SUMMARIZATION_SYSTEM_PROMPT = (
    "Summarize the following conversation between a developer and a coding assistant. "
    "Focus on: what was asked, what files/functions were found, and key technical findings. "
    "Be concise — bullet points preferred. This summary will be used as context in future turns."
)


class AgentLoop:
    def __init__(self, llm, embedder, store, repo_root: str, max_history_turns: int = 10):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.repo_root = repo_root
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, Any]] = []

    def _tool_handler(self, tool_name: str, tool_input: Dict) -> str:
        if tool_name == "search_codebase":
            chunks = search_codebase(tool_input["query"], self.embedder, self.store)
            return format_chunks(chunks)
        if tool_name == "trace_flow":
            chunks = trace_flow(tool_input["entry_point"], self.embedder, self.store)
            return format_chunks(chunks)
        if tool_name == "read_file":
            try:
                return read_file(tool_input["path"], self.repo_root)
            except (ValueError, FileNotFoundError) as e:
                return f"Error: {e}"
        return f"Unknown tool: {tool_name}"

    def _summarize_and_truncate_history(self) -> None:
        """Summarize the older half of history when it exceeds the limit.
        Falls back to dropping the old half if summarization fails."""
        max_messages = self.max_history_turns * 2
        if len(self.history) <= max_messages:
            return

        mid = len(self.history) // 2
        old_messages = self.history[:mid]
        recent_messages = self.history[mid:]

        try:
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=512,
                system=SUMMARIZATION_SYSTEM_PROMPT,
                messages=old_messages,
            )
            summary = response.content[0].text
            self.history = [
                {"role": "user", "content": f"[Earlier conversation summary: {summary}]"},
                {"role": "assistant", "content": "Understood, I have context from our earlier conversation."},
            ] + recent_messages
        except Exception:
            # Fallback: discard old half, keep recent
            self.history = recent_messages

    def ask(self, question: str, on_tool_call: Optional[Callable] = None) -> str:
        self.history.append({"role": "user", "content": question})
        answer = self.llm.respond(
            messages=list(self.history),
            tool_handler=self._tool_handler,
            on_tool_call=on_tool_call,
        )
        self.history.append({"role": "assistant", "content": answer})
        self._summarize_and_truncate_history()
        return answer

    def clear_history(self) -> None:
        self.history = []
