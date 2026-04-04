from typing import Any, Callable, Dict, List, Optional

from src.llm import BaseLLMClient, ToolCallParseError


class HybridClient(BaseLLMClient):
    """Tries OllamaClient first; falls back to ClaudeClient on failure.

    Set force_claude = True to skip Ollama for the session (--model claude flag).
    """

    def __init__(self, ollama: BaseLLMClient, claude: BaseLLMClient):
        self.ollama = ollama
        self.claude = claude
        self.force_claude: bool = False

    @property
    def client(self):
        """Expose the underlying Anthropic client for callers that bypass respond()."""
        return self.claude.client

    @property
    def model(self) -> str:
        """Expose the Claude model name for callers that bypass respond()."""
        return self.claude.model

    def respond(
        self,
        messages: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict], str],
        on_event: Optional[Callable[[str, Dict], None]] = None,
        max_iterations: int = 10,
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        kwargs = dict(
            messages=messages,
            tool_handler=tool_handler,
            on_event=on_event,
            max_iterations=max_iterations,
            system=system,
            tools=tools,
        )
        if self.force_claude:
            return self.claude.respond(**kwargs)

        try:
            return self.ollama.respond(**kwargs)
        except ToolCallParseError as e:
            if on_event:
                on_event("model_fallback", {"kind": "parse_error", "reason": str(e), "turns": len(e.partial)})
            return self.claude.respond(
                **{**kwargs, "messages": e.partial if e.partial else messages}
            )
        except Exception as e:
            if on_event:
                on_event("model_fallback", {"kind": "connection_error", "reason": str(e), "turns": 0})
            return self.claude.respond(**kwargs)
