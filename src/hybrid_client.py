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

    def respond(
        self,
        messages: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict], str],
        on_event: Optional[Callable[[str, Dict], None]] = None,
        max_iterations: int = 10,
    ) -> str:
        if self.force_claude:
            return self.claude.respond(
                messages=messages,
                tool_handler=tool_handler,
                on_event=on_event,
                max_iterations=max_iterations,
            )

        try:
            return self.ollama.respond(
                messages=messages,
                tool_handler=tool_handler,
                on_event=on_event,
                max_iterations=max_iterations,
            )
        except ToolCallParseError as e:
            if on_event:
                on_event("model_fallback", {"reason": str(e), "turns": len(e.partial)})
            return self.claude.respond(
                messages=e.partial if e.partial else messages,
                tool_handler=tool_handler,
                on_event=on_event,
                max_iterations=max_iterations,
            )
        except Exception as e:
            if on_event:
                on_event("model_fallback", {"reason": str(e), "turns": 0})
            return self.claude.respond(
                messages=messages,
                tool_handler=tool_handler,
                on_event=on_event,
                max_iterations=max_iterations,
            )
