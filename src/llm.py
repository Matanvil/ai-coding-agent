import os
import anthropic
from typing import List, Dict, Any, Callable, Optional

TOOL_DEFINITIONS = [
    {
        "name": "search_codebase",
        "description": "Search the indexed codebase for code chunks relevant to a query. Use this to find where functionality is implemented.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing what you're looking for",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "trace_flow",
        "description": "Trace where a function, class, or symbol is defined and used across the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entry_point": {
                    "type": "string",
                    "description": "Function name, class name, or symbol to trace",
                }
            },
            "required": ["entry_point"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the full contents of a specific file in the repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path from repo root",
                }
            },
            "required": ["path"],
        },
    },
]

SYSTEM_PROMPT = """You are an expert coding assistant with access to a codebase index.
Help developers understand their codebase by searching it, tracing code flows, and reading files.

When answering:
1. Search for relevant code before answering — never guess without checking
2. Trace functions or classes when asked about flow or relationships
3. Read specific files when you need full context
4. Always cite the file path and line number in your answer
5. Be concise and precise — developers value accuracy over verbosity"""


class ClaudeClient:
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: Optional[str] = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def respond(
        self,
        messages: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict], str],
        on_tool_call: Optional[Callable[[str, Dict], None]] = None,
        max_iterations: int = 10,
    ) -> str:
        """Run the ReAct loop until a final answer is produced."""
        current_messages = list(messages)

        for _ in range(max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=current_messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            if response.stop_reason == "tool_use":
                current_messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        if on_tool_call:
                            on_tool_call(block.name, block.input)
                        result = tool_handler(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                current_messages.append({"role": "user", "content": tool_results})

        return "Maximum iterations reached without a final answer."
