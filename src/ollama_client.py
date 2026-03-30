import json
import re
import requests
from typing import Any, Dict, List, Optional, Callable

from src.llm import BaseLLMClient, ToolCallParseError, TOOL_DEFINITIONS, SYSTEM_PROMPT


def _to_openai_tools(anthropic_tools: List[Dict]) -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_tools
    ]


def _parse_xml_tool_call(content: str) -> Optional[Dict]:
    name_match = re.search(r"<function=(\w+)>", content)
    if not name_match:
        return None
    name = name_match.group(1)
    params = {}
    for m in re.finditer(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", content, re.DOTALL):
        params[m.group(1)] = m.group(2).strip()
    return {"id": "xml_0", "name": name, "input": params}


def _parse_tool_call(message: Dict) -> Optional[Dict]:
    tool_calls = message.get("tool_calls")
    if tool_calls:
        tc = tool_calls[0]
        return {
            "id": tc.get("id", "call_0"),
            "name": tc["function"]["name"],
            "input": json.loads(tc["function"]["arguments"]),
        }
    content = (message.get("content") or "").strip()
    if "<function=" in content:
        return _parse_xml_tool_call(content)
    return None


def _to_ollama_messages(messages: List[Dict]) -> List[Dict]:
    """Convert Anthropic-format message history to OpenAI/Ollama format."""
    result = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            result.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
                tool_calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b["input"]),
                        },
                    }
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
                result.append({"role": "assistant", "tool_calls": tool_calls})
            elif any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": b["content"],
                        })
            else:
                text = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
                result.append({"role": msg["role"], "content": text})
        else:
            result.append(msg)
    return result


class OllamaClient(BaseLLMClient):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._openai_tools = _to_openai_tools(TOOL_DEFINITIONS)

    def respond(
        self,
        messages: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict], str],
        on_event: Optional[Callable[[str, Dict], None]] = None,
        max_iterations: int = 10,
    ) -> str:
        current_messages = list(messages)

        for _ in range(max_iterations):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + _to_ollama_messages(current_messages),
                        "tools": self._openai_tools,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                raise ToolCallParseError(f"Ollama request failed: {e}", partial=current_messages)

            body = resp.json()
            choice = body["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")

            tool_call = _parse_tool_call(message)

            if tool_call is None:
                if finish_reason == "stop":
                    return (message.get("content") or "").strip()
                raise ToolCallParseError(
                    f"Unparseable response (finish_reason={finish_reason!r})",
                    partial=current_messages,
                )

            if on_event:
                on_event("tool_call", {"tool": tool_call["name"], "input": tool_call["input"]})

            result = tool_handler(tool_call["name"], tool_call["input"])

            current_messages.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": tool_call["input"],
                }],
            })
            current_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": result,
                }],
            })

        return "Maximum iterations reached without a final answer."
