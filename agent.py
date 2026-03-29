import os
import sys
from pathlib import Path

from src.config import load_config, save_config
from src.embedder import OllamaEmbedder, EmbedderError
from src.store import VectorStore
from src.indexer import index_repo
from src.llm import ClaudeClient
from src.agent_loop import AgentLoop
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

BANNER = "AI Coding Agent — type 'help' for commands, 'exit' to quit."

HELP_TEXT = """
Commands:
  ask <question>           Ask a question about the codebase
  trace <symbol>           Trace where a function/class is defined and used
  index [--repo <path>]    Index or re-index a repository
  clear                    Clear conversation history
  help                     Show this help
  exit                     Quit

You can also type a question directly without 'ask'.
"""

DIVIDER = "─" * 48

_PROMPT_STYLE = Style.from_dict({"": "bg:#1a1a2e #ffffff"})


def build_components(config):
    embedder = OllamaEmbedder(model=config.embedding_model, base_url=config.ollama_url)
    store = VectorStore(chroma_path=config.chroma_path)
    llm = ClaudeClient(model=config.model)
    agent = AgentLoop(llm=llm, embedder=embedder, store=store, repo_root=config.repo_path)
    return embedder, store, llm, agent


def build_session() -> PromptSession:
    history_file = Path.home() / ".ai-agent-history"
    return PromptSession(
        history=FileHistory(str(history_file)),
        style=_PROMPT_STYLE,
    )


def run_index(rest: str, config, embedder, store, llm):
    args = rest.split()
    repo_path = config.repo_path
    if "--repo" in args:
        idx = args.index("--repo")
        if idx + 1 < len(args):
            repo_path = args[idx + 1]

    if not repo_path:
        print("Error: No repo configured. Run: index --repo <path>")
        return config, None

    repo_path = str(Path(repo_path).expanduser().resolve())
    if not Path(repo_path).exists():
        print(f"Error: Path does not exist: {repo_path}")
        return config, None

    config.repo_path = repo_path
    save_config(config)

    print(f"Indexing {repo_path}...")
    try:
        count = index_repo(repo_path, embedder, store, use_semantic=True)
        print(f"Done. Indexed {count} chunks.")
    except EmbedderError as e:
        print(f"Error: {e}")
        return config, None

    new_agent = AgentLoop(llm=llm, embedder=embedder, store=store, repo_root=repo_path)
    return config, new_agent


def handle_question(question: str, agent, store):
    if store.count() == 0:
        print("No index found. Run 'index' first.")
        return

    print("Thinking...")

    def on_tool_call(tool_name, tool_input):
        if tool_name == "search_codebase":
            print(f" → searching: \"{tool_input.get('query', '')}\"")
        elif tool_name == "trace_flow":
            print(f" → tracing: {tool_input.get('entry_point', '')}")
        elif tool_name == "read_file":
            print(f" → reading: {tool_input.get('path', '')}")

    answer = agent.ask(question, on_tool_call=on_tool_call)
    print(f"\n{answer}\n")
    print(DIVIDER)


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    config = load_config()
    embedder, store, llm, agent = build_components(config)

    print(BANNER)
    if config.repo_path and store.count() > 0:
        print(f"Repo: {config.repo_path} ({store.count()} chunks indexed)")
    elif config.repo_path:
        print(f"Repo: {config.repo_path} (not indexed — run 'index')")
    else:
        print("No repo configured. Run: index --repo <path>")
    print()

    session = build_session()

    while True:
        try:
            user_input = session.prompt("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if command == "exit":
            print("Goodbye.")
            break
        elif command == "help":
            print(HELP_TEXT)
        elif command == "clear":
            agent.clear_history()
            print("Conversation history cleared.")
        elif command == "index":
            config, new_agent = run_index(rest, config, embedder, store, llm)
            if new_agent:
                agent = new_agent
        elif command == "ask":
            if not rest:
                print("Usage: ask <question>")
            else:
                handle_question(rest, agent, store)
        elif command == "trace":
            if not rest:
                print("Usage: trace <symbol>")
            else:
                handle_question(f"Trace the flow of: {rest}", agent, store)
        else:
            handle_question(user_input, agent, store)


if __name__ == "__main__":
    main()
