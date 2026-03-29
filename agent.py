import os
import sys
from datetime import datetime
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
  ask <question>              Ask a question about the codebase
  trace <symbol>              Trace where a function/class is defined and used
  index [--repo <path>]       Index or re-index a repository
  use <repo>                  Switch to an indexed repo
  repos                       List all indexed repos
  clear                       Clear conversation history
  help                        Show this help
  exit                        Quit

You can also type a question directly without 'ask'.
"""

DIVIDER = "─" * 48

_PROMPT_STYLE = Style.from_dict({"": "bg:#1a1a2e #ffffff"})


def build_shared(config):
    """Create embedder and LLM — shared across all repos."""
    embedder = OllamaEmbedder(model=config.embedding_model, base_url=config.ollama_url)
    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    llm = ClaudeClient(model=config.model, api_key=api_key)
    return embedder, llm


def activate_repo(repo_name: str, config, embedder, llm):
    """Create store and agent for a named repo. repo_name must be in config.repos."""
    repo_info = config.repos[repo_name]
    store = VectorStore(chroma_path=config.chroma_path, collection_name=repo_name)
    agent = AgentLoop(llm=llm, embedder=embedder, store=store, repo_root=repo_info["path"])
    return store, agent


def build_session() -> PromptSession:
    history_file = Path.home() / ".ai-agent-history"
    return PromptSession(
        history=FileHistory(str(history_file)),
        style=_PROMPT_STYLE,
    )


def parse_cli_args() -> str:
    """Return repo name from CLI args, or '' if none provided.

    Accepts: python agent.py jarvis  OR  python agent.py --repo jarvis
    """
    args = sys.argv[1:]
    if not args:
        return ""
    if args[0] == "--repo" and len(args) > 1:
        return args[1]
    if not args[0].startswith("-"):
        return args[0]
    return ""


def print_startup(config, store):
    """Print banner, then active repo info or full repo list."""
    print(BANNER)
    if store is not None and store.count() > 0:
        print(f"Repo: {config.active_repo} ({store.count()} chunks)")
    elif config.repos:
        print()
        print("Indexed repos:")
        for name, info in config.repos.items():
            s = VectorStore(chroma_path=config.chroma_path, collection_name=name)
            count = s.count()
            indexed_at = info.get("indexed_at", "unknown")
            marker = "* " if name == config.active_repo else "  "
            print(f"  {marker}{name:<15} ({count} chunks, indexed {indexed_at})")
        print()
        print("No active repo. Type 'use <repo>' to select one.")
    else:
        print("No repos indexed yet. Run: index --repo <path>")
    print()


def run_index(rest: str, config, embedder, llm):
    """Index a repo. Returns (config, store, agent) or (config, None, None) on failure."""
    args = rest.split()
    repo_path = ""
    if "--repo" in args:
        idx = args.index("--repo")
        if idx + 1 < len(args):
            repo_path = args[idx + 1]
    elif config.active_repo and config.repos.get(config.active_repo):
        repo_path = config.repos[config.active_repo]["path"]

    if not repo_path:
        print("Error: No repo configured. Run: index --repo <path>")
        return config, None, None

    repo_path = str(Path(repo_path).expanduser().resolve())
    if not Path(repo_path).exists():
        print(f"Error: Path does not exist: {repo_path}")
        return config, None, None

    repo_name = Path(repo_path).name
    store = VectorStore(chroma_path=config.chroma_path, collection_name=repo_name)

    print(f"Indexing {repo_path}...")
    try:
        count = index_repo(repo_path, embedder, store, use_semantic=True)
        print(f"Done. Indexed {count} chunks.")
    except EmbedderError as e:
        print(f"Error: {e}")
        return config, None, None

    config.repos[repo_name] = {
        "path": repo_path,
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    config.active_repo = repo_name
    save_config(config)

    agent = AgentLoop(llm=llm, embedder=embedder, store=store, repo_root=repo_path)
    return config, store, agent


def run_use(repo_name: str, config, embedder, llm):
    """Switch to an already-indexed repo. Returns (config, store, agent) or (config, None, None)."""
    if repo_name not in config.repos:
        known = ", ".join(config.repos.keys()) or "none"
        print(f"Unknown repo: {repo_name}. Indexed repos: {known}")
        return config, None, None

    store = VectorStore(chroma_path=config.chroma_path, collection_name=repo_name)
    if store.count() == 0:
        repo_path = config.repos[repo_name]["path"]
        print(f"Repo '{repo_name}' is registered but not indexed. Run: index --repo {repo_path}")
        return config, None, None

    config.active_repo = repo_name
    save_config(config)

    repo_path = config.repos[repo_name]["path"]
    agent = AgentLoop(llm=llm, embedder=embedder, store=store, repo_root=repo_path)
    print(f"Switched to {repo_name} ({store.count()} chunks).")
    print(DIVIDER)
    return config, store, agent


def run_repos(config):
    """Print all indexed repos with chunk counts and timestamps."""
    if not config.repos:
        print("No repos indexed yet. Run: index --repo <path>")
        return
    print("Indexed repos:")
    for name, info in config.repos.items():
        store = VectorStore(chroma_path=config.chroma_path, collection_name=name)
        count = store.count()
        indexed_at = info.get("indexed_at", "unknown")
        marker = "* " if name == config.active_repo else "  "
        print(f"  {marker}{name:<15} ({count} chunks, indexed {indexed_at})")


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
    config = load_config()

    if not config.api_key and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: No API key found.")
        print('Add "api_key": "your_key" to config.json, or set ANTHROPIC_API_KEY in your environment.')
        sys.exit(1)

    embedder, llm = build_shared(config)

    store = None
    agent = None

    cli_repo = parse_cli_args()
    if cli_repo:
        if cli_repo not in config.repos:
            print(f"Unknown repo: {cli_repo}. Run: index --repo <path>")
            sys.exit(1)
        config.active_repo = cli_repo
        save_config(config)
        store, agent = activate_repo(cli_repo, config, embedder, llm)
    elif config.active_repo and config.active_repo in config.repos:
        store, agent = activate_repo(config.active_repo, config, embedder, llm)

    print_startup(config, store)

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
            if agent:
                agent.clear_history()
                print("Conversation history cleared.")
            else:
                print("No active repo. Type 'use <repo>' to select one.")
        elif command == "index":
            config, new_store, new_agent = run_index(rest, config, embedder, llm)
            if new_store:
                store = new_store
                agent = new_agent
        elif command == "use":
            if not rest:
                print("Usage: use <repo>")
            else:
                config, new_store, new_agent = run_use(rest, config, embedder, llm)
                if new_store:
                    store = new_store
                    agent = new_agent
        elif command == "repos":
            run_repos(config)
        elif command == "ask":
            if not rest:
                print("Usage: ask <question>")
            elif not agent:
                print("No active repo. Type 'use <repo>' to select one.")
            else:
                handle_question(rest, agent, store)
        elif command == "trace":
            if not rest:
                print("Usage: trace <symbol>")
            elif not agent:
                print("No active repo. Type 'use <repo>' to select one.")
            else:
                handle_question(f"Trace the flow of: {rest}", agent, store)
        else:
            if not agent:
                print("No active repo. Type 'use <repo>' to select one.")
            else:
                handle_question(user_input, agent, store)


if __name__ == "__main__":
    main()
