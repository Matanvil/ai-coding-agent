import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


@dataclass
class Config:
    active_repo: str    # name of currently active repo, "" if none
    repos: dict         # { name: { "path": str, "indexed_at": str } }
    model: str
    embedding_model: str
    ollama_url: str
    chroma_path: str
    max_results: int
    api_key: str
    local_model: str


def load_config() -> Config:
    data = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = json.load(f)
    return Config(
        active_repo=data.get("active_repo", ""),
        repos=data.get("repos", {}),
        model=data.get("model", "claude-haiku-4-5-20251001"),
        embedding_model=data.get("embedding_model", "nomic-embed-text"),
        ollama_url=data.get("ollama_url", "http://localhost:11434"),
        chroma_path=data.get("chroma_path", ".chroma"),
        max_results=data.get("max_results", 5),
        api_key=data.get("api_key", ""),
        local_model=data.get("local_model", ""),
    )


def save_config(config: Config) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "active_repo": config.active_repo,
            "repos": config.repos,
            "model": config.model,
            "embedding_model": config.embedding_model,
            "ollama_url": config.ollama_url,
            "chroma_path": config.chroma_path,
            "max_results": config.max_results,
            "api_key": config.api_key,
            "local_model": config.local_model,
        }, f, indent=2)
