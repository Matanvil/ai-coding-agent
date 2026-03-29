from pathlib import Path
from typing import List, Iterator
from src.models import Chunk

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".md", ".txt", ".json", ".yaml", ".yml",
}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__",
    ".venv", "venv", "dist", "build", ".chroma",
}
CHUNK_SIZE = 50  # lines per chunk


def scan_files(repo_path: str) -> Iterator[Path]:
    root = Path(repo_path)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def chunk_file_naive(file_path: Path, repo_root: str, chunk_size: int = CHUNK_SIZE) -> List[Chunk]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    lines = text.splitlines()
    relative_path = str(file_path.relative_to(repo_root))

    chunks = []

    for i in range(0, len(lines), chunk_size):
        chunk_text = "\n".join(lines[i : i + chunk_size]).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    file=relative_path,
                    start_line=i + 1,
                    score=0.0,
                    chunk_type="block",
                )
            )
    return chunks


def index_repo(repo_path: str, embedder, store, chunk_size: int = CHUNK_SIZE, use_semantic: bool = False) -> int:
    """Index all files in repo_path. Clears existing index first. Returns chunk count."""
    store.clear()
    total = 0
    for file_path in scan_files(repo_path):
        if use_semantic:
            from src.indexer import chunk_file_semantic
            chunks = chunk_file_semantic(file_path, repo_path)
        else:
            chunks = chunk_file_naive(file_path, repo_path, chunk_size)
        if not chunks:
            continue
        embeddings = [embedder.embed(c.text) for c in chunks]
        store.add(chunks, embeddings)
        total += len(chunks)
    return total
