import ast
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


def chunk_file_semantic(file_path: Path, repo_root: str) -> List[Chunk]:
    """Split a Python file by top-level function/class boundaries.
    Falls back to naive chunking for non-Python files or unparseable Python."""
    if file_path.suffix != ".py":
        return chunk_file_naive(file_path, repo_root)

    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return chunk_file_naive(file_path, repo_root)

    lines = source.splitlines()
    relative_path = str(file_path.relative_to(repo_root))
    chunks = []

    # iter_child_nodes gives only top-level definitions, not nested ones
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno
        end = node.end_lineno
        chunk_text = "\n".join(lines[start - 1 : end]).strip()
        if not chunk_text:
            continue
        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
        chunks.append(
            Chunk(
                text=chunk_text,
                file=relative_path,
                start_line=start,
                score=0.0,
                chunk_type=chunk_type,
            )
        )

    # If no top-level definitions found, fall back to naive
    if not chunks:
        return chunk_file_naive(file_path, repo_root)

    return chunks


def index_repo(repo_path: str, embedder, store, chunk_size: int = CHUNK_SIZE, use_semantic: bool = False) -> int:
    """Index all files in repo_path. Clears existing index first. Returns chunk count."""
    store.clear()
    total = 0
    for file_path in scan_files(repo_path):
        if use_semantic:
            chunks = chunk_file_semantic(file_path, repo_path)
        else:
            chunks = chunk_file_naive(file_path, repo_path, chunk_size)
        if not chunks:
            continue
        embeddings = [embedder.embed(c.text) for c in chunks]
        store.add(chunks, embeddings)
        total += len(chunks)
    return total
