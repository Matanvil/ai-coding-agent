from pathlib import Path
from typing import List
from src.models import Chunk


def search_codebase(query: str, embedder, store, n_results: int = 5) -> List[Chunk]:
    """Embed query and return most similar chunks from the index."""
    query_embedding = embedder.embed(query)
    return store.search(query_embedding, n_results=n_results)


def trace_flow(entry_point: str, embedder, store, n_results: int = 8) -> List[Chunk]:
    """Two-step retrieval: keyword search then semantic expansion."""
    keyword_results = store.keyword_search(entry_point, n_results=n_results)

    if not keyword_results:
        # No direct references found — fall back to pure semantic search
        return search_codebase(entry_point, embedder, store, n_results=n_results)

    # Semantic expansion: embed the top keyword match, find related chunks
    top_chunk = keyword_results[0]
    semantic_results = search_codebase(top_chunk.text, embedder, store, n_results=n_results)

    # Merge: keyword results first, deduplicate by (file, start_line)
    seen = {(c.file, c.start_line) for c in keyword_results}
    merged = list(keyword_results)
    for chunk in semantic_results:
        key = (chunk.file, chunk.start_line)
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    return merged[:n_results]


def read_file(path: str, repo_root: str) -> str:
    """Read a file within the repo root. Rejects path traversal attempts."""
    repo = Path(repo_root).resolve()
    abs_path = (repo / path).resolve()
    try:
        abs_path.relative_to(repo)
    except ValueError:
        raise ValueError("Path must be within the indexed repo root")
    if not abs_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return abs_path.read_text(encoding="utf-8", errors="ignore")
