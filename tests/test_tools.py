import pytest
import chromadb
from unittest.mock import MagicMock
from src.store import VectorStore
from src.tools import search_codebase, trace_flow, read_file
from src.models import Chunk


def make_store():
    client = chromadb.EphemeralClient()
    store = VectorStore(_client=client)
    chunks = [
        Chunk("def authenticate(user, token): ...", "auth.py", 1, 0.0, "block"),
        Chunk("def validate_token(token): return True", "auth.py", 10, 0.0, "block"),
        Chunk("class UserService: ...", "user.py", 1, 0.0, "block"),
    ]
    embeddings = [[float(i) / 10] * 768 for i in range(len(chunks))]
    store.add(chunks, embeddings)
    return store


def make_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [0.0] * 768
    return embedder


def test_search_codebase_returns_chunks():
    store = make_store()
    results = search_codebase("authentication", make_embedder(), store, n_results=2)
    assert len(results) == 2
    assert all(isinstance(r, Chunk) for r in results)


def test_search_codebase_returns_empty_for_empty_store():
    client = chromadb.EphemeralClient()
    store = VectorStore(_client=client)
    results = search_codebase("anything", make_embedder(), store)
    assert results == []


def test_trace_flow_finds_keyword_match():
    store = make_store()
    results = trace_flow("validate_token", make_embedder(), store)
    files = [r.file for r in results]
    assert "auth.py" in files


def test_trace_flow_falls_back_to_semantic_when_no_keyword_match():
    store = make_store()
    embedder = make_embedder()
    # "nonexistent_symbol" won't match any chunk text — should fall back to semantic
    results = trace_flow("nonexistent_symbol", embedder, store)
    assert isinstance(results, list)
    # Verify the fallback actually ran: embedder.embed must have been called
    assert embedder.embed.called


def test_trace_flow_deduplicates_results():
    store = make_store()
    embedder = make_embedder()
    results = trace_flow("validate_token", embedder, store)
    # No duplicate (file, start_line) pairs
    seen = set()
    for r in results:
        key = (r.file, r.start_line)
        assert key not in seen, f"Duplicate chunk: {key}"
        seen.add(key)


def test_read_file_returns_content(tmp_path):
    f = tmp_path / "auth.py"
    f.write_text("def foo(): pass")
    content = read_file("auth.py", str(tmp_path))
    assert content == "def foo(): pass"


def test_read_file_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="within the indexed repo root"):
        read_file("../../etc/passwd", str(tmp_path))


def test_read_file_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_file("nonexistent.py", str(tmp_path))
