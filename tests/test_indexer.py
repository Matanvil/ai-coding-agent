import pytest
import chromadb
from pathlib import Path
from unittest.mock import MagicMock
from src.indexer import scan_files, chunk_file_naive, index_repo
from src.store import VectorStore
from src.models import Chunk


def test_scan_files_finds_supported_extensions(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "readme.md").write_text("# Hello")
    (tmp_path / "data.csv").write_text("a,b,c")
    files = list(scan_files(str(tmp_path)))
    names = [f.name for f in files]
    assert "main.py" in names
    assert "readme.md" in names
    assert "data.csv" not in names


def test_scan_files_skips_node_modules(tmp_path):
    node_mod = tmp_path / "node_modules"
    node_mod.mkdir()
    (node_mod / "lib.js").write_text("module.exports = {}")
    (tmp_path / "app.js").write_text("const x = 1")
    files = list(scan_files(str(tmp_path)))
    names = [f.name for f in files]
    assert "lib.js" not in names
    assert "app.js" in names


def test_scan_files_skips_pycache(tmp_path):
    # Use a supported extension (.py) inside __pycache__ to test directory skipping,
    # not extension filtering.
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.py").write_text("x = 1")
    (tmp_path / "module.py").write_text("x = 1")
    files = list(scan_files(str(tmp_path)))
    # Only the root-level module.py should appear, not the one inside __pycache__
    matching = [f for f in files if f.name == "module.py"]
    assert len(matching) == 1
    assert matching[0].parent == tmp_path


def test_chunk_file_naive_produces_correct_count(tmp_path):
    code = "\n".join([f"line {i}" for i in range(120)])
    f = tmp_path / "big.py"
    f.write_text(code)
    # 120 lines / 50 per chunk = 3 chunks (50, 50, 20)
    chunks = chunk_file_naive(f, str(tmp_path), chunk_size=50)
    assert len(chunks) == 3


def test_chunk_file_naive_has_correct_start_lines(tmp_path):
    code = "\n".join([f"line {i}" for i in range(100)])
    f = tmp_path / "code.py"
    f.write_text(code)
    chunks = chunk_file_naive(f, str(tmp_path), chunk_size=50)
    assert chunks[0].start_line == 1
    assert chunks[1].start_line == 51


def test_chunk_file_naive_chunk_type_and_score(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    pass")
    chunks = chunk_file_naive(f, str(tmp_path))
    assert all(c.chunk_type == "block" for c in chunks)
    assert all(c.score == 0.0 for c in chunks)  # sentinel value


def test_chunk_file_naive_skips_empty_file(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("   \n  \n  ")
    chunks = chunk_file_naive(f, str(tmp_path))
    assert chunks == []


def test_chunk_file_naive_relative_path(tmp_path):
    sub = tmp_path / "src"
    sub.mkdir()
    f = sub / "auth.py"
    f.write_text("def authenticate(): pass")
    chunks = chunk_file_naive(f, str(tmp_path))
    assert chunks[0].file == "src/auth.py"


def test_index_repo_returns_chunk_count(tmp_path):
    (tmp_path / "a.py").write_text("\n".join([f"line {i}" for i in range(50)]))
    (tmp_path / "b.py").write_text("\n".join([f"line {i}" for i in range(50)]))

    client = chromadb.EphemeralClient()
    store = VectorStore(_client=client)
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768

    count = index_repo(str(tmp_path), embedder, store)
    assert count == 2  # one chunk per file (50 lines / chunk_size=50 = 1 chunk each)
    assert store.count() == 2
    assert embedder.embed.call_count == 2  # embed called once per chunk


from src.indexer import chunk_file_semantic


def test_semantic_chunking_extracts_top_level_definitions(tmp_path):
    code = '''
def authenticate(user, token):
    """Check credentials."""
    return validate_token(token)


def validate_token(token):
    """Validate a JWT token."""
    return token is not None


class AuthService:
    def login(self, user, password):
        return True
'''
    f = tmp_path / "auth.py"
    f.write_text(code)
    chunks = chunk_file_semantic(f, str(tmp_path))
    chunk_types = [c.chunk_type for c in chunks]
    texts = [c.text for c in chunks]
    # 3 top-level definitions: authenticate, validate_token, AuthService
    assert len(chunks) == 3
    assert "function" in chunk_types
    assert "class" in chunk_types
    assert any("authenticate" in t for t in texts)
    assert any("AuthService" in t for t in texts)


def test_semantic_chunking_sets_correct_start_line(tmp_path):
    code = "x = 1\n\ndef foo():\n    pass\n"
    f = tmp_path / "code.py"
    f.write_text(code)
    chunks = chunk_file_semantic(f, str(tmp_path))
    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) == 1
    assert func_chunks[0].start_line == 3  # def foo() is on line 3


def test_semantic_chunking_falls_back_for_js(tmp_path):
    f = tmp_path / "app.js"
    f.write_text("const x = 1;\nconst y = 2;")
    chunks = chunk_file_semantic(f, str(tmp_path))
    # JS falls back to naive — chunk_type is "block"
    assert all(c.chunk_type == "block" for c in chunks)


def test_semantic_chunking_falls_back_for_invalid_python(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def foo(:\n    broken syntax")
    # Should not raise — falls back to naive chunking
    chunks = chunk_file_semantic(f, str(tmp_path))
    assert isinstance(chunks, list)


def test_index_repo_uses_semantic_chunking(tmp_path):
    code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    (tmp_path / "code.py").write_text(code)

    client = chromadb.EphemeralClient()
    store = VectorStore(_client=client)
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768

    count = index_repo(str(tmp_path), embedder, store, use_semantic=True)
    assert count == 2  # two top-level functions = two chunks
    assert embedder.embed.call_count == 2
