# AI Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI-based AI coding agent from scratch that indexes a codebase, answers questions via RAG, and traces code flows via an agent loop — serving as a hands-on learning platform for embeddings, vector search, and ReAct reasoning.

**Architecture:** A thin-slice build: indexing pipeline (file scanner → chunker → embedder → ChromaDB) feeds a retrieval layer, which feeds a Claude-powered ReAct agent loop. The REPL connects everything. Each layer is improved after the full system works end-to-end.

**Tech Stack:** Python, `nomic-embed-text` via Ollama (local embeddings), Anthropic Claude Haiku API (LLM), ChromaDB (vector store), no AI frameworks.

---

## File Structure

```
ai-coding-agent/
├── agent.py                  # Entry point: REPL loop and command dispatcher
├── config.json               # User config: repo_path, model (gitignored)
├── requirements.txt
├── pyproject.toml            # pytest config: pythonpath
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── models.py             # Chunk dataclass — shared contract between all layers
│   ├── config.py             # Config loader: reads config.json + env vars
│   ├── embedder.py           # Ollama HTTP client: text → 768-dim vector
│   ├── store.py              # ChromaDB wrapper: add, search, keyword_search, clear
│   ├── indexer.py            # File scanner + naive chunker + indexing pipeline
│   ├── tools.py              # search_codebase, trace_flow, read_file
│   ├── llm.py                # Claude API client with ReAct tool-use loop
│   └── agent_loop.py         # Conversation history + tool dispatch wrapper
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_config.py
    ├── test_embedder.py
    ├── test_store.py
    ├── test_indexer.py
    ├── test_tools.py
    ├── test_llm.py
    └── test_agent_loop.py
```

---

## Chunk 1: Project Setup + Data Layer

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/models.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Initialize the project**

```bash
cd ~/dev/ai-coding-agent
git init
python3 -m venv .venv
source .venv/bin/activate
```

- [ ] **Step 2: Write `requirements.txt`**

```
chromadb>=0.5.0,<1.0.0
anthropic>=0.40.0
requests>=2.31.0
pytest>=7.4.0
pytest-mock>=3.12.0
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 4: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.chroma/
config.json
.env
```

- [ ] **Step 5: Write `.env.example`**

```
ANTHROPIC_API_KEY=your_key_here
```

- [ ] **Step 6: Create `src/__init__.py` and `tests/__init__.py`**

Both files are empty.

- [ ] **Step 7: Write `tests/test_models.py`**

```python
from src.models import Chunk


def test_chunk_creation():
    chunk = Chunk(text="def foo(): pass", file="src/foo.py", start_line=1, score=0.9, chunk_type="function")
    assert chunk.text == "def foo(): pass"
    assert chunk.file == "src/foo.py"
    assert chunk.start_line == 1
    assert chunk.score == 0.9
    assert chunk.chunk_type == "function"
```

- [ ] **Step 8: Write `tests/test_config.py`**

```python
import json
from src.config import load_config, save_config


def test_load_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.CONFIG_PATH", tmp_path / "config.json")
    config = load_config()
    assert config.model == "claude-haiku-4-5-20251001"
    assert config.embedding_model == "nomic-embed-text"
    assert config.ollama_url == "http://localhost:11434"
    assert config.max_results == 5
    assert config.repo_path == ""


def test_load_config_from_file(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"repo_path": "/my/repo", "model": "claude-sonnet-4-6"}))
    monkeypatch.setattr("src.config.CONFIG_PATH", config_path)
    config = load_config()
    assert config.repo_path == "/my/repo"
    assert config.model == "claude-sonnet-4-6"


def test_save_and_reload_config(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.CONFIG_PATH", tmp_path / "config.json")
    config = load_config()
    config.repo_path = "/new/repo"
    save_config(config)
    reloaded = load_config()
    assert reloaded.repo_path == "/new/repo"
```

- [ ] **Step 9: Run tests to confirm they fail**

```bash
pip install -r requirements.txt
pytest tests/test_models.py tests/test_config.py -v
```

Expected: ImportError — `src.models` and `src.config` do not exist yet.

- [ ] **Step 10: Write `src/models.py`**

```python
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str         # raw code or text content
    file: str         # relative path from repo root
    start_line: int   # line number in source file
    score: float      # similarity score 0–1; 0.0 when not applicable (sentinel)
    chunk_type: str   # "function" | "class" | "block" | "unknown"
```

- [ ] **Step 11: Write `src/config.py`**

```python
import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


@dataclass
class Config:
    repo_path: str
    model: str
    embedding_model: str
    ollama_url: str
    chroma_path: str
    max_results: int


def load_config() -> Config:
    data = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = json.load(f)
    return Config(
        repo_path=data.get("repo_path", ""),
        model=data.get("model", "claude-haiku-4-5-20251001"),
        embedding_model=data.get("embedding_model", "nomic-embed-text"),
        ollama_url=data.get("ollama_url", "http://localhost:11434"),
        chroma_path=data.get("chroma_path", ".chroma"),
        max_results=data.get("max_results", 5),
    )


def save_config(config: Config) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "repo_path": config.repo_path,
            "model": config.model,
            "embedding_model": config.embedding_model,
            "ollama_url": config.ollama_url,
            "chroma_path": config.chroma_path,
            "max_results": config.max_results,
        }, f, indent=2)
```

- [ ] **Step 12: Run tests to confirm they pass**

```bash
pytest tests/test_models.py tests/test_config.py -v
```

Expected: all tests PASS.

- [ ] **Step 13: Commit**

```bash
git add src/ tests/ requirements.txt pyproject.toml .gitignore .env.example
git commit -m "feat: project setup — models, config, tests"
```

---

### Task 2: Embedder

**Files:**
- Create: `src/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing tests — `tests/test_embedder.py`**

```python
import pytest
import requests
from unittest.mock import patch, MagicMock
from src.embedder import OllamaEmbedder, EmbedderError


def test_embed_returns_vector():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response) as mock_post:
        embedder = OllamaEmbedder()
        result = embedder.embed("hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "hello world"},
            timeout=30,
        )


def test_embed_raises_on_connection_error():
    with patch("requests.post", side_effect=requests.ConnectionError()):
        embedder = OllamaEmbedder()
        with pytest.raises(EmbedderError, match="Ollama is not running"):
            embedder.embed("test")


def test_embed_raises_on_timeout():
    with patch("requests.post", side_effect=requests.Timeout()):
        embedder = OllamaEmbedder()
        with pytest.raises(EmbedderError, match="timed out"):
            embedder.embed("test")


def test_embed_raises_on_missing_embedding_key():
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": "model not found"}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        embedder = OllamaEmbedder()
        with pytest.raises(EmbedderError, match="Unexpected response"):
            embedder.embed("test")


def test_embed_uses_configured_model_and_url():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.5]}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response) as mock_post:
        embedder = OllamaEmbedder(model="all-minilm", base_url="http://custom:9999")
        embedder.embed("text")
        mock_post.assert_called_once_with(
            "http://custom:9999/api/embeddings",
            json={"model": "all-minilm", "prompt": "text"},
            timeout=30,
        )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_embedder.py -v
```

Expected: ImportError — `src.embedder` does not exist yet.

- [ ] **Step 3: Write `src/embedder.py`**

```python
import requests
from typing import List


class EmbedderError(Exception):
    pass


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> List[float]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if "embedding" not in data:
                raise EmbedderError("Unexpected response from Ollama")
            return data["embedding"]
        except requests.ConnectionError:
            raise EmbedderError("Ollama is not running. Start it with: ollama serve")
        except requests.Timeout:
            raise EmbedderError("Ollama timed out. Is the model loaded?")
        except requests.HTTPError as e:
            raise EmbedderError(f"Embedding failed: {e}")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_embedder.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/embedder.py tests/test_embedder.py
git commit -m "feat: Ollama embedder with error handling"
```

---

### Task 3: Vector Store

**Files:**
- Create: `src/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests — `tests/test_store.py`**

```python
import chromadb
from src.store import VectorStore
from src.models import Chunk


def make_store():
    client = chromadb.EphemeralClient()
    return VectorStore(_client=client)


def make_chunk(text, file="test.py", line=1, chunk_type="block"):
    return Chunk(text=text, file=file, start_line=line, score=0.0, chunk_type=chunk_type)


def test_add_and_count():
    store = make_store()
    store.add([make_chunk("def foo(): pass")], [[0.1] * 768])
    assert store.count() == 1


def test_search_returns_matching_chunk():
    store = make_store()
    chunk = make_chunk("def authenticate(user): pass")
    store.add([chunk], [[0.1] * 768])
    results = store.search([0.1] * 768, n_results=1)
    assert len(results) == 1
    assert results[0].text == "def authenticate(user): pass"
    assert results[0].file == "test.py"


def test_search_score_is_between_0_and_1():
    store = make_store()
    store.add([make_chunk("hello world")], [[0.5] * 768])
    results = store.search([0.5] * 768, n_results=1)
    assert 0.0 <= results[0].score <= 1.0


def test_search_empty_store_returns_empty():
    store = make_store()
    results = store.search([0.1] * 768)
    assert results == []


def test_clear_removes_all_chunks():
    store = make_store()
    store.add([make_chunk("code")], [[0.1] * 768])
    store.clear()
    assert store.count() == 0


def test_keyword_search_finds_matching_text():
    store = make_store()
    chunks = [
        make_chunk("def validate_token(token): return True", file="auth.py", line=1),
        make_chunk("def get_user(id): return None", file="user.py", line=1),
    ]
    store.add(chunks, [[0.1] * 768, [0.2] * 768])
    results = store.keyword_search("validate_token")
    assert len(results) == 1
    assert results[0].file == "auth.py"


def test_keyword_search_empty_store_returns_empty():
    store = make_store()
    results = store.keyword_search("anything")
    assert results == []


def test_add_preserves_metadata():
    store = make_store()
    chunk = make_chunk("class AuthService:", file="services/auth.py", line=42, chunk_type="class")
    store.add([chunk], [[0.3] * 768])
    results = store.search([0.3] * 768, n_results=1)
    assert results[0].file == "services/auth.py"
    assert results[0].start_line == 42
    assert results[0].chunk_type == "class"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_store.py -v
```

Expected: ImportError — `src.store` does not exist yet.

- [ ] **Step 3: Write `src/store.py`**

```python
import chromadb
from typing import List, Optional
from src.models import Chunk

COLLECTION_NAME = "codebase"


class VectorStore:
    def __init__(self, chroma_path: str = ".chroma", _client=None):
        self.path = chroma_path
        self._client = _client  # injectable for testing
        self._collection = None

    def _get_collection(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.path)
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        collection = self._get_collection()
        ids = [f"{c.file}:{c.start_line}" for c in chunks]
        collection.add(
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            ids=ids,
            metadatas=[
                {"file": c.file, "start_line": c.start_line, "chunk_type": c.chunk_type}
                for c in chunks
            ],
        )

    def search(self, query_embedding: List[float], n_results: int = 5) -> List[Chunk]:
        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # cosine distance (0–2) → similarity score (0–1)
            score = round(1.0 - (dist / 2.0), 4)
            chunks.append(
                Chunk(
                    text=doc,
                    file=meta["file"],
                    start_line=meta["start_line"],
                    score=score,
                    chunk_type=meta["chunk_type"],
                )
            )
        return chunks

    def keyword_search(self, keyword: str, n_results: int = 10) -> List[Chunk]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        results = collection.get(
            where_document={"$contains": keyword},
            include=["documents", "metadatas"],
            limit=n_results,
        )
        return [
            Chunk(
                text=doc,
                file=meta["file"],
                start_line=meta["start_line"],
                score=0.0,
                chunk_type=meta["chunk_type"],
            )
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    def clear(self) -> None:
        self._get_collection()  # ensure client is initialized
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass  # collection did not exist — that's fine
        self._collection = None

    def count(self) -> int:
        return self._get_collection().count()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_store.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/store.py tests/test_store.py
git commit -m "feat: ChromaDB vector store with cosine similarity"
```

---

### Task 4: Naive Indexer

**Files:**
- Create: `src/indexer.py`
- Create: `tests/test_indexer.py`

- [ ] **Step 1: Write the failing tests — `tests/test_indexer.py`**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_indexer.py -v
```

Expected: ImportError — `src.indexer` does not exist yet.

- [ ] **Step 3: Write `src/indexer.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_indexer.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Run all tests so far**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/indexer.py tests/test_indexer.py
git commit -m "feat: naive file scanner and chunker with indexing pipeline"
```

---

## Chunk 2: Tools + Agent + REPL

### Task 5: Tools

**Files:**
- Create: `src/tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write the failing tests — `tests/test_tools.py`**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_tools.py -v
```

Expected: ImportError — `src.tools` does not exist yet.

- [ ] **Step 3: Write `src/tools.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_tools.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools.py tests/test_tools.py
git commit -m "feat: search_codebase, trace_flow, read_file tools"
```

---

### Task 6: LLM Client

**Files:**
- Create: `src/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests — `tests/test_llm.py`**

```python
from unittest.mock import MagicMock
from src.llm import ClaudeClient


def make_end_turn_response(text: str):
    response = MagicMock()
    response.stop_reason = "end_turn"
    block = MagicMock()
    block.text = text
    response.content = [block]
    return response


def make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "id123"):
    response = MagicMock()
    response.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    response.content = [block]
    return response


def test_respond_returns_final_answer():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    client.client.messages.create.return_value = make_end_turn_response("Auth is in auth.py")

    result = client.respond(
        messages=[{"role": "user", "content": "how does auth work?"}],
        tool_handler=lambda name, inp: "",
    )
    assert result == "Auth is in auth.py"


def test_respond_calls_tool_then_returns_answer():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    client.client.messages.create.side_effect = [
        make_tool_use_response("search_codebase", {"query": "auth"}, "id1"),
        make_end_turn_response("Auth is handled in auth.py"),
    ]

    tool_calls = []
    def tool_handler(name, inp):
        tool_calls.append((name, inp))
        return "auth.py:1 - def authenticate()..."

    result = client.respond(
        messages=[{"role": "user", "content": "how does auth work?"}],
        tool_handler=tool_handler,
    )
    assert result == "Auth is handled in auth.py"
    assert len(tool_calls) == 1
    assert tool_calls[0] == ("search_codebase", {"query": "auth"})


def test_on_tool_call_callback_is_invoked():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    client.client.messages.create.side_effect = [
        make_tool_use_response("read_file", {"path": "auth.py"}, "id2"),
        make_end_turn_response("auth.py contains..."),
    ]

    callbacks = []
    client.respond(
        messages=[{"role": "user", "content": "show auth.py"}],
        tool_handler=lambda name, inp: "content",
        on_tool_call=lambda name, inp: callbacks.append(name),
    )
    assert callbacks == ["read_file"]


def test_respond_sends_tool_result_as_user_message():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    client.client.messages.create.side_effect = [
        make_tool_use_response("search_codebase", {"query": "auth"}, "id3"),
        make_end_turn_response("done"),
    ]

    client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "result text",
    )

    second_call_messages = client.client.messages.create.call_args_list[1].kwargs["messages"]
    # Last message in second call should be the tool result from user
    last_message = second_call_messages[-1]
    assert last_message["role"] == "user"
    assert last_message["content"][0]["type"] == "tool_result"
    assert last_message["content"][0]["tool_use_id"] == "id3"
    assert last_message["content"][0]["content"] == "result text"


def test_respond_returns_fallback_after_max_iterations():
    client = ClaudeClient(api_key="test")
    client.client = MagicMock()
    # Always return tool_use — never terminates naturally
    client.client.messages.create.return_value = make_tool_use_response(
        "search_codebase", {"query": "x"}, "id99"
    )

    result = client.respond(
        messages=[{"role": "user", "content": "q"}],
        tool_handler=lambda name, inp: "result",
        max_iterations=3,
    )
    assert "Maximum iterations" in result
    assert client.client.messages.create.call_count == 3
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_llm.py -v
```

Expected: ImportError — `src.llm` does not exist yet.

- [ ] **Step 3: Write `src/llm.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_llm.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat: Claude API client with ReAct tool-use loop"
```

---

### Task 7: Agent Loop

**Files:**
- Create: `src/agent_loop.py`
- Create: `tests/test_agent_loop.py`

- [ ] **Step 1: Write the failing tests — `tests/test_agent_loop.py`**

```python
import chromadb
from unittest.mock import MagicMock
from src.agent_loop import AgentLoop
from src.store import VectorStore
from src.models import Chunk


def make_agent():
    llm = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = [0.0] * 768
    store = MagicMock()
    return AgentLoop(llm=llm, embedder=embedder, store=store, repo_root="/repo")


def test_ask_returns_answer():
    agent = make_agent()
    agent.llm.respond.return_value = "Auth is in auth.py"
    result = agent.ask("How does auth work?")
    assert result == "Auth is in auth.py"


def test_ask_adds_user_and_assistant_to_history():
    agent = make_agent()
    agent.llm.respond.return_value = "Auth is in auth.py"
    agent.ask("How does auth work?")
    assert len(agent.history) == 2
    assert agent.history[0] == {"role": "user", "content": "How does auth work?"}
    assert agent.history[1] == {"role": "assistant", "content": "Auth is in auth.py"}


def test_ask_passes_full_history_to_llm():
    agent = make_agent()
    agent.llm.respond.return_value = "answer"
    agent.ask("first question")
    agent.ask("follow up")
    # Second call must include 3 messages: user1, assistant1, user2
    second_call_messages = agent.llm.respond.call_args_list[1].kwargs["messages"]
    assert len(second_call_messages) == 3


def test_clear_history_empties_history():
    agent = make_agent()
    agent.llm.respond.return_value = "answer"
    agent.ask("a question")
    agent.clear_history()
    assert agent.history == []


def test_ask_passes_on_tool_call_callback():
    agent = make_agent()
    agent.llm.respond.return_value = "answer"
    callback = MagicMock()
    agent.ask("question", on_tool_call=callback)
    call_kwargs = agent.llm.respond.call_args.kwargs
    assert call_kwargs["on_tool_call"] is callback


def test_tool_handler_dispatches_search_codebase(tmp_path):
    # Use a real store to test tool dispatch end-to-end
    client = chromadb.EphemeralClient()
    store = VectorStore(_client=client)
    store.add(
        [Chunk("def foo(): pass", "foo.py", 1, 0.0, "block")],
        [[0.1] * 768],
    )
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    llm = MagicMock()

    agent = AgentLoop(llm=llm, embedder=embedder, store=store, repo_root=str(tmp_path))
    result = agent._tool_handler("search_codebase", {"query": "foo function"})
    assert "foo.py" in result


def test_tool_handler_dispatches_read_file(tmp_path):
    f = tmp_path / "auth.py"
    f.write_text("def authenticate(): pass")
    agent = make_agent()
    agent.repo_root = str(tmp_path)
    result = agent._tool_handler("read_file", {"path": "auth.py"})
    assert "authenticate" in result


def test_tool_handler_returns_error_for_unknown_tool():
    agent = make_agent()
    result = agent._tool_handler("nonexistent_tool", {})
    assert "Unknown tool" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_agent_loop.py -v
```

Expected: ImportError — `src.agent_loop` does not exist yet.

- [ ] **Step 3: Write `src/agent_loop.py`**

```python
from typing import List, Dict, Any, Callable, Optional
from src.models import Chunk
from src.tools import search_codebase, trace_flow, read_file


def format_chunks(chunks: List[Chunk]) -> str:
    if not chunks:
        return "No relevant code found."
    parts = []
    for c in chunks:
        header = f"[{c.file}:{c.start_line}]"
        if c.score > 0:
            header += f" (score: {c.score:.2f})"
        parts.append(f"{header}\n```\n{c.text}\n```")
    return "\n\n".join(parts)


class AgentLoop:
    def __init__(self, llm, embedder, store, repo_root: str, max_history_turns: int = 10):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.repo_root = repo_root
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, Any]] = []

    def _tool_handler(self, tool_name: str, tool_input: Dict) -> str:
        if tool_name == "search_codebase":
            chunks = search_codebase(tool_input["query"], self.embedder, self.store)
            return format_chunks(chunks)
        if tool_name == "trace_flow":
            chunks = trace_flow(tool_input["entry_point"], self.embedder, self.store)
            return format_chunks(chunks)
        if tool_name == "read_file":
            try:
                return read_file(tool_input["path"], self.repo_root)
            except (ValueError, FileNotFoundError) as e:
                return f"Error: {e}"
        return f"Unknown tool: {tool_name}"

    def _truncate_history(self) -> None:
        """Keep only the last max_history_turns exchanges (2 messages per turn)."""
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def ask(self, question: str, on_tool_call: Optional[Callable] = None) -> str:
        self.history.append({"role": "user", "content": question})
        answer = self.llm.respond(
            messages=self.history,
            tool_handler=self._tool_handler,
            on_tool_call=on_tool_call,
        )
        self.history.append({"role": "assistant", "content": answer})
        self._truncate_history()
        return answer

    def clear_history(self) -> None:
        self.history = []
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_agent_loop.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agent_loop.py tests/test_agent_loop.py
git commit -m "feat: agent loop with conversation history, tool dispatch, context management"
```

---

### Task 8: REPL Entry Point

**Files:**
- Create: `agent.py`

No automated tests for the REPL — verify manually.

- [ ] **Step 1: Write `agent.py`**

```python
import os
import sys
from pathlib import Path

from src.config import load_config, save_config
from src.embedder import OllamaEmbedder, EmbedderError
from src.store import VectorStore
from src.indexer import index_repo
from src.llm import ClaudeClient
from src.agent_loop import AgentLoop

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


def build_components(config):
    embedder = OllamaEmbedder(model=config.embedding_model, base_url=config.ollama_url)
    store = VectorStore(chroma_path=config.chroma_path)
    llm = ClaudeClient(model=config.model)
    agent = AgentLoop(llm=llm, embedder=embedder, store=store, repo_root=config.repo_path)
    return embedder, store, llm, agent


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

    while True:
        try:
            user_input = input("> ").strip()
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
```

- [ ] **Step 2: Manual smoke test**

```bash
export ANTHROPIC_API_KEY=your_key
python agent.py
```

Expected: Banner prints, prompt appears. Type `exit` to quit.

- [ ] **Step 3: Test indexing**

```
> index --repo ~/dev/jarvis
```

Expected: "Indexing... Done. Indexed N chunks."

- [ ] **Step 4: Test a question**

```
> ask what is this project?
```

Expected: Thinking... → tool calls visible → answer printed.

- [ ] **Step 5: Commit**

```bash
git add agent.py
git commit -m "feat: interactive REPL entry point"
```

---

## Chunk 3: Semantic Chunking + Verification

### Task 9: Verify trace command end-to-end

`trace_flow` is already implemented in `src/tools.py` and wired into the agent loop and REPL. This task verifies end-to-end behavior with the real system.

- [ ] **Step 1: Test trace via REPL**

```
> trace authenticate
```

Expected: Thinking... → "tracing: authenticate" visible → answer listing files/lines where `authenticate` appears.

- [ ] **Step 2: Test a follow-up referencing prior context**

```
> who calls it?
```

Expected: Agent uses conversation history to understand "it" refers to `authenticate`. Returns callers without needing a repeat prompt.

- [ ] **Step 3: Commit any fixes needed**

```bash
git add -p
git commit -m "fix: trace flow wiring verified"
```

---

### Task 10: Semantic Chunking

Replace naive line-splitting with AST-based chunking for Python files. Other file types fall back to naive.

**Files:**
- Modify: `src/indexer.py`
- Modify: `tests/test_indexer.py`

- [ ] **Step 1: Write failing tests for semantic chunking**

Add to `tests/test_indexer.py`:

```python
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

    from src.indexer import index_repo
    client = chromadb.EphemeralClient()
    store = VectorStore(_client=client)
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768

    count = index_repo(str(tmp_path), embedder, store, use_semantic=True)
    assert count == 2  # two top-level functions = two chunks
    assert embedder.embed.call_count == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_indexer.py::test_semantic_chunking_extracts_top_level_definitions -v
```

Expected: FAIL — `chunk_file_semantic` does not exist yet.

- [ ] **Step 3: Add semantic chunking to `src/indexer.py`**

Add import at top of file:

```python
import ast
```

Add these functions after `chunk_file_naive`:

```python
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
```

Also fix the self-referential import in `index_repo`:

```python
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
```

- [ ] **Step 4: Run all indexer tests**

```bash
pytest tests/test_indexer.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Compare quality — re-index with semantic chunking**

```
> index --repo ~/dev/jarvis
```

Ask a question that previously had mediocre results with naive chunking. Compare answer quality.

- [ ] **Step 6: Commit**

```bash
git add src/indexer.py tests/test_indexer.py
git commit -m "feat: semantic chunking for Python files via AST (top-level only)"
```

---

### Task 11: Context Management

Context management is already implemented in `AgentLoop._truncate_history` (added in Task 7). This task adds tests and verifies the behavior.

**Files:**
- Modify: `tests/test_agent_loop.py`

- [ ] **Step 1: Write failing tests for context truncation**

Add to `tests/test_agent_loop.py`:

```python
def test_history_is_capped_at_max_history_turns():
    agent = AgentLoop(
        llm=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
        repo_root="/repo",
        max_history_turns=3,
    )
    agent.llm.respond.return_value = "answer"

    for i in range(10):
        agent.ask(f"question {i}")

    # Exactly max_history_turns * 2 messages (3 exchanges = 6 messages)
    assert len(agent.history) == 3 * 2


def test_history_not_truncated_when_under_limit():
    agent = AgentLoop(
        llm=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
        repo_root="/repo",
        max_history_turns=10,
    )
    agent.llm.respond.return_value = "answer"

    agent.ask("q1")
    agent.ask("q2")

    assert len(agent.history) == 4  # 2 questions + 2 answers, under limit
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_agent_loop.py -v
```

Expected: all tests PASS (truncation was implemented in Task 7).

- [ ] **Step 3: Run full test suite**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_agent_loop.py
git commit -m "test: verify context window truncation behavior"
```

---

## What You've Built

By the end of this plan, you have:

| Layer | What it does | AI concept learned |
|---|---|---|
| `src/embedder.py` | Converts text to 768-dim vectors via Ollama | How embeddings represent meaning |
| `src/store.py` | Stores and searches vectors via ChromaDB | Vector similarity search |
| `src/indexer.py` | Chunks code files, builds the index | Chunking strategies and their impact on retrieval |
| `src/tools.py` | search, trace, read — retrieval tools | RAG retrieval pipeline |
| `src/llm.py` | Claude API with tool use loop | ReAct reasoning pattern |
| `src/agent_loop.py` | Conversation history + tool dispatch + truncation | Context window management |
| `agent.py` | Interactive REPL | How everything connects |

**Next experiments after v1:**
1. Compare `claude-haiku-4-5` vs `claude-sonnet-4-6` on the same question
2. Change `CHUNK_SIZE` in naive chunker and observe retrieval quality difference
3. Add a second embedding model and compare search results
4. Point the agent at a public repo you don't know well

**Deferred from this plan (future work):**
- More advanced context truncation strategies: summarizing old turns, dropping tool call details while keeping answers
- Team-level knowledge sharing and multi-repo support
