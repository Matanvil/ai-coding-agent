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


def test_collection_name_isolates_data():
    client = chromadb.EphemeralClient()
    store_a = VectorStore(collection_name="repo_a", _client=client)
    store_b = VectorStore(collection_name="repo_b", _client=client)
    store_a.add([make_chunk("def foo(): pass")], [[0.1] * 768])
    assert store_a.count() == 1
    assert store_b.count() == 0  # separate collection — not affected


def test_list_collections_returns_all_collections():
    client = chromadb.EphemeralClient()
    store_a = VectorStore(collection_name="repo_a", _client=client)
    store_b = VectorStore(collection_name="repo_b", _client=client)
    store_a.add([make_chunk("code a")], [[0.1] * 768])
    store_b.add([make_chunk("code b")], [[0.2] * 768])
    collections = store_a.list_collections()
    names = [c["name"] for c in collections]
    assert "repo_a" in names
    assert "repo_b" in names
    repo_a_info = next(c for c in collections if c["name"] == "repo_a")
    assert repo_a_info["count"] == 1


def test_list_collections_returns_empty_list():
    client = chromadb.EphemeralClient()
    # Delete all existing collections to ensure a clean slate
    for col in client.list_collections():
        name = col.name if hasattr(col, "name") else str(col)
        try:
            client.delete_collection(name)
        except Exception:
            pass
    store = VectorStore(collection_name="unused", _client=client)
    # No data added, no _get_collection() called — no collections exist
    collections = store.list_collections()
    assert collections == []
