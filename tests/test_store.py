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
