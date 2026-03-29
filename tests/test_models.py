from src.models import Chunk


def test_chunk_creation():
    chunk = Chunk(text="def foo(): pass", file="src/foo.py", start_line=1, score=0.9, chunk_type="function")
    assert chunk.text == "def foo(): pass"
    assert chunk.file == "src/foo.py"
    assert chunk.start_line == 1
    assert chunk.score == 0.9
    assert chunk.chunk_type == "function"
