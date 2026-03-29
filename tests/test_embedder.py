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
