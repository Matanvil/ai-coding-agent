import requests
from typing import List


class EmbedderError(Exception):
    pass


MAX_EMBED_CHARS = 6000  # nomic-embed-text context limit is 8192 tokens; ~6000 chars is safe


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> List[float]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text[:MAX_EMBED_CHARS]},
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
