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
