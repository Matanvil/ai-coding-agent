import json
from src.config import load_config, save_config


def test_load_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.CONFIG_PATH", tmp_path / "config.json")
    config = load_config()
    assert config.model == "claude-haiku-4-5-20251001"
    assert config.embedding_model == "nomic-embed-text"
    assert config.ollama_url == "http://localhost:11434"
    assert config.max_results == 5
    assert config.active_repo == ""
    assert config.repos == {}


def test_load_config_from_file(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "active_repo": "jarvis",
        "repos": {"jarvis": {"path": "/dev/jarvis", "indexed_at": "2026-03-29 22:30"}},
        "model": "claude-sonnet-4-6",
    }))
    monkeypatch.setattr("src.config.CONFIG_PATH", config_path)
    config = load_config()
    assert config.active_repo == "jarvis"
    assert config.repos["jarvis"]["path"] == "/dev/jarvis"
    assert config.repos["jarvis"]["indexed_at"] == "2026-03-29 22:30"
    assert config.model == "claude-sonnet-4-6"


def test_save_and_reload_config(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.CONFIG_PATH", tmp_path / "config.json")
    config = load_config()
    config.active_repo = "myproject"
    config.repos["myproject"] = {"path": "/dev/myproject", "indexed_at": "2026-03-29 10:00"}
    save_config(config)
    reloaded = load_config()
    assert reloaded.active_repo == "myproject"
    assert reloaded.repos["myproject"]["path"] == "/dev/myproject"
    assert reloaded.repos["myproject"]["indexed_at"] == "2026-03-29 10:00"


def test_old_repo_path_key_is_ignored(tmp_path, monkeypatch):
    # Migration: old config.json files with repo_path should load without error
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"repo_path": "/old/path", "model": "claude-haiku-4-5-20251001"}))
    monkeypatch.setattr("src.config.CONFIG_PATH", config_path)
    config = load_config()
    assert config.active_repo == ""
    assert config.repos == {}
