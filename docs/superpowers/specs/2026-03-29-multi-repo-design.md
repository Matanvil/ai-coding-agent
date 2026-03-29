# Multi-Repo Support — Design Spec

**Date:** 2026-03-29
**Status:** Approved

---

## Overview

Allow the agent to index and query multiple repos, switching between them without re-indexing. Each repo gets its own ChromaDB collection. The CLI tracks indexed repos with their paths and last-index timestamps in `config.json`.

---

## Architecture

```
config.json
  active_repo: "jarvis"
  repos:
    jarvis:    { path: "/Users/.../jarvis",    indexed_at: "2026-03-29 22:30" }
    myproject: { path: "/Users/.../myproject", indexed_at: "2026-03-28 14:15" }

ChromaDB (.chroma/)
  collection: "jarvis"     → 882 chunks
  collection: "myproject"  → 1,200 chunks
```

Switching repos = swapping which ChromaDB collection is active. No re-indexing required.

---

## Component Changes

### 1. `src/store.py`

**`VectorStore.__init__`** gains a `collection_name` parameter (default `"codebase"` for backward compatibility):

```python
def __init__(self, chroma_path=".chroma", collection_name="codebase", _client=None):
```

The hardcoded `COLLECTION_NAME = "codebase"` constant is removed. The instance uses `self.collection_name` everywhere instead.

**New method: `list_collections() -> list[dict]`**

Returns all collections in the ChromaDB store with their names and chunk counts:

```python
def list_collections(self) -> list[dict]:
    """Return [{name: str, count: int}, ...] for all collections."""
```

Uses `self._client.list_collections()` (ChromaDB API). Returns empty list if no collections exist.

All other methods (`add`, `search`, `keyword_search`, `clear`, `count`) are unchanged — they operate on whichever collection was set at construction time.

**Collection name derivation** (in `agent.py`, not `store.py`): `Path(repo_path).name` — the last component of the path. `/Users/matan/dev/jarvis` → `"jarvis"`.

---

### 2. `src/config.py`

**`Config` dataclass** — replace `repo_path: str` with:

```python
active_repo: str        # name of currently active repo, "" if none
repos: dict             # { name: { "path": str, "indexed_at": str } }
```

**`load_config`** defaults: `active_repo=""`, `repos={}`.

**`save_config`** writes the new structure. `repo_path` is fully removed from both read and write paths.

---

### 3. `agent.py`

**CLI argument parsing** — before the REPL starts, check `sys.argv` for a repo name:

```
python agent.py jarvis           → launch into "jarvis"
python agent.py --repo jarvis    → same
python agent.py                  → show repo list
```

If a CLI arg is given and the name isn't in `config.repos`, exit with:
`"Unknown repo: jarvis. Run: index --repo <path>"`

**Startup display:**

With active repo:
```
AI Coding Agent — type 'help' for commands, 'exit' to quit.
Repo: jarvis (882 chunks)
```

Without active repo (or no repos indexed):
```
AI Coding Agent — type 'help' for commands, 'exit' to quit.

Indexed repos:
  * jarvis      (882 chunks, indexed 2026-03-29 22:30)
    myproject   (1,200 chunks, indexed 2026-03-28 14:15)

No active repo. Type 'use <repo>' to select one.
```

`*` marks the repo matching `config.active_repo`. If `config.repos` is empty:
```
No repos indexed yet. Run: index --repo <path>
```

**New `use` command:**
```
> use jarvis
Switched to jarvis (882 chunks).
```
- Looks up `config.repos["jarvis"]` — errors if not found or not indexed
- Swaps active `VectorStore` (new instance with `collection_name="jarvis"`) and `AgentLoop`
- Updates `config.active_repo = "jarvis"`, saves config
- Clears conversation history (switching repos = new context)

Error cases:
- `use unknown` → `"Unknown repo: unknown. Indexed repos: jarvis, myproject"`
- `use notindexed` → `"Repo 'notindexed' is registered but not indexed. Run: index --repo <path>"`

**Updated `index` command:**

`index --repo ~/dev/myproject`:
1. Derives name: `Path(repo_path).name` → `"myproject"`
2. Indexes into collection `"myproject"` (replaces existing if present)
3. Adds/updates `config.repos["myproject"] = {"path": "/full/path", "indexed_at": "2026-03-29 22:45"}`
4. Sets `config.active_repo = "myproject"`, saves config
5. Swaps active store and agent to `myproject`

Timestamp format: `datetime.now().strftime("%Y-%m-%d %H:%M")`

**New `repos` command:**
```
> repos
  * jarvis      (882 chunks, indexed 2026-03-29 22:30)
    myproject   (1,200 chunks, indexed 2026-03-28 14:15)
```
Reads chunk counts live from ChromaDB (`VectorStore.count()` per repo). Reads timestamps from `config.repos`. `*` marks active repo.

If no repos: `"No repos indexed yet. Run: index --repo <path>"`

**Updated `help` text:**
```
Commands:
  ask <question>              Ask a question about the codebase
  trace <symbol>              Trace where a function/class is defined and used
  index [--repo <path>]       Index or re-index a repository
  use <repo>                  Switch to an indexed repo
  repos                       List all indexed repos
  clear                       Clear conversation history
  help                        Show this help
  exit                        Quit
```

---

## Error Handling

| Situation | Message |
|---|---|
| CLI arg not in config.repos | `"Unknown repo: X. Run: index --repo <path>"` |
| `use` with unknown name | `"Unknown repo: X. Indexed repos: A, B"` |
| `use` with registered but unindexed repo | `"Repo 'X' is registered but not indexed. Run: index --repo <path>"` |
| `index` with no --repo and no active repo | `"No repo configured. Run: index --repo <path>"` |
| `ask`/`trace` with no active repo | `"No active repo. Type 'use <repo>' to select one."` |

---

## Testing

- `test_config.py` — update existing tests for new schema; add tests for repos dict load/save
- `test_store.py` — add test that `VectorStore(collection_name="foo")` uses collection `"foo"` independently from `"bar"`; add test for `list_collections()`
- `agent.py` — no automated tests; manual verification of all new commands

---

## Migration

Existing `config.json` files with `repo_path` will have that key silently ignored on load (it's not in the new schema). The user will need to re-run `index --repo <path>` once to register their repo under the new structure. The existing ChromaDB data in `.chroma/` remains valid — it just won't be addressable until re-indexed under the named collection scheme.
