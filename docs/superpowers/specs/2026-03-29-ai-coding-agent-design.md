# AI Coding Agent — Design Spec

**Date:** 2026-03-29
**Status:** Approved

---

## Overview

A CLI-based AI coding agent that understands, analyzes, and answers questions about software codebases. Built from scratch in Python as a learning platform for core AI concepts: RAG, embeddings, vector search, agent reasoning loops, and context window management.

The system is repo-agnostic — it can be pointed at any codebase. Initial testing target is the Jarvis project.

---

## Goals

**Primary:** Learn AI fundamentals hands-on by building a working system.

**Secondary:** Produce a useful developer tool that can eventually integrate into Jarvis as a coding assistant module.

**Learning outcomes by completion:**
- How embeddings represent meaning numerically
- How chunking strategies affect retrieval quality
- How vector similarity search works
- How a ReAct agent loop reasons and uses tools
- How conversation history is managed in a context window

---

## Scope

### Included (v1)
- Codebase Q&A via RAG
- Agent reasoning loop with tool use
- Code flow tracing
- Conversation memory within a session

### Excluded (v1)
- Web UI
- Automatic code execution or direct file modification
- Fine-tuning or model training
- Multi-repo support
- Framework abstractions (LangChain, LlamaIndex)

---

## Architecture

```
Interactive CLI (REPL)
 └── Agent (LLM + ReAct loop)
      ├── search_codebase(query) → ChromaDB
      ├── trace_flow(entry_point) → ChromaDB
      └── read_file(path) → filesystem

Indexing pipeline (runs once)
 └── File scanner → Chunker → Embedder (Ollama) → ChromaDB
```

---

## Components

### 1. Indexing Pipeline

Runs on demand via the `index` command. Processes a repo and populates the vector store.

**Pipeline:**
1. Scan repo files (filter by extension: `.py`, `.js`, `.ts`, `.md`, etc.)
2. Chunk files into smaller pieces
3. Embed each chunk via `nomic-embed-text` (Ollama)
4. Store chunk text + embedding + metadata in ChromaDB

**Chunking strategy — two phases:**
- **Phase 1 (naive):** Split every N lines regardless of code structure. Fast to implement. Produces mediocre retrieval — intentionally, to demonstrate the problem.
- **Phase 2 (semantic):** Split at natural code boundaries — functions, classes, modules. Each chunk is a complete, meaningful unit. Measurably better retrieval quality.

**Chunk metadata stored:**
```python
{
  "text": "...",           # raw code/text
  "file": "src/auth.py",  # source file path
  "start_line": 42,       # position in file
  "type": "function"      # once semantic chunking is active
}
```

---

### 2. Vector Store

**Technology:** ChromaDB — persists to disk, easy to inspect, no separate server needed.

Stores embeddings and metadata. Exposes similarity search: given a query embedding, return the N most similar chunks.

**Re-index behavior:** Running `index` on an already-indexed repo replaces the existing collection entirely (delete + rebuild). No append behavior. This keeps retrieval results consistent and avoids duplicate chunks.

---

### 3. Embedding Model

**Technology:** `nomic-embed-text` via Ollama (local, free).

Converts text to a 768-dimensional vector. Used for both indexing (chunks) and retrieval (queries). Chosen for strong retrieval performance at small size, runs well on Ollama, open training process.

---

### 4. Agent — ReAct Loop

**Technology:** OpenAI (GPT-4o) or Anthropic (Claude) API.

The agent receives the user's question plus conversation history. It reasons step by step, deciding whether to call a tool or generate a final answer.

**Loop:**
```
User question + conversation history
 └── LLM: do I need more info?
      ├── YES → choose tool, call it, add result to context, loop
      └── NO → generate final answer
```

**Tools:**

| Tool | Signature | Purpose |
|------|-----------|---------|
| `search_codebase` | `(query: str) → list[Chunk]` | Semantic search via vector similarity |
| `trace_flow` | `(entry_point: str) → list[Chunk]` | Multi-hop retrieval of related code |
| `read_file` | `(path: str) → str` | Return raw file content |

**`trace_flow` mechanism:** Two-step retrieval. Step 1 — perform a keyword search across all stored chunk text for occurrences of `entry_point` as a string (e.g. function name, class name). This finds direct references: callers, import statements, definitions. Step 2 — embed the top result and run a vector similarity search to surface semantically related chunks that may not reference the name literally. The combined results are re-ranked by relevance score and returned. This teaches multi-hop retrieval: a single question resolved through two complementary search strategies.

**`Chunk` data structure** — the shared contract between retrieval and agent layers:

```python
@dataclass
class Chunk:
    text: str          # raw code or text content
    file: str          # relative path from repo root
    start_line: int    # line number in source file
    score: float       # similarity score from retrieval (0–1); 0.0 when not applicable
    chunk_type: str    # "function" | "class" | "block" | "unknown"
```

`chunk_type` is `"block"` during Phase 1 (naive chunking — chunks have no semantic type yet) and `"unknown"` for any chunk the semantic parser cannot classify. `score` is `0.0` as a sentinel when no similarity score applies (e.g. during Phase 1 before similarity search is meaningful). `read_file` returns `str` directly — it does not produce `Chunk` objects.

This is what `search_codebase` and `trace_flow` return. The agent formats these into context before sending to the LLM.

---

### 5. Conversation Memory

The full conversation history (user messages + agent responses + tool calls) is maintained in memory for the session and passed to the LLM on every turn.

As history grows, the context window fills. This is intentional — you will observe and learn to manage it. Context management is addressed in Step 7 (Tune agent). Until then, naive growing history is acceptable. Truncation strategies to implement in Step 7:
- Summarize old turns
- Drop tool call details, keep answers
- Limit history to last N turns

---

### 6. CLI — Interactive REPL

**Start:**
```bash
python agent.py
```

**Session:**
```
AI Coding Agent
Repo: ./jarvis (4,231 chunks indexed)
Type 'help' for commands, 'exit' to quit.

> ask "How does authentication work?"
Thinking...
 → searching codebase for "authentication"
 → tracing flow from validate_token()

Answer: Token validation is handled in src/auth/middleware.py...

> who calls it?
Thinking...
 → searching codebase for "validate_token callers"

Answer: validate_token() is called by...

> exit
```

**Commands:**
- `ask <question>` — Q&A via RAG + agent (no quotes needed)
- `trace <entry point>` — explicit flow trace
- `index [--repo <path>]` — (re)index a repo. `--repo` defaults to the path set in `config.json` at the project root. If no config exists and no `--repo` is provided, the command fails with a clear error: "No repo configured. Run: index --repo <path>"
- `help` — list commands
- `exit` — quit

**Error handling policy:** Fail fast with a clear, human-readable error message. Examples: Ollama not running → "Ollama is not running. Start it with: ollama serve"; file path outside repo → "Path must be within the indexed repo root"; repo not indexed → "No index found. Run 'index' first."

The agent's tool calls are visible during "Thinking..." — intentionally, for learning purposes.

---

## Build Order (Thin Slice First)

1. **Indexing pipeline** — naive chunking, embed, store in ChromaDB
2. **Basic retrieval** — `search_codebase` tool, similarity search working
3. **Thin agent** — LLM with one tool (`search_codebase`), basic prompt
4. **REPL** — interactive CLI, conversation history
5. **Flow tracing** — `trace_flow` tool, multi-hop retrieval
6. **Semantic chunking** — improve chunking, compare retrieval quality
7. **Tune agent** — improve prompt, reasoning quality, context management

Each step produces a working (if imperfect) system. Improvements are measurable.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| Embedding model | `nomic-embed-text` via Ollama |
| LLM | Anthropic Claude (API) |
| Vector store | ChromaDB |
| CLI | Python stdlib (`input()` loop) |
| No frameworks | LangChain/LlamaIndex explicitly excluded for learning |

**Infrastructure prerequisites:** Ollama must be running locally before indexing or querying. Install via `brew install ollama`, then `ollama pull nomic-embed-text`. An Anthropic API key must be set in the environment.

**Future LLM swap:** The agent layer will use an abstraction so the LLM provider can be swapped to OpenAI or a local Ollama model in a single config change. v1 starts with `claude-haiku-4-5` for fast, cheap iteration. Switch to `claude-sonnet-4-6` when evaluating reasoning quality or when multi-hop agent chains produce poor results. Model is a single config value.

---

## Future (Post v1)

- Integrate into Jarvis as a coding assistant module
- Multi-repo support
- Layer a framework (LangChain/LlamaIndex) on top to compare with from-scratch implementation
- Semi-autonomous edit proposals
