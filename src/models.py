from dataclasses import dataclass


@dataclass
class Chunk:
    text: str         # raw code or text content
    file: str         # relative path from repo root
    start_line: int   # line number in source file
    score: float      # similarity score 0–1; 0.0 when not applicable (sentinel)
    chunk_type: str   # "function" | "class" | "block" | "unknown"
