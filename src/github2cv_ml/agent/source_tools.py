"""Read-only, bounded tools for inspecting a frozen GitHub commit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from github2cv_ml.domain.repository import RepoSnapshot, RepoTreeEntryKind

_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
    ".next", ".tox", "vendor", "coverage", ".mypy_cache", ".pytest_cache",
}
_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz",
    ".woff", ".woff2", ".ttf", ".ico", ".mp4", ".mp3", ".onnx",
    ".gguf", ".pt", ".pth", ".bin", ".parquet", ".csv", ".lock",
    ".pem", ".key", ".p12", ".pfx", ".crt", ".db", ".sqlite",
}
_SKIP_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
_PRIORITY = {".py": 0, ".go": 1, ".ts": 2, ".tsx": 3, ".js": 4, ".rs": 5}


def allowed_source_path(path: str) -> bool:
    """Reject secrets, binary/artifact paths and unsafe path traversal."""
    if not path or "\x00" in path or "\\" in path:
        return False
    parts = PurePosixPath(path).parts
    if path.startswith("/") or any(p in {".", ".."} for p in parts):
        return False
    if any(p.lower() in _SKIP_DIRS for p in parts):
        return False
    name = parts[-1].lower()
    if name in _SKIP_NAMES or name.startswith(".env.") or "secret" in name:
        return False
    if "credential" in name or name.endswith((".token", ".bak")):
        return False
    return PurePosixPath(path).suffix.lower() not in _SKIP_SUFFIXES


@dataclass
class RepositoryBrowser:
    """GitHub API-backed tool execution at one immutable commit SHA.

    Code is only fetched for paths present in the frozen git tree. The browser
    never executes repository code and does not expose write/shell tools.
    """

    client: Any
    snapshot: RepoSnapshot
    commit_sha: str
    max_unique_files: int = 25
    max_file_bytes: int = 80_000
    max_read_lines: int = 120
    max_search_files: int = 12
    _cache: dict[str, list[str]] = field(default_factory=dict, init=False)
    _attempted: set[str] = field(default_factory=set, init=False)
    _files: dict[str, int | None] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if len(self.commit_sha) != 40 or any(c not in "0123456789abcdef" for c in self.commit_sha):
            raise ValueError("A full lowercase Git commit SHA is required")
        self._files = {
            e.path: e.size
            for e in self.snapshot.file_tree
            if e.kind == RepoTreeEntryKind.FILE and allowed_source_path(e.path)
        }

    def _load(self, path: str) -> list[str]:
        if path not in self._files:
            raise ValueError("File is absent from the frozen safe file tree")
        if path in self._cache:
            return self._cache[path]
        size = self._files[path]
        if size is None or size > self.max_file_bytes:
            raise ValueError("File has unknown or excessive size")
        if len(self._attempted) >= self.max_unique_files:
            raise ValueError("GitHub file fetch budget exhausted")
        self._attempted.add(path)
        repo = self.snapshot.repository
        content = self.client.get_file(repo.owner, repo.name, path, self.commit_sha)
        if content is None:
            raise ValueError("File cannot be fetched at the pinned commit")
        if len(content.encode("utf-8")) > self.max_file_bytes or "\x00" in content:
            raise ValueError("File is too large or binary")
        self._cache[path] = content.splitlines()
        return self._cache[path]

    def list_files(self, path: str = "") -> dict[str, Any]:
        path = path.strip("/")
        if path and (not allowed_source_path(path) or ".." in path.split("/")):
            raise ValueError("Invalid directory")
        paths = sorted(
            (p for p in self._files if not path or p.startswith(path + "/")),
            key=self._priority,
        )
        return {"files": paths[:80], "truncated": len(paths) > 80, "total": len(paths)}

    @staticmethod
    def _priority(path: str) -> tuple[int, int, str]:
        suffix = PurePosixPath(path).suffix.lower()
        return (_PRIORITY.get(suffix, 8), path.count("/"), path)

    def search_code(self, query: str) -> dict[str, Any]:
        """Literal, case-insensitive search; no regex execution or shell access."""
        if not isinstance(query, str) or not 2 <= len(query) <= 64 or "\n" in query:
            raise ValueError("Search query must be 2-64 characters on one line")
        matches: list[dict[str, Any]] = []
        scanned = 0
        for path in sorted(self._files, key=self._priority):
            if scanned >= self.max_search_files or len(matches) >= 12:
                break
            try:
                lines = self._load(path)
            except ValueError:
                continue
            scanned += 1
            for n, line in enumerate(lines, 1):
                if query.casefold() in line.casefold():
                    matches.append({"path": path, "line": n})
                    if len(matches) >= 12:
                        break
        return {"query": query, "matches": matches, "scanned_files": scanned,
                "scan_limit": self.max_search_files, "partial_search": True}

    def read_file(self, path: str, start_line: int = 1, end_line: int = 120) -> dict[str, Any]:
        if type(start_line) is not int or type(end_line) is not int:
            raise ValueError("Line numbers must be integers")
        if start_line < 1 or end_line < start_line or end_line - start_line + 1 > self.max_read_lines:
            raise ValueError(f"Invalid range or more than {self.max_read_lines} lines")
        lines = self._load(path)
        if start_line > len(lines):
            raise ValueError("Start line exceeds file length")
        end = min(end_line, len(lines))
        return {
            "path": path, "start_line": start_line, "end_line": end,
            "total_lines": len(lines),
            "numbered_text": "\n".join(
                f"{n}: {lines[n - 1]}" for n in range(start_line, end + 1)
            ),
        }

    def project_info(self) -> dict[str, Any]:
        signals = self.snapshot.signals
        return {
            "repo": self.snapshot.repository.url,
            "commit_sha": self.commit_sha,
            "description": (self.snapshot.description or "")[:500],
            "readme_excerpt": (self.snapshot.readme or "")[:1700],
            "languages": list(self.snapshot.languages)[:12],
            "dependencies": [x.name for x in signals.dependencies[:30]],
            "tooling": [x.kind.value for x in signals.tooling[:12]],
            "safe_files": len(self._files),
            "tree_truncated": self.snapshot.file_tree_truncated,
        }


EXPLORER_TOOLS = [
    {
        "type": "function", "function": {
            "name": "list_files", "description": "List safe repository source files or files in a directory.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Directory prefix, or empty for all"}
            }},
        }
    },
    {
        "type": "function", "function": {
            "name": "search_code", "description": "Literal search in a bounded subset of source files. Not regex; results are incomplete.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Case-insensitive literal token or identifier"}
            }, "required": ["query"]},
        }
    },
    {
        "type": "function", "function": {
            "name": "read_file", "description": "Read up to 120 numbered lines of one safe source file at the frozen commit; findings will be extracted automatically.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1}
            }, "required": ["path"]},
        }
    },
    {
        "type": "function", "function": {
            "name": "get_project_info", "description": "Show README excerpt, languages, dependencies and project metadata.",
            "parameters": {"type": "object", "properties": {}},
        }
    },
    {
        "type": "function", "function": {
            "name": "finish_analysis", "description": "Finish investigation and write the final report using gathered findings.",
            "parameters": {"type": "object", "properties": {}},
        }
    },
]
