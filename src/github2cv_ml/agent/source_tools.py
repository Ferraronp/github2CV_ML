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
# Source-code only: configuration files and unknown formats are not exposed to the model.
# This cannot guarantee that a source file contains no hard-coded credentials.
_SOURCE_SUFFIXES = {
    ".py", ".pyi", ".go", ".js", ".jsx", ".ts", ".tsx", ".rs",
    ".java", ".kt", ".kts", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs",
    ".rb", ".php",
}
_CREDENTIAL_NAMES = {
    ".env", ".env.local", ".env.production", ".npmrc", ".pypirc",
    ".netrc", "pip.conf", "id_rsa", "id_ed25519",
}
_PRIORITY = {".py": 0, ".go": 1, ".ts": 2, ".tsx": 3, ".js": 4, ".rs": 5}


def _allowed_path_parts(path: str) -> bool:
    """Validate an untrusted repository-relative path or directory prefix."""
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        return False
    if path.startswith("/") or "//" in path:
        return False
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    if any(part.lower() in _SKIP_DIRS for part in parts):
        return False
    for part in parts:
        lower = part.lower()
        if lower in _CREDENTIAL_NAMES or lower.startswith(".env."):
            return False
        if "secret" in lower or "credential" in lower or "private" in lower:
            return False
        if lower.endswith((".token", ".pem", ".key", ".p12", ".pfx")):
            return False
    return True


def allowed_source_path(path: str) -> bool:
    """Allow known source-code extensions only; reject credentials/configs."""
    if not _allowed_path_parts(path):
        return False
    return PurePosixPath(path).suffix.lower() in _SOURCE_SUFFIXES


@dataclass
class RepositoryBrowser:
    """GitHub API-backed tool execution at one immutable commit SHA.

    Code is only fetched for paths present in the frozen git tree. The browser
    never executes repository code and does not expose write/shell tools.
    """

    client: Any
    snapshot: RepoSnapshot
    commit_sha: str
    max_unique_files: int = 40
    max_search_unique_files: int = 24
    max_file_bytes: int = 80_000
    max_read_lines: int = 120
    max_excerpt_chars: int = 12_000
    max_search_files: int = 12
    _cache: dict[str, list[str]] = field(default_factory=dict, init=False)
    _attempted: set[str] = field(default_factory=set, init=False)
    _search_attempted: set[str] = field(default_factory=set, init=False)
    _files: dict[str, int | None] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if len(self.commit_sha) != 40 or any(c not in "0123456789abcdef" for c in self.commit_sha):
            raise ValueError("A full lowercase Git commit SHA is required")
        self._files = {
            e.path: e.size
            for e in self.snapshot.file_tree
            if e.kind == RepoTreeEntryKind.FILE and allowed_source_path(e.path)
        }

    def _load(self, path: str, *, search_fetch: bool = False) -> list[str]:
        if not isinstance(path, str):
            raise ValueError("File path must be a string")
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
        # Charge a search slot only when _load actually reaches the GitHub API.
        if search_fetch:
            self._search_attempted.add(path)
        repo = self.snapshot.repository
        content = self.client.get_file(repo.owner, repo.name, path, self.commit_sha)
        if content is None:
            raise ValueError("File cannot be fetched at the pinned commit")
        if len(content.encode("utf-8")) > self.max_file_bytes or "\x00" in content:
            raise ValueError("File is too large or binary")
        self._cache[path] = content.splitlines()
        return self._cache[path]

    def list_files(
        self, path: str = "", offset: int = 0, limit: int = 40
    ) -> dict[str, Any]:
        """Paginated safe file listing; offset is an index in the filtered file list."""
        if not isinstance(path, str):
            raise ValueError("Directory path must be a string")
        path = path.strip("/")
        if path and not _allowed_path_parts(path):
            raise ValueError("Invalid directory")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 80:
            raise ValueError("limit must be an integer between 1 and 80")
        paths = sorted(
            (p for p in self._files if not path or p.startswith(path + "/")),
            key=self._priority,
        )
        page = paths[offset:offset + limit]
        next_offset = offset + len(page) if offset + len(page) < len(paths) else None
        return {
            "files": page, "total": len(paths), "offset": offset,
            "next_offset": next_offset, "truncated": next_offset is not None,
        }

    @staticmethod
    def _priority(path: str) -> tuple[int, int, int, str]:
        suffix = PurePosixPath(path).suffix.lower()
        # Search implementation before similarly named evaluation/utility files.
        implementation = path.startswith(
            ("src/", "backend/", "frontend/src/", "app/", "lib/")
        )
        return (0 if implementation else 1, _PRIORITY.get(suffix, 8), path.count("/"), path)

    def search_code(
        self, query: str, path: str = "", offset: int = 0
    ) -> dict[str, Any]:
        """Paged literal search within an optional safe directory prefix.

        Offset advances over candidate *files*, not result lines. Search remains
        incomplete if the model does not follow next_offset, or fetch budget runs out.
        """
        if not isinstance(query, str) or not 2 <= len(query) <= 64 or "\n" in query:
            raise ValueError("Search query must be 2-64 characters on one line")
        if not isinstance(path, str):
            raise ValueError("Directory path must be a string")
        path = path.strip("/")
        if path and not _allowed_path_parts(path):
            raise ValueError("Invalid directory")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        candidates = sorted(
            (p for p in self._files if not path or p.startswith(path + "/")),
            key=self._priority,
        )
        matches: list[dict[str, Any]] = []
        considered = 0
        scanned = 0
        search_budget_exhausted = False
        for candidate in candidates[offset:offset + self.max_search_files]:
            # Metadata-rejected files do not attempt a fetch or consume quota.
            size = self._files[candidate]
            if size is None or size <= 0 or size > self.max_file_bytes:
                considered += 1
                continue
            # Reserve some GitHub fetches for exploratory read_file calls.
            if (
                candidate not in self._cache
                and candidate not in self._search_attempted
                and len(self._search_attempted) >= self.max_search_unique_files
            ):
                search_budget_exhausted = True
                break
            considered += 1
            try:
                lines = self._load(candidate, search_fetch=True)
            except ValueError:
                continue
            scanned += 1
            for line_number, line in enumerate(lines, 1):
                if query.casefold() in line.casefold():
                    matches.append({"path": candidate, "line": line_number})
                    if len(matches) >= 12:
                        break
            if len(matches) >= 12:
                break
        next_offset = (
            offset + considered
            if not search_budget_exhausted and offset + considered < len(candidates)
            else None
        )
        return {
            "query": query, "path": path, "offset": offset,
            "matches": matches, "scanned_files": scanned,
            "considered_files": considered, "total_candidate_files": len(candidates),
            "next_offset": next_offset, "scan_limit": self.max_search_files,
            "partial_search": True,
            "search_budget_exhausted": search_budget_exhausted,
        }

    def read_file(self, path: str, start_line: int = 1, end_line: int = 120) -> dict[str, Any]:
        if type(start_line) is not int or type(end_line) is not int:
            raise ValueError("Line numbers must be integers")
        if start_line < 1 or end_line < start_line or end_line - start_line + 1 > self.max_read_lines:
            raise ValueError(f"Invalid range or more than {self.max_read_lines} lines")
        lines = self._load(path)
        if start_line > len(lines):
            raise ValueError("Start line exceeds file length")
        end = min(end_line, len(lines))
        numbered = "\n".join(
            f"{n}: {lines[n - 1]}" for n in range(start_line, end + 1)
        )
        if len(numbered) > self.max_excerpt_chars:
            raise ValueError("Requested source range exceeds prompt character budget")
        return {
            "path": path, "start_line": start_line, "end_line": end,
            "total_lines": len(lines), "numbered_text": numbered,
        }

    def read_for_extraction(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 120,
        *,
        prefer_whole_file: bool = False,
    ) -> dict[str, Any]:
        """Experiment: expand an initial read to the whole file when safe.

        The public read_file tool remains bounded to max_read_lines. Only an
        initial 1..max_read_lines request may expand; explicit smaller/later
        ranges retain their meaning. Both modes share the same frozen source
        cache, line numbering and max_excerpt_chars safety budget.
        """
        fragment = self.read_file(path, start_line=start_line, end_line=end_line)
        if (
            not prefer_whole_file
            or start_line != 1
            or end_line != self.max_read_lines
            or fragment["end_line"] == fragment["total_lines"]
        ):
            return fragment

        lines = self._load(path)
        numbered = "\n".join(f"{n}: {line}" for n, line in enumerate(lines, 1))
        if len(numbered) > self.max_excerpt_chars:
            return fragment
        return {
            **fragment,
            "end_line": len(lines),
            "numbered_text": numbered,
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


    def source_paths(self) -> list[str]:
        """Files the browser can read according to size and path metadata.

        Actual contents can still be unavailable, binary or larger than the
        GitHub tree's reported size; this is a preflight, not a guarantee.
        """
        return [
            path for path, size in self._files.items()
            if size is not None and 0 < size <= self.max_file_bytes
        ]


EXPLORER_TOOLS = [
    {
        "type": "function", "function": {
            "name": "list_files", "description": "Page through safe source files. Follow next_offset for more; path is an optional directory prefix.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Directory prefix, or empty for all"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 80}
            }},
        }
    },
    {
        "type": "function", "function": {
            "name": "search_code", "description": "Paged literal search, optionally inside a source directory. Follow next_offset to scan another batch. Not regex and never exhaustive by default.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Case-insensitive literal token or identifier"},
                "path": {"type": "string", "description": "Optional directory prefix"},
                "offset": {"type": "integer", "minimum": 0}
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
