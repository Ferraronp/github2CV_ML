"""Collect normalized repository snapshots from GitHub."""

from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlparse

from github2cv_ml.collectors.errors import GitHubApiError, InvalidRepositoryReferenceError
from github2cv_ml.collectors.github_client import GitHubApiClient
from github2cv_ml.domain import (
    RepoSnapshot,
    RepoTreeEntry,
    RepoTreeEntryKind,
    RepositoryRef,
)


class GitHubRepositoryClient(Protocol):
    """Data source required by the repository collector."""

    def get_repository(self, owner: str, name: str) -> dict[str, Any]: ...

    def get_languages(self, owner: str, name: str) -> dict[str, int]: ...

    def get_readme(self, owner: str, name: str) -> str | None: ...

    def get_tree(self, owner: str, name: str, ref: str) -> dict[str, Any]: ...


class GitHubRepositoryCollector:
    """Build a RepoSnapshot from normalized GitHub API data."""

    def __init__(
        self,
        client: GitHubRepositoryClient | None = None,
        *,
        token: str | None = None,
    ) -> None:
        self.client = client or GitHubApiClient(token=token)

    def collect(self, repository: str) -> RepoSnapshot:
        owner, name = parse_repository_reference(repository)
        metadata = self.client.get_repository(owner, name)

        default_branch = metadata.get("default_branch")
        if not isinstance(default_branch, str) or not default_branch.strip():
            raise GitHubApiError("GitHub repository metadata has no default branch")

        languages = self.client.get_languages(owner, name)
        readme = self.client.get_readme(owner, name)
        tree_payload = self.client.get_tree(owner, name, default_branch)

        canonical_owner = _nested_string(metadata, "owner", "login") or owner
        canonical_name = _string(metadata, "name") or name
        repository_url = _string(metadata, "html_url") or (
            f"https://github.com/{canonical_owner}/{canonical_name}"
        )

        return RepoSnapshot(
            repository=RepositoryRef(
                owner=canonical_owner,
                name=canonical_name,
                url=repository_url,
                default_branch=default_branch,
            ),
            description=metadata.get("description"),
            topics=[str(topic) for topic in metadata.get("topics") or []],
            languages=languages,
            readme=readme,
            file_tree=_normalize_tree(tree_payload),
            file_tree_truncated=bool(tree_payload.get("truncated", False)),
            stars=int(metadata.get("stargazers_count", 0)),
            forks=int(metadata.get("forks_count", 0)),
            open_issues=int(metadata.get("open_issues_count", 0)),
            is_fork=bool(metadata.get("fork", False)),
            is_private=bool(metadata.get("private", False)),
            archived=bool(metadata.get("archived", False)),
            pushed_at=metadata.get("pushed_at"),
            captured_at=datetime.now(timezone.utc),
        )


def parse_repository_reference(reference: str) -> tuple[str, str]:
    """Parse `owner/repo` or a canonical GitHub repository URL."""

    candidate = reference.strip()
    if not candidate:
        raise InvalidRepositoryReferenceError("Repository reference cannot be empty")

    if candidate.startswith(("https://", "http://")):
        parsed = urlparse(candidate)
        if (parsed.hostname or "").lower() not in {"github.com", "www.github.com"}:
            raise InvalidRepositoryReferenceError("Only github.com repository URLs are supported")
        parts = [part for part in parsed.path.split("/") if part]
    else:
        for prefix in ("github.com/", "www.github.com/"):
            if candidate.lower().startswith(prefix):
                candidate = candidate[len(prefix) :]
                break
        parts = [part for part in candidate.strip("/").split("/") if part]

    if len(parts) != 2:
        raise InvalidRepositoryReferenceError("Repository must be provided as owner/repo or GitHub URL")

    owner, name = parts
    if name.endswith(".git"):
        name = name[:-4]
    if not owner or not name:
        raise InvalidRepositoryReferenceError("Repository owner and name cannot be empty")
    return owner, name


def _normalize_tree(payload: dict[str, Any]) -> list[RepoTreeEntry]:
    kind_map = {
        "blob": RepoTreeEntryKind.FILE,
        "tree": RepoTreeEntryKind.DIRECTORY,
        "commit": RepoTreeEntryKind.SUBMODULE,
    }
    entries: list[RepoTreeEntry] = []
    for raw_entry in payload.get("tree") or []:
        if not isinstance(raw_entry, dict):
            continue
        path = raw_entry.get("path")
        kind = kind_map.get(raw_entry.get("type"))
        if not isinstance(path, str) or not path.strip() or kind is None:
            continue
        entries.append(
            RepoTreeEntry(
                path=path,
                kind=kind,
                size=raw_entry.get("size"),
            )
        )
    return entries


def _string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _nested_string(payload: dict[str, Any], parent: str, key: str) -> str | None:
    nested = payload.get(parent)
    if not isinstance(nested, dict):
        return None
    return _string(nested, key)
