from typing import Any

import pytest

from github2cv_ml.collectors import (
    GitHubRepositoryCollector,
    InvalidRepositoryReferenceError,
    RepositoryNotFoundError,
    parse_repository_reference,
)
from github2cv_ml.domain import RepoSnapshot, RepoTreeEntryKind


class FakeGitHubClient:
    def __init__(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        languages: dict[str, int] | None = None,
        readme: str | None = "# Demo",
        tree: dict[str, Any] | None = None,
    ) -> None:
        self.metadata = metadata or _metadata()
        self.languages = languages if languages is not None else {"Python": 1200}
        self.readme = readme
        self.tree = tree or {
            "tree": [
                {"path": "src", "type": "tree"},
                {"path": "src/app.py", "type": "blob", "size": 123},
                {"path": "vendor/lib", "type": "commit"},
            ],
            "truncated": False,
        }

    def get_repository(self, owner: str, name: str) -> dict[str, Any]:
        return self.metadata

    def get_languages(self, owner: str, name: str) -> dict[str, int]:
        return self.languages

    def get_readme(self, owner: str, name: str) -> str | None:
        return self.readme

    def get_tree(self, owner: str, name: str, ref: str) -> dict[str, Any]:
        return self.tree


def _metadata(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "owner": {"login": "octocat"},
        "name": "hello-world",
        "html_url": "https://github.com/octocat/hello-world",
        "default_branch": "main",
        "description": "Example repository",
        "topics": ["python", "api"],
        "stargazers_count": 12,
        "forks_count": 3,
        "open_issues_count": 7,
        "fork": False,
        "private": False,
        "archived": False,
        "pushed_at": "2026-09-28T09:00:00Z",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("octocat/hello-world", ("octocat", "hello-world")),
        ("https://github.com/octocat/hello-world", ("octocat", "hello-world")),
        ("github.com/octocat/hello-world.git", ("octocat", "hello-world")),
    ],
)
def test_parse_repository_reference(reference: str, expected: tuple[str, str]) -> None:
    assert parse_repository_reference(reference) == expected


@pytest.mark.parametrize(
    "reference",
    ["", "hello-world", "https://gitlab.com/octocat/hello-world", "octocat/repo/issues/1"],
)
def test_parse_repository_reference_rejects_invalid_input(reference: str) -> None:
    with pytest.raises(InvalidRepositoryReferenceError):
        parse_repository_reference(reference)


def test_collect_returns_normalized_repo_snapshot() -> None:
    collector = GitHubRepositoryCollector(client=FakeGitHubClient())

    snapshot = collector.collect("https://github.com/octocat/hello-world")

    assert isinstance(snapshot, RepoSnapshot)
    assert snapshot.repository.owner == "octocat"
    assert snapshot.repository.default_branch == "main"
    assert snapshot.languages == {"Python": 1200}
    assert snapshot.readme == "# Demo"
    assert snapshot.topics == ["python", "api"]
    assert snapshot.stars == 12
    assert "open_issues" not in snapshot.model_dump()
    assert snapshot.pushed_at is not None
    assert snapshot.captured_at.tzinfo is not None
    assert [entry.kind for entry in snapshot.file_tree] == [
        RepoTreeEntryKind.DIRECTORY,
        RepoTreeEntryKind.FILE,
        RepoTreeEntryKind.SUBMODULE,
    ]


def test_collect_preserves_repository_tree_path() -> None:
    path = " docs/file.txt "
    collector = GitHubRepositoryCollector(
        client=FakeGitHubClient(
            tree={
                "tree": [{"path": path, "type": "blob", "size": 10}],
                "truncated": False,
            }
        )
    )

    snapshot = collector.collect("octocat/hello-world")

    assert snapshot.file_tree[0].path == path


def test_collect_handles_empty_repository() -> None:
    collector = GitHubRepositoryCollector(
        client=FakeGitHubClient(
            languages={},
            readme=None,
            tree={"tree": [], "truncated": False},
        )
    )

    snapshot = collector.collect("octocat/hello-world")

    assert snapshot.languages == {}
    assert snapshot.readme is None
    assert snapshot.file_tree == []
    assert snapshot.file_tree_truncated is False


def test_collect_preserves_private_repository_flag() -> None:
    collector = GitHubRepositoryCollector(
        client=FakeGitHubClient(metadata=_metadata(private=True))
    )

    snapshot = collector.collect("octocat/hello-world")

    assert snapshot.is_private is True


def test_collect_propagates_missing_repository_error() -> None:
    class MissingRepositoryClient(FakeGitHubClient):
        def get_repository(self, owner: str, name: str) -> dict[str, Any]:
            raise RepositoryNotFoundError("missing")

    collector = GitHubRepositoryCollector(client=MissingRepositoryClient())

    with pytest.raises(RepositoryNotFoundError):
        collector.collect("octocat/missing")
