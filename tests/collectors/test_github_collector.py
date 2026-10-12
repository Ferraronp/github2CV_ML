from typing import Any

import pytest

from github2cv_ml.collectors import (
    GitHubApiError,
    GitHubRepositoryCollector,
    InvalidRepositoryReferenceError,
    RepositoryNotFoundError,
    parse_repository_reference,
)
from github2cv_ml.domain import DependencyEcosystem, RepoSnapshot, RepoTreeEntryKind, ToolingKind


class FakeGitHubClient:
    def __init__(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        languages: dict[str, int] | None = None,
        readme: str | None = "# Demo",
        tree: dict[str, Any] | None = None,
        files: dict[str, str] | None = None,
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
        self.files = files or {}

    def get_repository(self, owner: str, name: str) -> dict[str, Any]:
        return self.metadata

    def get_languages(self, owner: str, name: str) -> dict[str, int]:
        return self.languages

    def get_readme(self, owner: str, name: str) -> str | None:
        return self.readme

    def get_tree(self, owner: str, name: str, ref: str) -> dict[str, Any]:
        return self.tree

    def get_file(self, owner: str, name: str, path: str, ref: str) -> str | None:
        return self.files.get(path)


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
        "size": 42,
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
    assert snapshot.schema_version == "3.0"
    assert snapshot.repository.owner == "octocat"
    assert snapshot.repository.default_branch == "main"
    assert snapshot.languages == {"Python": 1200}
    assert snapshot.readme == "# Demo"
    assert snapshot.topics == ["python", "api"]
    assert snapshot.stars == 12
    assert snapshot.signals.dependencies == []
    assert "open_issues" not in snapshot.model_dump()
    assert snapshot.pushed_at is not None
    assert snapshot.captured_at.tzinfo is not None
    assert [entry.kind for entry in snapshot.file_tree] == [
        RepoTreeEntryKind.DIRECTORY,
        RepoTreeEntryKind.FILE,
        RepoTreeEntryKind.SUBMODULE,
    ]


def test_collect_adds_deterministic_project_signals() -> None:
    collector = GitHubRepositoryCollector(
        client=FakeGitHubClient(
            tree={
                "tree": [
                    {"path": "pyproject.toml", "type": "blob", "size": 100},
                    {"path": "Dockerfile", "type": "blob", "size": 80},
                    {"path": ".github/workflows/ci.yml", "type": "blob", "size": 60},
                    {"path": "tests", "type": "tree"},
                ],
                "truncated": False,
            },
            files={
                "pyproject.toml": '[project]\ndependencies = ["fastapi>=0.115"]\n'
            },
        )
    )

    snapshot = collector.collect("octocat/hello-world")

    assert [(item.name, item.ecosystem) for item in snapshot.signals.dependencies] == [
        ("fastapi", DependencyEcosystem.PYTHON)
    ]
    assert {item.kind for item in snapshot.signals.tooling} == {
        ToolingKind.DOCKERFILE,
        ToolingKind.GITHUB_ACTIONS,
    }


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


def test_collect_handles_empty_repository_with_nonempty_pushed_at() -> None:
    collector = GitHubRepositoryCollector(
        client=FakeGitHubClient(
            metadata=_metadata(size=0, pushed_at="2013-12-31T23:08:15Z"),
            languages={},
            readme=None,
            tree={"tree": [], "truncated": False},
        )
    )

    snapshot = collector.collect("octocat/hello-world")

    assert snapshot.languages == {}
    assert snapshot.readme is None
    assert snapshot.file_tree == []
    assert snapshot.signals.dependencies == []
    assert snapshot.file_tree_truncated is False
    assert snapshot.pushed_at is not None


def test_collect_does_not_hide_tree_unavailable_error() -> None:
    class UnavailableTreeClient(FakeGitHubClient):
        def get_tree(self, owner: str, name: str, ref: str) -> dict[str, Any]:
            raise GitHubApiError("GitHub repository data is unavailable (HTTP 409)")

    collector = GitHubRepositoryCollector(client=UnavailableTreeClient())

    with pytest.raises(GitHubApiError, match="HTTP 409"):
        collector.collect("octocat/hello-world")


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


def test_collect_skips_bare_vcs_urls_and_keeps_requirements_duplicates_safe() -> None:
    collector = GitHubRepositoryCollector(
        client=FakeGitHubClient(
            tree={
                "tree": [{"path": "requirements.txt", "type": "blob", "size": 120}],
                "truncated": False,
            },
            files={
                "requirements.txt": (
                    "requests\nrequests>=2\n"
                    "https://example.com/pkg.whl\n"
                    "git+https://github.com/org/repo.git\n"
                ),
            },
        )
    )

    snapshot = collector.collect("octocat/hello-world")

    assert [(dep.name, dep.requirement) for dep in snapshot.signals.dependencies] == [
        ("requests", None),
        ("requests", ">=2"),
    ]


def test_collect_ignores_invalid_manifest_entries_and_nested_workflows() -> None:
    collector = GitHubRepositoryCollector(
        client=FakeGitHubClient(
            tree={
                "tree": [
                    {"path": "web/package.json", "type": "blob", "size": 150},
                    {"path": "pyproject.toml", "type": "blob", "size": 150},
                    {
                        "path": ".github/workflows/templates/ci.yml",
                        "type": "blob",
                        "size": 30,
                    },
                ],
                "truncated": False,
            },
            files={
                "web/package.json": '{"dependencies": {"": "^1.0", "react": "^19"}}',
                "pyproject.toml": (
                    '[tool.poetry.dependencies]\n'
                    '"" = "^1.0"\n'
                    'requests = "^2.32"\n'
                ),
            },
        )
    )

    snapshot = collector.collect("octocat/hello-world")

    assert {(dep.name, dep.requirement) for dep in snapshot.signals.dependencies} == {
        ("react", "^19"),
        ("requests", "^2.32"),
    }
    assert snapshot.signals.tooling == []
    assert len(snapshot.signals.manifests) == 2
