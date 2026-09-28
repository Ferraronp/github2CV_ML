from urllib.error import HTTPError

import pytest

import github2cv_ml.collectors.github_client as github_client
from github2cv_ml.collectors import (
    GitHubApiError,
    RepositoryAccessError,
    RepositoryNotFoundError,
)


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_client_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_headers: dict[str, str | None] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured_headers["authorization"] = request.get_header("Authorization")  # type: ignore[attr-defined]
        return FakeResponse(b'{"name": "private-repo"}')

    monkeypatch.setattr(github_client, "urlopen", fake_urlopen)

    client = github_client.GitHubApiClient(token="test-token")
    client.get_repository("owner", "private-repo")

    assert captured_headers["authorization"] == "Bearer test-token"


def test_client_treats_missing_readme_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        raise HTTPError("https://api.github.com/readme", 404, "Not Found", None, None)

    monkeypatch.setattr(github_client, "urlopen", fake_urlopen)

    client = github_client.GitHubApiClient()

    assert client.get_readme("owner", "repo") is None


def test_client_does_not_treat_tree_conflict_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        raise HTTPError("https://api.github.com/tree", 409, "Conflict", None, None)

    monkeypatch.setattr(github_client, "urlopen", fake_urlopen)

    client = github_client.GitHubApiClient()

    with pytest.raises(GitHubApiError, match="HTTP 409"):
        client.get_tree("owner", "repo", "main")


def test_client_maps_missing_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        raise HTTPError("https://api.github.com/repo", 404, "Not Found", None, None)

    monkeypatch.setattr(github_client, "urlopen", fake_urlopen)

    with pytest.raises(RepositoryNotFoundError):
        github_client.GitHubApiClient().get_repository("owner", "missing")


def test_client_maps_access_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        raise HTTPError("https://api.github.com/repo", 403, "Forbidden", None, None)

    monkeypatch.setattr(github_client, "urlopen", fake_urlopen)

    with pytest.raises(RepositoryAccessError):
        github_client.GitHubApiClient().get_repository("owner", "private")
