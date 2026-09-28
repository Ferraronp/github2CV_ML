"""Small GitHub REST client used by repository collectors."""

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from github2cv_ml.collectors.errors import (
    GitHubApiError,
    RepositoryAccessError,
    RepositoryNotFoundError,
)


class GitHubApiClient:
    """Fetch repository data from the GitHub REST API."""

    def __init__(self, token: str | None = None, timeout: float = 10.0) -> None:
        self.token = token
        self.timeout = timeout

    def get_repository(self, owner: str, name: str) -> dict[str, Any]:
        return self._request_json(self._repo_path(owner, name))

    def get_languages(self, owner: str, name: str) -> dict[str, int]:
        payload = self._request_json(f"{self._repo_path(owner, name)}/languages")
        return {str(language): int(bytes_count) for language, bytes_count in payload.items()}

    def get_readme(self, owner: str, name: str) -> str | None:
        payload = self._request_bytes(
            f"{self._repo_path(owner, name)}/readme",
            accept="application/vnd.github.raw+json",
            allow_not_found=True,
        )
        if payload is None:
            return None
        return payload.decode("utf-8", errors="replace")

    def get_tree(self, owner: str, name: str, ref: str) -> dict[str, Any]:
        encoded_ref = quote(ref, safe="")
        return self._request_json(
            f"{self._repo_path(owner, name)}/git/trees/{encoded_ref}?recursive=1",
            allow_empty_repository_conflict=True,
        )

    def get_file(self, owner: str, name: str, path: str, ref: str) -> str | None:
        """Fetch one repository text file, returning None if it disappeared or is absent."""

        encoded_path = quote(path, safe="/")
        encoded_ref = quote(ref, safe="")
        payload = self._request_bytes(
            f"{self._repo_path(owner, name)}/contents/{encoded_path}?ref={encoded_ref}",
            accept="application/vnd.github.raw+json",
            allow_not_found=True,
        )
        if payload is None:
            return None
        return payload.decode("utf-8", errors="replace")

    @staticmethod
    def _repo_path(owner: str, name: str) -> str:
        encoded_owner = quote(owner, safe="")
        encoded_name = quote(name, safe="")
        return f"/repos/{encoded_owner}/{encoded_name}"

    def _request_json(
        self,
        path: str,
        *,
        allow_empty_repository_conflict: bool = False,
    ) -> dict[str, Any]:
        payload = self._request_bytes(
            path,
            allow_empty_repository_conflict=allow_empty_repository_conflict,
        )
        if payload is None:
            return {}
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise GitHubApiError("GitHub returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise GitHubApiError("GitHub returned an unexpected JSON payload")
        return decoded

    def _request_bytes(
        self,
        path: str,
        *,
        accept: str = "application/vnd.github+json",
        allow_not_found: bool = False,
        allow_empty_repository_conflict: bool = False,
    ) -> bytes | None:
        request = Request(
            f"https://api.github.com{path}",
            headers=self._headers(accept),
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404 and allow_not_found:
                return None
            if exc.code == 404:
                raise RepositoryNotFoundError(
                    "Repository was not found or is not visible with the supplied credentials"
                ) from exc
            if exc.code in {401, 403}:
                raise RepositoryAccessError("GitHub rejected repository access") from exc
            if exc.code == 409:
                if allow_empty_repository_conflict and _is_empty_repository_conflict(exc):
                    return b'{"tree": [], "truncated": false}'
                raise GitHubApiError("GitHub repository data is unavailable (HTTP 409)") from exc
            raise GitHubApiError(f"GitHub API request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise GitHubApiError("GitHub API request failed") from exc

    def _headers(self, accept: str) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": "github2cv-ml",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


def _is_empty_repository_conflict(error: HTTPError) -> bool:
    """Return True only for GitHub's explicit empty Git repository conflict."""

    try:
        payload = json.loads(error.read())
    except (json.JSONDecodeError, OSError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("message") == "Git Repository is empty."
