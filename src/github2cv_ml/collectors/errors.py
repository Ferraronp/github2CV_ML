"""Collector-specific exceptions."""


class GitHubCollectorError(RuntimeError):
    """Base error for GitHub repository collection failures."""


class InvalidRepositoryReferenceError(ValueError):
    """Raised when a repository reference cannot be parsed."""


class RepositoryNotFoundError(GitHubCollectorError):
    """Raised when a repository is missing or not visible to the caller."""


class RepositoryAccessError(GitHubCollectorError):
    """Raised when GitHub rejects access to a repository."""


class GitHubApiError(GitHubCollectorError):
    """Raised for unexpected GitHub API or transport failures."""
