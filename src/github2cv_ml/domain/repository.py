"""Repository-level domain contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from github2cv_ml.domain.base import ContractModel, NonEmptyLocator


class RepositoryRef(ContractModel):
    """Normalized repository identity used across pipeline stages."""

    owner: str = Field(min_length=1)
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    default_branch: str = Field(min_length=1)


class RepoTreeEntryKind(StrEnum):
    """Normalized kinds of entries in a repository tree."""

    FILE = "file"
    DIRECTORY = "directory"
    SUBMODULE = "submodule"


class RepoTreeEntry(ContractModel):
    """One normalized entry from a repository file tree."""

    path: NonEmptyLocator
    kind: RepoTreeEntryKind
    size: int | None = Field(default=None, ge=0)


class RepoSnapshot(ContractModel):
    """Normalized repository metadata collected from GitHub."""

    schema_version: Literal["1.0"] = "1.0"
    repository: RepositoryRef
    description: str | None = None
    topics: list[str] = Field(default_factory=list)
    languages: dict[str, int] = Field(default_factory=dict)
    readme: str | None = None
    file_tree: list[RepoTreeEntry] = Field(default_factory=list)
    file_tree_truncated: bool = False
    stars: int = Field(default=0, ge=0)
    forks: int = Field(default=0, ge=0)
    open_issues: int = Field(default=0, ge=0)
    is_fork: bool = False
    is_private: bool = False
    archived: bool = False
    pushed_at: datetime | None = None
    captured_at: datetime
