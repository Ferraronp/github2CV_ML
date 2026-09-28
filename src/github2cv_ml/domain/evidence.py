"""Evidence and provenance domain contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from github2cv_ml.domain.base import (
    CommitSha,
    ContractModel,
    NonEmptyId,
    NonEmptyLocator,
    RepositoryPath,
)
from github2cv_ml.domain.repository import RepositoryRef


class EvidenceKind(StrEnum):
    """Kinds of repository sources that can support an observation."""

    REPOSITORY_METADATA = "repository_metadata"
    FILE = "file"
    COMMIT = "commit"
    DEPENDENCY_MANIFEST = "dependency_manifest"
    LANGUAGE_STATS = "language_stats"
    ISSUE = "issue"
    PULL_REQUEST = "pull_request"


class EvidenceSource(ContractModel):
    """Locator that points back to the source of repository evidence."""

    kind: EvidenceKind
    url: NonEmptyLocator | None = None
    path: RepositoryPath | None = None
    commit_sha: CommitSha | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_locator(self) -> "EvidenceSource":
        if self.url is None and self.path is None and self.commit_sha is None:
            raise ValueError("at least one source locator is required")
        if (self.line_start is not None or self.line_end is not None) and self.path is None:
            raise ValueError("path is required when source lines are provided")
        if self.line_end is not None and self.line_start is None:
            raise ValueError("line_start is required when line_end is provided")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class RepoEvidence(ContractModel):
    """One normalized observation backed by a repository source."""

    schema_version: Literal["1.0"] = "1.0"
    id: NonEmptyId
    repository: RepositoryRef
    category: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    value: JsonValue | None = None
    source: EvidenceSource
    captured_at: datetime
