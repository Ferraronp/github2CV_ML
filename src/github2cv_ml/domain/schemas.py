"""Domain contracts for the github2CV ML pipeline."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

NonEmptyId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonEmptyLocator = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ContractModel(BaseModel):
    """Base model for stable internal contracts."""

    model_config = ConfigDict(extra="forbid")


class RepositoryRef(ContractModel):
    """Normalized repository identity used across pipeline stages."""

    owner: str = Field(min_length=1)
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    default_branch: str = Field(min_length=1)


class RepoSnapshot(ContractModel):
    """Normalized repository metadata collected from GitHub."""

    schema_version: Literal["1.0"] = "1.0"
    repository: RepositoryRef
    description: str | None = None
    topics: list[str] = Field(default_factory=list)
    languages: dict[str, int] = Field(default_factory=dict)
    stars: int = Field(default=0, ge=0)
    forks: int = Field(default=0, ge=0)
    open_issues: int = Field(default=0, ge=0)
    is_fork: bool = False
    archived: bool = False
    captured_at: datetime


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
    path: NonEmptyLocator | None = None
    commit_sha: str | None = Field(default=None, min_length=7)
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


class ClaimKind(StrEnum):
    """Candidate-profile claim categories."""

    SKILL = "skill"
    PROJECT = "project"
    RESPONSIBILITY = "responsibility"
    ACHIEVEMENT = "achievement"
    SUMMARY = "summary"


class CandidateIdentity(ContractModel):
    """Candidate identity metadata, separate from inferred claims."""

    github_login: str = Field(min_length=1)
    display_name: str | None = None


class CandidateClaim(ContractModel):
    """A profile claim that must point to supporting repository evidence."""

    id: NonEmptyId
    kind: ClaimKind
    text: str = Field(min_length=1)
    evidence_ids: list[NonEmptyId] = Field(min_length=1)


class CandidateProfile(ContractModel):
    """Evidence-backed candidate profile synthesized from repositories."""

    schema_version: Literal["1.0"] = "1.0"
    candidate: CandidateIdentity
    claims: list[CandidateClaim] = Field(default_factory=list)
    generated_at: datetime

    @model_validator(mode="after")
    def validate_unique_claim_ids(self) -> "CandidateProfile":
        claim_ids = [claim.id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim ids must be unique within a candidate profile")
        return self


class ResumeItem(ContractModel):
    """Resume text traced back to one or more profile claims."""

    text: str = Field(min_length=1)
    claim_ids: list[NonEmptyId] = Field(min_length=1)


class ResumeSection(ContractModel):
    """A named resume section containing evidence-backed items."""

    title: str = Field(min_length=1)
    items: list[ResumeItem] = Field(min_length=1)


class ResumeDocument(ContractModel):
    """Structured resume output with traceability to candidate claims."""

    schema_version: Literal["1.0"] = "1.0"
    candidate_name: str = Field(min_length=1)
    headline: ResumeItem | None = None
    summary: ResumeItem | None = None
    sections: list[ResumeSection] = Field(default_factory=list)
    generated_at: datetime
