"""Version 2 ML job contract for the existing ResumeHub RabbitMQ topology.

The backend's *transport* (exchange/queue/routing) remains unchanged. Its
current text-only v1 DTO must be upgraded to v2 to accept structured evidence,
a Profile and ResumeDocument. No user credentials or private source are sent.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from github2cv_ml.agent.runner import RepoAnalysis
from github2cv_ml.domain.base import ContractModel
from github2cv_ml.domain.profile import CandidateIdentity, CandidateProfile
from github2cv_ml.domain.resume import ResumeDocument
from github2cv_ml.pipeline import PipelineProblem, PipelineStatus, normalize_repository

SHA_PATTERN = r"^[a-f0-9]{40}$"
CONTEXT_PATTERN = r"^[a-f0-9]{64}$"


class RequestedRepository(ContractModel):
    repo_id: str = Field(pattern=r"^[0-9]{1,20}$")
    repository_url: str
    commit_sha: str = Field(pattern=SHA_PATTERN)
    context_hash: str = Field(pattern=CONTEXT_PATTERN)
    # These are user declarations and are not proof of authorship.
    declared_role: str | None = Field(default=None, max_length=3000)
    started_at: date | None = None
    ended_at: date | None = None

    @field_validator("repository_url")
    @classmethod
    def canonical_url(cls, value: str) -> str:
        key = normalize_repository(value)
        if not value.startswith("https://github.com/"):
            raise ValueError("Only canonical https://github.com/owner/repo URLs")
        return f"https://github.com/{key}"

    @model_validator(mode="after")
    def period(self) -> "RequestedRepository":
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("ended_at precedes started_at")
        return self


class AnalysisOptions(ContractModel):
    max_steps: int = Field(default=28, ge=1, le=100)
    max_reads: int = Field(default=12, ge=1, le=50)
    min_planned_reads: int = Field(default=8, ge=0, le=50)
    max_skills: int = Field(default=20, ge=1, le=20)
    max_input_chars: int = Field(default=20000, ge=1000, le=200000)
    force_refresh: bool = False

    @model_validator(mode="after")
    def validate_budgets(self) -> "AnalysisOptions":
        if self.max_reads > self.max_steps or self.min_planned_reads > self.max_reads:
            raise ValueError("Inconsistent exploration budgets")
        return self


class MLJobRequest(ContractModel):
    schema_version: Literal[2] = 2
    type: Literal["github.report.requested"] = "github.report.requested"
    message_id: UUID
    job_id: UUID
    input_revision: int = Field(ge=0)
    candidate: CandidateIdentity
    repositories: list[RequestedRepository] = Field(min_length=1, max_length=20)
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)

    @model_validator(mode="after")
    def unique_repositories(self) -> "MLJobRequest":
        ids = [item.repo_id for item in self.repositories]
        names = [normalize_repository(item.repository_url) for item in self.repositories]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("Requested repositories must be unique by GitHub ID and URL")
        return self


class MLRepositoryResult(ContractModel):
    repo_id: str
    repository_url: str
    commit_sha: str
    context_hash: str
    status: Literal["completed", "partial", "failed"]
    cache_hit: bool = False
    analysis: RepoAnalysis | None = None
    problems: list[PipelineProblem] = Field(default_factory=list)


class MLJobResult(ContractModel):
    schema_version: Literal[2] = 2
    type: Literal["github.report.completed", "github.report.failed"]
    # Deterministic response UUID allows downstream idempotent inbox storage.
    message_id: UUID
    request_message_id: UUID
    job_id: UUID
    input_revision: int
    status: PipelineStatus
    repositories: list[MLRepositoryResult]
    profile: CandidateProfile | None = None
    resume: ResumeDocument | None = None
    problems: list[PipelineProblem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Optional, bounded correction metadata; never raw LLM/source responses.
    correction_events: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def valid_result(self) -> "MLJobResult":
        if self.type == "github.report.completed":
            if self.resume is None or self.profile is None or self.error is not None:
                raise ValueError("Completed event requires profile and resume, no error")
            if self.status not in {PipelineStatus.COMPLETED, PipelineStatus.PARTIAL}:
                raise ValueError("Completed event requires completed or partial status")
        elif self.resume is not None or self.error is None or self.status != PipelineStatus.FAILED:
            raise ValueError("Failed event requires error without a finished resume")
        return self
