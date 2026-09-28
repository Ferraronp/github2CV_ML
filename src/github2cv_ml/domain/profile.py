"""Candidate profile domain contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from github2cv_ml.domain.base import ContractModel, NonEmptyId


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
