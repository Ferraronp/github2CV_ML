"""Resume document domain contracts."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from github2cv_ml.domain.base import ContractModel, NonEmptyId


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
