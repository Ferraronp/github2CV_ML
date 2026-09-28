"""Shared primitives for domain contracts."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonEmptyLocator = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CommitSha = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=7, pattern=r"^[0-9a-fA-F]+$"),
]


class ContractModel(BaseModel):
    """Base model for stable internal contracts."""

    model_config = ConfigDict(extra="forbid")
