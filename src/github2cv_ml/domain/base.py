"""Shared primitives for domain contracts."""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

NonEmptyId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonEmptyLocator = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CommitSha = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=7, pattern=r"^[0-9a-fA-F]+$"),
]


def _validate_repository_path(value: str) -> str:
    if not value.strip():
        raise ValueError("repository path cannot be whitespace-only")
    return value


RepositoryPath = Annotated[
    str,
    StringConstraints(min_length=1),
    AfterValidator(_validate_repository_path),
]


class ContractModel(BaseModel):
    """Base model for stable internal contracts."""

    model_config = ConfigDict(extra="forbid")
