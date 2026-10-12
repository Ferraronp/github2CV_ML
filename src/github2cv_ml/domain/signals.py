"""Deterministic repository signal contracts."""

from enum import StrEnum

from pydantic import Field

from github2cv_ml.domain.base import ContractModel, RepositoryPath


class DependencyEcosystem(StrEnum):
    """Dependency ecosystems supported by deterministic extraction."""

    PYTHON = "python"
    NPM = "npm"
    GO = "go"


class DependencySignal(ContractModel):
    """One dependency declaration extracted from a repository manifest."""

    name: str = Field(min_length=1)
    ecosystem: DependencyEcosystem
    requirement: str | None = None
    group: str = Field(default="runtime", min_length=1)
    source_path: RepositoryPath


class ManifestKind(StrEnum):
    """Dependency manifest formats recognized without an LLM."""

    PYPROJECT = "pyproject"
    REQUIREMENTS = "requirements"
    PACKAGE_JSON = "package_json"
    GO_MOD = "go_mod"


class ManifestSignal(ContractModel):
    """Presence of a recognized dependency manifest."""

    kind: ManifestKind
    path: RepositoryPath


class ToolingKind(StrEnum):
    """Repository tooling detected from conventional file locations."""

    DOCKERFILE = "dockerfile"
    DOCKER_COMPOSE = "docker_compose"
    GITHUB_ACTIONS = "github_actions"


class ToolingSignal(ContractModel):
    """Presence of deterministic infrastructure/tooling metadata."""

    kind: ToolingKind
    path: RepositoryPath


class StructureKind(StrEnum):
    """Conventional top-level project directory layouts."""

    SOURCE = "source"
    TESTS = "tests"
    FRONTEND = "frontend"
    BACKEND = "backend"
    GO_CMD = "go_cmd"
    GO_INTERNAL = "go_internal"
    GO_PKG = "go_pkg"


class StructureSignal(ContractModel):
    """One conventional project structure signal."""

    kind: StructureKind
    path: RepositoryPath


class RepositorySignals(ContractModel):
    """Deterministic signals extracted before any LLM analysis."""

    dependencies: list[DependencySignal] = Field(default_factory=list)
    manifests: list[ManifestSignal] = Field(default_factory=list)
    tooling: list[ToolingSignal] = Field(default_factory=list)
    structures: list[StructureSignal] = Field(default_factory=list)
