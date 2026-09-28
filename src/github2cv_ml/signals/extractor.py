"""Deterministic project signal extraction from repository files and tree entries."""

from pathlib import PurePosixPath

from github2cv_ml.domain.repository import RepoTreeEntry, RepoTreeEntryKind
from github2cv_ml.domain.signals import (
    ManifestSignal,
    RepositorySignals,
    StructureKind,
    StructureSignal,
    ToolingKind,
    ToolingSignal,
)
from github2cv_ml.signals.dependencies import manifest_kind_for_path, parse_dependency_manifest

_COMPOSE_FILENAMES = {
    "compose.yml",
    "compose.yaml",
    "docker-compose.yml",
    "docker-compose.yaml",
}
_STRUCTURE_KINDS = {
    "src": StructureKind.SOURCE,
    "tests": StructureKind.TESTS,
    "test": StructureKind.TESTS,
    "frontend": StructureKind.FRONTEND,
    "backend": StructureKind.BACKEND,
    "cmd": StructureKind.GO_CMD,
    "internal": StructureKind.GO_INTERNAL,
    "pkg": StructureKind.GO_PKG,
}


def dependency_manifest_paths(file_tree: list[RepoTreeEntry]) -> list[str]:
    """Return deterministic, sorted manifest paths that need content fetching."""

    return sorted(
        entry.path
        for entry in file_tree
        if entry.kind == RepoTreeEntryKind.FILE and manifest_kind_for_path(entry.path) is not None
    )


def extract_repository_signals(
    file_tree: list[RepoTreeEntry], file_contents: dict[str, str]
) -> RepositorySignals:
    """Extract repository signals without invoking an LLM."""

    manifests: list[ManifestSignal] = []
    dependencies = []
    tooling: list[ToolingSignal] = []
    structures: list[StructureSignal] = []

    for entry in sorted(file_tree, key=lambda item: item.path):
        if entry.kind == RepoTreeEntryKind.FILE:
            manifest_kind = manifest_kind_for_path(entry.path)
            if manifest_kind is not None:
                manifests.append(ManifestSignal(kind=manifest_kind, path=entry.path))
                content = file_contents.get(entry.path)
                if content is not None:
                    dependencies.extend(parse_dependency_manifest(entry.path, content))

            tooling_kind = _tooling_kind_for_path(entry.path)
            if tooling_kind is not None:
                tooling.append(ToolingSignal(kind=tooling_kind, path=entry.path))

        elif entry.kind == RepoTreeEntryKind.DIRECTORY and "/" not in entry.path:
            structure_kind = _STRUCTURE_KINDS.get(entry.path.lower())
            if structure_kind is not None:
                structures.append(StructureSignal(kind=structure_kind, path=entry.path))

    dependencies.sort(
        key=lambda item: (
            item.ecosystem.value,
            item.name,
            item.group,
            item.source_path,
            item.requirement or "",
        )
    )
    return RepositorySignals(
        dependencies=dependencies,
        manifests=manifests,
        tooling=tooling,
        structures=structures,
    )


def _tooling_kind_for_path(path: str) -> ToolingKind | None:
    pure_path = PurePosixPath(path)
    name = pure_path.name.lower()
    if name == "dockerfile" or name.startswith("dockerfile."):
        return ToolingKind.DOCKERFILE
    if name in _COMPOSE_FILENAMES:
        return ToolingKind.DOCKER_COMPOSE
    if path.startswith(".github/workflows/") and name.endswith((".yml", ".yaml")):
        return ToolingKind.GITHUB_ACTIONS
    return None
