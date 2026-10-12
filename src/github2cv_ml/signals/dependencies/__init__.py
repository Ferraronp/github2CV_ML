"""Dependency manifest recognition and parser dispatch."""

import re
from pathlib import PurePosixPath

from github2cv_ml.domain.signals import DependencySignal, ManifestKind
from github2cv_ml.signals.dependencies.go import parse_go_mod
from github2cv_ml.signals.dependencies.node import parse_package_json
from github2cv_ml.signals.dependencies.python import parse_pyproject, parse_requirements

_REQUIREMENTS_RE = re.compile(r"^requirements.*\.txt$", re.IGNORECASE)


def manifest_kind_for_path(path: str) -> ManifestKind | None:
    """Return the supported manifest kind for a repository path."""

    lowered = PurePosixPath(path).name.lower()
    if lowered == "pyproject.toml":
        return ManifestKind.PYPROJECT
    if _REQUIREMENTS_RE.fullmatch(lowered):
        return ManifestKind.REQUIREMENTS
    if lowered == "package.json":
        return ManifestKind.PACKAGE_JSON
    if lowered == "go.mod":
        return ManifestKind.GO_MOD
    return None


def parse_dependency_manifest(path: str, content: str) -> list[DependencySignal]:
    """Parse a supported manifest; malformed/unsupported content yields no dependencies."""

    kind = manifest_kind_for_path(path)
    if kind == ManifestKind.PYPROJECT:
        return parse_pyproject(path, content)
    if kind == ManifestKind.REQUIREMENTS:
        return parse_requirements(path, content)
    if kind == ManifestKind.PACKAGE_JSON:
        return parse_package_json(path, content)
    if kind == ManifestKind.GO_MOD:
        return parse_go_mod(path, content)
    return []


__all__ = ["manifest_kind_for_path", "parse_dependency_manifest"]
