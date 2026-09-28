"""Dependency manifest recognition and deterministic parsers."""

import json
import re
import tomllib
from pathlib import PurePosixPath
from typing import Any

from github2cv_ml.domain.signals import (
    DependencyEcosystem,
    DependencySignal,
    ManifestKind,
)

_REQUIREMENTS_RE = re.compile(r"^requirements.*\.txt$", re.IGNORECASE)
_PYTHON_REQUIREMENT_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]+\])?\s*(.*)$"
)


def manifest_kind_for_path(path: str) -> ManifestKind | None:
    """Return the supported manifest kind for a repository path."""

    name = PurePosixPath(path).name
    lowered = name.lower()
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
    if kind is ManifestKind.PYPROJECT:
        return _parse_pyproject(path, content)
    if kind is ManifestKind.REQUIREMENTS:
        return _parse_requirements(path, content)
    if kind is ManifestKind.PACKAGE_JSON:
        return _parse_package_json(path, content)
    if kind is ManifestKind.GO_MOD:
        return _parse_go_mod(path, content)
    return []


def _parse_pyproject(path: str, content: str) -> list[DependencySignal]:
    try:
        payload = tomllib.loads(content)
    except (tomllib.TOMLDecodeError, TypeError):
        return []

    dependencies: list[DependencySignal] = []
    project = payload.get("project")
    if isinstance(project, dict):
        runtime = project.get("dependencies")
        if isinstance(runtime, list):
            dependencies.extend(_python_requirements(runtime, path, "runtime"))

        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            for group, requirements in optional.items():
                if isinstance(group, str) and isinstance(requirements, list):
                    dependencies.extend(
                        _python_requirements(requirements, path, f"optional:{group}")
                    )

    tool = payload.get("tool")
    poetry = tool.get("poetry") if isinstance(tool, dict) else None
    if isinstance(poetry, dict):
        dependencies.extend(_poetry_dependency_table(poetry.get("dependencies"), path, "runtime"))
        dependencies.extend(_poetry_dependency_table(poetry.get("dev-dependencies"), path, "dev"))
        groups = poetry.get("group")
        if isinstance(groups, dict):
            for group, group_payload in groups.items():
                if isinstance(group, str) and isinstance(group_payload, dict):
                    dependencies.extend(
                        _poetry_dependency_table(
                            group_payload.get("dependencies"),
                            path,
                            f"group:{group}",
                        )
                    )

    return _deduplicate_dependencies(dependencies)


def _python_requirements(
    requirements: list[Any], path: str, group: str
) -> list[DependencySignal]:
    dependencies: list[DependencySignal] = []
    for requirement in requirements:
        if not isinstance(requirement, str):
            continue
        parsed = _parse_python_requirement(requirement)
        if parsed is None:
            continue
        name, spec = parsed
        dependencies.append(
            DependencySignal(
                name=name,
                ecosystem=DependencyEcosystem.PYTHON,
                requirement=spec,
                group=group,
                source_path=path,
            )
        )
    return dependencies


def _poetry_dependency_table(table: Any, path: str, group: str) -> list[DependencySignal]:
    if not isinstance(table, dict):
        return []
    dependencies: list[DependencySignal] = []
    for name, requirement in table.items():
        if not isinstance(name, str) or name.lower() == "python":
            continue
        spec: str | None
        if isinstance(requirement, str):
            spec = requirement
        elif isinstance(requirement, dict) and isinstance(requirement.get("version"), str):
            spec = requirement["version"]
        else:
            spec = None
        dependencies.append(
            DependencySignal(
                name=name,
                ecosystem=DependencyEcosystem.PYTHON,
                requirement=spec,
                group=group,
                source_path=path,
            )
        )
    return dependencies


def _parse_requirements(path: str, content: str) -> list[DependencySignal]:
    group = _requirements_group(path)
    dependencies: list[DependencySignal] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split(" #", 1)[0].strip()
        egg_match = re.search(r"[#&]egg=([^&]+)", line)
        if egg_match is not None and ("://" in line or line.startswith(("git+", "-e "))):
            name, spec = egg_match.group(1), line
        elif line.startswith("-"):
            continue
        else:
            parsed = _parse_python_requirement(line)
            if parsed is None:
                continue
            name, spec = parsed
        dependencies.append(
            DependencySignal(
                name=name,
                ecosystem=DependencyEcosystem.PYTHON,
                requirement=spec,
                group=group,
                source_path=path,
            )
        )
    return _deduplicate_dependencies(dependencies)


def _requirements_group(path: str) -> str:
    stem = PurePosixPath(path).name[:-4]
    suffix = stem[len("requirements") :].lstrip("-_.")
    return suffix or "runtime"


def _parse_python_requirement(requirement: str) -> tuple[str, str | None] | None:
    match = _PYTHON_REQUIREMENT_RE.match(requirement)
    if match is None:
        return None
    name = match.group(1)
    extras = match.group(2) or ""
    remainder = match.group(3).strip()
    spec = f"{extras}{remainder}" or None
    return name, spec


def _parse_package_json(path: str, content: str) -> list[DependencySignal]:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []

    dependencies: list[DependencySignal] = []
    groups = {
        "dependencies": "runtime",
        "devDependencies": "dev",
        "peerDependencies": "peer",
        "optionalDependencies": "optional",
    }
    for key, group in groups.items():
        table = payload.get(key)
        if not isinstance(table, dict):
            continue
        for name, requirement in table.items():
            if not isinstance(name, str) or not isinstance(requirement, str):
                continue
            dependencies.append(
                DependencySignal(
                    name=name,
                    ecosystem=DependencyEcosystem.NPM,
                    requirement=requirement or None,
                    group=group,
                    source_path=path,
                )
            )
    return _deduplicate_dependencies(dependencies)


def _parse_go_mod(path: str, content: str) -> list[DependencySignal]:
    dependencies: list[DependencySignal] = []
    in_require_block = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line == "require (":
            in_require_block = True
            continue
        if in_require_block and line == ")":
            in_require_block = False
            continue

        if line.startswith("require "):
            declaration = line[len("require ") :].strip()
        elif in_require_block:
            declaration = line
        else:
            continue

        group = "runtime"
        if "//" in declaration:
            declaration, comment = declaration.split("//", 1)
            if comment.strip().startswith("indirect"):
                group = "indirect"
        parts = declaration.split()
        if len(parts) < 2:
            continue
        dependencies.append(
            DependencySignal(
                name=parts[0],
                ecosystem=DependencyEcosystem.GO,
                requirement=parts[1],
                group=group,
                source_path=path,
            )
        )
    return _deduplicate_dependencies(dependencies)


def _deduplicate_dependencies(dependencies: list[DependencySignal]) -> list[DependencySignal]:
    unique: dict[tuple[str, str, str, str, str | None], DependencySignal] = {}
    for dependency in dependencies:
        key = (
            dependency.ecosystem.value,
            dependency.name,
            dependency.group,
            dependency.source_path,
            dependency.requirement,
        )
        unique[key] = dependency
    return [unique[key] for key in sorted(unique)]
