"""Python dependency manifest parsers."""

import re
import tomllib
from pathlib import PurePosixPath
from typing import Any

from github2cv_ml.domain.signals import DependencyEcosystem, DependencySignal
from github2cv_ml.signals.dependencies.common import deduplicate_dependencies

_PYTHON_REQUIREMENT_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]+\])?\s*(.*)$"
)


def parse_pyproject(path: str, content: str) -> list[DependencySignal]:
    """Extract PEP 621 and common Poetry dependency declarations."""

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

    return deduplicate_dependencies(dependencies)


def parse_requirements(path: str, content: str) -> list[DependencySignal]:
    """Extract common requirement lines while ignoring pip options/includes."""

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
    return deduplicate_dependencies(dependencies)


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
