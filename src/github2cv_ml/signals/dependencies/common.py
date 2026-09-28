"""Shared dependency parser helpers."""

from github2cv_ml.domain.signals import DependencySignal


def deduplicate_dependencies(dependencies: list[DependencySignal]) -> list[DependencySignal]:
    """Return dependencies in a stable order without exact duplicates."""

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
