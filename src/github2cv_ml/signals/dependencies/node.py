"""Node/npm dependency manifest parser."""

import json

from github2cv_ml.domain.signals import DependencyEcosystem, DependencySignal
from github2cv_ml.signals.dependencies.common import deduplicate_dependencies


def parse_package_json(path: str, content: str) -> list[DependencySignal]:
    """Extract dependency groups from package.json."""

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
    return deduplicate_dependencies(dependencies)
