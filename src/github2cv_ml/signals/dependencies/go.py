"""Go module dependency parser."""

from github2cv_ml.domain.signals import DependencyEcosystem, DependencySignal
from github2cv_ml.signals.dependencies.common import deduplicate_dependencies


def parse_go_mod(path: str, content: str) -> list[DependencySignal]:
    """Extract direct and indirect require declarations from go.mod."""

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
    return deduplicate_dependencies(dependencies)
