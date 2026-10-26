"""Deterministic source-map planning, not an inferred architecture description."""

from __future__ import annotations

from pathlib import PurePosixPath

ROLE_ORDER = ("pipeline", "retrieval", "generation", "entrypoint", "storage", "core", "evaluation")


def _role_for(path: str) -> str:
    name = PurePosixPath(path).stem.lower()
    lower = path.lower()
    if "test" in lower or "evaluat" in lower or "benchmark" in lower:
        return "evaluation"
    if any(token in name for token in ("pipeline", "workflow", "orchestrat", "engine")):
        return "pipeline"
    if any(token in name for token in ("retriev", "rerank", "search", "vector", "embed")):
        return "retrieval"
    if any(token in name for token in ("generat", "writer", "inference", "llm")):
        return "generation"
    if name in {"main", "app", "api", "server", "cli"} or any(
        token in name for token in ("route", "handler", "controller")
    ):
        return "entrypoint"
    if any(token in name for token in ("database", "storage", "store", "repository", "db")):
        return "storage"
    return "core"


def build_exploration_plan(paths: list[str], limit: int = 7) -> list[dict[str, str]]:
    """Select role-diverse candidate files from the safe source tree.

    Roles are *filename heuristics*, not supported claims. The model should
    inspect the listed files before assuming they implement those roles.
    """
    if limit < 1:
        raise ValueError("Plan limit must be positive")

    buckets: dict[str, list[str]] = {role: [] for role in ROLE_ORDER}
    for path in sorted(set(paths)):
        role = _role_for(path)
        buckets[role].append(path)

    def score(path: str) -> tuple[int, int, int, str]:
        p = PurePosixPath(path)
        source = path.startswith(("src/", "backend/", "frontend/src/", "app/", "lib/"))
        boilerplate = p.stem.lower() in {"__init__", "conftest", "types"}
        return (1 if boilerplate else 0, 0 if source else 1, len(p.parts), path)

    for bucket in buckets.values():
        bucket.sort(key=score)

    selected: list[dict[str, str]] = []
    # First pass: distribute attention over distinct potential components.
    # Further passes: add additional candidates when budget allows.
    while len(selected) < limit:
        selected_any = False
        for role in ROLE_ORDER:
            if buckets[role] and len(selected) < limit:
                selected.append({"path": buckets[role].pop(0), "candidate_role": role})
                selected_any = True
        if not selected_any:
            break
    return selected
