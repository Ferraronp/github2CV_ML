"""Deterministic source-map planning, not an inferred architecture description."""

from __future__ import annotations

from pathlib import PurePosixPath

# Prefer implementation/behavior over thin API wrappers, configuration and tests.
PRIMARY_ROLES = (
    "pipeline", "retrieval", "generation", "service", "data_access",
    "security", "infrastructure", "entrypoint",
)
SECONDARY_ROLES = ("core", "evaluation", "migration")
ROLE_ORDER = PRIMARY_ROLES + SECONDARY_ROLES


def _role_for(path: str) -> str:
    p = PurePosixPath(path)
    name = p.stem.lower()
    parents = {part.lower() for part in p.parts[:-1]}
    # Migration filenames often contain "pipeline", "generation" or "service",
    # but schema changes are not proof of application implementations.
    if parents & {"migrations", "alembic", "versions"}:
        return "migration"
    if (
        "tests" in parents or "test" in parents or "evaluation" in parents
        or name.startswith("test_") or "benchmark" in name or name.startswith("eval_")
    ):
        return "evaluation"
    if any(token in name for token in ("pipeline", "workflow", "orchestrat", "engine")):
        return "pipeline"
    if any(token in name for token in ("retriev", "rerank", "search", "vector", "embed")):
        return "retrieval"
    if any(token in name for token in ("generat", "writer", "inference", "llm")):
        return "generation"
    if (
        name in {"service", "services", "usecase", "use_case"}
        or name.endswith(("_service", "_services"))
    ):
        return "service"
    if name in {"repo", "repository", "dao", "models"} or name.endswith(
        ("_repo", "_repository", "_dao", "_models")
    ):
        return "data_access"
    if (
        name in {"security", "auth", "tokens", "jwt", "permissions"}
        or any(token in name for token in ("password", "crypto"))
    ):
        return "security"
    if (
        (name in {"session", "client", "connection", "pool"} and parents
         & {"db", "database", "redis", "cache", "storage", "postgres"})
        or name in {"redis", "database", "db", "storage", "cache"}
    ):
        return "infrastructure"
    if name in {"main", "app", "api", "server", "cli"} or any(
        token in name for token in ("route", "handler", "controller")
    ):
        return "entrypoint"
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

    def score(path: str) -> tuple[int, int, int, int, str]:
        p = PurePosixPath(path)
        source = path.startswith(("src/", "backend/", "frontend/src/", "app/", "lib/"))
        boilerplate = p.stem.lower() in {"__init__", "conftest", "types"}
        # A repository/query implementation is more informative than ORM
        # declarations alone when we have to choose one data-access file.
        data_glue = p.stem.lower() in {"models", "model", "schemas"}
        return (
            1 if boilerplate else 0, 1 if data_glue else 0,
            0 if source else 1, len(p.parts), path,
        )

    for bucket in buckets.values():
        bucket.sort(key=score)

    selected: list[dict[str, str]] = []

    def take_one(role: str) -> None:
        if buckets[role] and len(selected) < limit:
            selected.append({
                "path": buckets[role].pop(0), "candidate_role": role,
            })

    # Cover distinct implementation layers before adding more files from
    # the same layer. Routers alone do not demonstrate auth/business logic.
    for role in PRIMARY_ROLES:
        take_one(role)

    # Prefer a second service/repository/infra module to boilerplate when
    # the budget cannot accommodate the entire repository.
    while len(selected) < limit and any(buckets[role] for role in PRIMARY_ROLES):
        for role in PRIMARY_ROLES:
            take_one(role)
            if len(selected) >= limit:
                break

    # Migrations, tests and core utilities are useful, but must not displace
    # source implementations when a read budget is tight.
    # Tests and core utilities are useful, but should not displace service
    # implementations if the budget is small.
    while len(selected) < limit and any(buckets[role] for role in SECONDARY_ROLES):
        for role in SECONDARY_ROLES:
            take_one(role)
    return selected
