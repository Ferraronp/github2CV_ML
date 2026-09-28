from github2cv_ml.domain import (
    DependencyEcosystem,
    ManifestKind,
    RepoTreeEntry,
    RepoTreeEntryKind,
    StructureKind,
    ToolingKind,
)
from github2cv_ml.signals import dependency_manifest_paths, extract_repository_signals


def _file(path: str) -> RepoTreeEntry:
    return RepoTreeEntry(path=path, kind=RepoTreeEntryKind.FILE)


def _directory(path: str) -> RepoTreeEntry:
    return RepoTreeEntry(path=path, kind=RepoTreeEntryKind.DIRECTORY)


def test_extracts_python_dependency_and_structure_signals() -> None:
    tree = [
        _file("pyproject.toml"),
        _file("requirements-dev.txt"),
        _directory("src"),
        _directory("tests"),
    ]
    contents = {
        "pyproject.toml": """
[project]
dependencies = ["fastapi>=0.115", "pydantic[email]>=2.11"]

[project.optional-dependencies]
dev = ["pytest>=8"]
""",
        "requirements-dev.txt": "ruff>=0.12\n",
    }

    signals = extract_repository_signals(tree, contents)

    assert [(item.kind, item.path) for item in signals.manifests] == [
        (ManifestKind.PYPROJECT, "pyproject.toml"),
        (ManifestKind.REQUIREMENTS, "requirements-dev.txt"),
    ]
    assert {
        (item.name, item.ecosystem, item.group, item.requirement)
        for item in signals.dependencies
    } == {
        ("fastapi", DependencyEcosystem.PYTHON, "runtime", ">=0.115"),
        ("pydantic", DependencyEcosystem.PYTHON, "runtime", "[email]>=2.11"),
        ("pytest", DependencyEcosystem.PYTHON, "optional:dev", ">=8"),
        ("ruff", DependencyEcosystem.PYTHON, "dev", ">=0.12"),
    }
    assert [(item.kind, item.path) for item in signals.structures] == [
        (StructureKind.SOURCE, "src"),
        (StructureKind.TESTS, "tests"),
    ]


def test_extracts_node_tooling_and_frontend_signals() -> None:
    tree = [
        _file("package.json"),
        _file("Dockerfile"),
        _file("compose.yaml"),
        _file(".github/workflows/ci.yml"),
        _directory("frontend"),
    ]
    contents = {
        "package.json": """
{
  "dependencies": {"react": "^19.0.0"},
  "devDependencies": {"vite": "^7.0.0"},
  "peerDependencies": {"typescript": ">=5"}
}
"""
    }

    signals = extract_repository_signals(tree, contents)

    assert {
        (item.name, item.group, item.requirement) for item in signals.dependencies
    } == {
        ("react", "runtime", "^19.0.0"),
        ("vite", "dev", "^7.0.0"),
        ("typescript", "peer", ">=5"),
    }
    assert [(item.kind, item.path) for item in signals.tooling] == [
        (ToolingKind.GITHUB_ACTIONS, ".github/workflows/ci.yml"),
        (ToolingKind.DOCKERFILE, "Dockerfile"),
        (ToolingKind.DOCKER_COMPOSE, "compose.yaml"),
    ]
    assert [(item.kind, item.path) for item in signals.structures] == [
        (StructureKind.FRONTEND, "frontend")
    ]


def test_extracts_go_dependencies_and_layout() -> None:
    tree = [
        _file("go.mod"),
        _directory("cmd"),
        _directory("internal"),
        _directory("pkg"),
    ]
    contents = {
        "go.mod": """
module example.com/service

go 1.25

require (
    github.com/gin-gonic/gin v1.10.0
    golang.org/x/text v0.29.0 // indirect
)
"""
    }

    signals = extract_repository_signals(tree, contents)

    assert {
        (item.name, item.ecosystem, item.group, item.requirement)
        for item in signals.dependencies
    } == {
        ("github.com/gin-gonic/gin", DependencyEcosystem.GO, "runtime", "v1.10.0"),
        ("golang.org/x/text", DependencyEcosystem.GO, "indirect", "v0.29.0"),
    }
    assert [(item.kind, item.path) for item in signals.structures] == [
        (StructureKind.GO_CMD, "cmd"),
        (StructureKind.GO_INTERNAL, "internal"),
        (StructureKind.GO_PKG, "pkg"),
    ]


def test_unknown_and_malformed_formats_do_not_break_extraction() -> None:
    tree = [
        _file("pyproject.toml"),
        _file("package.json"),
        _file("Cargo.toml"),
        _file("notes.txt"),
    ]
    contents = {
        "pyproject.toml": "not = [valid toml",
        "package.json": "{not json}",
        "Cargo.toml": "[dependencies]",
        "notes.txt": "ignored",
    }

    signals = extract_repository_signals(tree, contents)

    assert signals.dependencies == []
    assert [(item.kind, item.path) for item in signals.manifests] == [
        (ManifestKind.PACKAGE_JSON, "package.json"),
        (ManifestKind.PYPROJECT, "pyproject.toml"),
    ]


def test_only_supported_manifest_paths_are_requested() -> None:
    tree = [
        _file("services/api/pyproject.toml"),
        _file("web/package.json"),
        _file("go.mod"),
        _file("Cargo.toml"),
        _file("package-lock.json"),
    ]

    assert dependency_manifest_paths(tree) == [
        "go.mod",
        "services/api/pyproject.toml",
        "web/package.json",
    ]
