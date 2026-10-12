"""Regression tests for Python requirements extraction."""

from github2cv_ml.signals.dependencies import parse_dependency_manifest


def test_pinned_and_unpinned_requirements_sort_and_deduplicate() -> None:
    dependencies = parse_dependency_manifest(
        "requirements.txt",
        "requests>=2\nrequests\nrequests>=2\nrequests\n",
    )

    assert [(item.name, item.requirement) for item in dependencies] == [
        ("requests", None),
        ("requests", ">=2"),
    ]
    assert all(item.source_path == "requirements.txt" for item in dependencies)


def test_bare_urls_and_vcs_sources_do_not_become_fake_dependencies() -> None:
    dependencies = parse_dependency_manifest(
        "requirements.txt",
        (
            "https://example.com/pkg.whl\n"
            "http://example.com/pkg.tar.gz\n"
            "git+https://github.com/org/repo.git\n"
            "git+ssh://git@github.com/org/repo.git\n"
            "git@github.com:org/repo.git\n"
            "requests>=2\n"
        ),
    )

    assert [(item.name, item.requirement) for item in dependencies] == [
        ("requests", ">=2")
    ]


def test_named_direct_references_and_egg_sources_are_preserved() -> None:
    dependencies = parse_dependency_manifest(
        "requirements.txt",
        (
            "widget @ https://example.com/widget.whl\n"
            "library @ git+https://github.com/org/library.git\n"
            "git+https://github.com/org/legacy.git#egg=legacy\n"
        ),
    )

    assert {(item.name, item.requirement) for item in dependencies} == {
        ("widget", "@ https://example.com/widget.whl"),
        ("library", "@ git+https://github.com/org/library.git"),
        ("legacy", "git+https://github.com/org/legacy.git#egg=legacy"),
    }


def test_poetry_empty_or_whitespace_names_do_not_hide_valid_dependencies() -> None:
    dependencies = parse_dependency_manifest(
        "pyproject.toml",
        """
[tool.poetry.dependencies]
python = "^3.11"
"" = "^1.0"
"   " = "^2.0"
requests = "^2.32"

[tool.poetry.group.dev.dependencies]
"" = "^1.0"
pytest = "^8"
""",
    )

    assert [(item.name, item.requirement, item.group)
            for item in dependencies] == [
        ("pytest", "^8", "group:dev"),
        ("requests", "^2.32", "runtime"),
    ]
