"""Regression tests for malformed dependency entries in npm manifests."""

from github2cv_ml.signals.dependencies import parse_dependency_manifest


def test_invalid_npm_dependency_names_do_not_hide_valid_packages() -> None:
    dependencies = parse_dependency_manifest(
        "web/package.json",
        """{
  "dependencies": {"": "^1.0", "  ": "^2.0", "react": "^19"},
  "devDependencies": {"": "^1.0", "vite": "^7"},
  "optionalDependencies": {"": "^1.0", "fsevents": "^2"}
}""",
    )

    assert [(item.name, item.requirement, item.group, item.source_path)
            for item in dependencies] == [
        ("fsevents", "^2", "optional", "web/package.json"),
        ("react", "^19", "runtime", "web/package.json"),
        ("vite", "^7", "dev", "web/package.json"),
    ]


def test_invalid_npm_dependency_values_do_not_hide_valid_packages() -> None:
    dependencies = parse_dependency_manifest(
        "package.json",
        '{"dependencies": {"bad": null, "also-bad": 7, "react": "^19"}}',
    )

    assert [(item.name, item.requirement) for item in dependencies] == [
        ("react", "^19")
    ]
