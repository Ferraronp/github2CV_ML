import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from github2cv_ml.domain import RepoSnapshot, RepoTreeEntry, RepoTreeEntryKind

EXAMPLES = json.loads(
    (Path(__file__).parents[2] / "examples" / "domain_contracts.json").read_text()
)


def test_repo_snapshot_example_validates() -> None:
    RepoSnapshot.model_validate(EXAMPLES["repo_snapshot"])


def test_snapshot_rejects_raw_github_fields() -> None:
    payload = {**EXAMPLES["repo_snapshot"], "raw_github_response": {"id": 123}}

    with pytest.raises(ValidationError):
        RepoSnapshot.model_validate(payload)


def test_repo_tree_entry_preserves_path_whitespace() -> None:
    path = " docs/file.txt "

    entry = RepoTreeEntry(path=path, kind=RepoTreeEntryKind.FILE)

    assert entry.path == path


def test_repo_tree_entry_rejects_whitespace_only_path() -> None:
    with pytest.raises(ValidationError):
        RepoTreeEntry(path="   ", kind=RepoTreeEntryKind.FILE)
