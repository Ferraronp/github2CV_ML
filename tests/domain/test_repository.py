import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from github2cv_ml.domain import RepoSnapshot

EXAMPLES = json.loads(
    (Path(__file__).parents[2] / "examples" / "domain_contracts.json").read_text()
)


def test_repo_snapshot_example_validates() -> None:
    RepoSnapshot.model_validate(EXAMPLES["repo_snapshot"])


def test_snapshot_rejects_raw_github_fields() -> None:
    payload = {**EXAMPLES["repo_snapshot"], "raw_github_response": {"id": 123}}

    with pytest.raises(ValidationError):
        RepoSnapshot.model_validate(payload)
