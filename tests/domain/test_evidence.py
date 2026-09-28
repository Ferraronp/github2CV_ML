import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from github2cv_ml.domain import EvidenceSource, RepoEvidence

EXAMPLES = json.loads(
    (Path(__file__).parents[2] / "examples" / "domain_contracts.json").read_text()
)


def test_repo_evidence_example_validates() -> None:
    RepoEvidence.model_validate(EXAMPLES["repo_evidence"])


def test_evidence_source_requires_locator() -> None:
    with pytest.raises(ValidationError):
        EvidenceSource.model_validate({"kind": "file"})


def test_evidence_source_lines_require_path() -> None:
    with pytest.raises(ValidationError):
        EvidenceSource.model_validate(
            {
                "kind": "file",
                "url": "https://github.com/octocat/hello-world/blob/main/README.md",
                "line_start": 10,
            }
        )


def test_evidence_source_rejects_blank_commit_sha_locator() -> None:
    with pytest.raises(ValidationError):
        EvidenceSource.model_validate(
            {
                "kind": "commit",
                "commit_sha": "       ",
            }
        )
