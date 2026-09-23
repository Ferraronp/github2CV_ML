import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from github2cv_ml.domain import (
    CandidateClaim,
    CandidateProfile,
    EvidenceSource,
    RepoEvidence,
    RepoSnapshot,
    ResumeDocument,
    ResumeItem,
)

EXAMPLES = json.loads(
    (Path(__file__).parents[1] / "examples" / "domain_contracts.json").read_text()
)


@pytest.mark.parametrize(
    ("key", "model"),
    [
        ("repo_snapshot", RepoSnapshot),
        ("repo_evidence", RepoEvidence),
        ("candidate_profile", CandidateProfile),
        ("resume_document", ResumeDocument),
    ],
)
def test_json_examples_validate(key: str, model: type) -> None:
    model.model_validate(EXAMPLES[key])


def test_snapshot_rejects_raw_github_fields() -> None:
    payload = {**EXAMPLES["repo_snapshot"], "raw_github_response": {"id": 123}}

    with pytest.raises(ValidationError):
        RepoSnapshot.model_validate(payload)


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


def test_candidate_claim_rejects_blank_evidence_id() -> None:
    with pytest.raises(ValidationError):
        CandidateClaim.model_validate(
            {
                "id": "claim-python-api",
                "kind": "project",
                "text": "Built a Python API.",
                "evidence_ids": ["   "],
            }
        )


def test_resume_item_rejects_blank_claim_id() -> None:
    with pytest.raises(ValidationError):
        ResumeItem.model_validate(
            {
                "text": "Built a Python API.",
                "claim_ids": [""],
            }
        )


def test_candidate_profile_rejects_duplicate_claim_ids() -> None:
    claim = EXAMPLES["candidate_profile"]["claims"][0]
    payload = {
        **EXAMPLES["candidate_profile"],
        "claims": [claim, claim],
    }

    with pytest.raises(ValidationError):
        CandidateProfile.model_validate(payload)


def test_example_provenance_chain_is_connected() -> None:
    evidence = RepoEvidence.model_validate(EXAMPLES["repo_evidence"])
    profile = CandidateProfile.model_validate(EXAMPLES["candidate_profile"])
    resume = ResumeDocument.model_validate(EXAMPLES["resume_document"])

    evidence_ids = {evidence.id}
    claim_ids = {claim.id for claim in profile.claims}

    assert all(set(claim.evidence_ids) <= evidence_ids for claim in profile.claims)
    assert resume.headline is not None
    assert set(resume.headline.claim_ids) <= claim_ids
    assert resume.summary is not None
    assert set(resume.summary.claim_ids) <= claim_ids
    assert all(set(item.claim_ids) <= claim_ids for section in resume.sections for item in section.items)
