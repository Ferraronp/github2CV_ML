import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from github2cv_ml.domain import CandidateProfile, RepoEvidence, RepoSnapshot, ResumeDocument


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
