import json
from pathlib import Path

from github2cv_ml.domain import CandidateProfile, RepoEvidence, ResumeDocument

EXAMPLES = json.loads(
    (Path(__file__).parents[2] / "examples" / "domain_contracts.json").read_text()
)


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
