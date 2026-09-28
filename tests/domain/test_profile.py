import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from github2cv_ml.domain import CandidateClaim, CandidateProfile

EXAMPLES = json.loads(
    (Path(__file__).parents[2] / "examples" / "domain_contracts.json").read_text()
)


def test_candidate_profile_example_validates() -> None:
    CandidateProfile.model_validate(EXAMPLES["candidate_profile"])


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


def test_candidate_profile_rejects_duplicate_claim_ids() -> None:
    claim = EXAMPLES["candidate_profile"]["claims"][0]
    payload = {
        **EXAMPLES["candidate_profile"],
        "claims": [claim, claim],
    }

    with pytest.raises(ValidationError):
        CandidateProfile.model_validate(payload)
