import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from github2cv_ml.domain import ResumeDocument, ResumeItem

EXAMPLES = json.loads(
    (Path(__file__).parents[2] / "examples" / "domain_contracts.json").read_text()
)


def test_resume_document_example_validates() -> None:
    ResumeDocument.model_validate(EXAMPLES["resume_document"])


def test_resume_item_rejects_blank_claim_id() -> None:
    with pytest.raises(ValidationError):
        ResumeItem.model_validate(
            {
                "text": "Built a Python API.",
                "claim_ids": [""],
            }
        )
