"""Regression cases derived from real Qwen/Gemma Colab failure shapes."""

import json

import pytest
from test_cv_writer import profile

from github2cv_ml.cv_writer import write_resume
from github2cv_ml.model_recovery import RecoveringModel
from github2cv_ml.profile_builder import RepositoryProposal

EVIDENCE_A = "ev-05626e77ac47874b"
EVIDENCE_B = "ev-f2a8a7976431d384"
UNKNOWN_EVIDENCE = "ev-e32ad8ac6429d3b2"
PROJECT_CLAIM = "claim-project-c2044627e8546295"


class Responses:
    def __init__(self, *values):
        self.answers = list(values)
        self.calls = []

    def complete(self, messages, *, tools=None, json_mode=False):
        self.calls.append({
            "system": messages[0]["content"],
            "tools": tools,
            "json_mode": json_mode,
        })
        assert self.answers, "Model was unexpectedly queried again"
        answer = self.answers.pop(0)
        return {"content": json.dumps(answer, ensure_ascii=False)}


def request(observations=None):
    observations = observations or [
        {"id": EVIDENCE_A, "observation": "Retrieval uses Qdrant"},
        {"id": EVIDENCE_B, "observation": "Answers are reranked"},
    ]
    return [
        {"role": "system", "content": "Write evidence-backed repository JSON"},
        {"role": "user", "content": json.dumps({
            "repository": "ferraronp/military-charters-rag",
            "observations": observations,
        })},
    ]


def proposal(*, skills=None, project_ids=None):
    return {
        "skills": (
            [{"name": "Qdrant", "evidence_ids": [EVIDENCE_A]}]
            if skills is None else skills
        ),
        "project": {
            "paragraph": "Проект содержит RAG и поиск через Qdrant.",
            "evidence_ids": [EVIDENCE_A] if project_ids is None else project_ids,
        },
    }


def test_qwen_unknown_evidence_recovered_only_with_real_input_id():
    llm = Responses(
        proposal(skills=[{"name": "Qdrant", "evidence_ids": [UNKNOWN_EVIDENCE]}]),
        proposal(),
    )
    recovering = RecoveringModel(llm)
    response = recovering.complete(request(), json_mode=True)
    validated = RepositoryProposal.model_validate(json.loads(response["content"]))
    assert validated.skills[0].evidence_ids == [EVIDENCE_A]
    assert len(llm.calls) == 2
    assert "Unknown evidence_ids" in llm.calls[1]["system"]
    assert len(recovering.events) == 1


def test_gemma_missing_project_ids_then_empty_skill_recovered_by_omission():
    llm = Responses(
        {"skills": [{"name": "Qdrant", "evidence_ids": [EVIDENCE_A]}],
         "project": {"paragraph": "RAG работает."}},
        proposal(skills=[{"name": "FastAPI", "evidence_ids": []}]),
        proposal(skills=[
            {"name": "Qdrant", "evidence_ids": [EVIDENCE_A]},
            {"name": "FastAPI", "evidence_ids": []},
        ]),
    )
    recovering = RecoveringModel(llm)
    data = json.loads(recovering.complete(request(), json_mode=True)["content"])
    assert data["project"]["evidence_ids"] == [EVIDENCE_A]
    assert [item["name"] for item in data["skills"]] == ["Qdrant"]
    assert recovering.events[-1]["kind"] == "omitted_unsupported_skills"
    assert recovering.events[-1]["skill_names"] == ["FastAPI"]
    assert len(llm.calls) == 3
    assert "project.evidence_ids" in llm.calls[1]["system"]
    assert "skills.0.evidence_ids" in llm.calls[2]["system"]


def test_bad_project_evidence_still_fails_closed_after_all_retries():
    invalid = proposal(project_ids=[UNKNOWN_EVIDENCE])
    llm = Responses(invalid, invalid, invalid)
    recovering = RecoveringModel(llm)
    answer = recovering.complete(request(), json_mode=True)
    with pytest.raises(ValueError, match="Unknown evidence_ids"):
        recovering._supported_proposal(
            json.loads(answer["content"]), {EVIDENCE_A, EVIDENCE_B}
        )
    assert not any(
        event["kind"] == "omitted_unsupported_skills"
        for event in recovering.events
    )


def test_invalid_alias_reconciliation_keeps_all_original_skills():
    data = {"task": "skill_reconciliation", "skills": [
        {"name": "Groq API"}, {"name": "LLM"}, {"name": "Groq"},
    ]}
    llm = Responses({"merges": [
        {"canonical": "Groq API", "aliases": ["Groq API"],
         "reason": "These two spellings refer to the same API."},
        {"canonical": "LLM", "aliases": ["LLM"],
         "reason": "Identical tokens are not an alias group."},
        {"canonical": "Groq API", "aliases": ["Groq"],
         "reason": "Both names identify the same provider API."},
        {"canonical": "Imaginary", "aliases": ["Groq"],
         "reason": "Invented name must never enter candidate skills."},
    ]})
    recovering = RecoveringModel(llm)
    messages = [
        {"role": "system", "content": "Reconcile identical tags only."},
        {"role": "user", "content": json.dumps(data)},
    ]
    output = json.loads(recovering.complete(messages, json_mode=True)["content"])
    assert len(output["merges"]) == 1
    assert output["merges"][0]["canonical"] == "Groq API"
    assert output["merges"][0]["aliases"] == ["Groq"]
    assert recovering.events[-1]["skipped"] == 3


def test_invalid_reconciliation_json_yields_no_merges_not_fake_aliases():
    class InvalidJson:
        def complete(self, messages, *, tools=None, json_mode=False):
            return {"content": "invalid {"}

    recovering = RecoveringModel(InvalidJson())
    messages = [
        {"role": "system", "content": "Reconcile tags"},
        {"role": "user", "content": json.dumps({
            "task": "skill_reconciliation", "skills": [{"name": "Python"}]
        })},
    ]
    assert json.loads(recovering.complete(messages, json_mode=True)["content"]) == {
        "merges": [],
    }


def test_gemma_claim_ids_not_printed_in_final_resume():
    input_profile = profile()
    input_profile.claims[3].id = PROJECT_CLAIM
    summary = {
        "summary": {
            "text": (
                "В портфолио представлены RAG-проекты "
                f"({PROJECT_CLAIM}). Также используется Python."
            ),
            "claim_ids": [PROJECT_CLAIM, "s1"],
        },
        "headline": None,
    }
    result = write_resume(input_profile, model=Responses(summary))
    assert PROJECT_CLAIM not in result.summary.text
    assert "()" not in result.summary.text
    assert "RAG-проекты" in result.summary.text
    assert PROJECT_CLAIM in result.summary.claim_ids
    assert result.summary.claim_ids.count(PROJECT_CLAIM) == 1


def test_gemma_inline_known_id_missing_from_array_is_added_not_guessed():
    input_profile = profile()
    input_profile.claims[3].id = PROJECT_CLAIM
    result = write_resume(input_profile, model=Responses({
        "summary": {
            "text": f"Проекты включают API ({PROJECT_CLAIM}).",
            "claim_ids": ["s1"],
        },
        "headline": None,
    }))
    assert result.summary.claim_ids == ["s1", PROJECT_CLAIM]
    assert "claim-" not in result.summary.text


def test_unknown_inline_claim_token_is_rejected():
    invalid = "claim-project-ffffffffffffffff"
    with pytest.raises(ValueError, match="Unknown inline"):
        write_resume(profile(), model=Responses({
            "summary": {"text": f"В коде найдено API ({invalid}).", "claim_ids": ["s1"]},
            "headline": None,
        }))


def test_persistent_invalid_cv_uses_conservative_summary_with_existing_id():
    input_profile = profile()
    llm = Responses(*[
        {"summary": {
            "text": "Выдуманный senior опыт.", "claim_ids": ["not-real"]
        }} for _ in range(3)
    ])
    recovering = RecoveringModel(llm)
    resume = write_resume(input_profile, model=recovering)
    assert resume.summary is not None
    assert "senior" not in resume.summary.text
    assert set(resume.summary.claim_ids) <= {c.id for c in input_profile.claims}
    assert recovering.events[-1]["kind"] == "conservative_fallback"


def test_regular_tool_calls_do_not_trigger_schema_repair():
    class Explorer:
        def complete(self, messages, *, tools=None, json_mode=False):
            return {"tool_calls": [{"function": {"name": "read_file"}}]}

    recovering = RecoveringModel(Explorer())
    output = recovering.complete(
        [{"role": "system", "content": "Explore repo"},
         {"role": "user", "content": json.dumps({
             "repository": "example/api", "observations": []
         })}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    assert output["tool_calls"][0]["function"]["name"] == "read_file"
    assert recovering.events == []


def test_gemma_square_bracket_claim_citations_are_not_visible():
    """Real 03743d68 zip embeds one or several claims in [brackets]."""
    p = profile()
    p.claims[3].id = PROJECT_CLAIM
    skill_id = "claim-skill-1111111111111111"
    p.claims[0].id = skill_id
    llm = Responses({
        "summary": {
            "text": (
                f"RAG использует векторный поиск [{PROJECT_CLAIM}]. "
                f"Технологии доступны в проектах [{PROJECT_CLAIM}, {skill_id}]."
            ),
            "claim_ids": [PROJECT_CLAIM, skill_id],
        },
        "headline": None,
    })
    resume = write_resume(p, model=llm)
    assert "claim-" not in resume.summary.text
    assert "[" not in resume.summary.text
    assert "]" not in resume.summary.text
    assert "[]" not in resume.summary.text
    assert set(resume.summary.claim_ids) == {PROJECT_CLAIM, skill_id}


def test_gemma_inline_evidence_id_stays_in_metadata_not_profile_or_cv():
    """The 03743d68 RAG project had [ev-9fb871477ccac05e] in its prose."""
    from test_pipeline import FakeGitHub, FakeThreeStageModel
    from test_pipeline import request as pipeline_request

    from github2cv_ml.pipeline import PipelineStatus, run_pipeline

    class InlineEvidence(FakeThreeStageModel):
        def complete(self, messages, *, tools=None, json_mode=False):
            answer = super().complete(messages, tools=tools, json_mode=json_mode)
            if "You create an evidence-based technical candidate profile" in messages[0]["content"]:
                parsed = json.loads(answer["content"])
                parsed["project"]["paragraph"] += (
                    f" [{parsed['project']['evidence_ids'][0]}]"
                )
                return {"content": json.dumps(parsed, ensure_ascii=False)}
            return answer

    result = run_pipeline(
        pipeline_request("example/api", max_steps=4, max_reads=2,
                         min_planned_reads=1),
        github_client=FakeGitHub(), model=InlineEvidence(),
    )
    assert result.status == PipelineStatus.COMPLETED
    assert result.resume is not None
    project_claims = [claim for claim in result.profile.claims if claim.kind == "project"]
    assert len(project_claims) == 1
    assert project_claims[0].evidence_ids
    assert "ev-" not in project_claims[0].text
    project_section = next(x for x in result.resume.sections if x.title == "Проекты")
    assert "ev-" not in project_section.items[0].text


def test_unreferenced_inline_evidence_id_is_not_silently_dropped():
    bad = proposal()
    bad["project"]["paragraph"] += f" [{UNKNOWN_EVIDENCE}]"
    llm = Responses(bad, bad, bad)
    recovery = RecoveringModel(llm)
    answer = recovery.complete(request(), json_mode=True)
    with pytest.raises(ValueError, match="Unreferenced inline evidence_ids"):
        recovery._supported_proposal(
            json.loads(answer["content"]), {EVIDENCE_A, EVIDENCE_B}
        )
    assert len(recovery.events) == 3
