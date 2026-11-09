"""Live demo regression tests: cached JSON/ZIP, real model-call boundary and outputs."""

import json
from datetime import datetime, timezone
from zipfile import ZipFile

import pytest

from github2cv_ml.agent.runner import RepoAnalysis
from github2cv_ml.domain.evidence import EvidenceKind, EvidenceSource, RepoEvidence
from github2cv_ml.domain.profile import CandidateIdentity
from github2cv_ml.domain.repository import RepositoryRef
from github2cv_ml.profile_demo import (
    MAX_ARTIFACT_BYTES,
    generate_profile_artifacts,
    load_analysis,
    render_profile_preview,
)

SHA = "a" * 40


def _analysis(project: str, evidence_id: str, statement: str) -> RepoAnalysis:
    repo = RepositoryRef(
        owner="example", name=project,
        url=f"https://github.com/example/{project}",
        default_branch="main",
    )
    return RepoAnalysis(
        repository=repo,
        commit_sha=SHA,
        project_summary=f"{project} test project",
        status="completed",
        evidence=[RepoEvidence(
            id=evidence_id,
            repository=repo,
            category="architecture",
            observation=statement,
            source=EvidenceSource(
                kind=EvidenceKind.FILE,
                path="src/main.py", commit_sha=SHA,
                url=f"https://github.com/example/{project}/blob/{SHA}/src/main.py#L1-L4",
                line_start=1, line_end=4,
            ),
            captured_at=datetime(2026, 11, 9, tzinfo=timezone.utc),
        )],
        generated_at=datetime(2026, 11, 9, tzinfo=timezone.utc),
    )


class FakeLiveModel:
    def __init__(self):
        self.calls = []

    def complete(self, messages, *, tools=None, json_mode=False):
        assert tools is None
        assert json_mode
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload)
        if payload.get("task") == "skill_reconciliation":
            return {"content": json.dumps({"merges": []})}
        identifier = payload["observations"][0]["id"]
        project = payload["repository"]
        skill = "Nix Flakes" if project.endswith("/build") else "FastAPI"
        return {"content": json.dumps({
            "skills": [{"name": skill, "evidence_ids": [identifier]}],
            "project": {
                "paragraph": f"Репозиторий {project} использует {skill}.",
                "evidence_ids": [identifier],
            },
        }, ensure_ascii=False)}


def test_demo_reads_json_and_zip_and_exposes_real_model_proposals(tmp_path):
    api = _analysis("api", "a", "Implements HTTP API using FastAPI")
    build = _analysis("build", "b", "Builds reproducible Nix Flakes projects")
    raw = tmp_path / "analysis.json"
    raw.write_text(api.model_dump_json(), encoding="utf-8")
    archive = tmp_path / "cached_build.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("analysis.json", build.model_dump_json())

    assert load_analysis(raw).commit_sha == SHA
    assert load_analysis(archive).repository.name == "build"
    llm = FakeLiveModel()
    output = tmp_path / "demo"
    result = generate_profile_artifacts(
        [raw, archive],
        candidate=CandidateIdentity(github_login="example"),
        model=llm,
        output_dir=output,
    )
    assert [c.text for c in result.claims if c.kind == "skill"] == [
        "FastAPI", "Nix Flakes",
    ]
    assert len(llm.calls) == 3  # two repository calls and one reconciliation
    assert len([c for c in result.claims if c.kind == "project"]) == 2

    stored = json.loads((output / "candidate_profile.json").read_text("utf-8"))
    assert stored["candidate"]["github_login"] == "example"
    assert len(stored["claims"]) == 4
    trace = json.loads((output / "llm_profile_trace.json").read_text("utf-8"))
    assert len(trace) == 3
    assert [call["repository"] for call in trace if "repository" in call] == [
        "example/api", "example/build",
    ]
    assert trace[-1]["stage"] == "skill_reconciliation"
    assert trace[-1]["input_skills"] == ["FastAPI", "Nix Flakes"]
    assert trace[-1]["llm_proposed_merges"] == []
    assert trace[0]["evidence_count"] == 1
    assert trace[1]["llm_proposed_skills"] == ["Nix Flakes"]
    assert trace[0]["duration_seconds"] >= 0
    assert "Репозиторий example" in trace[0]["llm_proposed_project_paragraph"]
    assert "numbered_source" not in json.dumps(trace)

    preview = (output / "profile_preview.md").read_text("utf-8")
    assert "Навыки (LLM)" in preview
    assert "`Nix Flakes`" in preview
    assert "### example/api" in preview
    assert f"https://github.com/example/api/blob/{SHA}/src/main.py" in preview
    assert "не доказывают личное авторство" in preview


def test_demo_rejects_missing_analysis_member_and_oversized_artifacts(tmp_path):
    broken = tmp_path / "broken.zip"
    with ZipFile(broken, "w") as z:
        z.writestr("report.md", "not a snapshot")
    with pytest.raises(ValueError, match="lacks a valid analysis.json"):
        load_analysis(broken)

    too_large = tmp_path / "large.json"
    too_large.write_bytes(b"x" * (MAX_ARTIFACT_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        load_analysis(too_large)
    zipbomb = tmp_path / "too_large.zip"
    with ZipFile(zipbomb, "w") as z:
        z.writestr("analysis.json", b"x" * (MAX_ARTIFACT_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        load_analysis(zipbomb)


def test_demo_never_writes_partial_outputs_on_model_failure(tmp_path):
    api = _analysis("api", "a", "Uses FastAPI")
    path = tmp_path / "analysis.json"
    path.write_text(api.model_dump_json(), encoding="utf-8")

    class BrokenModel:
        def complete(self, messages, *, tools=None, json_mode=False):
            return {"content": '{"skills": []}'}

    output = tmp_path / "demo"
    with pytest.raises(ValueError, match="invalid JSON/schema"):
        generate_profile_artifacts(
            [path], candidate=CandidateIdentity(github_login="developer"),
            model=BrokenModel(), output_dir=output,
        )
    assert not output.exists()


def test_demo_requires_input_and_keeps_external_urls_as_plain_text(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        generate_profile_artifacts(
            [], candidate=CandidateIdentity(github_login="developer"),
            model=FakeLiveModel(), output_dir=tmp_path / "out",
        )
    output = tmp_path / "analysis.json"
    analysis = _analysis("api", "a", "Uses FastAPI")
    output.write_text(analysis.model_dump_json(), encoding="utf-8")
    profile = generate_profile_artifacts(
        [output], candidate=CandidateIdentity(github_login="developer"),
        model=FakeLiveModel(), output_dir=tmp_path / "out",
    )
    preview = render_profile_preview(profile, [analysis])
    assert "[src/main.py](https://github.com/example/api/" in preview
