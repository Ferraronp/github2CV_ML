"""Offline vertical-slice fixture for the backend-friendly MVP pipeline.

Tests use the REAL collector, browser, RepoAnalyzer, profile builder and
CV Writer, with only GitHub HTTP and the language model replaced by fakes.
"""

import json

import pytest

from github2cv_ml.agent.runner import RepoAnalysis
from github2cv_ml.domain.profile import CandidateIdentity
from github2cv_ml.pipeline import (
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
    normalize_repository,
    run_pipeline,
)
from github2cv_ml.pipeline_artifacts import write_pipeline_artifacts

SHA = "a" * 40
CODE = "from fastapi import FastAPI\napp = FastAPI()\n"


class FakeGitHub:
    def __init__(self, failing=()):
        self.failing = set(failing)
        self.branch_refs = []
        self.source_refs = []

    def get_repository(self, owner, name):
        if name in self.failing:
            raise RuntimeError("GitHub unavailable for test repo")
        return {
            "default_branch": "main",
            "owner": {"login": owner},
            "name": name,
            "html_url": f"https://github.com/{owner}/{name}",
            "description": "Small API demo",
            "topics": [],
        }

    def get_branch_head_sha(self, owner, name, branch):
        self.branch_refs.append((owner, name, branch))
        return SHA

    def get_languages(self, owner, name):
        return {"Python": len(CODE)}

    def get_readme(self, owner, name):
        raise AssertionError("Frozen collection must not fetch moving README")

    def get_tree(self, owner, name, ref):
        assert ref == SHA
        return {
            "truncated": False,
            "tree": [{"path": "src/main.py", "type": "blob", "size": len(CODE)}],
        }

    def get_file(self, owner, name, path, ref):
        self.source_refs.append((owner, name, path, ref))
        assert ref == SHA
        return CODE if path == "src/main.py" else None


class FakeThreeStageModel:
    def __init__(self, *, no_evidence=False, broken_profile=False, broken_cv=False):
        self.calls = []
        self.no_evidence = no_evidence
        self.broken_profile = broken_profile
        self.broken_cv = broken_cv

    def complete(self, messages, *, tools=None, json_mode=False):
        system = messages[0]["content"]
        data = json.loads(messages[-1]["content"])
        self.calls.append({
            "stage": system[:60], "data": data,
            "tool": tools is not None, "json_mode": json_mode,
        })
        if tools is not None:
            previous = data["file_summaries"]
            name = "finish_analysis" if previous else "read_file"
            args = {} if previous else {
                "path": "src/main.py", "start_line": 1, "end_line": 2,
            }
            return {"tool_calls": [{"function": {
                "name": name, "arguments": json.dumps(args),
            }}]}
        if "Extract up to five" in system:
            return {"content": json.dumps({
                "file_summary": "A minimal FastAPI service",
                "observations": [] if self.no_evidence else [{
                    "claim": "Source defines a FastAPI app object.",
                    "category": "technology",
                    "source_path": data["source_path"],
                    "line_start": 2, "line_end": 2,
                }],
            })}
        if "Write a short project summary" in system:
            return {"content": json.dumps({
                "project_summary": "Simple HTTP API",
                "project_type": "API",
            })}
        if "You create an evidence-based technical candidate profile" in system:
            if self.broken_profile:
                return {"content": '{"skills": ['}
            evidence_id = data["observations"][0]["id"]
            repository = data["repository"]
            return {"content": json.dumps({
                "skills": [{"name": "FastAPI", "evidence_ids": [evidence_id]}],
                "project": {
                    "paragraph": f"Проект {repository} реализует API на FastAPI.",
                    "evidence_ids": [evidence_id],
                },
            }, ensure_ascii=False)}
        if "You reconcile skill-tag names" in system:
            return {"content": json.dumps({"merges": []})}
        if "You are a professional CV editor" in system:
            if self.broken_cv:
                return {"content": json.dumps({
                    "summary": {"text": "Придуманное утверждение.", "claim_ids": ["fake"]}
                })}
            project_ids = [c["id"] for c in data["claims"] if c["kind"] == "project"]
            skill_ids = [c["id"] for c in data["claims"] if c["kind"] == "skill"]
            return {"content": json.dumps({
                "summary": {
                    "text": "В портфолио представлены API-проекты на FastAPI.",
                    "claim_ids": skill_ids + project_ids,
                },
                "headline": None,
            }, ensure_ascii=False)}
        raise AssertionError(f"Unexpected stage: {system[:80]}")


def request(*repos, **kwargs):
    return PipelineRequest(
        candidate=CandidateIdentity(
            github_login="developer", display_name="Тестовый разработчик"
        ),
        repositories=list(repos),
        **kwargs,
    )


def test_real_vertical_slice_accepts_two_repos_and_preserves_all_provenance(tmp_path):
    github = FakeGitHub()
    model = FakeThreeStageModel()
    result = run_pipeline(
        request("https://github.com/example/api", "example/worker",
                max_steps=5, max_reads=2, min_planned_reads=1),
        github_client=github, model=model,
    )
    assert result.status == PipelineStatus.COMPLETED
    assert not result.problems
    assert len(result.snapshots) == 2
    assert len(result.analyses) == 2
    assert len(result.evidence_index) == 2
    assert result.profile is not None
    assert result.resume is not None
    assert result.resume.candidate_name == "Тестовый разработчик"
    assert len([c for c in result.profile.claims if c.kind == "project"]) == 2
    assert len([c for c in result.profile.claims if c.kind == "skill"]) == 1
    assert len(result.resume.sections) == 2
    assert result.resume.sections[0].title == "Ключевые навыки"
    assert len(result.resume.sections[1].items) == 2
    assert len(result.profile_trace) == 2  # same skill: no alias reconciliation
    assert len(result.cv_trace) == 1
    assert len(github.branch_refs) == 2
    assert all(ref[-1] == SHA for ref in github.source_refs)
    assert all(a.commit_sha == SHA for a in result.analyses)
    ids = set(result.evidence_index)
    assert all(set(claim.evidence_ids) <= ids for claim in result.profile.claims)
    claim_ids = {c.id for c in result.profile.claims}
    assert all(
        set(item.claim_ids) <= claim_ids
        for section in result.resume.sections for item in section.items
    )
    assert set(result.resume.summary.claim_ids) <= claim_ids

    output = tmp_path / "results"
    paths = write_pipeline_artifacts(result, output)
    assert "pipeline_result.json" in paths
    restored = PipelineResult.model_validate_json(
        (output / "pipeline_result.json").read_text("utf-8")
    )
    assert restored == result
    manifest = json.loads((output / "pipeline_manifest.json").read_text("utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["has_resume"] is True
    assert manifest["evidence_count"] == 2
    assert "cv/resume.json" in manifest["artifacts"]
    assert "provenance/evidence_index.json" in manifest["artifacts"]
    index = json.loads(
        (output / "provenance" / "evidence_index.json").read_text("utf-8")
    )
    assert set(index) == set(result.evidence_index)
    assert all(entry["source"]["commit_sha"] == SHA for entry in index.values())
    assert (output / "cv" / "resume_preview.md").is_file()
    assert (output / "profile" / "candidate_profile.json").is_file()
    assert len(list((output / "snapshots").glob("*.json"))) == 2
    assert len(list((output / "analyses").glob("*/analysis.json"))) == 2
    with pytest.raises(ValueError, match="must be empty"):
        write_pipeline_artifacts(result, output)  # no stale overwrite


def test_partial_github_failure_does_not_discard_healthy_repo():
    result = run_pipeline(
        request("example/down", "example/api", max_steps=5,
                max_reads=2, min_planned_reads=1),
        github_client=FakeGitHub(failing={"down"}),
        model=FakeThreeStageModel(),
    )
    assert result.status == PipelineStatus.PARTIAL
    assert result.resume is not None
    assert len(result.analyses) == 1
    assert result.problems[0].stage == PipelineStage.COLLECT
    assert result.problems[0].repository == "example/down"


def test_zero_evidence_fails_closed_without_fake_cv(tmp_path):
    result = run_pipeline(
        request("example/api", max_steps=4, max_reads=2, min_planned_reads=1),
        github_client=FakeGitHub(), model=FakeThreeStageModel(no_evidence=True),
    )
    assert result.status == PipelineStatus.FAILED
    assert result.profile is None and result.resume is None
    assert any(p.code == "no_evidence" for p in result.problems)
    output = tmp_path / "failed"
    write_pipeline_artifacts(result, output)
    assert (output / "pipeline_result.json").exists()
    assert not (output / "cv" / "resume.json").exists()


def test_failed_profile_preserves_analysis_and_identifies_stage(tmp_path):
    result = run_pipeline(
        request("example/api", max_steps=4, max_reads=2, min_planned_reads=1),
        github_client=FakeGitHub(),
        model=FakeThreeStageModel(broken_profile=True),
    )
    assert result.status == PipelineStatus.FAILED
    assert len(result.analyses) == 1
    assert result.profile is None and result.resume is None
    assert result.problems[-1].stage == PipelineStage.PROFILE
    assert result.problems[-1].code == "profile_failed"
    assert len(result.profile_trace) == 2  # failed response and retry
    assert len(result.profile_error_responses) == 2
    assert result.profile_error_responses[0]["raw_content"] == '{"skills": ['
    output = tmp_path / "failed"
    write_pipeline_artifacts(result, output)
    assert len(list((output / "analyses").glob("*/analysis.json"))) == 1
    assert not (output / "cv" / "resume.json").exists()
    debug = json.loads(
        (output / "profile" / "llm_profile_error.json").read_text("utf-8")
    )
    assert debug["problems"][0]["code"] == "profile_failed"
    assert len(debug["responses"]) == 2


def test_failed_cv_preserves_profile_with_stage_context():
    result = run_pipeline(
        request("example/api", max_steps=4, max_reads=2, min_planned_reads=1),
        github_client=FakeGitHub(),
        model=FakeThreeStageModel(broken_cv=True),
    )
    assert result.status == PipelineStatus.FAILED
    assert result.profile is not None
    assert result.resume is None
    assert result.problems[-1].stage == PipelineStage.RESUME
    assert result.problems[-1].code == "resume_failed"


def test_cached_analysis_is_reusable_without_github_network():
    initial = run_pipeline(
        request("example/api", max_steps=4, max_reads=2, min_planned_reads=1),
        github_client=FakeGitHub(),
        model=FakeThreeStageModel(),
    )
    cached = RepoAnalysis.model_validate_json(initial.analyses[0].model_dump_json())
    result = run_pipeline(
        request(), github_client=None,
        model=FakeThreeStageModel(),
        cached_analyses=[cached],
    )
    assert result.status == PipelineStatus.COMPLETED
    assert result.resume is not None
    assert result.snapshots == []  # cached result has no original RepoSnapshot
    assert result.analyses == [cached]
    assert set(result.evidence_index) == {cached.evidence[0].id}


def test_cached_and_fresh_repo_collision_fails_before_any_network():
    github = FakeGitHub()
    previous = run_pipeline(
        request("example/api", max_steps=4, max_reads=2, min_planned_reads=1),
        github_client=github, model=FakeThreeStageModel(),
    )
    with pytest.raises(ValueError, match="Conflicting fresh/cached"):
        run_pipeline(
            request("EXAMPLE/API"), github_client=github,
            model=FakeThreeStageModel(), cached_analyses=previous.analyses,
        )


@pytest.mark.parametrize("bad", [
    "http://github.com/user/repo",
    "https://evil.example/user/repo",
    "https://github.com/user/repo?x=1",
    "https://github.com/user/repo#readme",
    "https://user:password@github.com/user/repo",
    "https://github.com/user/repo/other",
    "https://github.com/user/repo:8080",
    "not-a-repository", "",
])
def test_backend_request_rejects_bad_repository_reference(bad):
    with pytest.raises(ValueError):
        request(bad)


def test_duplicate_requests_and_inconsistent_budgets_rejected():
    with pytest.raises(ValueError, match="Duplicate repository"):
        request("example/api", "https://github.com/EXAMPLE/API")
    with pytest.raises(ValueError, match="min_planned_reads"):
        request("example/api", max_reads=2)
    with pytest.raises(ValueError, match="max_reads"):
        request("example/api", max_reads=12, max_steps=5)
    with pytest.raises(ValueError, match="At least one"):
        run_pipeline(request(), github_client=None, model=FakeThreeStageModel())


def test_backend_request_is_strict_json_and_source_index_does_not_invent_links():
    input_data = {
        "candidate": {"github_login": "developer"},
        "repositories": ["example/api", "example/worker"],
        "max_steps": 8, "max_reads": 4, "min_planned_reads": 2,
    }
    payload = PipelineRequest.model_validate_json(json.dumps(input_data))
    assert payload.repositories == input_data["repositories"]
    with pytest.raises(ValueError):
        PipelineRequest.model_validate({**input_data, "unknown_field": True})
    assert normalize_repository("https://github.com/Example/Api.git") == "example/api"


def test_cli_runs_one_command_and_preserves_stage_metadata(tmp_path, monkeypatch):
    from github2cv_ml import pipeline_cli

    monkeypatch.setattr(pipeline_cli, "GitHubApiClient", lambda token=None: FakeGitHub())
    monkeypatch.setattr(
        pipeline_cli, "LocalOpenAIModel",
        lambda **kwargs: FakeThreeStageModel(),
    )
    output = tmp_path / "generated"
    rc = pipeline_cli.main([
        "--github-login", "developer",
        "--repo", "example/api",
        "--repo", "example/worker",
        "--max-steps", "5", "--max-reads", "2",
        "--min-planned-reads", "1",
        "--output", str(output),
    ])
    assert rc == 0
    assert (output / "cv" / "resume.json").is_file()
    assert (output / "pipeline_manifest.json").is_file()

    # Backend can alternatively write a JSON request and call the same CLI.
    request_path = tmp_path / "request.json"
    request_path.write_text(
        request("example/api", max_steps=5, max_reads=2,
                min_planned_reads=1).model_dump_json(), encoding="utf-8",
    )
    other_output = tmp_path / "generated_json"
    rc = pipeline_cli.main([
        "--request", str(request_path), "--output", str(other_output),
    ])
    assert rc == 0
    assert (other_output / "pipeline_result.json").is_file()


def test_cli_accepts_cached_analysis_without_new_git_requests(tmp_path, monkeypatch):
    from github2cv_ml import pipeline_cli

    initial = run_pipeline(
        request("example/api", max_steps=5, max_reads=2, min_planned_reads=1),
        github_client=FakeGitHub(), model=FakeThreeStageModel(),
    )
    path = tmp_path / "analysis.json"
    path.write_text(initial.analyses[0].model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        pipeline_cli, "GitHubApiClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("No GitHub calls")),
    )
    monkeypatch.setattr(
        pipeline_cli, "LocalOpenAIModel",
        lambda **kwargs: FakeThreeStageModel(),
    )
    output = tmp_path / "cached_run"
    rc = pipeline_cli.main([
        "--github-login", "developer", "--analysis", str(path),
        "--output", str(output),
    ])
    assert rc == 0
    assert (output / "cv/resume.json").is_file()
    assert (output / "snapshots").exists() is False



def test_cached_repositories_count_toward_global_limit():
    previous = run_pipeline(
        request("example/api", max_steps=4, max_reads=2, min_planned_reads=1),
        github_client=FakeGitHub(), model=FakeThreeStageModel(),
    )
    with pytest.raises(ValueError, match="at most 20"):
        run_pipeline(
            request(*[f"owner/repo{i:02d}" for i in range(20)]),
            github_client=FakeGitHub(), model=FakeThreeStageModel(),
            cached_analyses=previous.analyses,
        )


def test_cli_failure_returns_code_two_and_still_saves_artifacts(tmp_path, monkeypatch):
    from github2cv_ml import pipeline_cli

    monkeypatch.setattr(pipeline_cli, "GitHubApiClient", lambda token=None: FakeGitHub())
    monkeypatch.setattr(
        pipeline_cli, "LocalOpenAIModel",
        lambda **kwargs: FakeThreeStageModel(no_evidence=True),
    )
    output = tmp_path / "incomplete"
    exit_code = pipeline_cli.main([
        "--github-login", "dev", "--repo", "example/api",
        "--max-steps", "4", "--max-reads", "2",
        "--min-planned-reads", "1", "--output", str(output),
    ])
    assert exit_code == 2
    assert (output / "pipeline_result.json").is_file()
    assert not (output / "cv" / "resume.json").exists()
