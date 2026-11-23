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


def test_pipeline_honors_pinned_sha_and_saves_each_repo_incrementally():
    client = FakeGitHub()
    saved = []
    result = run_pipeline(
        request("example/api", "example/worker", max_steps=4,
                max_reads=2, min_planned_reads=1),
        github_client=client,
        model=FakeThreeStageModel(),
        pinned_shas={"example/api": SHA, "example/worker": SHA},
        on_analysis=saved.append,
    )
    assert result.resume is not None
    assert len(saved) == 2
    assert not client.branch_refs  # Pinned SHA must not resolve a moving HEAD.
    assert all(a.commit_sha == SHA for a in saved)
    with pytest.raises(ValueError, match="Pinned SHA"):
        run_pipeline(
            request("example/api"), github_client=client,
            model=FakeThreeStageModel(),
            pinned_shas={"example/api": "bad"},
        )
    with pytest.raises(ValueError, match="unique and requested"):
        run_pipeline(
            request("example/api"), github_client=client,
            model=FakeThreeStageModel(),
            pinned_shas={"example/unrequested": SHA},
        )


def _worker_request(*repositories, message_id=None, candidate="developer", role=None):
    from uuid import uuid4

    from github2cv_ml.worker_contracts import (
        AnalysisOptions,
        MLJobRequest,
        RequestedRepository,
    )

    return MLJobRequest(
        message_id=message_id or uuid4(),
        job_id=uuid4(),
        input_revision=1,
        candidate=CandidateIdentity(github_login=candidate),
        repositories=[
            RequestedRepository(
                repo_id=str(i + 100),
                repository_url=f"https://github.com/example/{repo}",
                commit_sha=SHA,
                context_hash="f" * 64,
                declared_role=role,
            ) for i, repo in enumerate(repositories)
        ],
        options=AnalysisOptions(max_steps=5, max_reads=2, min_planned_reads=1),
    )


def test_worker_message_contract_rejects_extra_fields_bad_sha_and_duplicates():
    from github2cv_ml.worker_contracts import MLJobRequest

    incoming = _worker_request("api", "worker")
    assert MLJobRequest.model_validate_json(incoming.model_dump_json()) == incoming
    assert incoming.schema_version == 2
    with pytest.raises(ValueError):
        MLJobRequest.model_validate({**incoming.model_dump(), "unexpected": "value"})
    with pytest.raises(ValueError):
        _worker_request("api", "api")
    altered = incoming.model_dump()
    altered["repositories"][0]["commit_sha"] = "not-a-sha"
    with pytest.raises(ValueError):
        MLJobRequest.model_validate(altered)
    altered = incoming.model_dump()
    altered["repositories"][0]["repository_url"] = "https://evil.example/repo"
    with pytest.raises(ValueError):
        MLJobRequest.model_validate(altered)


def test_worker_cache_reuses_repo_analyses_with_new_candidate_context(tmp_path):
    from uuid import uuid4

    from github2cv_ml.worker_service import JobProcessor
    from github2cv_ml.worker_store import WorkerStore

    db = tmp_path / "ml-cache.db"
    gh = FakeGitHub()
    model = FakeThreeStageModel()
    processor = JobProcessor(
        db_path=db, github_client=gh, model=model, model_id="qwen-3.5"
    )
    first = _worker_request("api", "worker")
    result1 = processor.process(first)
    assert result1.type == "github.report.completed"
    assert result1.status == PipelineStatus.COMPLETED
    assert len(result1.repositories) == 2
    assert all(r.cache_hit is False for r in result1.repositories)
    assert not gh.branch_refs
    assert all(ref[-1] == SHA for ref in gh.source_refs)
    first_reads = len(gh.source_refs)
    first_calls = len(model.calls)
    assert first_reads > 0

    # The broker can redeliver the exact same message after losing publisher ACK.
    assert processor.process(first) == result1
    assert len(model.calls) == first_calls
    assert len(gh.source_refs) == first_reads

    # New request/user context reuses source facts, re-runs personal CV stages.
    second = _worker_request("api", "worker", message_id=uuid4(),
                             candidate="another-person", role="Reviewer")
    result2 = processor.process(second)
    assert result2.type == "github.report.completed"
    assert all(r.cache_hit is True for r in result2.repositories)
    assert len(gh.source_refs) == first_reads
    assert len(model.calls) > first_calls
    assert result2.profile.candidate.github_login == "another-person"
    with WorkerStore(db) as store:
        assert store.claim(first) == result1
    assert result2.message_id != result1.message_id


def test_worker_partial_failure_returns_full_repository_status_set(tmp_path):
    from github2cv_ml.worker_service import JobProcessor

    result = JobProcessor(
        db_path=tmp_path / "worker.sqlite",
        github_client=FakeGitHub(failing={"down"}),
        model=FakeThreeStageModel(),
        model_id="fake",
    ).process(_worker_request("down", "api"))
    assert result.type == "github.report.completed"
    assert result.status == PipelineStatus.PARTIAL
    assert [(r.repo_id, r.status) for r in result.repositories] == [
        ("100", "failed"), ("101", "completed"),
    ]
    assert result.resume is not None
    assert any(p.stage == PipelineStage.COLLECT for p in result.problems)


def test_worker_rejects_result_replay_with_different_request_payload(tmp_path):
    from github2cv_ml.worker_service import JobProcessor

    request_v1 = _worker_request("api")
    processor = JobProcessor(
        db_path=tmp_path / "worker.sqlite",
        github_client=FakeGitHub(),
        model=FakeThreeStageModel(), model_id="fake",
    )
    processor.process(request_v1)
    # Same message_id, changed job_id: fail instead of reusing somebody else's CV.
    request_v2 = _worker_request("api", message_id=request_v1.message_id)
    with pytest.raises(ValueError, match="reused"):
        processor.process(request_v2)


def test_worker_persists_repository_analysis_even_when_profile_fails(tmp_path):
    from github2cv_ml.worker_service import JobProcessor

    db = tmp_path / "worker.sqlite"
    gh = FakeGitHub()
    bad = JobProcessor(
        db_path=db, github_client=gh,
        model=FakeThreeStageModel(broken_profile=True), model_id="fake",
    )
    first = bad.process(_worker_request("api"))
    assert first.type == "github.report.failed"
    assert first.resume is None
    assert len(first.repositories) == 1
    assert first.repositories[0].analysis is not None
    read_count = len(gh.source_refs)

    good = JobProcessor(
        db_path=db, github_client=gh,
        model=FakeThreeStageModel(), model_id="fake",
    )
    second = good.process(_worker_request("api"))
    assert second.type == "github.report.completed"
    assert second.repositories[0].cache_hit
    assert len(gh.source_refs) == read_count


def test_store_processing_lease_prevents_concurrent_duplicate_until_released(tmp_path):
    from github2cv_ml.worker_store import JobBusy, WorkerStore

    request_msg = _worker_request("api")
    with WorkerStore(tmp_path / "ml.sqlite3") as store:
        assert store.claim(request_msg) is None
        assert store.attempt_count(str(request_msg.message_id)) == 1
        with pytest.raises(JobBusy):
            store.claim(request_msg)
        store.release(str(request_msg.message_id))
        assert store.claim(request_msg) is None
        assert store.attempt_count(str(request_msg.message_id)) == 2


def test_worker_cache_keys_invalidate_sha_or_model_or_budget():
    from github2cv_ml.worker_contracts import AnalysisOptions
    from github2cv_ml.worker_store import analysis_cache_key

    options = AnalysisOptions()
    x = analysis_cache_key("example/api", SHA, options, "qwen")
    assert x == analysis_cache_key("https://github.com/EXAMPLE/API", SHA, options, "qwen")
    assert x != analysis_cache_key("example/api", "b" * 40, options, "qwen")
    assert x != analysis_cache_key("example/api", SHA, options, "gemma")
    assert x != analysis_cache_key("example/api", SHA, options.model_copy(update={
        "max_reads": 15,
    }), "qwen")


def test_failed_response_if_event_exceeds_transport_limit(tmp_path):
    from github2cv_ml.worker_service import JobProcessor

    processor = JobProcessor(
        db_path=tmp_path / "ml.sqlite3", github_client=FakeGitHub(),
        model=FakeThreeStageModel(), model_id="fake", max_event_bytes=100,
    )
    result = processor.process(_worker_request("api"))
    assert result.type == "github.report.failed"
    assert "exceeded" in result.error
    assert result.resume is None


def test_worker_retries_internal_error_only_three_times(tmp_path):
    from github2cv_ml.worker_service import JobProcessor

    class BrokenProcessor(JobProcessor):
        def _run(self, request, store):
            raise RuntimeError("Deliberate failure")

    processor = BrokenProcessor(
        db_path=tmp_path / "broken.sqlite3",
        github_client=FakeGitHub(),
        model=FakeThreeStageModel(),
        model_id="fake",
    )
    request_msg = _worker_request("api")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Deliberate"):
            processor.process(request_msg)
    terminal = processor.process(request_msg)
    assert terminal.type == "github.report.failed"
    assert "3 attempts" in terminal.error
    assert terminal.status == PipelineStatus.FAILED
    assert processor.process(request_msg) == terminal


def test_pipeline_public_only_even_if_github_token_has_private_access():
    class PrivateGitHub(FakeGitHub):
        def get_repository(self, owner, name):
            metadata = super().get_repository(owner, name)
            return {**metadata, "private": True}

    github = PrivateGitHub()
    result = run_pipeline(
        request("example/api", max_steps=4, max_reads=2,
                min_planned_reads=1),
        github_client=github, model=FakeThreeStageModel(),
        pinned_shas={"example/api": SHA},
    )
    assert result.status == PipelineStatus.FAILED
    assert result.snapshots == [] and result.analyses == []
    assert result.problems[0].stage == PipelineStage.COLLECT
    assert not github.source_refs
