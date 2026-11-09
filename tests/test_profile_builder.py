"""Regressions for model-driven multi-repository candidate profile synthesis."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from github2cv_ml.agent.runner import RepoAnalysis, RepoAnalyzer
from github2cv_ml.domain.evidence import EvidenceKind, EvidenceSource, RepoEvidence
from github2cv_ml.domain.profile import CandidateIdentity, ClaimKind
from github2cv_ml.domain.repository import RepositoryRef
from github2cv_ml.profile_builder import (
    build_candidate_profile,
    build_profile_from_analyses,
)

SHA = "a" * 40
OTHER_SHA = "b" * 40
NOW = datetime(2026, 11, 9, tzinfo=timezone.utc)
CANDIDATE = CandidateIdentity(github_login="developer", display_name="Developer")


def repo(name: str) -> RepositoryRef:
    return RepositoryRef(
        owner="Example",
        name=name,
        url=f"https://github.com/Example/{name}",
        default_branch="main",
    )


def evidence(
    identifier: str,
    project: str,
    text: str,
    category: str = "technology",
    *,
    path: str = "src/main.py",
    start: int = 1,
    commit: str = SHA,
) -> RepoEvidence:
    repository = repo(project)
    return RepoEvidence(
        id=identifier,
        repository=repository,
        category=category,
        observation=text,
        source=EvidenceSource(
            kind=EvidenceKind.FILE,
            url=f"https://github.com/Example/{project}/blob/{commit}/{path}#L{start}",
            path=path,
            commit_sha=commit,
            line_start=start,
            line_end=start + 2,
        ),
        captured_at=NOW,
    )


def analyze(project: str, items: list[RepoEvidence], commit: str = SHA) -> RepoAnalysis:
    return RepoAnalysis(
        repository=repo(project),
        commit_sha=commit,
        project_summary=f"{project} implementation",
        evidence=items,
        status="completed",
        generated_at=NOW,
    )


def proposal(
    skills: list[tuple[str, list[str]]],
    paragraph: str,
    project_evidence: list[str],
) -> dict:
    return {
        "skills": [
            {"name": name, "evidence_ids": refs} for name, refs in skills
        ],
        "project": {"paragraph": paragraph, "evidence_ids": project_evidence},
    }


class FakeProfileModel:
    """Fixed structured LLM responses, selected by repository (no keyword engine)."""

    def __init__(self, responses: dict[str, dict] | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []

    def complete(self, messages, *, tools=None, json_mode=False):
        assert tools is None
        assert json_mode is True
        assert "NO predetermined list" in messages[0]["content"]
        user_input = json.loads(messages[1]["content"])
        self.calls.append(user_input)
        return {
            "content": json.dumps(self.responses[user_input["repository"]]),
        }


def test_multi_repository_profile_uses_model_selected_skills_and_source_ids():
    items = [
        evidence("a1", "api", "FastAPI route and Redis integration", "architecture"),
        evidence("a2", "api", "FastAPI endpoint implementation"),
        evidence("a3", "api", "SQLAlchemy session management"),
        evidence("b1", "worker", "FastAPI integration with PostgreSQL"),
        evidence("b2", "worker", "PostgreSQL-specific upsert", "architecture"),
        evidence("c1", "tests", "Redis integration is tested with Pytest", "testing"),
    ]
    model = FakeProfileModel({
        "example/api": proposal(
            [("FastAPI", ["a1", "a2"]), ("Redis", ["a1"]), ("SQLAlchemy", ["a3"])],
            "The repository implements a FastAPI service with Redis integration.",
            ["a1", "a3"],
        ),
        "example/worker": proposal(
            [("FastAPI", ["b1"]), ("PostgreSQL", ["b1", "b2"])],
            "The repository implements a worker using PostgreSQL upserts.",
            ["b1", "b2"],
        ),
        "example/tests": proposal(
            [("Redis", ["c1"]), ("Pytest", ["c1"])],
            "The repository contains Pytest checks for Redis integration.",
            ["c1"],
        ),
    })
    profile = build_candidate_profile(items, candidate=CANDIDATE, model=model)

    assert profile.candidate == CANDIDATE
    assert profile.schema_version == "1.0"
    assert profile.generated_at.tzinfo is not None
    skills = [c for c in profile.claims if c.kind == ClaimKind.SKILL]
    projects = [c for c in profile.claims if c.kind == ClaimKind.PROJECT]
    assert [c.text for c in skills] == [
        "FastAPI", "Redis", "PostgreSQL", "Pytest", "SQLAlchemy",
    ]
    assert next(c for c in skills if c.text == "FastAPI").evidence_ids == [
        "a1", "b1",
    ]
    assert len(projects) == 3
    assert projects[0].text.startswith("example/api: The repository")
    assert all("I built" not in c.text for c in projects)
    assert len(model.calls) == 3
    assert [call["repository"] for call in model.calls] == [
        "example/api", "example/tests", "example/worker",
    ]
    assert model.calls[0]["observations"][0]["id"] == "a1"
    for claim in profile.claims:
        assert set(claim.evidence_ids) <= {item.id for item in items}
    assert type(profile).model_validate_json(profile.model_dump_json()).claims == profile.claims


def test_open_vocabulary_skill_can_be_extracted_without_code_changes():
    item = evidence("nix", "infra", "Project builds with Nix Flakes", "architecture")
    model = FakeProfileModel({
        "example/infra": proposal(
            [("Nix Flakes", ["nix"])],
            "The project uses Nix Flakes to manage builds.",
            ["nix"],
        ),
    })
    profile = build_candidate_profile([item], candidate=CANDIDATE, model=model)
    assert [c.text for c in profile.claims if c.kind == ClaimKind.SKILL] == [
        "Nix Flakes",
    ]
    assert profile.claims[0].evidence_ids == ["nix"]


def test_skill_rank_counts_repositories_not_repeated_mentions():
    repeated = [
        evidence(f"p{i}", "one", "FastAPI route", start=i + 1)
        for i in range(15)
    ]
    cross_repo = [
        evidence("r1", "two", "Redis caching", "architecture"),
        evidence("r2", "three", "Redis caching", "architecture"),
    ]
    model = FakeProfileModel({
        "example/one": proposal(
            [("FastAPI", [f"p{i}" for i in range(15)])],
            "This service has FastAPI routes.", ["p0", "p1"],
        ),
        "example/two": proposal(
            [("Redis", ["r1"])], "This service uses Redis caching.", ["r1"],
        ),
        "example/three": proposal(
            [("Redis", ["r2"])], "This service uses Redis caching.", ["r2"],
        ),
    })
    profile = build_candidate_profile(
        repeated + cross_repo, candidate=CANDIDATE, model=model
    )
    skills = [c for c in profile.claims if c.kind == ClaimKind.SKILL]
    assert [c.text for c in skills] == ["Redis", "FastAPI"]
    assert skills[0].evidence_ids == ["r2", "r1"]
    assert skills[1].evidence_ids == ["p0"]


def test_duplicate_skill_names_are_merged_case_insensitively():
    items = [
        evidence("a", "api", "FastAPI HTTP API", "technology"),
        evidence("b", "api", "FastAPI application", "architecture", start=4),
    ]
    model = FakeProfileModel({
        "example/api": proposal(
            [("FastAPI", ["a"]), (" fastapi ", ["b"])],
            "The repository exposes an HTTP API.", ["a", "b"],
        )
    })
    profile = build_candidate_profile(items, candidate=CANDIDATE, model=model)
    skills = [c for c in profile.claims if c.kind == ClaimKind.SKILL]
    assert len(skills) == 1
    assert skills[0].text == "FastAPI"
    assert skills[0].evidence_ids == ["a"]


def test_project_ranking_prefers_diversity_over_repeated_technology():
    repetitive = [
        evidence(f"r{i}", "noisy", "FastAPI endpoint " + str(i), start=i + 1)
        for i in range(12)
    ]
    diverse = [
        evidence("d1", "diverse", "Uses PostgreSQL row locking", "architecture"),
        evidence("d2", "diverse", "Uses Redis cache", "technology"),
        evidence("d3", "diverse", "Pytest covers retries", "testing"),
    ]
    model = FakeProfileModel({
        "example/noisy": proposal(
            [("FastAPI", ["r0"])], "An API service.", ["r0"],
        ),
        "example/diverse": proposal(
            [("PostgreSQL", ["d1"]), ("Redis", ["d2"]), ("Pytest", ["d3"])],
            "The service handles row locking and cache operations with tests.",
            ["d1", "d2", "d3"],
        ),
    })
    result = build_candidate_profile(
        repetitive + diverse, candidate=CANDIDATE, model=model
    )
    projects = [c for c in result.claims if c.kind == ClaimKind.PROJECT]
    assert projects[0].text.startswith("example/diverse:")
    assert set(projects[0].evidence_ids) == {"d1", "d2", "d3"}


def test_input_order_does_not_change_output_with_stable_model_responses():
    items = [
        evidence("a1", "api", "FastAPI and Redis", "architecture"),
        evidence("b1", "worker", "SQLAlchemy and Redis"),
        evidence("a2", "api", "PostgreSQL indexes", "technology", start=3),
    ]
    answers = {
        "example/api": proposal(
            [("FastAPI", ["a1"]), ("Redis", ["a1"])],
            "The API uses FastAPI and Redis.", ["a1", "a2"],
        ),
        "example/worker": proposal(
            [("Redis", ["b1"]), ("SQLAlchemy", ["b1"])],
            "The worker uses SQLAlchemy and Redis.", ["b1"],
        ),
    }
    p1 = build_candidate_profile(items, candidate=CANDIDATE, model=FakeProfileModel(answers))
    p2 = build_candidate_profile(
        reversed(items), candidate=CANDIDATE, model=FakeProfileModel(answers)
    )
    assert p1.claims == p2.claims


def test_cached_and_fresh_analyses_share_one_pipeline_without_github_calls():
    api = analyze("api", [evidence("a", "api", "FastAPI async endpoints")])
    worker = analyze("worker", [evidence("b", "worker", "Redis task queue")])
    responses = {
        "example/api": proposal(
            [("FastAPI", ["a"])], "The project implements an API.", ["a"]
        ),
        "example/worker": proposal(
            [("Redis", ["b"])], "The project implements a task queue.", ["b"]
        ),
    }
    cached_api = RepoAnalysis.model_validate_json(api.model_dump_json())
    model = FakeProfileModel(responses)
    profile = build_profile_from_analyses(
        [cached_api, worker], candidate=CANDIDATE, model=model
    )
    expected = build_candidate_profile(
        api.evidence + worker.evidence,
        candidate=CANDIDATE, model=FakeProfileModel(responses),
    )
    assert profile.claims == expected.claims
    assert len(model.calls) == 2
    assert len([c for c in profile.claims if c.kind == ClaimKind.PROJECT]) == 2


def test_duplicate_cached_evidence_is_idempotent():
    item = evidence("id", "api", "FastAPI")
    model = FakeProfileModel({
        "example/api": proposal(
            [("FastAPI", ["id"])], "The repository uses FastAPI.", ["id"],
        ),
    })
    profile = build_profile_from_analyses(
        [analyze("api", [item]), analyze("api", [item])],
        candidate=CANDIDATE, model=model,
    )
    assert len(profile.claims) == 2
    assert len(model.calls) == 1
    assert all(claim.evidence_ids == ["id"] for claim in profile.claims)


def test_empty_inputs_skip_llm_and_validate_limits():
    model = FakeProfileModel()
    empty = build_candidate_profile([], candidate=CANDIDATE, model=model)
    assert empty.claims == []
    assert not model.calls
    with pytest.raises(ValueError, match="max_skills"):
        build_candidate_profile([], candidate=CANDIDATE, model=model, max_skills=21)
    with pytest.raises(ValueError, match="max_projects"):
        build_candidate_profile([], candidate=CANDIDATE, model=model, max_projects=0)
    with pytest.raises(ValueError, match="max_input_chars"):
        build_candidate_profile([], candidate=CANDIDATE, model=model, max_input_chars=999)


def test_max_skills_and_projects_apply_after_model_inference():
    items = [
        evidence("a", "api", "FastAPI Redis PostgreSQL"),
        evidence("b", "worker", "Pytest Docker Python", "testing"),
    ]
    model = FakeProfileModel({
        "example/api": proposal(
            [("FastAPI", ["a"]), ("Redis", ["a"]), ("PostgreSQL", ["a"])],
            "An API backed by databases and cache.", ["a"],
        ),
        "example/worker": proposal(
            [("Pytest", ["b"]), ("Docker", ["b"]), ("Python", ["b"])],
            "A Python worker tested using Pytest.", ["b"],
        ),
    })
    profile = build_candidate_profile(
        items, candidate=CANDIDATE, model=model, max_skills=2, max_projects=1
    )
    assert len([c for c in profile.claims if c.kind == ClaimKind.SKILL]) == 2
    assert len([c for c in profile.claims if c.kind == ClaimKind.PROJECT]) == 1


@pytest.mark.parametrize(
    "wrong",
    [
        proposal([("FastAPI", ["missing"])], "API implementation.", ["a"]),
        proposal([("FastAPI", ["a"])], "API implementation.", ["missing"]),
        {"skills": [{"name": "FastAPI", "evidence_ids": ["a"]}]},
        proposal([(" ", ["a"])], "API implementation.", ["a"]),
    ],
)
def test_model_cannot_use_unknown_ids_or_malformed_output(wrong):
    model = FakeProfileModel({"example/api": wrong})
    with pytest.raises(ValueError):
        build_candidate_profile(
            [evidence("a", "api", "FastAPI")], candidate=CANDIDATE, model=model
        )


def test_model_output_size_is_bounded_per_repository():
    item = evidence("large", "api", "A" * 21_000)
    model = FakeProfileModel()
    with pytest.raises(ValueError, match="exceeds 20000 characters"):
        build_candidate_profile([item], candidate=CANDIDATE, model=model)
    assert model.calls == []


def test_conflicting_evidence_id_cannot_join_different_repositories():
    a = evidence("collision", "api", "FastAPI")
    b = evidence("collision", "worker", "FastAPI")
    with pytest.raises(ValueError, match="Conflicting RepoEvidence"):
        build_candidate_profile(
            [a, b], candidate=CANDIDATE, model=FakeProfileModel()
        )


def test_mixed_commits_for_same_repository_are_rejected():
    a = evidence("one", "api", "FastAPI", commit=SHA)
    b = evidence("two", "api", "Redis", commit=OTHER_SHA)
    with pytest.raises(ValueError, match="Multiple commits"):
        build_candidate_profile(
            [a, b], candidate=CANDIDATE, model=FakeProfileModel()
        )


def test_analysis_rejects_foreign_repository_or_mismatched_commit():
    foreign = analyze("api", [evidence("wrong", "other", "FastAPI")])
    with pytest.raises(ValueError, match="another repository"):
        build_profile_from_analyses(
            [foreign], candidate=CANDIDATE, model=FakeProfileModel()
        )
    stale = analyze("api", [evidence("old", "api", "FastAPI", commit=OTHER_SHA)])
    with pytest.raises(ValueError, match="commit SHA do not match"):
        build_profile_from_analyses(
            [stale], candidate=CANDIDATE, model=FakeProfileModel()
        )


def test_extractor_ids_are_distinct_for_same_claim_in_two_repositories():
    """Existing Repo Analyzer fix prevents identical source-claim collisions."""

    class FixedExtractor:
        def complete(self, messages, *, tools=None, json_mode=False):
            return {"content": json.dumps({
                "file_summary": "FastAPI routes",
                "observations": [{
                    "claim": "Uses FastAPI for HTTP routing",
                    "category": "technology",
                    "source_path": "src/main.py",
                    "line_start": 1,
                    "line_end": 2,
                }],
            })}

    fragment = {
        "path": "src/main.py", "start_line": 1, "end_line": 2,
        "numbered_text": "1: from fastapi import FastAPI\n2: app = FastAPI()",
    }

    def extracted(project):
        browser = SimpleNamespace(
            snapshot=SimpleNamespace(repository=repo(project)), commit_sha=SHA
        )
        _, result, rejected = RepoAnalyzer(FixedExtractor(), browser)._extract_file(fragment)
        assert rejected == 0
        return result[0]

    first = extracted("api")
    second = extracted("worker")
    assert first.id != second.id
    model = FakeProfileModel({
        "example/api": proposal(
            [("FastAPI", [first.id])], "An API.", [first.id],
        ),
        "example/worker": proposal(
            [("FastAPI", [second.id])], "A worker.", [second.id],
        ),
    })
    profile = build_candidate_profile(
        [first, second], candidate=CANDIDATE, model=model
    )
    fastapi = next(c for c in profile.claims if c.kind == ClaimKind.SKILL)
    assert set(fastapi.evidence_ids) == {first.id, second.id}
