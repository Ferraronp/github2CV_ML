"""Regression tests for evidence-only multi-repository candidate synthesis."""

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


def test_aggregates_multiple_repositories_into_valid_profile_with_provenance():
    items = [
        evidence("a1", "api", "FastAPI route and Redis integration", "architecture"),
        evidence("a2", "api", "FastAPI endpoint implementation"),
        evidence("a3", "api", "SQLAlchemy session management"),
        evidence("b1", "worker", "FastAPI integration with PostgreSQL"),
        evidence("b2", "worker", "PostgreSQL-specific upsert", "architecture"),
        evidence("c1", "tests", "Redis integration is tested with Pytest", "testing"),
    ]
    profile = build_candidate_profile(items, candidate=CANDIDATE)

    assert profile.candidate == CANDIDATE
    assert profile.schema_version == "1.0"
    assert profile.generated_at.tzinfo is not None
    skill_claims = [c for c in profile.claims if c.kind == ClaimKind.SKILL]
    project_claims = [c for c in profile.claims if c.kind == ClaimKind.PROJECT]
    assert [c.text for c in skill_claims] == [
        "FastAPI", "Redis", "PostgreSQL", "Pytest", "SQLAlchemy",
    ]
    # Same skill in many files of one repo gives ONE source, not extra votes.
    fastapi = next(c for c in skill_claims if c.text == "FastAPI")
    assert fastapi.evidence_ids == ["a1", "b1"]
    postgres = next(c for c in skill_claims if c.text == "PostgreSQL")
    assert postgres.evidence_ids == ["b1"]
    assert len(project_claims) == 3
    assert project_claims[0].text.startswith("Repository example/api:")
    assert all("I built" not in c.text for c in project_claims)

    ids = {item.id for item in items}
    assert all(set(claim.evidence_ids) <= ids for claim in profile.claims)
    by_id = {item.id: item for item in items}
    for claim in project_claims:
        assert len({_repo_key_for_test(by_id[i]) for i in claim.evidence_ids}) == 1

    # JSON round-trip proves the canonical CandidateProfile contract is used.
    assert type(profile).model_validate_json(profile.model_dump_json()).claims == profile.claims


def _repo_key_for_test(item: RepoEvidence) -> str:
    return f"{item.repository.owner}/{item.repository.name}"


def test_skill_rank_is_by_distinct_repositories_not_number_of_mentions():
    repeated = [
        evidence(f"p{i}", "one", "FastAPI route", start=i + 1)
        for i in range(15)
    ]
    cross_repo = [
        evidence("r1", "two", "Redis caching", "architecture"),
        evidence("r2", "three", "Redis caching", "architecture"),
    ]
    profile = build_candidate_profile(repeated + cross_repo, candidate=CANDIDATE)
    skills = [c for c in profile.claims if c.kind == ClaimKind.SKILL]
    assert [c.text for c in skills] == ["Redis", "FastAPI"]
    assert skills[0].evidence_ids == ["r2", "r1"]  # stable order by repository key
    assert len(skills[1].evidence_ids) == 1


def test_skills_only_from_explicit_observations_not_paths_or_repo_metadata():
    inputs = [
        evidence("e1", "python-and-docker", "Supports repository metadata",
                 path="src/redis_fastapi.py"),
        evidence("e2", "api", "Uses PostgreSQL without MongoDB"),
    ]
    profile = build_candidate_profile(inputs, candidate=CANDIDATE)
    skills = [c.text for c in profile.claims if c.kind == ClaimKind.SKILL]
    assert skills == ["PostgreSQL"]
    assert not any(c.text == "MongoDB" for c in profile.claims)


def test_project_ranking_prefers_diverse_categories_over_repeated_technology():
    repetitive = [
        evidence(f"r{i}", "noisy", "FastAPI endpoint " + str(i), start=i + 1)
        for i in range(12)
    ]
    diverse = [
        evidence("d1", "diverse", "Uses PostgreSQL row locking", "architecture"),
        evidence("d2", "diverse", "Uses Redis cache", "technology"),
        evidence("d3", "diverse", "Pytest covers retries", "testing"),
    ]
    profile = build_candidate_profile(repetitive + diverse, candidate=CANDIDATE)
    projects = [c for c in profile.claims if c.kind == ClaimKind.PROJECT]
    assert projects[0].text.startswith("Repository example/diverse:")
    assert len(projects[0].evidence_ids) == 3
    assert set(projects[0].evidence_ids) == {"d1", "d2", "d3"}


def test_repeated_statements_in_one_project_are_not_doubled_in_paragraph():
    items = [
        evidence("e1", "api", "Uses Redis for caching.", "technology", start=1),
        evidence("e2", "api", "  Uses Redis   for caching  ", "architecture", start=4),
        evidence("e3", "api", "Implements task retries", "architecture", start=8),
    ]
    profile = build_candidate_profile(items, candidate=CANDIDATE)
    project_claim = next(c for c in profile.claims if c.kind == ClaimKind.PROJECT)
    assert project_claim.text.count("Uses Redis for caching") == 1
    assert len(project_claim.evidence_ids) == 2


def test_input_order_does_not_affect_claim_ids_text_or_rank():
    inputs = [
        evidence("a1", "api", "FastAPI and Redis", "architecture"),
        evidence("b1", "worker", "SQLAlchemy and Redis"),
        evidence("a2", "api", "PostgreSQL indexes", "technology", start=3),
    ]
    p1 = build_candidate_profile(inputs, candidate=CANDIDATE)
    p2 = build_candidate_profile(reversed(inputs), candidate=CANDIDATE)
    assert p1.claims == p2.claims


def test_cached_and_fresh_analyses_use_same_builder_without_github_calls():
    api = analyze("api", [evidence("a", "api", "FastAPI async endpoints")])
    worker = analyze("worker", [evidence("b", "worker", "Redis task queue")])
    cached_api = RepoAnalysis.model_validate_json(api.model_dump_json())
    profile = build_profile_from_analyses(
        [cached_api, worker], candidate=CANDIDATE,
    )
    expected = build_candidate_profile(api.evidence + worker.evidence, candidate=CANDIDATE)
    assert profile.claims == expected.claims
    assert len([c for c in profile.claims if c.kind == ClaimKind.PROJECT]) == 2


def test_duplicate_cached_evidence_is_idempotent():
    e = evidence("id", "api", "FastAPI")
    profile = build_profile_from_analyses(
        [analyze("api", [e]), analyze("api", [e])],
        candidate=CANDIDATE,
    )
    assert len(profile.claims) == 2  # one skill and one project
    assert all(claim.evidence_ids == ["id"] for claim in profile.claims)


def test_same_evidence_id_in_different_repositories_is_rejected():
    a = evidence("collision", "api", "FastAPI")
    b = evidence("collision", "worker", "FastAPI")
    with pytest.raises(ValueError, match="Conflicting RepoEvidence id"):
        build_candidate_profile([a, b], candidate=CANDIDATE)


def test_mixed_revisions_for_one_repository_are_not_silently_merged():
    a = evidence("one", "api", "FastAPI", commit=SHA)
    b = evidence("two", "api", "Redis", commit=OTHER_SHA)
    with pytest.raises(ValueError, match="Multiple commits"):
        build_candidate_profile([a, b], candidate=CANDIDATE)


def test_analysis_rejects_foreign_repository_or_different_commit_sha():
    foreign = analyze("api", [evidence("wrong", "other", "FastAPI")])
    with pytest.raises(ValueError, match="another repository"):
        build_profile_from_analyses([foreign], candidate=CANDIDATE)

    stale = analyze("api", [evidence("old", "api", "FastAPI", commit=OTHER_SHA)])
    with pytest.raises(ValueError, match="commit SHA do not match"):
        build_profile_from_analyses([stale], candidate=CANDIDATE)


def test_empty_partial_and_limited_outputs_are_explicit_and_valid():
    empty = build_candidate_profile([], candidate=CANDIDATE)
    assert empty.claims == []
    assert empty.candidate.github_login == "developer"

    inputs = [
        evidence("one", "api", "FastAPI Redis PostgreSQL SQLAlchemy"),
        evidence("two", "worker", "Pytest Docker Python", "testing"),
    ]
    profile = build_candidate_profile(
        inputs, candidate=CANDIDATE, max_skills=2, max_projects=1,
    )
    assert len([c for c in profile.claims if c.kind == ClaimKind.SKILL]) == 2
    assert len([c for c in profile.claims if c.kind == ClaimKind.PROJECT]) == 1

    with pytest.raises(ValueError, match="max_skills"):
        build_candidate_profile(inputs, candidate=CANDIDATE, max_skills=21)
    with pytest.raises(ValueError, match="max_projects"):
        build_candidate_profile(inputs, candidate=CANDIDATE, max_projects=0)


def test_extractor_generates_distinct_ids_for_same_claim_in_different_repos():
    """Identical source paths and claims must not collide during aggregation."""

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
        "path": "src/main.py",
        "start_line": 1,
        "end_line": 2,
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
    profile = build_candidate_profile([first, second], candidate=CANDIDATE)
    fastapi = next(c for c in profile.claims if c.kind == ClaimKind.SKILL)
    assert set(fastapi.evidence_ids) == {first.id, second.id}
