"""Deterministic synthesis of evidence-backed candidate claims.

This stage does not verify source semantics, infer authorship or use an LLM.
It consumes RepoEvidence from fresh or cached analyses; the CV Writer is
responsible for prose and presentation.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Protocol

from github2cv_ml.domain.evidence import RepoEvidence
from github2cv_ml.domain.profile import (
    CandidateClaim,
    CandidateIdentity,
    CandidateProfile,
    ClaimKind,
)
from github2cv_ml.domain.repository import RepositoryRef

# Deliberately conservative vocabulary: a skill tag requires an explicit
# mention in an observation, not a guess based on file extensions or metadata.
# Longest/specific forms are listed before the generic ones for readability.
_SKILL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("GitHub Actions", r"\bgithub[ -]actions\b"),
    ("Docker Compose", r"\bdocker[ -]compose\b"),
    ("REST API", r"\brest(?:ful)?[ -]+api\b"),
    ("PostgreSQL", r"\bpostgres(?:ql)?\b"),
    ("SQLAlchemy", r"\bsqlalchemy\b"),
    ("TypeScript", r"\btypescript\b"),
    ("JavaScript", r"\bjavascript\b"),
    ("Kubernetes", r"\b(?:kubernetes|k8s)\b"),
    ("Python", r"\bpython\b"),
    ("FastAPI", r"\bfastapi\b"),
    ("Pydantic", r"\bpydantic\b"),
    ("Redis", r"\bredis\b"),
    ("Docker", r"\bdocker\b"),
    ("RabbitMQ", r"\brabbitmq\b"),
    ("SQL", r"\bsql\b"),
    ("Pytest", r"\bpytest\b"),
    ("AsyncIO", r"\basyncio\b"),
    ("JWT", r"\bjwt\b"),
    ("GraphQL", r"\bgraphql\b"),
    ("MongoDB", r"\bmongodb\b"),
    ("Kafka", r"\bkafka\b"),
    ("React", r"\breact\b"),
    ("Node.js", r"\bnode(?:\.js|js)\b"),
    ("Go", r"\bgolang\b"),
    ("Rust", r"\brust\b"),
    ("Java", r"\bjava\b"),
    ("C++", r"(?<!\w)c\+\+(?!\w)"),
    ("C#", r"(?<!\w)c#(?!\w)"),
    ("PyTorch", r"\bpytorch\b"),
    ("TensorFlow", r"\btensorflow\b"),
    ("RAG", r"\brag\b"),
    ("LLM", r"\bllms?\b"),
    ("gRPC", r"\bgrpc\b"),
    ("Terraform", r"\bterraform\b"),
    ("Alembic", r"\balembic\b"),
)
_SKILLS = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _SKILL_PATTERNS)
_NEGATION = re.compile(
    r"\b(?:not|no|without|never|doesn't|isn't|не|без|отсутствует)\b"
    r"(?:\W+\w+){0,3}\W*$",
    re.IGNORECASE,
)
_CATEGORY_ORDER = ("architecture", "technology", "testing", "deployment", "project_type")


class EvidenceAnalysis(Protocol):
    """Minimal read-only interface shared by fresh and cached RepoAnalysis."""

    repository: RepositoryRef
    commit_sha: str
    evidence: list[RepoEvidence]


def _repo_key(repo: RepositoryRef) -> str:
    return f"{repo.owner.casefold()}/{repo.name.casefold()}"


def _clean_observation(text: str) -> str:
    return " ".join(text.split()).strip()


def _evidence_sort_key(evidence: RepoEvidence) -> tuple[str, str, int, str]:
    return (
        _repo_key(evidence.repository),
        evidence.source.path or "",
        evidence.source.line_start or 0,
        evidence.id,
    )


def _unique_evidence(items: Iterable[RepoEvidence]) -> list[RepoEvidence]:
    by_id: dict[str, RepoEvidence] = {}
    commits: dict[str, str] = {}
    for item in items:
        if not isinstance(item, RepoEvidence):
            raise TypeError("All inputs must be RepoEvidence instances")
        repo = _repo_key(item.repository)
        sha = item.source.commit_sha
        if sha:
            previous_sha = commits.setdefault(repo, sha.casefold())
            if previous_sha != sha.casefold():
                raise ValueError(
                    f"Multiple commits for {repo}: upstream must select one cached "
                    "or newly analyzed version per repository"
                )
        previous = by_id.get(item.id)
        if previous is None:
            by_id[item.id] = item
        elif previous.model_dump(exclude={"captured_at"}) != item.model_dump(
            exclude={"captured_at"}
        ):
            raise ValueError(
                f"Conflicting RepoEvidence id {item.id!r}; cannot resolve "
                "source provenance across repositories"
            )
    return sorted(by_id.values(), key=_evidence_sort_key)


def _skills_in(text: str) -> set[str]:
    found: set[str] = set()
    for skill, pattern in _SKILLS:
        for match in pattern.finditer(text):
            if not _NEGATION.search(text[max(0, match.start() - 36):match.start()]):
                found.add(skill)
                break
    return found


def _claim_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"claim-{kind}-{digest}"


def _representative_evidence(items: list[RepoEvidence], limit: int = 3) -> list[RepoEvidence]:
    """Prefer distinct statements and source categories over repeated mentions."""
    buckets: dict[str, list[RepoEvidence]] = defaultdict(list)
    seen: set[str] = set()
    for item in sorted(items, key=_evidence_sort_key):
        normalized = _clean_observation(item.observation).casefold().rstrip(" .;")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        buckets[item.category].append(item)

    selected: list[RepoEvidence] = []
    ordered_categories = list(_CATEGORY_ORDER)
    ordered_categories += sorted(category for category in buckets if category not in ordered_categories)
    # Round-robin: diverse evidence before several similar statements.
    while len(selected) < limit:
        found = False
        for category in ordered_categories:
            if buckets[category]:
                selected.append(buckets[category].pop(0))
                found = True
                if len(selected) >= limit:
                    break
        if not found:
            break
    return selected


def build_candidate_profile(
    evidence: Iterable[RepoEvidence],
    *,
    candidate: CandidateIdentity,
    max_skills: int = 20,
    max_projects: int | None = None,
) -> CandidateProfile:
    """Aggregate supplied evidence without inventing claims or ownership.

    Skill scores count *distinct repositories* rather than mentions.
    Project ranking emphasizes variety of observed source categories; it is
    not a performance, quality, authorship or semantic-truth score.

    Evidence must remain available to downstream components by its original
    IDs. A reused ID with different provenance is rejected, not silently
    attributed to the wrong repository.
    """
    if not 1 <= max_skills <= 20:
        raise ValueError("max_skills must be between 1 and 20")
    if max_projects is not None and max_projects < 1:
        raise ValueError("max_projects must be positive or None")

    unique = _unique_evidence(evidence)
    if not unique:
        return CandidateProfile(
            candidate=candidate, claims=[], generated_at=datetime.now(timezone.utc)
        )

    projects: dict[str, list[RepoEvidence]] = defaultdict(list)
    skills: dict[str, dict[str, RepoEvidence]] = defaultdict(dict)
    for item in unique:
        repository = _repo_key(item.repository)
        projects[repository].append(item)
        for skill in _skills_in(item.observation):
            # Only one representative mention per skill and repository.
            skills[skill].setdefault(repository, item)

    ranked_skills = sorted(skills, key=lambda skill: (-len(skills[skill]), skill.casefold()))
    claims: list[CandidateClaim] = []
    for skill in ranked_skills[:max_skills]:
        sources = [skills[skill][repo] for repo in sorted(skills[skill])]
        claims.append(CandidateClaim(
            id=_claim_id("skill", skill.casefold()),
            kind=ClaimKind.SKILL,
            text=skill,
            evidence_ids=[source.id for source in sources],
        ))

    def project_score(items: list[RepoEvidence]) -> tuple[int, int, int]:
        categories = {item.category for item in items}
        observed = {
            _clean_observation(item.observation).casefold().rstrip(" .;")
            for item in items
        }
        # Multiple references to the same technology do not scale the score.
        implementation_categories = categories & set(_CATEGORY_ORDER)
        return (
            len(implementation_categories),
            int("architecture" in categories) + int("testing" in categories),
            min(len(observed), 3),
        )

    ranked_repositories = sorted(
        projects, key=lambda repo: (
            *tuple(-value for value in project_score(projects[repo])),
            repo,
        )
    )
    if max_projects is not None:
        ranked_repositories = ranked_repositories[:max_projects]

    for repository in ranked_repositories:
        chosen = _representative_evidence(projects[repository])
        if not chosen:
            continue
        # Only source statements are included, as a draft for CV Writer.
        # No "built", "led" or first-person authorship inference.
        descriptions = [_clean_observation(item.observation).rstrip(" .;") for item in chosen]
        paragraph = f"Repository {repository}: " + ". ".join(descriptions) + "."
        claims.append(CandidateClaim(
            id=_claim_id("project", repository + "|" + "|".join(item.id for item in chosen)),
            kind=ClaimKind.PROJECT,
            text=paragraph,
            evidence_ids=[item.id for item in chosen],
        ))

    return CandidateProfile(
        candidate=candidate,
        claims=claims,
        generated_at=datetime.now(timezone.utc),
    )


def build_profile_from_analyses(
    analyses: Iterable[EvidenceAnalysis],
    *,
    candidate: CandidateIdentity,
    max_skills: int = 20,
    max_projects: int | None = None,
) -> CandidateProfile:
    """Adapter for RepoAnalysis outputs, including backend-cached instances.

    Backend selects cached or fresh data *before* calling this function.
    The adapter never fetches repository files and does not store source code.
    """
    all_evidence: list[RepoEvidence] = []
    for analysis in analyses:
        if not isinstance(analysis.repository, RepositoryRef):
            raise TypeError("Analysis repository must be RepositoryRef")
        repo_key = _repo_key(analysis.repository)
        for item in analysis.evidence:
            if not isinstance(item, RepoEvidence):
                raise TypeError("Analysis evidence must contain RepoEvidence")
            if _repo_key(item.repository) != repo_key:
                raise ValueError("RepoAnalysis contains evidence from another repository")
            if (
                item.source.commit_sha is not None
                and item.source.commit_sha.casefold() != analysis.commit_sha.casefold()
            ):
                raise ValueError("RepoAnalysis and evidence commit SHA do not match")
            all_evidence.append(item)
    return build_candidate_profile(
        all_evidence,
        candidate=candidate,
        max_skills=max_skills,
        max_projects=max_projects,
    )
