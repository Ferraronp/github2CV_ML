"""Single entry point for the MVP GitHub -> evidence -> candidate -> CV flow.

This is an in-process API for a backend worker, not a web server. The backend
owns authentication, jobs and HTTP. Repos are analyzed sequentially with one
pinned commit each. Unexpected stage failures become structured diagnostics;
partial evidence is never silently represented as a complete analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable, Literal

from pydantic import Field, model_validator

from github2cv_ml.agent.runner import ChatModel, RepoAnalysis, RepoAnalyzer
from github2cv_ml.agent.source_tools import RepositoryBrowser
from github2cv_ml.collectors.github_collector import (
    GitHubRepositoryCollector,
    parse_repository_reference,
)
from github2cv_ml.cv_demo import TracingWriterModel
from github2cv_ml.cv_writer import write_resume
from github2cv_ml.domain.base import ContractModel
from github2cv_ml.domain.evidence import RepoEvidence
from github2cv_ml.domain.profile import CandidateIdentity, CandidateProfile
from github2cv_ml.domain.repository import RepoSnapshot
from github2cv_ml.domain.resume import ResumeDocument
from github2cv_ml.model_recovery import RecoveringModel
from github2cv_ml.profile_builder import build_profile_from_analyses
from github2cv_ml.profile_demo import TracingProfileModel

MAX_REPOSITORIES = 20


class PipelineStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class PipelineStage(StrEnum):
    INPUT = "input"
    COLLECT = "collect"
    ANALYZE = "analyze"
    PROFILE = "profile"
    RESUME = "resume"


class PipelineRequest(ContractModel):
    """Validated backend payload; repository names are GitHub owner/repo or URLs."""

    candidate: CandidateIdentity
    repositories: list[str] = Field(default_factory=list, max_length=MAX_REPOSITORIES)
    max_steps: int = Field(default=28, ge=1, le=100)
    max_reads: int = Field(default=12, ge=1, le=50)
    min_planned_reads: int = Field(default=8, ge=0, le=50)
    max_skills: int = Field(default=20, ge=1, le=20)
    max_projects: int | None = Field(default=None, ge=1)
    max_input_chars: int = Field(default=20_000, ge=1000, le=200_000)

    @model_validator(mode="after")
    def validate_budgets_and_repos(self) -> "PipelineRequest":
        if self.max_reads > self.max_steps:
            raise ValueError("max_reads must not exceed max_steps")
        if self.min_planned_reads > self.max_reads:
            raise ValueError("min_planned_reads must not exceed max_reads")
        seen: set[str] = set()
        for reference in self.repositories:
            key = normalize_repository(reference)
            if key in seen:
                raise ValueError(f"Duplicate repository in request: {reference}")
            seen.add(key)
        return self


class PipelineProblem(ContractModel):
    """Non-sensitive, stage-scoped diagnostic for backend display/logging."""

    stage: PipelineStage
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    repository: str | None = None


class PipelineResult(ContractModel):
    """One complete, partially complete or failed pipeline attempt."""

    schema_version: Literal["1.0"] = "1.0"
    status: PipelineStatus
    request: PipelineRequest
    snapshots: list[RepoSnapshot] = Field(default_factory=list)
    analyses: list[RepoAnalysis] = Field(default_factory=list)
    profile: CandidateProfile | None = None
    resume: ResumeDocument | None = None
    problems: list[PipelineProblem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    profile_trace: list[dict[str, Any]] = Field(default_factory=list)
    cv_trace: list[dict[str, Any]] = Field(default_factory=list)
    correction_events: list[dict[str, Any]] = Field(default_factory=list)
    # Present only when Profile Builder failed; never reuse invalid model text
    # as trusted candidate data. Useful for diagnosing Qwen JSON truncation.
    profile_error_responses: list[dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def evidence_index(self) -> dict[str, RepoEvidence]:
        """Backend can link ResumeItem -> CandidateClaim -> RepoEvidence."""
        return {
            evidence.id: evidence
            for analysis in self.analyses
            for evidence in analysis.evidence
        }


def normalize_repository(reference: str) -> str:
    """Fail closed on malformed or deceptive URL forms before GitHub requests."""
    from urllib.parse import urlsplit

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Repository reference must not be blank")
    value = reference.strip()
    if value.startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https" or parsed.hostname != "github.com"
            or parsed.username or parsed.password or parsed.port is not None
            or parsed.query or parsed.fragment
        ):
            raise ValueError(f"Expected direct HTTPS github.com/owner/repo: {value}")
        value = parsed.path.strip("/")
    owner, name = parse_repository_reference(value)
    import re

    if not re.fullmatch(r"[A-Za-z0-9-]{1,39}", owner) or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,100}", name
    ):
        raise ValueError(f"Invalid GitHub owner/repository name: {reference}")
    return f"{owner.casefold()}/{name.casefold()}"


def _problem(
    stage: PipelineStage,
    code: str,
    exc: Exception | str,
    *,
    repository: str | None = None,
) -> PipelineProblem:
    # Exception strings can include untrusted model text; never include a
    # traceback or an HTTP authorization header in backend-facing diagnostics.
    msg = str(exc).splitlines()[0][:320]
    return PipelineProblem(
        stage=stage, code=code, message=msg or code, repository=repository
    )


def run_pipeline(
    request: PipelineRequest,
    *,
    github_client: Any | None,
    model: ChatModel,
    cached_analyses: Iterable[RepoAnalysis] = (),
    pinned_shas: Mapping[str, str] | None = None,
    on_analysis: Callable[[RepoAnalysis], None] | None = None,
) -> PipelineResult:
    """Run sequential multi-repo collection and the two independent LLM stages.

    The client/model are injected for backend control and offline testing.
    The function never writes files, makes no HTTP server, and does not run
    untrusted repository code. The caller can persist all artifacts separately.
    Failed repos are reported but do not prevent healthy repos reaching CV.
    """
    if not isinstance(request, PipelineRequest):
        raise TypeError("run_pipeline requires PipelineRequest")
    seen = {normalize_repository(ref) for ref in request.repositories}
    supplied_pins = pinned_shas or {}
    pinned: dict[str, str] = {}
    import re

    for ref, sha in supplied_pins.items():
        key = normalize_repository(ref)
        if key not in seen or key in pinned:
            raise ValueError(f"Pinned repository must be unique and requested: {ref}")
        if not isinstance(sha, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", sha):
            raise ValueError(f"Pinned SHA must be 40 hex characters: {ref}")
        pinned[key] = sha.lower()
    retained: list[RepoAnalysis] = []
    for analysis in cached_analyses:
        if len(seen) >= MAX_REPOSITORIES:
            raise ValueError(f"Pipeline supports at most {MAX_REPOSITORIES} repositories")
        if not isinstance(analysis, RepoAnalysis):
            raise TypeError("cached_analyses must contain RepoAnalysis")
        key = normalize_repository(
            f"{analysis.repository.owner}/{analysis.repository.name}"
        )
        if key in seen:
            raise ValueError(f"Conflicting fresh/cached analysis for repository {key}")
        if any(
            normalize_repository(f"{a.repository.owner}/{a.repository.name}") == key
            for a in retained
        ):
            raise ValueError(f"Duplicate cached analysis for repository {key}")
        seen.add(key)
        retained.append(analysis)
    if not seen:
        raise ValueError("At least one repository or cached analysis is required")
    if request.repositories and github_client is None:
        raise ValueError("A GitHub client is required for fresh repositories")

    result = PipelineResult(
        status=PipelineStatus.FAILED, request=request, analyses=list(retained)
    )
    for ref in request.repositories:
        key = normalize_repository(ref)
        owner, name = key.split("/", 1)
        try:
            metadata = github_client.get_repository(owner, name)
            if metadata.get("private") is True or metadata.get("visibility") not in (
                None, "public",
            ):
                raise ValueError("Only public GitHub repositories are supported")
            branch = metadata.get("default_branch")
            if not isinstance(branch, str) or not branch.strip():
                raise ValueError("GitHub default_branch is missing")
            sha = pinned.get(key)
            if sha is None:
                sha = github_client.get_branch_head_sha(owner, name, branch)
            snapshot = GitHubRepositoryCollector(client=github_client).collect(
                f"{owner}/{name}", commit_sha=sha
            )
        except Exception as exc:
            result.problems.append(
                _problem(PipelineStage.COLLECT, "collect_failed", exc, repository=key)
            )
            continue
        result.snapshots.append(snapshot)
        try:
            browser = RepositoryBrowser(
                client=github_client, snapshot=snapshot, commit_sha=sha
            )
            analysis = RepoAnalyzer(
                model=model, browser=browser,
                max_steps=request.max_steps,
                max_read_calls=request.max_reads,
                min_planned_reads=request.min_planned_reads,
            ).run()
        except Exception as exc:
            result.problems.append(
                _problem(PipelineStage.ANALYZE, "analyze_failed", exc, repository=key)
            )
            continue
        result.analyses.append(analysis)
        if on_analysis is not None:
            # Persist every completed repository before starting the next one.
            # A worker restart must not lose all previous expensive LLM calls.
            on_analysis(analysis)
    # Apply exactly the same quality checks to fresh and cached analyses.
    # A cached analysis is not automatically a *completed* analysis.
    for analysis in result.analyses:
        key = normalize_repository(
            f"{analysis.repository.owner}/{analysis.repository.name}"
        )
        if not analysis.evidence:
            result.problems.append(PipelineProblem(
                stage=PipelineStage.ANALYZE,
                code="no_evidence",
                message="No source evidence was extracted from this repository",
                repository=key,
            ))
        if analysis.status != "completed":
            result.warnings.append(
                f"{key}: exploration status={analysis.status}; "
                "analysis may be incomplete"
            )

    good = [analysis for analysis in result.analyses if analysis.evidence]
    if not good:
        result.problems.append(PipelineProblem(
            stage=PipelineStage.PROFILE,
            code="no_evidence",
            message="No evidence-backed repository available to build a CV",
        ))
        return result

    recovering = RecoveringModel(model)
    observed_profile = TracingProfileModel(recovering)
    try:
        profile = build_profile_from_analyses(
            good,
            candidate=request.candidate,
            model=observed_profile,
            max_skills=request.max_skills,
            max_projects=request.max_projects,
            max_input_chars=request.max_input_chars,
        )
        result.profile = profile
    except Exception as exc:
        result.problems.append(
            _problem(PipelineStage.PROFILE, "profile_failed", exc)
        )
        result.profile_error_responses = [
            {**entry, "raw_content": entry["raw_content"][:8000]}
            for entry in observed_profile.raw_responses[-2:]
        ]
        return result
    finally:
        result.profile_trace = observed_profile.calls
        result.correction_events = list(recovering.events)

    observed_writer = TracingWriterModel(recovering)
    try:
        result.resume = write_resume(
            result.profile,
            model=observed_writer,
            max_skills=request.max_skills,
            max_projects=request.max_projects,
            max_input_chars=request.max_input_chars,
        )
    except Exception as exc:
        result.problems.append(
            _problem(PipelineStage.RESUME, "resume_failed", exc)
        )
        return result
    finally:
        result.cv_trace = observed_writer.calls
        result.correction_events = list(recovering.events)

    if any(
        event["kind"] in {
            "omitted_unsupported_skills", "ignored_invalid_merges",
            "conservative_fallback",
        }
        for event in result.correction_events
    ):
        result.warnings.append(
            "Model output required evidence-safe omissions, skipped alias groups "
            "or a conservative CV overview; inspect correction_events"
        )

    result.status = (
        PipelineStatus.PARTIAL
        if result.problems or result.warnings
        else PipelineStatus.COMPLETED
    )
    return result
