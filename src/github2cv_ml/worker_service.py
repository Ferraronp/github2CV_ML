"""Pure processing adapter between ResumeHub MQ v2 envelopes and ML pipeline.

No RabbitMQ dependencies here: tests exercise this layer without a broker.
Individual RepoAnalysis snapshots and the idempotent result are durable before
the consumer publishes a completed/failed event. A backend can migrate the
store from SQLite to a shared artifact service without touching CV contracts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from github2cv_ml.agent.runner import ChatModel, RepoAnalysis
from github2cv_ml.pipeline import (
    PipelineProblem,
    PipelineRequest,
    PipelineResult,
    PipelineStatus,
    normalize_repository,
    run_pipeline,
)
from github2cv_ml.worker_contracts import (
    MLJobRequest,
    MLJobResult,
    MLRepositoryResult,
)
from github2cv_ml.worker_store import WorkerStore, analysis_cache_key

logger = logging.getLogger(__name__)
MAX_EVENT_BYTES = 4_000_000


def response_id(request: MLJobRequest):
    return uuid5(NAMESPACE_URL, f"resumehub.ml.v2:{request.message_id}:{request.job_id}")


def _public_problems(problems: list[PipelineProblem]) -> list[PipelineProblem]:
    """No raw exception strings or untrusted repository text in MQ responses."""
    return [
        PipelineProblem(
            stage=item.stage,
            code=item.code,
            message=(
                "No source evidence found"
                if item.code == "no_evidence"
                else "Processing failed; inspect ML worker logs with job_id"
            ),
            repository=item.repository,
        )
        for item in problems
    ]


def _result_for(
    request: MLJobRequest,
    pipeline: PipelineResult,
    hits: set[str],
) -> MLJobResult:
    by_repo = {
        normalize_repository(
            f"{analysis.repository.owner}/{analysis.repository.name}"
        ): analysis for analysis in pipeline.analyses
    }
    problems = _public_problems(pipeline.problems)
    repositories: list[MLRepositoryResult] = []
    for original in request.repositories:
        key = normalize_repository(original.repository_url)
        analysis = by_repo.get(key)
        issues = [p for p in problems if p.repository == key]
        if analysis is None or not analysis.evidence:
            status = "failed"
        elif analysis.status != "completed":
            status = "partial"
        else:
            status = "completed"
        repositories.append(MLRepositoryResult(
            repo_id=original.repo_id,
            repository_url=original.repository_url,
            commit_sha=original.commit_sha,
            context_hash=original.context_hash,
            status=status,
            cache_hit=key in hits,
            analysis=analysis,
            problems=issues,
        ))
    has_resume = pipeline.resume is not None
    return MLJobResult(
        type="github.report.completed" if has_resume else "github.report.failed",
        message_id=response_id(request),
        request_message_id=request.message_id,
        job_id=request.job_id,
        input_revision=request.input_revision,
        status=pipeline.status if has_resume else PipelineStatus.FAILED,
        repositories=repositories,
        profile=pipeline.profile if has_resume else None,
        resume=pipeline.resume if has_resume else None,
        problems=problems,
        warnings=pipeline.warnings,
        error=None if has_resume else "ML pipeline did not produce an evidence-backed CV",
    )


class JobProcessor:
    """Backend-facing processing facade, runnable in a single background thread."""

    def __init__(
        self,
        *,
        db_path: Path,
        github_client: Any,
        model: ChatModel,
        model_id: str,
        max_event_bytes: int = MAX_EVENT_BYTES,
    ):
        self.db_path = db_path
        self.github_client = github_client
        self.model = model
        self.model_id = model_id
        self.max_event_bytes = max_event_bytes

    def process(self, request: MLJobRequest) -> MLJobResult:
        started = time.monotonic()
        with WorkerStore(self.db_path) as store:
            previous = store.claim(request)
            if previous is not None:
                logger.info(
                    "ml_job_replayed",
                    extra={"job_id": str(request.job_id), "message_id": str(request.message_id)},
                )
                return previous
            try:
                result = self._run(request, store)
                # A large response should not get stuck in an infinite republish
                # cycle: failed event is small, while per-repo cache stays local.
                if len(result.model_dump_json().encode("utf-8")) > self.max_event_bytes:
                    result = MLJobResult(
                        type="github.report.failed",
                        message_id=response_id(request),
                        request_message_id=request.message_id,
                        job_id=request.job_id,
                        input_revision=request.input_revision,
                        status=PipelineStatus.FAILED,
                        repositories=[
                            MLRepositoryResult(
                                repo_id=item.repo_id,
                                repository_url=item.repository_url,
                                commit_sha=item.commit_sha,
                                context_hash=item.context_hash,
                                status="failed",
                            ) for item in request.repositories
                        ],
                        error="Result exceeded RabbitMQ message limit; cached analyses retained",
                    )
                store.save_result(request, result)
            except Exception as exc:
                if store.attempt_count(str(request.message_id)) >= 3:
                    # Poison jobs are terminal after three durable processing
                    # attempts; publishing may still retry using the same result.
                    result = MLJobResult(
                        type="github.report.failed",
                        message_id=response_id(request),
                        request_message_id=request.message_id,
                        job_id=request.job_id,
                        input_revision=request.input_revision,
                        status=PipelineStatus.FAILED,
                        repositories=[
                            MLRepositoryResult(
                                repo_id=item.repo_id,
                                repository_url=item.repository_url,
                                commit_sha=item.commit_sha,
                                context_hash=item.context_hash,
                                status="failed",
                            ) for item in request.repositories
                        ],
                        error="ML worker failed after 3 attempts; inspect worker logs",
                    )
                    store.save_result(request, result)
                    logger.error(
                        "ml_job_exhausted",
                        extra={"job_id": str(request.job_id), "stage": type(exc).__name__},
                    )
                else:
                    store.release(str(request.message_id))
                    raise
        logger.info(
            "ml_job_processed",
            extra={
                "job_id": str(request.job_id),
                "message_id": str(request.message_id),
                "status": result.status,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "repositories": len(request.repositories),
            },
        )
        return result

    def _run(self, request: MLJobRequest, store: WorkerStore) -> MLJobResult:
        keys = {
            normalize_repository(item.repository_url): analysis_cache_key(
                item.repository_url, item.commit_sha, request.options, self.model_id
            )
            for item in request.repositories
        }
        hits: set[str] = set()
        cached: list[RepoAnalysis] = []
        fresh: list[str] = []
        pinned: dict[str, str] = {}

        for item in request.repositories:
            key = normalize_repository(item.repository_url)
            found = None
            if not request.options.force_refresh:
                found = store.find_analysis(
                    keys[key], repository=key, sha=item.commit_sha
                )
            if found is not None:
                hits.add(key)
                cached.append(found)
            else:
                fresh.append(item.repository_url)
                pinned[key] = item.commit_sha

        logger.info(
            "ml_cache_plan",
            extra={
                "job_id": str(request.job_id),
                "cache_hits": len(hits),
                "cache_misses": len(fresh),
            },
        )

        def persist(analysis: RepoAnalysis) -> None:
            key = normalize_repository(
                f"{analysis.repository.owner}/{analysis.repository.name}"
            )
            store.save_analysis(keys[key], analysis)

        request_core = PipelineRequest(
            candidate=request.candidate,
            repositories=fresh,
            max_steps=request.options.max_steps,
            max_reads=request.options.max_reads,
            min_planned_reads=request.options.min_planned_reads,
            max_skills=request.options.max_skills,
            max_input_chars=request.options.max_input_chars,
        )
        pipeline = run_pipeline(
            request_core,
            github_client=self.github_client if fresh else None,
            model=self.model,
            cached_analyses=cached,
            pinned_shas=pinned,
            on_analysis=persist,
        )
        return _result_for(request, pipeline, hits)
