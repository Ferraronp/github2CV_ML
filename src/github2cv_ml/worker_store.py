"""Small durable per-repository cache and at-least-once job inbox.

Use a persistent volume, not container-local ephemeral storage. SQLite is an
MVP single-instance store; use shared PostgreSQL/object storage before scaling
ML workers across hosts. Payload hashes and leases prevent accidental reuse
of an idempotency key for different requests.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from github2cv_ml.agent.runner import RepoAnalysis
from github2cv_ml.pipeline import normalize_repository
from github2cv_ml.worker_contracts import AnalysisOptions, MLJobRequest, MLJobResult

ANALYZER_REVISION = "analyzer-v3-12read-2026-11"
MAX_ANALYSIS_BYTES = 8_000_000


class JobBusy(RuntimeError):
    """Another worker currently holds the unexpired message lease."""


def analysis_cache_key(
    url: str,
    commit_sha: str,
    options: AnalysisOptions,
    model_id: str,
    *,
    revision: str = ANALYZER_REVISION,
) -> str:
    """Only repository facts enter this key: never candidate/profile context."""
    data = {
        "repo": normalize_repository(url),
        "sha": commit_sha.lower(),
        "revision": revision,
        "model": model_id,
        "max_steps": options.max_steps,
        "max_reads": options.max_reads,
        "min_planned_reads": options.min_planned_reads,
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class WorkerStore:
    """Transactional SQLite inbox and RepoAnalysis cache for one ML worker."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=15, isolation_level=None)
        self.db.execute("PRAGMA busy_timeout = 15000")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                cache_key TEXT PRIMARY KEY,
                repo_key TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                payload_sha256 TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('processing', 'ready')),
                lease_until INTEGER NOT NULL,
                result_json TEXT,
                created_at INTEGER NOT NULL
            );
            """
        )

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "WorkerStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def find_analysis(
        self, key: str, *, repository: str, sha: str
    ) -> RepoAnalysis | None:
        row = self.db.execute(
            "SELECT repo_key, commit_sha, analysis_json FROM analyses WHERE cache_key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row[0] != normalize_repository(repository) or row[1] != sha.lower():
            raise ValueError("Cache identity mismatch: refusing wrong repository/SHA")
        result = RepoAnalysis.model_validate_json(row[2])
        actual = normalize_repository(
            f"{result.repository.owner}/{result.repository.name}"
        )
        if result.commit_sha.lower() != sha.lower() or actual != row[0]:
            raise ValueError("Cached RepoAnalysis has inconsistent source identity")
        return result

    def save_analysis(self, key: str, analysis: RepoAnalysis) -> None:
        raw = analysis.model_dump_json()
        if len(raw.encode("utf-8")) > MAX_ANALYSIS_BYTES:
            raise ValueError("RepoAnalysis exceeds cache storage limit")
        repo_key = normalize_repository(
            f"{analysis.repository.owner}/{analysis.repository.name}"
        )
        self.db.execute(
            """
            INSERT INTO analyses(cache_key, repo_key, commit_sha, analysis_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                analysis_json=excluded.analysis_json
            WHERE analyses.repo_key=excluded.repo_key
              AND analyses.commit_sha=excluded.commit_sha
            """,
            (key, repo_key, analysis.commit_sha.lower(), raw, int(time.time())),
        )

    def claim(self, request: MLJobRequest, *, lease_seconds: int = 3600) -> MLJobResult | None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        payload_hash = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        now = int(time.time())
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT payload_sha256, state, lease_until, result_json "
                "FROM messages WHERE message_id=?", (str(request.message_id),)
            ).fetchone()
            if row is not None:
                if row[0] != payload_hash:
                    raise ValueError("message_id reused for a different ML request")
                if row[1] == "ready":
                    return MLJobResult.model_validate_json(row[3])
                if row[2] > now:
                    raise JobBusy("ML job is already being processed")
                self.db.execute(
                    "UPDATE messages SET lease_until=? WHERE message_id=?",
                    (now + lease_seconds, str(request.message_id)),
                )
            else:
                self.db.execute(
                    "INSERT INTO messages VALUES(?, ?, 'processing', ?, NULL, ?)",
                    (str(request.message_id), payload_hash, now + lease_seconds, now),
                )
            return None
        finally:
            self.db.execute("COMMIT")

    def save_result(self, request: MLJobRequest, result: MLJobResult) -> None:
        if result.request_message_id != request.message_id or result.job_id != request.job_id:
            raise ValueError("Result is not for this job")
        cur = self.db.execute(
            """
            UPDATE messages SET state='ready', result_json=?, lease_until=0
            WHERE message_id=? AND state='processing'
            """,
            (result.model_dump_json(), str(request.message_id)),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Could not persist result for the claimed message")
