"""Offline broker delivery semantics: fake AMQP exchange and incoming message."""

import asyncio
import json
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

from test_pipeline import _worker_request

from github2cv_ml.pipeline import PipelineStatus
from github2cv_ml.rabbitmq_worker import RabbitMLWorker, RabbitSettings
from github2cv_ml.worker_contracts import MLJobRequest, MLJobResult, MLRepositoryResult
from github2cv_ml.worker_store import IdempotencyConflict


class Exchange:
    def __init__(self):
        self.published = []

    async def publish(self, outgoing, *, routing_key, mandatory, timeout):
        assert mandatory and timeout == 20
        self.published.append((outgoing, routing_key))


class Incoming:
    def __init__(self, body, *, message_id=None, correlation_id=None, kind=None):
        self.body = body
        self.message_id = message_id
        self.correlation_id = correlation_id
        self.type = kind
        self.acked = 0
        self.nacked = 0

    async def ack(self):
        self.acked += 1

    async def nack(self, *, requeue):
        assert requeue is True
        self.nacked += 1


def _failure(request):
    return MLJobResult(
        type="github.report.failed",
        message_id=uuid5(NAMESPACE_URL, "offline-test-result"),
        request_message_id=request.message_id,
        job_id=request.job_id,
        input_revision=request.input_revision,
        status=PipelineStatus.FAILED,
        repositories=[
            MLRepositoryResult(
                repo_id=repo.repo_id,
                repository_url=repo.repository_url,
                commit_sha=repo.commit_sha,
                context_hash=repo.context_hash,
                status="failed",
            ) for repo in request.repositories
        ],
        error="No source evidence",
    )


def test_worker_only_acks_after_publishing_a_valid_result(tmp_path):
    request = _worker_request("api")
    calls = []
    result = _failure(request)
    service = SimpleNamespace(
        db_path=tmp_path / "worker.sqlite",
        process=lambda r: (calls.append(r), result)[1],
    )
    worker = RabbitMLWorker(RabbitSettings(url="amqp://dummy/"), service)
    exchange, invalid = Exchange(), Exchange()
    worker._exchange, worker._dead_exchange = exchange, invalid
    incoming = Incoming(
        request.model_dump_json().encode(),
        message_id=str(request.message_id),
        correlation_id=str(request.job_id),
        kind=request.type,
    )
    asyncio.run(worker.consume(incoming))
    assert incoming.acked == 1 and incoming.nacked == 0
    assert calls == [request]
    assert len(exchange.published) == 1
    outgoing, routing_key = exchange.published[0]
    assert routing_key == "github.report.failed"
    assert outgoing.message_id == str(result.message_id)
    assert outgoing.correlation_id == str(request.job_id)
    assert outgoing.delivery_mode is not None
    assert MLJobResult.model_validate_json(outgoing.body) == result
    assert not invalid.published


def test_worker_dead_letters_invalid_payload_without_copying_sensitive_content(tmp_path):
    service = SimpleNamespace(
        db_path=tmp_path / "unused.sqlite",
        process=lambda r: (_ for _ in ()).throw(AssertionError("must not process")),
    )
    worker = RabbitMLWorker(RabbitSettings(url="amqp://dummy/"), service)
    worker._exchange, worker._dead_exchange = Exchange(), Exchange()
    payload = b'{"token":"secret-value","schema_version":1}'
    incoming = Incoming(payload)
    asyncio.run(worker.consume(incoming))
    assert incoming.acked == 1 and incoming.nacked == 0
    assert not worker._exchange.published
    outgoing, key = worker._dead_exchange.published[0]
    assert key == "invalid"
    info = json.loads(outgoing.body)
    assert info["type"] == "ml.input.invalid"
    assert info["bytes"] == len(payload)
    assert b"secret-value" not in outgoing.body


def test_worker_rejects_mismatched_correlation_header(tmp_path):
    request = _worker_request("api")
    service = SimpleNamespace(
        db_path=tmp_path / "unused.sqlite",
        process=lambda r: (_ for _ in ()).throw(AssertionError("must not process")),
    )
    worker = RabbitMLWorker(RabbitSettings(url="amqp://dummy/"), service)
    worker._exchange, worker._dead_exchange = Exchange(), Exchange()
    incoming = Incoming(
        request.model_dump_json().encode(),
        correlation_id="not-the-job-id",
    )
    asyncio.run(worker.consume(incoming))
    assert incoming.acked == 1
    assert worker._dead_exchange.published[0][1] == "invalid"


def test_envelope_roundtrip_produces_valid_json_and_known_routing():
    request = _worker_request("api", "worker")
    loaded = MLJobRequest.model_validate_json(request.model_dump_json())
    assert loaded.repositories[0].commit_sha == "a" * 40
    assert loaded.repositories[1].repo_id == "101"
    result = _failure(loaded)
    assert MLJobResult.model_validate_json(result.model_dump_json()) == result
    assert result.schema_version == 2



def test_idempotency_conflict_is_dead_lettered_without_processing_loop(tmp_path):
    request = _worker_request("api")

    def conflict(_request):
        raise IdempotencyConflict("UUID already used for a different payload")

    worker = RabbitMLWorker(
        RabbitSettings(url="amqp://dummy/"),
        SimpleNamespace(db_path=tmp_path / "unused.sqlite", process=conflict),
    )
    worker._exchange, worker._dead_exchange = Exchange(), Exchange()
    incoming = Incoming(request.model_dump_json().encode())
    asyncio.run(worker.consume(incoming))
    assert incoming.acked == 1 and incoming.nacked == 0
    assert not worker._exchange.published
    outgoing, key = worker._dead_exchange.published[0]
    assert key == "invalid"
    assert json.loads(outgoing.body)["reason"] == "idempotency_conflict"



def test_oversized_result_is_terminal_dlq_not_infinite_requeue(tmp_path):
    """Even the compact failed MLJobResult can exceed a tiny broker limit."""
    request = _worker_request("api")
    assert len(request.model_dump_json().encode()) < 1024

    # 20 result entries, but only one incoming repo: we exercise transport's
    # publish boundary independently of the already-validated model pipeline.
    repositories = [
        MLRepositoryResult(
            repo_id=str(i + 100),
            repository_url=f"https://github.com/example/very-long-repository-{i:02d}",
            commit_sha="a" * 40,
            context_hash="b" * 64,
            status="failed",
        ) for i in range(20)
    ]
    oversized = _failure(request).model_copy(update={"repositories": repositories})
    assert len(oversized.model_dump_json().encode()) > 1024
    service = SimpleNamespace(
        db_path=tmp_path / "unused.sqlite3",
        process=lambda _: oversized,  # Persisted ready result on every replay.
    )
    worker = RabbitMLWorker(RabbitSettings(url="amqp://dummy/", message_limit=1024), service)
    worker._exchange, worker._dead_exchange = Exchange(), Exchange()
    payload = request.model_dump_json().encode()

    for _ in range(2):
        incoming = Incoming(payload, message_id=str(request.message_id))
        asyncio.run(worker.consume(incoming))
        assert incoming.acked == 1
        assert incoming.nacked == 0

    assert not worker._exchange.published
    assert len(worker._dead_exchange.published) == 2
    for outgoing, key in worker._dead_exchange.published:
        assert key == "invalid"
        diagnostic = json.loads(outgoing.body)
        assert diagnostic["reason"] == "oversized_response"
        assert diagnostic["payload_sha256"]
        assert len(outgoing.body) < 1024


def test_small_limit_for_many_repository_request_dead_letters_without_loop(tmp_path):
    """If even the request exceeds the cap it must never run the model."""
    request = _worker_request(*(f"repo-{i}" for i in range(20)))
    payload = request.model_dump_json().encode()
    assert len(payload) > 1024
    service = SimpleNamespace(
        db_path=tmp_path / "unused.sqlite3",
        process=lambda _: (_ for _ in ()).throw(AssertionError("no inference")),
    )
    worker = RabbitMLWorker(RabbitSettings(url="amqp://dummy/", message_limit=1024), service)
    worker._exchange, worker._dead_exchange = Exchange(), Exchange()
    incoming = Incoming(payload)
    asyncio.run(worker.consume(incoming))
    assert incoming.acked == 1 and incoming.nacked == 0
    assert worker._dead_exchange.published[0][1] == "invalid"
