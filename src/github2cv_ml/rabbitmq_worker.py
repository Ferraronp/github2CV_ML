"""RabbitMQ ML Core consumer for ResumeHub's existing durable topology.

Install the worker extra: pip install -e '.[worker]'.
Uses the backend's exact exchange and routing keys, but *v2* structured
message bodies. A backend DTO migration is needed before replacing demo-core.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
from pydantic import ValidationError

from github2cv_ml.agent.runner import LocalOpenAIModel
from github2cv_ml.collectors.github_client import GitHubApiClient
from github2cv_ml.worker_contracts import MLJobRequest
from github2cv_ml.worker_service import MAX_EVENT_BYTES, JobProcessor
from github2cv_ml.worker_store import IdempotencyConflict, JobBusy, LeaseLost

logger = logging.getLogger("github2cv_ml.rabbitmq_worker")


@dataclass(frozen=True)
class RabbitSettings:
    url: str
    exchange: str = "resumehub.ml"
    request_queue: str = "resumehub.ml.requests"
    result_queue: str = "resumehub.ml.results"
    dead_letter_queue: str = "resumehub.ml.invalid-results"
    request_key: str = "github.report.requested"
    completed_key: str = "github.report.completed"
    failed_key: str = "github.report.failed"
    message_limit: int = MAX_EVENT_BYTES

    @classmethod
    def from_env(cls) -> "RabbitSettings":
        return cls(
            url=os.getenv(
                "RABBITMQ_URL",
                "amqp://resumehub:resumehub@localhost:5672/",
            ),
            exchange=os.getenv("ML_EXCHANGE", "resumehub.ml"),
            request_queue=os.getenv("ML_REQUEST_QUEUE", "resumehub.ml.requests"),
            result_queue=os.getenv("ML_RESULT_QUEUE", "resumehub.ml.results"),
            dead_letter_queue=os.getenv(
                "ML_DEAD_LETTER_QUEUE", "resumehub.ml.invalid-results"
            ),
            request_key=os.getenv("ML_REQUEST_KEY", "github.report.requested"),
            completed_key=os.getenv("ML_COMPLETED_KEY", "github.report.completed"),
            failed_key=os.getenv("ML_FAILED_KEY", "github.report.failed"),
            message_limit=int(os.getenv("ML_MESSAGE_BYTES", str(MAX_EVENT_BYTES))),
        )


class JsonFormatter(logging.Formatter):
    """No payloads, raw errors, tokens or source excerpts in normal logs."""

    def format(self, record: logging.LogRecord) -> str:
        result = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for field in (
            "job_id", "message_id", "repository", "stage",
            "status", "elapsed_seconds", "cache_hits", "cache_misses",
            "repositories",
        ):
            value = getattr(record, field, None)
            if value is not None:
                result[field] = value
        return json.dumps(result, ensure_ascii=False)


class RabbitMLWorker:
    def __init__(self, settings: RabbitSettings, processor: JobProcessor):
        self.settings = settings
        self.processor = processor
        self._exchange = None
        self._dead_exchange = None

    async def _publish_invalid(self, data: bytes, reason: str) -> None:
        """Dead-letter invalid input without copying untrusted raw JSON/secrets."""
        if self._dead_exchange is None:
            raise RuntimeError("RabbitMQ dead-letter exchange is unavailable")
        body = json.dumps({
            "type": "ml.input.invalid",
            "reason": reason,
            "payload_sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }).encode()
        await self._dead_exchange.publish(
            aio_pika.Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                type="ml.input.invalid",
            ),
            routing_key="invalid",
            mandatory=True,
            timeout=20,
        )

    async def consume(self, message: AbstractIncomingMessage) -> None:
        try:
            if len(message.body) > self.settings.message_limit:
                raise ValueError("message_too_large")
            request = MLJobRequest.model_validate_json(message.body)
            if (
                (message.type and message.type != request.type)
                or (message.message_id and message.message_id != str(request.message_id))
                or (message.correlation_id and message.correlation_id != str(request.job_id))
            ):
                raise ValueError("message_headers_mismatch")
        except (ValueError, ValidationError):
            try:
                await self._publish_invalid(message.body, "invalid_schema_or_headers")
                await message.ack()
            except Exception:
                await asyncio.sleep(3)
                await message.nack(requeue=True)
            return

        job_id = str(request.job_id)
        try:
            result = await asyncio.to_thread(self.processor.process, request)
            body = result.model_dump_json().encode()
            if len(body) > self.settings.message_limit:
                # Even the compact MLJobResult can exceed a tiny limit when
                # the request has many repositories. This is permanent, not
                # a transport failure: put a bounded diagnostic in DLQ and
                # ACK only after the DLQ publisher confirms.
                await self._publish_invalid(message.body, "oversized_response")
                await message.ack()
                logger.error(
                    "ml_result_oversized_dead_lettered",
                    extra={"job_id": job_id, "message_id": str(request.message_id)},
                )
                return
            routing_key = (
                self.settings.completed_key
                if result.type == "github.report.completed"
                else self.settings.failed_key
            )
            if self._exchange is None:
                raise RuntimeError("RabbitMQ exchange not ready")
            await self._exchange.publish(
                aio_pika.Message(
                    body=body,
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    type=result.type,
                    message_id=str(result.message_id),
                    correlation_id=job_id,
                ),
                routing_key=routing_key,
                mandatory=True,
                timeout=20,
            )
            # Publisher confirm completed: backend will separately ACK after DB commit.
            await message.ack()
            logger.info(
                "ml_result_published",
                extra={"job_id": job_id, "message_id": str(request.message_id),
                       "status": result.status},
            )
        except IdempotencyConflict:
            # A different payload with the same UUID is a permanent error.
            # Never ACK until the invalid-event publisher confirm succeeds.
            try:
                await self._publish_invalid(message.body, "idempotency_conflict")
                await message.ack()
            except Exception:
                await asyncio.sleep(5)
                await message.nack(requeue=True)
        except LeaseLost:
            # A successor has reclaimed the message. Never publish the stale
            # result; the successor (or the persisted ready result) will win.
            logger.warning("ml_stale_attempt_discarded", extra={"job_id": job_id})
            await asyncio.sleep(5)
            await message.nack(requeue=True)
        except JobBusy:
            logger.warning("ml_duplicate_inflight", extra={"job_id": job_id})
            await asyncio.sleep(5)
            await message.nack(requeue=True)
        except Exception as exc:
            # Transient broker/DB/process errors are requeued, not silently lost.
            # Persisted result is replayed after successful publisher confirm.
            logger.warning(
                "ml_job_retry",
                extra={"job_id": job_id, "stage": type(exc).__name__},
            )
            await asyncio.sleep(5)
            await message.nack(requeue=True)
    async def run(self) -> None:
        connection = await aio_pika.connect_robust(self.settings.url)
        try:
            channel = await connection.channel(
                publisher_confirms=True, on_return_raises=True
            )
            await channel.set_qos(prefetch_count=1)
            exchange = await channel.declare_exchange(
                self.settings.exchange, aio_pika.ExchangeType.DIRECT, durable=True
            )
            dead = await channel.declare_exchange(
                self.settings.exchange + ".dlx",
                aio_pika.ExchangeType.DIRECT,
                durable=True,
            )
            rejected = await channel.declare_queue(
                self.settings.dead_letter_queue, durable=True
            )
            await rejected.bind(dead, routing_key="invalid")
            requests = await channel.declare_queue(
                self.settings.request_queue, durable=True
            )
            await requests.bind(exchange, routing_key=self.settings.request_key)
            responses = await channel.declare_queue(
                self.settings.result_queue,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": dead.name,
                    "x-dead-letter-routing-key": "invalid",
                },
            )
            for key in (self.settings.completed_key, self.settings.failed_key):
                await responses.bind(exchange, routing_key=key)
            self._exchange = exchange
            self._dead_exchange = dead
            tag = await requests.consume(self.consume)
            logger.info("ml_worker_ready")
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                with suppress(NotImplementedError, RuntimeError):
                    loop.add_signal_handler(sig, stop.set)
            try:
                await stop.wait()
            finally:
                await requests.cancel(tag)
        finally:
            await connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=Path,
        default=Path(os.getenv("ML_STORE_PATH", "results/worker/ml.sqlite3")),
    )
    parser.add_argument(
        "--base-url", default=os.getenv("ML_MODEL_BASE_URL", "http://127.0.0.1:8080/v1"),
    )
    parser.add_argument("--model", default=os.getenv("ML_MODEL_ID", "qwen-3.5"))
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--allow-remote-model", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger().handlers[0].setFormatter(JsonFormatter())
    logging.getLogger("aio_pika").setLevel(logging.CRITICAL)
    logging.getLogger("aiormq").setLevel(logging.CRITICAL)

    processor = JobProcessor(
        db_path=args.store,
        github_client=GitHubApiClient(token=os.getenv("GITHUB_TOKEN")),
        model=LocalOpenAIModel(
            base_url=args.base_url, model=args.model,
            temperature=args.temperature, max_tokens=args.max_output_tokens,
            allow_remote=args.allow_remote_model
            or os.getenv("ML_ALLOW_REMOTE_MODEL") == "1",
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
        ),
        model_id=args.model,
        max_event_bytes=RabbitSettings.from_env().message_limit,
    )
    try:
        asyncio.run(RabbitMLWorker(RabbitSettings.from_env(), processor).run())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("ml_worker_stopped", extra={"stage": type(exc).__name__})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
