"""Embedded application integration with a fake async gateway client.

The fake client stands in for an existing direct SDK, third-party gateway, or
internal application service. It performs no retries: ABL owns transport and
application-level retries in this example.

Run without credentials or network access:

    uv run python examples/example_callable_application.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from async_batch_llm import (
    ArtifactIdentity,
    BatchResult,
    CallableStrategy,
    CallOutcome,
    GuardrailConfig,
    JsonlArtifactStore,
    ProcessorConfig,
    ResumePolicy,
    RetryConfig,
    RetryState,
    TokenTrackingError,
    WorkItemResult,
    process_stream,
)


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    version: int


@dataclass(frozen=True)
class FakeGatewayResponse:
    text: str
    usage: Mapping[str, int]
    route: str = "local-demo"


@dataclass
class DemoReport:
    config: ProcessorConfig
    first_client_calls: int
    resumed_client_calls: int
    first_results: list[WorkItemResult[dict[str, str], Any]]
    resumed_results: list[WorkItemResult[dict[str, str], Any]]
    checkpoint_verified_ids: list[str]
    feedback_by_document: dict[str, list[str | None]]
    first_client_prepared: int
    first_client_closed: int
    artifact_item_records: int


class DocumentRepository:
    """Small stand-in for a paginated database repository."""

    def __init__(self) -> None:
        self._documents = [
            Document(f"doc-{index}", f"application text {index}", version=1) for index in range(5)
        ]

    async def iter_documents(self, *, page_size: int = 2) -> AsyncIterator[Document]:
        for offset in range(0, len(self._documents), page_size):
            await asyncio.sleep(0)  # one asynchronous page fetch
            for document in self._documents[offset : offset + page_size]:
                yield document


class FakeApplicationGateway:
    """A local fake with the shape of an existing asynchronous dependency."""

    _invalid_first = frozenset({"doc-1", "doc-3"})

    def __init__(self) -> None:
        self.calls = 0
        self.prepared = 0
        self.closed = 0
        self._calls_by_document: dict[str, int] = {}
        self.feedback_by_document: dict[str, list[str | None]] = {}

    async def open(self) -> None:
        self.prepared += 1

    async def close(self) -> None:
        self.closed += 1

    async def classify(
        self,
        prompt: str,
        *,
        feedback: str | None,
        timeout: float,
    ) -> FakeGatewayResponse:
        """Return billed text; retry policy deliberately belongs to ABL."""
        payload = json.loads(prompt)
        document_id = str(payload["document_id"])
        if feedback is not None and document_id not in feedback:
            raise AssertionError("retry feedback leaked between work items")

        self.calls += 1
        call_number = self._calls_by_document.get(document_id, 0) + 1
        self._calls_by_document[document_id] = call_number
        self.feedback_by_document.setdefault(document_id, []).append(feedback)

        # Deterministic latency makes completion order differ from input order.
        delay = {"doc-0": 0.03, "doc-2": 0.005, "doc-4": 0.001}.get(document_id, 0)
        await asyncio.sleep(min(delay, timeout / 2))

        if document_id in self._invalid_first and call_number == 1:
            return FakeGatewayResponse(
                text="not-json",
                usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )
        return FakeGatewayResponse(
            text=json.dumps({"document_id": document_id, "label": "keep"}),
            usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        )


class TransactionalResultSink:
    """Stand-in for transactional database writes made by a stream consumer."""

    def __init__(self, artifact_path: Path) -> None:
        self.artifact_path = artifact_path
        self.rows: dict[str, dict[str, str]] = {}
        self.write_order: list[str] = []
        self.checkpoint_verified_ids: list[str] = []

    async def save(self, result: WorkItemResult[dict[str, str], Any]) -> None:
        # JsonlArtifactStore flushes before publication, so the item record is
        # already visible when application code receives the streamed result.
        records = [
            json.loads(line) for line in self.artifact_path.read_text(encoding="utf-8").splitlines()
        ]
        assert any(
            record.get("record_type") == "item" and record.get("item_id") == result.item_id
            for record in records
        )
        self.checkpoint_verified_ids.append(result.item_id)

        await asyncio.sleep(0)  # begin/commit an asynchronous transaction
        if result.success and result.output is not None:
            self.rows[result.item_id] = result.output
        self.write_order.append(result.item_id)


IDENTITY = ArtifactIdentity(
    provider="application-gateway",
    model="local-classifier-route",
    prompt_version="document-label-v1",
    parser_version="json-label-v1",
    application_version="embedded-demo-v1",
)


def build_strategy(client: FakeApplicationGateway) -> CallableStrategy[dict[str, str]]:
    async def invoke(
        prompt: str,
        *,
        attempt: int,
        timeout: float,
        state: RetryState | None,
    ) -> CallOutcome[dict[str, str]]:
        feedback = None if state is None else state.get("validation_feedback")
        response = await client.classify(prompt, feedback=feedback, timeout=timeout)
        try:
            parsed = json.loads(response.text)
            if not isinstance(parsed, dict) or set(parsed) != {"document_id", "label"}:
                raise ValueError("expected document_id and label")
        except (json.JSONDecodeError, ValueError) as exc:
            document_id = str(json.loads(prompt)["document_id"])
            billed = TokenTrackingError(
                f"billed response failed validation for {document_id}",
                token_usage=dict(response.usage),
            )
            billed.__dict__["document_id"] = document_id
            raise billed from exc
        return CallOutcome(
            output={str(key): str(value) for key, value in parsed.items()},
            token_usage=response.usage,
            metadata={"route": response.route, "logical_attempt": attempt},
        )

    def on_error(error: Exception, attempt: int, state: RetryState | None) -> None:
        if state is not None and isinstance(error, TokenTrackingError):
            document_id = str(error.__dict__["document_id"])
            state.set(
                "validation_feedback",
                f"For {document_id}, return one JSON object with document_id and label.",
            )

    return CallableStrategy(
        invoke,
        identity=IDENTITY,
        prepare=client.open,
        cleanup=client.close,
        on_error=on_error,
        max_concurrency=3,
        concurrency_scope=client,
    )


CONFIG = ProcessorConfig(
    concurrency=3,
    max_queue_size=4,
    max_result_queue_size=2,
    attempt_timeout=1,
    retry=RetryConfig(
        max_attempts=2,
        initial_wait=0.001,
        max_wait=0.001,
        jitter=False,
    ),
    guardrails=GuardrailConfig(
        total_timeout_per_item=5,
        batch_timeout=30,
        # Authentication is safe to fail fast because retries cannot repair it.
        abort_on_error_categories=frozenset({"authentication"}),
    ),
)


async def prompt_source(repository: DocumentRepository) -> AsyncIterator[tuple[str, str, int]]:
    async for document in repository.iter_documents():
        prompt = json.dumps(
            {"document_id": document.document_id, "text": document.text}, sort_keys=True
        )
        yield document.document_id, prompt, document.version


def _artifact_item_count(path: Path) -> int:
    return sum(
        json.loads(line).get("record_type") == "item"
        for line in path.read_text(encoding="utf-8").splitlines()
    )


async def run_demo(
    artifact_path: Path,
    *,
    verbose: bool = True,
    progress: bool = False,
) -> DemoReport:
    repository = DocumentRepository()
    first_client = FakeApplicationGateway()
    first_sink = TransactionalResultSink(artifact_path)
    first_results: list[WorkItemResult[dict[str, str], Any]] = []

    async for result in process_stream(
        build_strategy(first_client),
        prompt_source(repository),
        config=CONFIG,
        progress=progress,
        artifact_store=JsonlArtifactStore(artifact_path),
    ):
        await first_sink.save(result)
        first_results.append(result)  # only for this small demo's summary

    # The same identity and inputs replay compatible successes. The callable is
    # never invoked and replay records are not appended a second time.
    resumed_client = FakeApplicationGateway()
    resumed_sink = TransactionalResultSink(artifact_path)
    resumed_results: list[WorkItemResult[dict[str, str], Any]] = []
    async for result in process_stream(
        build_strategy(resumed_client),
        prompt_source(DocumentRepository()),
        config=CONFIG,
        progress=progress,
        artifact_store=JsonlArtifactStore(artifact_path),
        resume=ResumePolicy.REUSE_SUCCESSES,
    ):
        await resumed_sink.save(result)
        resumed_results.append(result)

    report = DemoReport(
        config=CONFIG,
        first_client_calls=first_client.calls,
        resumed_client_calls=resumed_client.calls,
        first_results=first_results,
        resumed_results=resumed_results,
        checkpoint_verified_ids=first_sink.checkpoint_verified_ids,
        feedback_by_document=first_client.feedback_by_document,
        first_client_prepared=first_client.prepared,
        first_client_closed=first_client.closed,
        artifact_item_records=_artifact_item_count(artifact_path),
    )

    if verbose:
        batch = BatchResult(results=first_results)
        print(batch.summary())
        print("completion order:", [result.item_id for result in first_results])
        print("input order:", [result.item_id for result in batch.in_input_order().results])
        print("validation retries recovered: doc-1, doc-3")
        print("live client calls:", first_client.calls)
        print("resume client calls:", resumed_client.calls)
    return report


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="abl-callable-demo-") as directory:
        await run_demo(Path(directory) / "documents.jsonl", progress=True)


if __name__ == "__main__":
    asyncio.run(main())
