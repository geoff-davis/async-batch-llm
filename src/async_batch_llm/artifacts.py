"""Versioned JSONL audit artifacts and compatible result replay.

The JSONL implementation is concurrency-safe within one store instance and
process. It does not claim cross-process append safety; applications needing
multiple writers must provide an :class:`ArtifactStore` with real file locking
or a transactional backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, TextIO, TypeAlias

from .base import BatchResult, BatchTermination, LLMWorkItem, WorkItemResult
from .serialization import (
    JSONValue,
    ResultSerializationError,
    ValueDecoder,
    ValueEncoder,
    _to_fingerprint_value,
    to_json_value,
    work_item_result_from_dict,
    work_item_result_to_dict,
)

ARTIFACT_SCHEMA_VERSION = 1
CostCalculator: TypeAlias = Callable[[WorkItemResult[Any, Any]], float | None]
ContextFingerprinter: TypeAlias = Callable[[Any], str]
_ReplayKey: TypeAlias = tuple[str, str, str | None, str, str]


class ArtifactError(RuntimeError):
    """Base class for artifact preparation, format, and persistence failures."""


class ArtifactSerializationError(ArtifactError):
    """An artifact identity/input/result could not be canonically serialized."""


class ArtifactIOError(ArtifactError):
    """An artifact could not be read, written, flushed, or closed."""


class ArtifactFormatError(ArtifactError):
    """An artifact is malformed or uses an unsupported schema version."""


@dataclass(frozen=True)
class ArtifactIdentity:
    """Caller-supplied provenance used to decide whether replay is compatible."""

    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    parser_version: str | None = None
    application_version: str | None = None
    extra: Mapping[str, JSONValue] = field(default_factory=dict)


# Deterministic provider labels for the built-in model classes, used when
# inferring an identity from a strategy (zero-config artifacts, issue #99).
_PROVIDER_BY_MODEL_CLASS = {
    "GeminiModel": "gemini",
    "GeminiCachedModel": "gemini",
    "OpenAIModel": "openai",
    "OpenRouterModel": "openrouter",
    "DeepSeekModel": "deepseek",
}

_UNVERSIONED = "unversioned"


def infer_artifact_identity(strategy: Any) -> ArtifactIdentity:
    """Derive a deterministic :class:`ArtifactIdentity` from a strategy.

    Used by :class:`JsonlArtifactStore` when no explicit identity is given
    (v0.19.0, issue #99). ``provider`` and ``model`` come from the strategy's
    wrapped model (built-in model classes map to their provider name; other
    models use their class name); the version fields default to
    ``"unversioned"``. The result is deterministic for the same strategy
    setup across processes, so resume keeps working — and changing the model
    changes the identity fingerprint, which invalidates reuse.

    Prompt (and, by default, context) always participate in the per-item
    compatibility fingerprint regardless of identity, so a changed prompt
    never silently replays a stale result even with a defaulted identity.
    """
    provider: str | None = None
    model_id: str | None = None

    model_obj = getattr(strategy, "model", None)
    if model_obj is not None:
        raw_model = getattr(model_obj, "_model", None)
        if isinstance(raw_model, str) and raw_model:
            model_id = raw_model
        provider = _PROVIDER_BY_MODEL_CLASS.get(type(model_obj).__name__)
        if provider is None:
            provider = type(model_obj).__name__
    else:
        # PydanticAIStrategy and similar wrappers expose an agent.
        agent = getattr(strategy, "agent", None)
        agent_model = getattr(agent, "model", None) if agent is not None else None
        if isinstance(agent_model, str) and agent_model:
            model_id = agent_model
        elif agent_model is not None:
            name = getattr(agent_model, "model_name", None)
            if isinstance(name, str) and name:
                model_id = name

    if provider is None:
        provider = type(strategy).__name__

    return ArtifactIdentity(
        provider=provider,
        model=model_id or "unknown",
        prompt_version=_UNVERSIONED,
        parser_version=_UNVERSIONED,
        application_version=_UNVERSIONED,
    )


class ResumePolicy(str, Enum):
    """Which compatible terminal artifact records may bypass provider work."""

    NONE = "none"
    REUSE_SUCCESSES = "reuse_successes"
    REUSE_ALL = "reuse_all"


@dataclass(frozen=True)
class _ItemFingerprint:
    prompt: str
    context: str | None
    combined: str


class ArtifactStore(Protocol):
    """Provider-neutral asynchronous checkpoint/replay store."""

    async def prepare_item(self, work_item: LLMWorkItem[Any, Any, Any]) -> Any:
        """Prepare the run and validate/fingerprint an item before execution."""

    async def lookup(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        policy: ResumePolicy,
    ) -> WorkItemResult[Any, Any] | None:
        """Return the newest compatible reusable result, if any."""

    async def append(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        result: WorkItemResult[Any, Any],
    ) -> None:
        """Durably append one newly executed terminal result."""

    def iter_results(self, *, successes_only: bool = False) -> AsyncIterator[WorkItemResult]:
        """Iterate stored results without starting a processor."""

    async def close(self) -> None:
        """Flush and close the store; repeated calls are safe."""


def _package_version() -> str:
    try:
        return version("async-batch-llm")
    except PackageNotFoundError:
        return "0.0.0+dev"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: JSONValue) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256(value: JSONValue) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _identity_mapping(
    identity: ArtifactIdentity, *, for_fingerprint: bool = False
) -> dict[str, JSONValue]:
    try:
        converter = _to_fingerprint_value if for_fingerprint else to_json_value
        extra = converter(identity.extra, path="$.identity.extra")
    except ResultSerializationError as exc:
        raise ArtifactSerializationError(str(exc)) from exc
    if not isinstance(extra, dict):  # Mapping above always normalizes to dict.
        raise ArtifactSerializationError("ArtifactIdentity.extra must serialize to an object")
    return {
        "provider": identity.provider,
        "model": identity.model,
        "prompt_version": identity.prompt_version,
        "parser_version": identity.parser_version,
        "application_version": identity.application_version,
        "extra": extra,
    }


def _read_artifact_records(path: Path, *, allow_create: bool) -> tuple[list[dict[str, Any]], bool]:
    """Read and validate an artifact, optionally treating missing/empty as new."""
    try:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
    except OSError as exc:
        raise ArtifactIOError(f"Could not inspect artifact {path}: {exc}") from exc
    if not exists:
        if allow_create:
            return [], True
        raise ArtifactIOError(f"Artifact does not exist: {path}")
    if size == 0:
        if allow_create:
            return [], True
        raise ArtifactFormatError(f"Artifact is empty: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactIOError(f"Could not read artifact {path}: {exc}") from exc
    segments = raw.split(b"\n")
    has_trailing_newline = raw.endswith(b"\n")
    records: list[dict[str, Any]] = []
    manifest_seen = False
    for index, segment in enumerate(segments):
        if not segment:
            continue
        line_number = index + 1
        try:
            value = json.loads(segment)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_truncated_final = index == len(segments) - 1 and not has_trailing_newline
            if is_truncated_final:
                break
            raise ArtifactFormatError(
                f"Malformed artifact JSON at non-final line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ArtifactFormatError(f"Artifact line {line_number} must be a JSON object")
        schema = value.get("artifact_schema_version")
        if schema != ARTIFACT_SCHEMA_VERSION:
            if isinstance(schema, int) and schema > ARTIFACT_SCHEMA_VERSION:
                raise ArtifactFormatError(
                    f"Unsupported future artifact schema version {schema} at line {line_number}"
                )
            raise ArtifactFormatError(
                f"Unsupported artifact schema version {schema!r} at line {line_number}"
            )
        record_type = value.get("record_type")
        if not manifest_seen:
            if record_type != "manifest":
                raise ArtifactFormatError("The first complete artifact record must be a manifest")
            manifest_seen = True
            continue
        if record_type != "item":
            raise ArtifactFormatError(
                f"Unsupported artifact record_type {record_type!r} at line {line_number}"
            )
        records.append(value)
    if not manifest_seen:
        raise ArtifactFormatError("Artifact has no complete manifest record")
    return records, False


class JsonlArtifactStore:
    """Append-only version-1 JSONL artifact store.

    Prompt and context text are excluded by default; their SHA-256 hashes are
    always recorded for compatibility. Output and metadata are included by
    default because successful replay requires output. Set ``include_output``
    false for audit-only artifacts; those successful records are not replayable.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        identity: ArtifactIdentity | None = None,
        user_metadata: Mapping[str, JSONValue] | None = None,
        include_output: bool = True,
        include_metadata: bool = True,
        include_prompt: bool = False,
        include_context: bool = False,
        context_in_identity: bool = True,
        encoder: ValueEncoder | None = None,
        output_decoder: ValueDecoder | None = None,
        context_decoder: ValueDecoder | None = None,
        context_fingerprinter: ContextFingerprinter | None = None,
        cost_calculator: CostCalculator | None = None,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.identity = identity
        self.user_metadata = user_metadata or {}
        self.include_output = include_output
        self.include_metadata = include_metadata
        self.include_prompt = include_prompt
        self.include_context = include_context
        self.context_in_identity = context_in_identity
        self.encoder = encoder
        self.output_decoder = output_decoder
        self.context_decoder = context_decoder
        self.context_fingerprinter = context_fingerprinter
        self.cost_calculator = cost_calculator
        self.fsync = fsync

        # With no explicit identity, resolution is deferred to the first
        # prepare_item() call, which infers provider/model from the item's
        # strategy (zero-config artifacts, v0.19.0). Until then the
        # fingerprint is unset and lookup/append refuse to run.
        self._identity_value: dict[str, JSONValue] | None = None
        self.identity_fingerprint: str | None = None
        if identity is not None:
            self._identity_value = _identity_mapping(identity)
            self.identity_fingerprint = _sha256(_identity_mapping(identity, for_fingerprint=True))
        try:
            metadata_value = to_json_value(self.user_metadata, path="$.user_metadata")
        except ResultSerializationError as exc:
            raise ArtifactSerializationError(str(exc)) from exc
        if not isinstance(metadata_value, dict):
            raise ArtifactSerializationError("user_metadata must serialize to an object")
        self._user_metadata_value = metadata_value

        self._lock = asyncio.Lock()
        self._prepared = False
        self._closed = False
        self._handle: TextIO | None = None
        self._records: list[dict[str, Any]] = []
        self._latest_replayable: dict[_ReplayKey, dict[str, Any]] = {}
        self._latest_success: dict[_ReplayKey, dict[str, Any]] = {}
        self._next_sequence = 0
        self._detached_io_tasks: set[asyncio.Task[Any]] = set()
        self._detached_io_errors: list[Exception] = []

    async def prepare_item(self, work_item: LLMWorkItem[Any, Any, Any]) -> _ItemFingerprint:
        """Create/validate the artifact and fingerprint input before provider work."""
        self._resolve_identity_from(work_item.strategy)
        await self._prepare()
        return self._fingerprint_item(work_item)

    def _resolve_identity_from(self, strategy: Any) -> None:
        """Infer and pin the identity from the first item's strategy (v0.19.0)."""
        if self.identity is not None:
            return
        inferred = infer_artifact_identity(strategy)
        self.identity = inferred
        self._identity_value = _identity_mapping(inferred)
        self.identity_fingerprint = _sha256(_identity_mapping(inferred, for_fingerprint=True))

    def _require_resolved_fingerprint(self) -> str:
        if self.identity_fingerprint is None:
            raise ArtifactError(
                "Artifact identity is not resolved yet. Pass "
                "identity=ArtifactIdentity(...) to JsonlArtifactStore, or run the "
                "store through a processor so prepare_item() can infer the "
                "identity from the strategy."
            )
        return self.identity_fingerprint

    def _fingerprint_item(self, work_item: LLMWorkItem[Any, Any, Any]) -> _ItemFingerprint:
        prompt_hash = hashlib.sha256(work_item.prompt.encode("utf-8")).hexdigest()
        context_hash: str | None = None
        if self.context_in_identity:
            if self.context_fingerprinter is not None:
                try:
                    context_hash = self.context_fingerprinter(work_item.context)
                except Exception as exc:
                    raise ArtifactSerializationError(
                        f"Context fingerprinter failed for item {work_item.item_id!r}: {exc}"
                    ) from exc
                if not isinstance(context_hash, str) or not context_hash:
                    raise ArtifactSerializationError(
                        "context_fingerprinter must return a non-empty string"
                    )
            else:
                try:
                    context_value = _to_fingerprint_value(
                        work_item.context,
                        encoder=self.encoder,
                        path=f"$.items[{work_item.item_id!r}].context",
                    )
                except ResultSerializationError as exc:
                    raise ArtifactSerializationError(str(exc)) from exc
                context_hash = _sha256(context_value)
        combined_hash = _sha256(
            {
                "item_id": work_item.item_id,
                "prompt_fingerprint": prompt_hash,
                "context_fingerprint": context_hash,
            }
        )
        return _ItemFingerprint(prompt=prompt_hash, context=context_hash, combined=combined_hash)

    async def lookup(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        policy: ResumePolicy,
    ) -> WorkItemResult[Any, Any] | None:
        if policy is ResumePolicy.NONE:
            return None
        fingerprint = self._coerce_fingerprint(prepared_item)
        await self._prepare()
        key = self._replay_key(work_item, fingerprint)
        records = (
            self._latest_success
            if policy is ResumePolicy.REUSE_SUCCESSES
            else self._latest_replayable
        )
        record = records.get(key)
        if record is None or not self._compatible(record, work_item, fingerprint):
            return None
        try:
            result = work_item_result_from_dict(
                record["result"],
                output_decoder=self.output_decoder,
                context_decoder=self.context_decoder,
            )
        except (KeyError, ResultSerializationError) as exc:
            raise ArtifactFormatError(
                f"Malformed stored result for item {work_item.item_id!r}: {exc}"
            ) from exc
        result.context = work_item.context
        result.submission_index = work_item.submission_index
        result.replayed_from_artifact = True
        result.exception = None
        return result

    def _replay_key(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        fingerprint: _ItemFingerprint,
    ) -> _ReplayKey:
        return (
            work_item.item_id,
            fingerprint.prompt,
            fingerprint.context,
            fingerprint.combined,
            self._require_resolved_fingerprint(),
        )

    @staticmethod
    def _record_replay_key(record: Mapping[str, Any]) -> _ReplayKey | None:
        item_id = record.get("item_id")
        prompt = record.get("prompt_fingerprint")
        context = record.get("context_fingerprint")
        combined = record.get("input_fingerprint")
        identity = record.get("identity_fingerprint")
        if (
            not isinstance(item_id, str)
            or not isinstance(prompt, str)
            or (context is not None and not isinstance(context, str))
            or not isinstance(combined, str)
            or not isinstance(identity, str)
        ):
            return None
        return item_id, prompt, context, combined, identity

    def _index_record(self, record: dict[str, Any]) -> None:
        if not record.get("replay_eligible", False):
            return
        key = self._record_replay_key(record)
        if key is None:
            return
        self._latest_replayable[key] = record
        if record.get("success") is True:
            self._latest_success[key] = record

    def _rebuild_replay_index(self) -> None:
        self._latest_replayable.clear()
        self._latest_success.clear()
        for record in self._records:
            self._index_record(record)

    def _compatible(
        self,
        record: Mapping[str, Any],
        work_item: LLMWorkItem[Any, Any, Any],
        fingerprint: _ItemFingerprint,
    ) -> bool:
        return (
            record.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION
            and record.get("item_id") == work_item.item_id
            and record.get("prompt_fingerprint") == fingerprint.prompt
            and record.get("context_fingerprint") == fingerprint.context
            and record.get("input_fingerprint") == fingerprint.combined
            and record.get("identity_fingerprint") == self.identity_fingerprint
        )

    async def append(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        result: WorkItemResult[Any, Any],
    ) -> None:
        fingerprint = self._coerce_fingerprint(prepared_item)
        await self._prepare()
        try:
            serialized_result = work_item_result_to_dict(
                result,
                encoder=self.encoder,
                include_output=self.include_output,
                include_context=False,
                include_metadata=self.include_metadata,
            )
            raw_context = (
                to_json_value(work_item.context, encoder=self.encoder, path="$.raw_context")
                if self.include_context
                else None
            )
            raw_prompt = (
                to_json_value(work_item.prompt, path="$.raw_prompt")
                if self.include_prompt
                else None
            )
            cost = self.cost_calculator(result) if self.cost_calculator is not None else None
            if cost is not None:
                cost = float(cost)
                to_json_value(cost, path="$.calculated_cost")
        except (ResultSerializationError, TypeError, ValueError) as exc:
            raise ArtifactSerializationError(
                f"Could not serialize artifact result for item {work_item.item_id!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise ArtifactSerializationError(
                f"Cost calculator failed for item {work_item.item_id!r}: {exc}"
            ) from exc

        strategy_type = type(work_item.strategy)
        identity_fingerprint = self._require_resolved_fingerprint()
        identity = self.identity
        assert identity is not None  # resolved together with the fingerprint
        record: dict[str, Any] = {
            "record_type": "item",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "recorded_at": _utc_now(),
            # Assigned under the append lock so concurrent workers cannot
            # observe and reuse the same sequence value.
            "record_sequence": None,
            "item_id": work_item.item_id,
            "submission_index": result.submission_index,
            "prompt_fingerprint": fingerprint.prompt,
            "context_fingerprint": fingerprint.context,
            "input_fingerprint": fingerprint.combined,
            "identity_fingerprint": identity_fingerprint,
            "strategy_class": f"{strategy_type.__module__}.{strategy_type.__qualname__}",
            "provider": identity.provider,
            "model": identity.model,
            "prompt_version": identity.prompt_version,
            "parser_version": identity.parser_version,
            "application_version": identity.application_version,
            "identity": self._identity_value,
            "success": result.success,
            "error_category": result.error_category,
            "token_usage": serialized_result["token_usage"],
            "timing": serialized_result["timing"],
            "calculated_cost": cost,
            "replay_eligible": (not result.success) or self.include_output,
            "raw_prompt": raw_prompt,
            "raw_context": raw_context,
            "result": serialized_result,
        }
        await self._append_record(record)

    @staticmethod
    def _coerce_fingerprint(value: Any) -> _ItemFingerprint:
        if not isinstance(value, _ItemFingerprint):
            raise ArtifactSerializationError("Artifact item was not prepared by this store")
        return value

    def _finish_detached_io(self, task: asyncio.Task[Any]) -> None:
        """Retain a cancelled caller's I/O failure for the next store operation."""
        if task not in self._detached_io_tasks:
            return
        self._detached_io_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._detached_io_errors.append(exc)

    def _raise_detached_io_error(self) -> None:
        for task in list(self._detached_io_tasks):
            if task.done():
                self._finish_detached_io(task)
        if self._detached_io_errors:
            raise self._detached_io_errors.pop(0)

    async def _run_lock_owner(self, awaitable: Any) -> Any:
        """Keep the store lock owned until threaded I/O ends after cancellation."""
        task = asyncio.create_task(awaitable)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # asyncio.to_thread() cannot stop an already-running thread. The
            # child task must therefore retain the lock after caller
            # cancellation so another append/close cannot overlap the handle.
            self._detached_io_tasks.add(task)
            task.add_done_callback(self._finish_detached_io)
            raise

    async def _prepare(self) -> None:
        await self._run_lock_owner(self._prepare_locked())

    async def _prepare_locked(self) -> None:
        async with self._lock:
            self._raise_detached_io_error()
            if self._prepared:
                if self._closed:
                    raise ArtifactIOError(f"Artifact store is closed: {self.path}")
                return
            if self._closed:
                raise ArtifactIOError(f"Artifact store is closed: {self.path}")
            try:
                records, needs_manifest = await asyncio.to_thread(
                    _read_artifact_records,
                    self.path,
                    allow_create=True,
                )
                self._records = records
                self._rebuild_replay_index()
                self._next_sequence = (
                    max((int(record.get("record_sequence", -1)) for record in records), default=-1)
                    + 1
                )
                if needs_manifest and self._identity_value is None:
                    raise ArtifactError(
                        f"Cannot create a new artifact {self.path} without a "
                        "resolved identity. Pass identity=ArtifactIdentity(...) "
                        "to JsonlArtifactStore, or run the store through a "
                        "processor so it can be inferred from the strategy."
                    )
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self.path.open("a", encoding="utf-8", newline="\n")
                if needs_manifest:
                    manifest = {
                        "record_type": "manifest",
                        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                        "created_at": _utc_now(),
                        "package_version": _package_version(),
                        "identity": self._identity_value,
                        "identity_fingerprint": self.identity_fingerprint,
                        "user_metadata": self._user_metadata_value,
                    }
                    await asyncio.to_thread(self._write_record_sync, manifest)
                self._prepared = True
            except ArtifactError:
                raise
            except OSError as exc:
                raise ArtifactIOError(f"Could not prepare artifact {self.path}: {exc}") from exc

    async def _append_record(self, record: dict[str, Any]) -> None:
        await self._run_lock_owner(self._append_record_locked(record))

    async def _append_record_locked(self, record: dict[str, Any]) -> None:
        async with self._lock:
            self._raise_detached_io_error()
            if self._closed or self._handle is None:
                raise ArtifactIOError(f"Artifact store is not writable: {self.path}")
            record["record_sequence"] = self._next_sequence
            try:
                await asyncio.to_thread(self._write_record_sync, record)
            except OSError as exc:
                raise ArtifactIOError(
                    f"Could not append artifact record for item {record.get('item_id')!r}: {exc}"
                ) from exc
            self._records.append(record)
            self._index_record(record)
            self._next_sequence += 1

    def _write_record_sync(self, record: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise OSError("artifact file is not open")
        line = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._handle.write(line + "\n")
        self._handle.flush()
        if self.fsync:
            os.fsync(self._handle.fileno())

    async def iter_results(
        self, *, successes_only: bool = False
    ) -> AsyncIterator[WorkItemResult[Any, Any]]:
        if not self._prepared:
            await self._prepare()
        for record in list(self._records):
            if successes_only and not record.get("success"):
                continue
            try:
                yield work_item_result_from_dict(
                    record["result"],
                    output_decoder=self.output_decoder,
                    context_decoder=self.context_decoder,
                )
            except (KeyError, ResultSerializationError) as exc:
                raise ArtifactFormatError(f"Malformed stored result: {exc}") from exc

    async def close(self) -> None:
        await self._run_lock_owner(self._close_locked())

    async def _close_locked(self) -> None:
        async with self._lock:
            if self._closed:
                self._raise_detached_io_error()
                return
            self._closed = True
            handle = self._handle
            self._handle = None
            if handle is None:
                self._raise_detached_io_error()
                return
            try:
                await asyncio.to_thread(self._close_sync, handle)
            except OSError as exc:
                raise ArtifactIOError(f"Could not close artifact {self.path}: {exc}") from exc
            self._raise_detached_io_error()

    def _close_sync(self, handle: TextIO) -> None:
        handle.flush()
        if self.fsync:
            os.fsync(handle.fileno())
        handle.close()

    @classmethod
    def read_results(
        cls,
        path: str | Path,
        *,
        successes_only: bool = False,
        output_decoder: ValueDecoder | None = None,
        context_decoder: ValueDecoder | None = None,
    ) -> BatchResult[Any, Any]:
        """Read stored results without opening a writer or calling a provider."""
        records, _ = _read_artifact_records(Path(path), allow_create=False)
        results: list[WorkItemResult[Any, Any]] = []
        for record in records:
            if successes_only and not record.get("success"):
                continue
            try:
                results.append(
                    work_item_result_from_dict(
                        record["result"],
                        output_decoder=output_decoder,
                        context_decoder=context_decoder,
                    )
                )
            except (KeyError, ResultSerializationError) as exc:
                raise ArtifactFormatError(f"Malformed stored result: {exc}") from exc
        return BatchResult(results=results, termination=BatchTermination())


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactFormatError",
    "ArtifactIOError",
    "ArtifactIdentity",
    "ArtifactSerializationError",
    "ArtifactStore",
    "JsonlArtifactStore",
    "ResumePolicy",
]
