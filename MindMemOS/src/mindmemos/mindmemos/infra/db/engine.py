"""Collection-agnostic Qdrant primitives shared by all repositories.

``QdrantEngine`` owns the :class:`AsyncQdrantClient` and every Qdrant operation
that carries no memory/skill business semantics: it deals only in collection
names, point structs, records and project-scoped filters. Typed adapters such as
``QdrantStore`` and ``SkillVersionRepository`` compose one engine and delegate
the low-level work here, so connection handling, payload sanitisation, record
mapping and project scoping live in exactly one place.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ...config import QdrantConfig
from ...logging import get_logger
from ..retry import AsyncRetryProxy
from .concurrency import AsyncClientConcurrencyLimiter, capped_db_client_concurrency
from .errors import MemoryDbConfigurationError, MemoryDbValidationError
from .filters import match_value
from .models import (
    QdrantCollectionSpec,
    QdrantRecord,
    QdrantSearchRecord,
    SparseVectorData,
)
from .qdrant_batch_writer import QdrantBatchWriter

logger = get_logger(__name__)


class QdrantEngine:
    """Thin, business-agnostic wrapper over ``AsyncQdrantClient``."""

    def __init__(self, cfg: QdrantConfig, *, client: AsyncQdrantClient | None = None) -> None:
        self._cfg = cfg
        max_client_concurrency = capped_db_client_concurrency(
            cfg.max_client_concurrency,
            cap=cfg.max_client_concurrency_cap,
        )
        if client is not None:
            raw_client = client
        else:
            kwargs: dict[str, Any] = {
                "url": cfg.url,
                "api_key": cfg.api_key,
                "grpc_port": cfg.grpc_port,
                "prefer_grpc": cfg.prefer_grpc,
                "timeout": int(cfg.timeout),
                "trust_env": False,
            }
            # pool_size caps concurrent in-flight HTTP requests; 0 keeps the client default.
            if cfg.pool_size:
                kwargs["pool_size"] = min(cfg.pool_size, max_client_concurrency)
            raw_client = AsyncQdrantClient(**kwargs)
        limited_client = AsyncClientConcurrencyLimiter(raw_client, max_concurrency=max_client_concurrency)
        self._client = AsyncRetryProxy(
            limited_client,
            operation_name="qdrant",
            max_attempts=cfg.max_retries,
            base_delay=cfg.retry_base_delay,
            retryable=_is_retryable_qdrant_error,
        )
        self._batch_writer = (
            QdrantBatchWriter(
                self._raw_upsert,
                batch_size=cfg.batch_upsert_size,
                flush_interval_ms=cfg.batch_upsert_flush_interval_ms,
                max_queue_size=cfg.batch_upsert_max_queue_size,
                max_inflight_batches=cfg.batch_upsert_max_inflight_batches,
            )
            if cfg.batch_upsert_enabled
            else None
        )

    @property
    def client(self) -> AsyncQdrantClient:
        """Underlying async Qdrant client, owned and closed by this engine."""

        return self._client

    @property
    def cfg(self) -> QdrantConfig:
        """Qdrant configuration backing this engine."""

        return self._cfg

    async def ensure_collection(self, spec: QdrantCollectionSpec) -> None:
        """Create one collection and its payload indexes (no ``auto_create`` gate).

        The gate lives in the caller's ``ensure_schema`` so this stays a pure
        primitive. A collection with neither dense nor sparse vectors is created
        with an empty ``vectors_config`` (payload + filter only).
        """

        exists = await self._client.collection_exists(spec.name)
        if not exists:
            vectors_config: dict[str, qmodels.VectorParams] = {}
            sparse_vectors_config: dict[str, qmodels.SparseVectorParams] | None = None
            if spec.enable_dense:
                vectors_config[spec.dense_vector_name] = qmodels.VectorParams(
                    size=spec.vector_size,
                    distance=self._distance(spec.distance),
                )
            if spec.enable_sparse:
                sparse_vectors_config = {
                    spec.sparse_vector_name: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
                }
            await self._client.create_collection(
                collection_name=spec.name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                on_disk_payload=spec.on_disk_payload,
            )
        elif spec.on_disk_payload is not None:
            await self._client.update_collection(
                collection_name=spec.name,
                collection_params=qmodels.CollectionParamsDiff(on_disk_payload=spec.on_disk_payload),
            )

        for index in spec.payload_indexes:
            try:
                await self._client.create_payload_index(
                    collection_name=spec.name,
                    field_name=index.field_name,
                    field_schema=index.field_schema,
                )
            except Exception as exc:
                logger.debug(
                    "qdrant payload index create skipped",
                    collection=spec.name,
                    field_name=index.field_name,
                    error=str(exc),
                )

    async def upsert(self, collection: str, points: list[qmodels.PointStruct]) -> None:
        """Upsert point structs into one collection (no-op on empty input)."""

        if not points:
            return
        if self._batch_writer is not None:
            await self._batch_writer.upsert(collection, points)
            return
        await self._raw_upsert(collection, points)

    async def retrieve(self, collection: str, ids: list[str], *, with_vectors: bool = False) -> list[QdrantRecord]:
        """Retrieve points by id and map them to :class:`QdrantRecord`."""

        if not ids:
            return []
        records = await self._client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [self.record_from(record) for record in records]

    async def scroll(
        self,
        collection: str,
        *,
        scroll_filter: qmodels.Filter | None,
        limit: int,
        offset: Any | None = None,
        order_by: Any | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[QdrantRecord], Any | None]:
        """Scroll a collection with an already-built filter."""

        records, next_offset = await self._client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            order_by=order_by,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [self.record_from(record) for record in records], next_offset

    async def query(
        self,
        collection: str,
        *,
        source: str,
        query: Any,
        using: str | None = None,
        prefetch: Any | None = None,
        query_filter: qmodels.Filter | None = None,
        limit: int,
        with_payload: bool = True,
        score_threshold: float | None = None,
    ) -> list[QdrantSearchRecord]:
        """Run ``query_points`` and map the response to search records."""

        response = await self._client.query_points(
            collection_name=collection,
            query=query,
            using=using,
            prefetch=prefetch,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            score_threshold=score_threshold,
        )
        return self.hits_from_response(response, source=source)

    async def set_payload(self, collection: str, point_id: str, payload: dict[str, Any]) -> None:
        """Set payload fields on one point (payload is sanitised first)."""

        await self._client.set_payload(
            collection_name=collection,
            points=[point_id],
            payload=self.safe_payload(payload),
        )

    async def delete(self, collection: str, point_ids: list[str]) -> None:
        """Delete points by id (no-op on empty input)."""

        if not point_ids:
            return
        await self._client.delete(collection_name=collection, points_selector=point_ids)

    async def batch_update(self, collection: str, operations: list[qmodels.UpdateOperation]) -> None:
        """Apply a batch of update operations against one collection."""

        await self._client.batch_update_points(collection_name=collection, update_operations=operations)

    async def close(self) -> None:
        """Close the underlying client."""

        if self._batch_writer is not None:
            await self._batch_writer.close()
        await self._client.close()

    async def _raw_upsert(self, collection: str, points: list[qmodels.PointStruct]) -> None:
        await self._client.upsert(collection_name=collection, points=points)

    def zero_vector(self) -> list[float]:
        """Configured-size all-zero dense vector (placeholder for vector-less points)."""

        return [0.0] * self._cfg.vector_size

    @staticmethod
    def project_filter(
        project_id: str,
        *,
        filter_: qmodels.Filter | None = None,
        conditions: list[Any] | None = None,
    ) -> qmodels.Filter:
        """Build a project-scoped filter.

        ``project_id`` is always enforced as the first ``must`` condition. Pass
        ``filter_`` to extend a caller-supplied filter, or ``conditions`` to add a
        flat list of extra ``must`` conditions; the two are mutually exclusive in
        practice (callers use one or the other).
        """

        project_condition = match_value("project_id", project_id)
        if filter_ is not None:
            must = list(filter_.must or [])
            if not any(_is_same_project_condition(condition, project_id) for condition in must):
                must.insert(0, project_condition)
            return qmodels.Filter(must=must, should=filter_.should, must_not=filter_.must_not)
        must = [project_condition, *(conditions or [])]
        return qmodels.Filter(must=must)

    @staticmethod
    def to_qdrant_sparse(vector: SparseVectorData) -> qmodels.SparseVector:
        """Convert internal sparse data to a Qdrant sparse vector."""

        if len(vector.indices) != len(vector.values):
            raise MemoryDbValidationError("sparse vector index/value length mismatch")
        return qmodels.SparseVector(indices=vector.indices, values=vector.values)

    @staticmethod
    def safe_payload(patch: dict[str, Any]) -> dict[str, Any]:
        """Recursively coerce a payload patch into Qdrant-safe primitives."""

        return {key: _payload_safe_value(value) for key, value in patch.items()}

    @staticmethod
    def record_from(record: Any) -> QdrantRecord:
        """Map a raw Qdrant point/record to :class:`QdrantRecord`."""

        return QdrantRecord(
            point_id=str(getattr(record, "id")),
            payload=dict(getattr(record, "payload", None) or {}),
            vectors=getattr(record, "vector", None),
        )

    @staticmethod
    def hits_from_response(response: Any, *, source: str) -> list[QdrantSearchRecord]:
        """Map a ``query_points`` response to ranked search records."""

        points: Sequence[Any] = getattr(response, "points", []) or []
        hits: list[QdrantSearchRecord] = []
        for rank, point in enumerate(points, start=1):
            hits.append(
                QdrantSearchRecord(
                    point_id=str(getattr(point, "id")),
                    score=float(getattr(point, "score", 0.0)),
                    payload=dict(getattr(point, "payload", None) or {}),
                    vectors=getattr(point, "vector", None),
                    source=source,
                    debug={"rank": rank},
                )
            )
        return hits

    @staticmethod
    def first_project_match(records: list[QdrantRecord], project_id: str) -> QdrantRecord | None:
        """Return the first record whose payload matches ``project_id``."""

        for record in records:
            if record.payload.get("project_id") == project_id:
                return record
        return None

    @staticmethod
    def _distance(value: str) -> qmodels.Distance:
        try:
            return qmodels.Distance(value)
        except ValueError as exc:
            raise MemoryDbConfigurationError(f"unsupported qdrant distance: {value}") from exc


def _payload_safe_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _payload_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_payload_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_payload_safe_value(item) for item in value]
    return value


def _is_retryable_qdrant_error(exc: Exception) -> bool:
    if isinstance(exc, ResponseHandlingException):
        return True
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is None or exc.status_code == 408 or exc.status_code == 429 or exc.status_code >= 500
    return _is_http_transport_error(exc)


def _is_same_project_condition(condition: object, project_id: str) -> bool:
    return (
        isinstance(condition, qmodels.FieldCondition)
        and condition.key == "project_id"
        and isinstance(condition.match, qmodels.MatchValue)
        and condition.match.value == project_id
    )


def _is_http_transport_error(exc: Exception) -> bool:
    module = type(exc).__module__
    return module.startswith("httpx") or module.startswith("httpcore") or isinstance(exc, TimeoutError)
