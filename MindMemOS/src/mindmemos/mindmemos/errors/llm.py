from .base import MindMemOSError


class EmbeddingDimensionError(MindMemOSError):
    """Raised when an embedding vector length != configured collection vector_size.

    Indicates the provider or litellm silently dropped the ``dimensions`` request
    parameter (litellm ``drop_params=True``), or the embedding model was switched
    to one with a different native dimension. The Qdrant collection dimension is
    immutable after creation, so mismatched writes are silently rejected while
    ``add`` still reports success with zero points; this error forces that failure
    to surface immediately instead of becoming silent data loss.
    """

    def __init__(self, *, expected: int, actual: int, model: str, task: str) -> None:
        self.expected = expected
        self.actual = actual
        self.model = model
        self.task = task
        msg = (
            f"Embedding dimension mismatch (task={task}, model={model}): "
            f"expected {expected} (= database.qdrant.vector_size), got {actual}. "
            "This usually means the `dimensions` request param was silently dropped by the "
            "provider or litellm (drop_params=True), or the embedding model was switched to one "
            "with a different native dimension. The Qdrant collection dimension is immutable after "
            "creation; restore the previous model, set endpoints[].dimensions to match vector_size, "
            "or drop and recreate the collection."
        )
        super().__init__(msg)
