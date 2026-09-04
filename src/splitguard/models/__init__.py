"""Model adapters exposed by SplitGuard Vision."""

from splitguard.models.embedder import (
    DEFAULT_DINOV2_DIMENSION,
    DEFAULT_DINOV2_MODEL,
    DEFAULT_DINOV2_PREPROCESSING,
    DEFAULT_DINOV2_REVISION,
    DinoV2Embedder,
    Embedder,
    FakeEmbedder,
    normalize_embeddings,
)

__all__ = [
    "DEFAULT_DINOV2_DIMENSION",
    "DEFAULT_DINOV2_MODEL",
    "DEFAULT_DINOV2_PREPROCESSING",
    "DEFAULT_DINOV2_REVISION",
    "DinoV2Embedder",
    "Embedder",
    "FakeEmbedder",
    "normalize_embeddings",
]
