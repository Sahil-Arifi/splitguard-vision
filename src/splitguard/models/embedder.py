"""Typed local image-embedding adapters."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

DEFAULT_DINOV2_MODEL = "facebook/dinov2-small"
DEFAULT_DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DEFAULT_DINOV2_PREPROCESSING = "hf-auto-image-processor-v1"
DEFAULT_DINOV2_DIMENSION = 384
_IMMUTABLE_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

DeviceRequest = Literal["auto", "cpu", "cuda"]
Float32Array = npt.NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Minimal contract implemented by production and offline embedders."""

    @property
    def model_identity(self) -> str: ...

    @property
    def preprocessing_version(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, images: Sequence[Image.Image]) -> Float32Array: ...


def normalize_embeddings(
    vectors: npt.ArrayLike,
    *,
    expected_rows: int | None = None,
    expected_dimension: int | None = None,
) -> Float32Array:
    """Return a finite, contiguous, float32 matrix with unit-length rows."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError("embedding row count does not match the input batch")
    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise ValueError("embedding dimension does not match the embedder contract")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")
    if matrix.shape[0] == 0:
        return np.ascontiguousarray(matrix, dtype=np.float32)

    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0.0):
        raise ValueError("zero-length embeddings cannot be normalized")
    normalized = matrix / norms[:, np.newaxis]
    if not np.isfinite(normalized).all():
        raise ValueError("normalized embeddings must contain only finite values")
    return np.ascontiguousarray(normalized, dtype=np.float32)


class FakeEmbedder:
    """Deterministic pixel-derived embedder for offline tests and demos."""

    def __init__(
        self,
        *,
        dimension: int = 16,
        seed: int = 0,
        preprocessing_version: str = "fake-rgb-bytes-v1",
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not preprocessing_version.strip():
            raise ValueError("preprocessing_version cannot be empty")
        self._dimension = dimension
        self._seed = seed
        self._preprocessing_version = preprocessing_version.strip()

    @property
    def model_identity(self) -> str:
        return (
            "fake:sha256-expand-v1"
            f":seed={self._seed}:dimension={self._dimension}:l2=float32-v1"
        )

    @property
    def preprocessing_version(self) -> str:
        return self._preprocessing_version

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, images: Sequence[Image.Image]) -> Float32Array:
        rows: list[Float32Array] = []
        seed_bytes = str(self._seed).encode("ascii")
        for image in images:
            rgb = image.convert("RGB")
            try:
                width, height = rgb.size
                payload = hashlib.sha256()
                payload.update(width.to_bytes(8, "big", signed=False))
                payload.update(height.to_bytes(8, "big", signed=False))
                payload.update(rgb.tobytes())
                image_digest = payload.digest()
            finally:
                rgb.close()

            expanded = bytearray()
            counter = 0
            while len(expanded) < self._dimension:
                block = hashlib.sha256(
                    seed_bytes + counter.to_bytes(8, "big", signed=False) + image_digest
                ).digest()
                expanded.extend(block)
                counter += 1
            row = np.frombuffer(bytes(expanded[: self._dimension]), dtype=np.uint8).astype(
                np.float32
            )
            rows.append(row - np.float32(127.5))

        if not rows:
            return np.empty((0, self._dimension), dtype=np.float32)
        return normalize_embeddings(
            np.stack(rows),
            expected_rows=len(images),
            expected_dimension=self._dimension,
        )


class DinoV2Embedder:
    """Lazily loaded Hugging Face DINOv2 pooled-image embedder."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_DINOV2_MODEL,
        revision: str = DEFAULT_DINOV2_REVISION,
        preprocessing_version: str = DEFAULT_DINOV2_PREPROCESSING,
        device: DeviceRequest = "auto",
        dimension: int = DEFAULT_DINOV2_DIMENSION,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if _IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
            raise ValueError("revision must be an immutable 40- or 64-character commit SHA")
        if not preprocessing_version.strip():
            raise ValueError("preprocessing_version cannot be empty")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self._model_name = model_name.strip()
        self._revision = revision
        self._preprocessing_version = preprocessing_version.strip()
        self._dimension = dimension
        self._device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self._device == "auto":
            self._device = "cpu"
        self._processor: Any | None = None
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    @property
    def model_identity(self) -> str:
        return (
            f"huggingface:{self._model_name}@{self._revision}"
            f":pool=pooler_output:dimension={self._dimension}:l2=float32-v1"
        )

    @property
    def preprocessing_version(self) -> str:
        return self._preprocessing_version

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def device(self) -> Literal["cpu", "cuda"]:
        return cast(Literal["cpu", "cuda"], self._device)

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        with self._load_lock:
            if self._processor is None or self._model is None:
                # Hugging Face model files may be fetched on first use, but library
                # telemetry is disabled before its modules are imported. Image bytes
                # are never passed to the model-loading client.
                os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
                transformers = importlib.import_module("transformers")
                processor = transformers.AutoImageProcessor.from_pretrained(
                    self._model_name,
                    revision=self._revision,
                    trust_remote_code=False,
                )
                model = transformers.AutoModel.from_pretrained(
                    self._model_name,
                    revision=self._revision,
                    trust_remote_code=False,
                    use_safetensors=True,
                )
                configured_dimension = getattr(getattr(model, "config", None), "hidden_size", None)
                if configured_dimension != self._dimension:
                    raise RuntimeError("loaded model dimension does not match the adapter contract")
                model.to(self._device)
                model.eval()
                self._processor = processor
                self._model = model
        return self._processor, self._model

    def embed(self, images: Sequence[Image.Image]) -> Float32Array:
        if not images:
            return np.empty((0, self._dimension), dtype=np.float32)

        processor, model = self._ensure_loaded()
        rgb_images = [image.convert("RGB") for image in images]
        try:
            encoded = cast(
                Mapping[str, torch.Tensor],
                processor(images=rgb_images, return_tensors="pt"),
            )
        finally:
            for image in rgb_images:
                image.close()
        model_inputs = {name: tensor.to(self._device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            outputs = model(**model_inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if not isinstance(pooled, torch.Tensor):
            raise RuntimeError("DINOv2 output did not contain a pooled representation")
        matrix = np.asarray(pooled.detach().to(device="cpu", dtype=torch.float32).numpy())
        return normalize_embeddings(
            matrix,
            expected_rows=len(images),
            expected_dimension=self._dimension,
        )


__all__ = [
    "DEFAULT_DINOV2_DIMENSION",
    "DEFAULT_DINOV2_MODEL",
    "DEFAULT_DINOV2_PREPROCESSING",
    "DEFAULT_DINOV2_REVISION",
    "DeviceRequest",
    "DinoV2Embedder",
    "Embedder",
    "FakeEmbedder",
    "Float32Array",
    "normalize_embeddings",
]
