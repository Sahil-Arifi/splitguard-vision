from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never, cast

import numpy as np
import pytest
import torch
from PIL import Image

from splitguard.embeddings import (
    EmbeddingCache,
    EmbeddingInputError,
    embed_records,
    embedding_cache_key,
)
from splitguard.models.embedder import (
    DEFAULT_DINOV2_MODEL,
    DEFAULT_DINOV2_REVISION,
    DinoV2Embedder,
    Embedder,
    FakeEmbedder,
    Float32Array,
    normalize_embeddings,
)
from splitguard.schemas import ImageRecord, Split, stable_id


class CountingFakeEmbedder(FakeEmbedder):
    def __init__(
        self,
        *,
        dimension: int = 8,
        seed: int = 0,
        preprocessing_version: str = "fake-rgb-bytes-v1",
    ) -> None:
        super().__init__(
            dimension=dimension,
            seed=seed,
            preprocessing_version=preprocessing_version,
        )
        self.calls = 0

    def embed(self, images: Sequence[Image.Image]) -> Float32Array:
        self.calls += 1
        return super().embed(images)


class Float64FakeEmbedder(FakeEmbedder):
    def embed(self, images: Sequence[Image.Image]) -> Float32Array:
        return cast(Float32Array, super().embed(images).astype(np.float64))


def make_record(
    root: Path,
    relative_path: str,
    *,
    color: tuple[int, int, int] = (10, 20, 30),
) -> ImageRecord:
    path = root.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color).save(path, format="PNG")
    contents = path.read_bytes()
    return ImageRecord(
        id=stable_id("img", relative_path),
        path=relative_path,
        split=Split.TRAIN,
        label="class-a",
        byte_sha256=hashlib.sha256(contents).hexdigest(),
        byte_size=len(contents),
        width=12,
        height=10,
        format="png",
    )


def test_fake_embedder_is_deterministic_normalized_and_protocol_compatible() -> None:
    embedder = FakeEmbedder(dimension=12, seed=7)
    first = Image.new("RGB", (4, 3), (10, 20, 30))
    second = Image.new("RGB", (4, 3), (30, 20, 10))
    try:
        forward = embedder.embed((first, second))
        repeated = embedder.embed((first, second))
    finally:
        first.close()
        second.close()

    assert isinstance(embedder, Embedder)
    assert forward.shape == (2, 12)
    assert forward.dtype == np.float32
    np.testing.assert_array_equal(forward, repeated)
    np.testing.assert_allclose(np.linalg.norm(forward, axis=1), 1.0, atol=1e-6)
    assert not np.array_equal(forward[0], forward[1])
    assert embedder.embed(()).shape == (0, 12)


@pytest.mark.parametrize(
    "values,expected_rows,expected_dimension,match",
    [
        ([1.0, 2.0], None, None, "two-dimensional"),
        ([[1.0, 2.0]], 2, None, "row count"),
        ([[1.0, 2.0]], None, 3, "dimension"),
        ([[0.0, 0.0]], None, None, "zero-length"),
        ([[float("nan"), 1.0]], None, None, "finite"),
    ],
)
def test_normalization_rejects_invalid_matrices(
    values: list[float] | list[list[float]],
    expected_rows: int | None,
    expected_dimension: int | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_embeddings(
            values,
            expected_rows=expected_rows,
            expected_dimension=expected_dimension,
        )


def test_embed_records_is_id_aligned_cached_and_read_only(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    later = make_record(root, "train/class-a/z.png", color=(2, 3, 4))
    earlier = make_record(root, "train/class-a/a.png", color=(5, 6, 7))
    cache_dir = tmp_path / "cache"
    embedder = CountingFakeEmbedder(dimension=10)

    first = embed_records(
        root,
        (later, earlier),
        embedder=embedder,
        cache_dir=cache_dir,
        batch_size=1,
    )
    calls_after_first = embedder.calls
    second = embed_records(
        root,
        (earlier, later),
        embedder=embedder,
        cache_dir=cache_dir,
        batch_size=2,
    )

    assert first.record_ids == tuple(sorted((earlier.id, later.id)))
    assert first.cache_hits == 0
    assert first.cache_misses == 2
    assert second.cache_hits == 2
    assert second.cache_misses == 0
    assert embedder.calls == calls_after_first
    assert second.vectors.dtype == np.float32
    assert second.vectors.flags.writeable is False
    np.testing.assert_allclose(np.linalg.norm(second.vectors, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(first.vectors, second.vectors)


def test_embed_records_deduplicates_identical_content_within_a_run(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    first = make_record(root, "train/class-a/a.png")
    second_path = root / "test" / "class-a" / "copy.png"
    second_path.parent.mkdir(parents=True)
    second_path.write_bytes((root / "train" / "class-a" / "a.png").read_bytes())
    second = ImageRecord(
        id=stable_id("img", "test/class-a/copy.png"),
        path="test/class-a/copy.png",
        split=Split.TEST,
        label="class-a",
        byte_sha256=first.byte_sha256,
        byte_size=first.byte_size,
        width=first.width,
        height=first.height,
        format=first.format,
    )
    embedder = CountingFakeEmbedder()

    result = embed_records(
        root,
        (first, second),
        embedder=embedder,
        cache_dir=tmp_path / "cache",
    )

    assert result.cache_misses == 1
    assert result.deduplicated_records == 1
    assert embedder.calls == 1
    np.testing.assert_array_equal(result.vectors[0], result.vectors[1])


def test_embed_records_rejects_protocol_violating_dtype(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    record = make_record(root, "train/class-a/a.png")

    with pytest.raises(ValueError, match="float32 NumPy"):
        embed_records(
            root,
            (record,),
            embedder=Float64FakeEmbedder(),
            cache_dir=tmp_path / "cache",
        )


def test_cache_key_invalidates_on_every_identity_component() -> None:
    first = FakeEmbedder(seed=1, preprocessing_version="prep-v1")
    changed_model = FakeEmbedder(seed=2, preprocessing_version="prep-v1")
    changed_preprocessing = FakeEmbedder(seed=1, preprocessing_version="prep-v2")
    sha = "a" * 64

    baseline = embedding_cache_key(sha, first.model_identity, first.preprocessing_version)

    assert baseline != embedding_cache_key(
        "b" * 64, first.model_identity, first.preprocessing_version
    )
    assert baseline != embedding_cache_key(
        sha, changed_model.model_identity, changed_model.preprocessing_version
    )
    assert baseline != embedding_cache_key(
        sha,
        changed_preprocessing.model_identity,
        changed_preprocessing.preprocessing_version,
    )


def test_corrupt_cache_is_ignored_and_replaced(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    record = make_record(root, "train/class-a/a.png")
    cache_dir = tmp_path / "cache"
    embedder = CountingFakeEmbedder()
    first = embed_records(root, (record,), embedder=embedder, cache_dir=cache_dir)
    key = embedding_cache_key(
        record.byte_sha256,
        embedder.model_identity,
        embedder.preprocessing_version,
    )
    path = EmbeddingCache(cache_dir).path_for_key(key)
    path.write_bytes(b"not an npz archive")

    second = embed_records(root, (record,), embedder=embedder, cache_dir=cache_dir)

    assert first.cache_misses == 1
    assert second.cache_hits == 0
    assert second.cache_misses == 1
    assert embedder.calls == 2
    assert EmbeddingCache(cache_dir).load(record.byte_sha256, embedder) is not None


@pytest.mark.parametrize("corruption", ["dtype", "dimension", "nan", "norm", "metadata"])
def test_cache_validates_vector_and_metadata(tmp_path: Path, corruption: str) -> None:
    cache = EmbeddingCache(tmp_path / "cache")
    embedder = FakeEmbedder(dimension=4)
    sha = "a" * 64
    vector = normalize_embeddings([[1.0, 2.0, 3.0, 4.0]])[0]
    cache.store(sha, embedder, vector)
    key = embedding_cache_key(sha, embedder.model_identity, embedder.preprocessing_version)
    path = cache.path_for_key(key)
    with np.load(path, allow_pickle=False) as archive:
        metadata = str(archive["metadata"].item())
    candidate: np.ndarray[Any, Any] = vector
    if corruption == "dtype":
        candidate = vector.astype(np.float64)
    elif corruption == "dimension":
        candidate = np.ones(3, dtype=np.float32)
    elif corruption == "nan":
        candidate = vector.copy()
        candidate[0] = np.nan
    elif corruption == "norm":
        candidate = vector * np.float32(2.0)
    else:
        payload = json.loads(metadata)
        payload["preprocessing_version"] = "different"
        metadata = json.dumps(payload)
    np.savez(path, metadata=np.asarray(metadata), vector=candidate)

    assert cache.load(sha, embedder) is None


def test_cache_write_is_atomic_and_metadata_contains_no_paths(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "private" / "cache")
    embedder = FakeEmbedder(dimension=4)
    sha = "b" * 64
    vector = normalize_embeddings([[4.0, 3.0, 2.0, 1.0]])[0]

    cache.store(sha, embedder, vector)

    key = embedding_cache_key(sha, embedder.model_identity, embedder.preprocessing_version)
    path = cache.path_for_key(key)
    with np.load(path, allow_pickle=False) as archive:
        metadata = str(archive["metadata"].item())
    assert str(tmp_path) not in metadata
    assert not tuple(cache.root.rglob("*.tmp"))
    np.testing.assert_array_equal(cache.load(sha, embedder), vector)


def test_embed_records_rejects_path_escape_without_leaking_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(outside)
    contents = outside.read_bytes()
    unsafe = ImageRecord.model_construct(
        id=stable_id("img", "unsafe"),
        path="../outside.png",
        split=Split.TRAIN,
        label="class-a",
        byte_sha256=hashlib.sha256(contents).hexdigest(),
        byte_size=len(contents),
        width=2,
        height=2,
        format="png",
        phash=None,
    )

    with pytest.raises(EmbeddingInputError) as error:
        embed_records(
            root,
            (unsafe,),
            embedder=FakeEmbedder(),
            cache_dir=tmp_path / "cache",
        )

    assert str(tmp_path) not in str(error.value)


def test_embed_records_rejects_content_changed_after_validation(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    record = make_record(root, "train/class-a/a.png")
    Image.new("RGB", (12, 10), (200, 100, 50)).save(root / record.path)

    with pytest.raises(EmbeddingInputError, match="content changed"):
        embed_records(
            root,
            (record,),
            embedder=FakeEmbedder(),
            cache_dir=tmp_path / "cache",
        )


def test_fake_workflow_never_imports_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    record = make_record(root, "train/class-a/a.png")

    def fail_import(_name: str) -> Never:
        raise AssertionError("offline FakeEmbedder path imported transformers")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    result = embed_records(
        root,
        (record,),
        embedder=FakeEmbedder(),
        cache_dir=tmp_path / "cache",
    )

    assert result.vectors.shape == (1, 16)


def test_dinov2_constructor_is_lazy_and_cuda_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_name: str) -> Never:
        raise AssertionError("constructor imported transformers")

    monkeypatch.setattr(importlib, "import_module", fail_import)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cpu = DinoV2Embedder(device="cpu")
    automatic = DinoV2Embedder(device="auto")

    assert cpu.device == "cpu"
    assert automatic.device == "cpu"
    assert cpu.model_identity.startswith(
        f"huggingface:{DEFAULT_DINOV2_MODEL}@{DEFAULT_DINOV2_REVISION}:"
    )
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        DinoV2Embedder(device="cuda")
    with pytest.raises(ValueError, match="immutable"):
        DinoV2Embedder(revision="main")


def test_dinov2_loader_uses_safe_options_eval_and_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeProcessor:
        def __call__(
            self, *, images: Sequence[Image.Image], return_tensors: str
        ) -> dict[str, torch.Tensor]:
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones((len(images), 3, 2, 2))}

    class ProcessorFactory:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: object) -> FakeProcessor:
            calls["processor_loads"] = calls.get("processor_loads", 0) + 1
            calls["processor"] = (model_name, kwargs)
            return FakeProcessor()

    class FakeModel:
        config = SimpleNamespace(hidden_size=4)

        def to(self, device: str) -> FakeModel:
            calls["device"] = device
            return self

        def eval(self) -> FakeModel:
            calls["eval"] = True
            return self

        def __call__(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
            calls["inference_mode"] = torch.is_inference_mode_enabled()
            rows = pixel_values.shape[0]
            return SimpleNamespace(pooler_output=torch.arange(1, 5).repeat(rows, 1))

    class ModelFactory:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: object) -> FakeModel:
            calls["model_loads"] = calls.get("model_loads", 0) + 1
            calls["model"] = (model_name, kwargs)
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoImageProcessor=ProcessorFactory,
        AutoModel=ModelFactory,
    )

    def import_fake_transformers(name: str) -> SimpleNamespace:
        assert name == "transformers"
        return fake_transformers

    monkeypatch.setattr(importlib, "import_module", import_fake_transformers)
    embedder = DinoV2Embedder(device="cpu", dimension=4)
    image = Image.new("RGB", (3, 3), (1, 2, 3))
    try:
        vectors = embedder.embed((image,))
        embedder.embed((image,))
    finally:
        image.close()

    processor_name, processor_kwargs = calls["processor"]
    model_name, model_kwargs = calls["model"]
    assert processor_name == DEFAULT_DINOV2_MODEL
    assert model_name == DEFAULT_DINOV2_MODEL
    assert processor_kwargs == {
        "revision": DEFAULT_DINOV2_REVISION,
        "trust_remote_code": False,
    }
    assert model_kwargs == {
        "revision": DEFAULT_DINOV2_REVISION,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    assert calls["device"] == "cpu"
    assert calls["eval"] is True
    assert calls["inference_mode"] is True
    assert calls["processor_loads"] == 1
    assert calls["model_loads"] == 1
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_dinov2_auto_selects_cuda_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert DinoV2Embedder(device="auto").device == "cuda"
