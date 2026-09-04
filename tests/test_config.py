"""Tests for strict, portable SplitGuard configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from splitguard.config import (
    ConfigLoadError,
    IOConfig,
    NeighborsConfig,
    RepairConfig,
    ReportConfig,
    SplitGuardConfig,
    canonical_config_hash,
    load_config,
)


def test_defaults_are_complete_and_nested_models_are_frozen() -> None:
    config = SplitGuardConfig()

    assert set(config.canonical_dict()) == {
        "io",
        "phash",
        "embeddings",
        "neighbors",
        "policy",
        "repair",
        "report",
    }
    with pytest.raises(ValidationError, match="frozen_instance"):
        config.io.thumbnail_size = 512
    with pytest.raises(ValidationError, match="frozen_instance"):
        config.report = ReportConfig(output_dir="reports")


def test_loads_all_required_yaml_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "default.yaml"
    config_path.write_text(
        """
io:
  max_image_pixels: 40000000
  thumbnail_size: 192
  cache_dir: .cache/splitguard
phash:
  enabled: true
  hamming_threshold: 7
embeddings:
  enabled: true
  model: facebook/dinov2-small
  device: cpu
  batch_size: 16
neighbors:
  index: hnsw
  k: 10
  cosine_threshold: 0.96
  hnsw_m: 24
  hnsw_ef_construction: 160
  hnsw_ef_search: 48
policy:
  exact_is_duplicate: true
  phash_is_duplicate: true
  embedding_only_requires_review: true
repair:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  split_size_weight: 1.5
  class_balance_weight: 2.0
  random_seed: 7
  local_improvement_iterations: 120
  local_improvement_patience: 20
report:
  output_dir: artifacts/report
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.io.max_image_pixels == 40_000_000
    assert config.neighbors.hnsw_m == 24
    assert config.repair.local_improvement_iterations == 120
    assert config.report.output_dir == Path("artifacts/report")


def test_committed_default_configuration_is_valid() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"

    config = load_config(config_path)

    assert config.embeddings.model == "facebook/dinov2-small"
    assert config.repair.train_ratio == 0.8
    assert config.report.output_dir == Path("artifacts")


@pytest.mark.parametrize(
    ("section", "payload"),
    [
        ("root", {"unexpected": True}),
        ("io", {"io": {"skip_bad_images": True}}),
        ("neighbors", {"neighbors": {"metric": "cosine"}}),
        ("repair", {"repair": {"allow_group_splitting": True}}),
    ],
)
def test_unknown_keys_are_rejected(section: str, payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden") as error:
        SplitGuardConfig.model_validate(payload)

    assert section in str(error.value) or section == "root"


def test_ratio_validation_uses_decimal_tolerance() -> None:
    near_exact = RepairConfig(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1000000005,
    )
    repeating = RepairConfig(
        train_ratio=0.3333333333333333,
        val_ratio=0.3333333333333333,
        test_ratio=0.3333333333333333,
    )

    assert near_exact.test_ratio == 0.1000000005
    assert repeating.train_ratio == pytest.approx(1 / 3)

    with pytest.raises(ValidationError, match="must sum to 1"):
        RepairConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.10000001)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_ratio_values_must_be_finite_and_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        RepairConfig(train_ratio=value, val_ratio=0.1, test_ratio=0.1)


def test_repair_objective_and_local_improvement_are_validated() -> None:
    with pytest.raises(ValidationError, match="objective weight"):
        RepairConfig(split_size_weight=0.0, class_balance_weight=0.0)
    with pytest.raises(ValidationError, match="patience cannot exceed"):
        RepairConfig(local_improvement_iterations=10, local_improvement_patience=11)


def test_hnsw_parameters_are_validated_without_restricting_flat_reference() -> None:
    with pytest.raises(ValidationError, match="at least hnsw_m"):
        NeighborsConfig(index="hnsw", hnsw_m=64, hnsw_ef_construction=32)
    with pytest.raises(ValidationError, match="at least k"):
        NeighborsConfig(index="hnsw", k=65, hnsw_ef_search=64)

    flat = NeighborsConfig(index="flat_ip", k=100, hnsw_ef_search=1)
    assert flat.index == "flat_ip"


@pytest.mark.parametrize(
    "path",
    [
        "/private/images",
        r"C:\private\images",
        r"\\server\share\images",
        r"\rooted-on-current-drive",
        "../outside",
        "~/.cache/splitguard",
    ],
)
def test_machine_specific_or_escaping_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValidationError, match="path must"):
        IOConfig(cache_dir=path)
    with pytest.raises(ValidationError, match="path must"):
        ReportConfig(output_dir=path)


def test_canonical_serialization_is_portable_and_contains_no_loader_path(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "machine-specific-location" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "io:\n  cache_dir: cache\\embeddings\nreport:\n  output_dir: artifacts\\report\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    payload = json.loads(config.canonical_json())

    assert payload["io"]["cache_dir"] == "cache/embeddings"
    assert payload["report"]["output_dir"] == "artifacts/report"
    assert str(tmp_path) not in config.canonical_json()


def test_canonical_hash_is_stable_and_sensitive_to_effective_values() -> None:
    first = SplitGuardConfig.model_validate(
        {"phash": {"hamming_threshold": 6}, "io": {"thumbnail_size": 128}}
    )
    reordered = SplitGuardConfig.model_validate(
        {"io": {"thumbnail_size": 128}, "phash": {"hamming_threshold": 6}}
    )
    changed = SplitGuardConfig.model_validate(
        {"io": {"thumbnail_size": 128}, "phash": {"hamming_threshold": 7}}
    )

    assert first.canonical_json() == reordered.canonical_json()
    assert first.config_hash == canonical_config_hash(reordered)
    assert len(first.config_hash) == 64
    assert first.config_hash != changed.config_hash


def test_direct_settings_instantiation_supports_prefixed_nested_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPLITGUARD_PHASH__HAMMING_THRESHOLD", "5")

    config = SplitGuardConfig()

    assert config.phash.hamming_threshold == 5


@pytest.mark.parametrize("document", ["- not\n- a\n- mapping\n", "[unterminated"])
def test_invalid_yaml_document_is_reported(tmp_path: Path, document: str) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(document, encoding="utf-8")

    with pytest.raises(ConfigLoadError):
        load_config(config_path)


def test_non_string_yaml_keys_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-keys.yaml"
    config_path.write_text("1: value\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="keys must be strings"):
        load_config(config_path)
