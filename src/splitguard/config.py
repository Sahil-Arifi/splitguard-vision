"""Validated, immutable configuration for SplitGuard Vision."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_RATIO_SUM_TOLERANCE = Decimal("1e-9")


class ConfigLoadError(ValueError):
    """Raised when a configuration document cannot be read or decoded."""


def _portable_relative_path(value: object) -> Path:
    """Validate and normalize a path without resolving it against this machine."""
    if isinstance(value, Path):
        raw = str(value)
    elif isinstance(value, str):
        raw = value
    else:
        raise ValueError("path must be a string or pathlib.Path")

    raw = raw.strip()
    if not raw or "\x00" in raw:
        raise ValueError("path must not be empty")

    windows_path = PureWindowsPath(raw)
    portable_path = PurePosixPath(raw.replace("\\", "/"))
    if windows_path.drive or windows_path.root or portable_path.is_absolute():
        raise ValueError("path must be relative; absolute paths are not portable")
    if ".." in portable_path.parts:
        raise ValueError("path must not traverse outside the project")
    if portable_path == PurePosixPath(".") or portable_path.parts[0].startswith("~"):
        raise ValueError("path must name a project-relative location")

    return Path(*portable_path.parts)


class FrozenStrictModel(BaseModel):
    """Base class shared by every nested configuration section."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class IOConfig(FrozenStrictModel):
    """Input validation, thumbnail, and local-cache settings."""

    max_image_pixels: int = Field(default=50_000_000, gt=0)
    thumbnail_size: int = Field(default=256, ge=16, le=4096)
    cache_dir: Path = Field(default=Path(".splitguard/cache"))

    @field_validator("cache_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls, value: object) -> Path:
        return _portable_relative_path(value)

    @field_serializer("cache_dir")
    def serialize_cache_dir(self, value: Path) -> str:
        return value.as_posix()


class PHashConfig(FrozenStrictModel):
    """Perceptual-hash detector settings."""

    enabled: bool = True
    hamming_threshold: int = Field(default=8, ge=0, le=64)


class EmbeddingsConfig(FrozenStrictModel):
    """Deep image-embedding settings."""

    enabled: bool = True
    model: str = Field(default="facebook/dinov2-small", min_length=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(default=32, gt=0)


class NeighborsConfig(FrozenStrictModel):
    """FAISS exact-reference and HNSW approximate-search settings."""

    index: Literal["hnsw", "flat", "flat_ip"] = "hnsw"
    k: int = Field(default=20, gt=0)
    cosine_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    hnsw_m: int = Field(default=32, ge=2)
    hnsw_ef_construction: int = Field(default=200, ge=2)
    hnsw_ef_search: int = Field(default=64, ge=1)

    @model_validator(mode="after")
    def validate_hnsw_parameters(self) -> NeighborsConfig:
        if self.index != "hnsw":
            return self
        if self.hnsw_ef_construction < self.hnsw_m:
            raise ValueError("hnsw_ef_construction must be at least hnsw_m")
        if self.hnsw_ef_search < self.k:
            raise ValueError("hnsw_ef_search must be at least k")
        return self


class PolicyConfig(FrozenStrictModel):
    """Evidence-classification policy settings."""

    exact_is_duplicate: bool = True
    phash_is_duplicate: bool = True
    embedding_only_requires_review: bool = True


class RepairConfig(FrozenStrictModel):
    """Group-aware split repair objective and local-search settings."""

    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    split_size_weight: float = Field(default=1.0, ge=0.0)
    class_balance_weight: float = Field(default=1.0, ge=0.0)
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    local_improvement_iterations: int = Field(default=250, gt=0)
    local_improvement_patience: int = Field(default=25, gt=0)

    @model_validator(mode="after")
    def validate_repair_objective(self) -> RepairConfig:
        decimal_ratios = (
            Decimal(str(self.train_ratio)),
            Decimal(str(self.val_ratio)),
            Decimal(str(self.test_ratio)),
        )
        ratio_sum = sum(decimal_ratios, start=Decimal(0))
        if abs(ratio_sum - Decimal(1)) > _RATIO_SUM_TOLERANCE:
            raise ValueError(
                "train_ratio, val_ratio, and test_ratio must sum to 1 "
                f"within {_RATIO_SUM_TOLERANCE}; got {ratio_sum}"
            )
        if self.split_size_weight == 0.0 and self.class_balance_weight == 0.0:
            raise ValueError("at least one repair objective weight must be positive")
        if self.local_improvement_patience > self.local_improvement_iterations:
            raise ValueError(
                "local_improvement_patience cannot exceed local_improvement_iterations"
            )
        return self


class ReportConfig(FrozenStrictModel):
    """Static report output settings."""

    output_dir: Path = Field(default=Path("artifacts"))

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, value: object) -> Path:
        return _portable_relative_path(value)

    @field_serializer("output_dir")
    def serialize_output_dir(self, value: Path) -> str:
        return value.as_posix()


class SplitGuardConfig(BaseSettings):
    """Complete effective configuration for a SplitGuard run."""

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
        env_prefix="SPLITGUARD_",
        env_nested_delimiter="__",
        env_ignore_empty=True,
    )

    io: IOConfig = Field(default_factory=IOConfig)
    phash: PHashConfig = Field(default_factory=PHashConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    neighbors: NeighborsConfig = Field(default_factory=NeighborsConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SplitGuardConfig:
        """Load one YAML mapping and validate the complete effective configuration."""
        config_path = Path(path)
        try:
            document = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            detail = exc.strerror or exc.__class__.__name__
            raise ConfigLoadError(
                f"could not read configuration file {config_path.name!r}: {detail}"
            ) from exc

        try:
            loaded: object = yaml.safe_load(document)
        except yaml.YAMLError as exc:
            message = f"invalid YAML in configuration file {config_path.name!r}"
            raise ConfigLoadError(message) from exc

        if loaded is None:
            payload: dict[str, object] = {}
        elif isinstance(loaded, dict):
            if not all(isinstance(key, str) for key in loaded):
                raise ConfigLoadError("configuration mapping keys must be strings")
            payload = cast(dict[str, object], loaded)
        else:
            raise ConfigLoadError("configuration document must contain a YAML mapping")

        # model_validate deliberately validates the file payload itself. BaseSettings
        # environment sources remain available when callers instantiate the class directly.
        return cls.model_validate(payload)

    def canonical_dict(self) -> dict[str, object]:
        """Return the complete, portable representation used for reproducibility."""
        return cast(dict[str, object], self.model_dump(mode="json"))

    def canonical_json(self) -> str:
        """Serialize deterministically without machine-specific absolute paths."""
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical effective configuration."""
        return canonical_config_hash(self)


def canonical_config_hash(config: SplitGuardConfig) -> str:
    """Return the stable SHA-256 identity of an effective configuration."""
    return hashlib.sha256(config.canonical_json().encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> SplitGuardConfig:
    """Load a SplitGuard configuration from YAML."""
    return SplitGuardConfig.from_yaml(path)


__all__ = [
    "ConfigLoadError",
    "EmbeddingsConfig",
    "IOConfig",
    "NeighborsConfig",
    "PHashConfig",
    "PolicyConfig",
    "RepairConfig",
    "ReportConfig",
    "SplitGuardConfig",
    "canonical_config_hash",
    "load_config",
]
