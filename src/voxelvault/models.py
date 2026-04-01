"""VoxelVault core domain models.

All models are immutable (frozen=True) Pydantic v2 models representing the
fundamental concepts of the spatiotemporal raster cube engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Variable(BaseModel):
    """Describes a single measured quantity (e.g. temperature, reflectance)."""

    model_config = ConfigDict(frozen=True)

    name: str
    unit: str
    dtype: str
    nodata: float | int | None = None
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_must_be_identifier(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError(f"Variable name must be a valid Python identifier, got {v!r}")
        return v

    @field_validator("dtype")
    @classmethod
    def _dtype_must_be_valid(cls, v: str) -> str:
        try:
            np.dtype(v)
        except TypeError as exc:
            raise ValueError(f"Invalid numpy dtype string: {v!r}") from exc
        return v


class BandDefinition(BaseModel):
    """Maps a variable to a 1-based GeoTIFF band index."""

    model_config = ConfigDict(frozen=True)

    band_index: int
    variable: Variable
    component: Literal["scalar", "real", "imaginary"] = "scalar"

    @field_validator("band_index")
    @classmethod
    def _band_index_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"band_index must be >= 1 (GDAL convention), got {v}")
        return v


class SpatialExtent(BaseModel):
    """Geographic bounding box referenced by EPSG code."""

    model_config = ConfigDict(frozen=True)

    west: float
    south: float
    east: float
    north: float
    epsg: int

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.west >= self.east:
            raise ValueError(f"west ({self.west}) must be less than east ({self.east})")
        if self.south >= self.north:
            raise ValueError(f"south ({self.south}) must be less than north ({self.north})")
        return self


class TemporalExtent(BaseModel):
    """Time range — both endpoints must be timezone-aware."""

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.start.tzinfo is None:
            raise ValueError("start must be timezone-aware")
        if self.end.tzinfo is None:
            raise ValueError("end must be timezone-aware")
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")
        return self


class GridDefinition(BaseModel):
    """Spatial grid parameters including affine transform (6 coefficients)."""

    model_config = ConfigDict(frozen=True)

    width: int
    height: int
    epsg: int
    transform: tuple[float, ...]

    @property
    def resolution(self) -> tuple[float, float]:
        """Pixel resolution as (x_res, y_res) from the affine transform."""
        return (self.transform[0], self.transform[4])

    @property
    def bounds(self) -> SpatialExtent:
        """Compute bounding box from grid dimensions and affine transform."""
        # transform: (x_res, x_skew, x_origin, y_skew, y_res, y_origin)
        x_origin = self.transform[2]
        y_origin = self.transform[5]
        x_res = self.transform[0]
        y_res = self.transform[4]

        x_end = x_origin + self.width * x_res
        y_end = y_origin + self.height * y_res

        west = min(x_origin, x_end)
        east = max(x_origin, x_end)
        south = min(y_origin, y_end)
        north = max(y_origin, y_end)

        return SpatialExtent(west=west, south=south, east=east, north=north, epsg=self.epsg)


class CubeDescriptor(BaseModel):
    """Complete raster cube schema — the blueprint for a set of COG files."""

    model_config = ConfigDict(frozen=True)

    name: str
    bands: list[BandDefinition]
    grid: GridDefinition
    temporal_resolution: timedelta | None = None
    description: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("bands")
    @classmethod
    def _unique_band_indices(cls, v: list[BandDefinition]) -> list[BandDefinition]:
        indices = [b.band_index for b in v]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate band_index values: {indices}")
        return v

    @property
    def variables(self) -> list[Variable]:
        """Extract the Variable from each band definition."""
        return [b.variable for b in self.bands]

    @property
    def band_count(self) -> int:
        return len(self.bands)


class StorageConfig(BaseModel):
    """Storage backend and codec configuration.

    Controls the raster container format, compression codec, tiling, and
    overview settings used when writing files into a vault.

    Supported format/codec combinations:
        geotiff: deflate, lzw, zstd, none
        jp2k:    jp2k_lossless
    """

    model_config = ConfigDict(frozen=True)

    format: Literal["geotiff", "jp2k"] = "geotiff"
    codec: str = "deflate"
    codec_level: int | None = 6
    tile_size: int = 256
    overview_levels: list[int] = Field(default_factory=lambda: [2, 4, 8, 16])

    _VALID_CODECS: dict[str, frozenset[str]] = {
        "geotiff": frozenset({"deflate", "lzw", "zstd", "none"}),
        "jp2k": frozenset({"jp2k_lossless"}),
    }

    @model_validator(mode="after")
    def _validate_codec_for_format(self) -> Self:
        allowed = self._VALID_CODECS.get(self.format, frozenset())
        if self.codec not in allowed:
            raise ValueError(
                f"Codec {self.codec!r} is not valid for format {self.format!r}. "
                f"Valid codecs: {sorted(allowed)}"
            )
        return self


class VaultConfig(BaseModel):
    """Vault instance configuration — controls storage behaviour.

    Accepts both the new structured ``storage`` field and legacy flat fields
    (``compression``, ``compression_level``, ``tile_size``, ``overview_levels``)
    for backward compatibility with existing vault.json files.
    """

    model_config = ConfigDict(frozen=True)

    storage: StorageConfig = Field(default_factory=StorageConfig)
    default_epsg: int = 4326

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Migrate flat compression/tile fields into the nested storage config."""
        if not isinstance(data, dict):
            return data
        if "storage" in data:
            return data

        storage: dict[str, Any] = {}
        if "compression" in data:
            codec = data.pop("compression")
            storage["codec"] = codec
        if "compression_level" in data:
            storage["codec_level"] = data.pop("compression_level")
        if "tile_size" in data:
            storage["tile_size"] = data.pop("tile_size")
        if "overview_levels" in data:
            storage["overview_levels"] = data.pop("overview_levels")
        if storage:
            data["storage"] = storage
        return data

    # --- Legacy property accessors (backward compatibility) ---

    @property
    def compression(self) -> str:
        """Codec name (legacy accessor for ``storage.codec``)."""
        return self.storage.codec

    @property
    def compression_level(self) -> int:
        """Codec compression level (legacy accessor)."""
        return self.storage.codec_level if self.storage.codec_level is not None else 6

    @property
    def tile_size(self) -> int:
        """Tile size in pixels (legacy accessor for ``storage.tile_size``)."""
        return self.storage.tile_size

    @property
    def overview_levels(self) -> list[int]:
        """Overview decimation levels (legacy accessor)."""
        return self.storage.overview_levels


class FileRecord(BaseModel):
    """Metadata for a single ingested COG file in the vault."""

    model_config = ConfigDict(frozen=True)

    file_id: str
    cube_name: str
    relative_path: str
    spatial_extent: SpatialExtent
    temporal_extent: TemporalExtent
    band_count: int
    file_size_bytes: int
    created_at: datetime
    checksum: str

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v
