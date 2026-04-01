"""Raster storage backends for VoxelVault.

Provides a pluggable backend architecture for reading and writing raster files.
Built-in backends:
  - GeoTiffBackend: Cloud Optimized GeoTIFF (COG) via rasterio
  - JP2KBackend:    JPEG 2000 lossless via GDAL's JP2OpenJPEG driver

The storage layer knows nothing about vaults, schemas, or indexes — it
converts between numpy arrays and georeferenced raster files on disk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds as window_from_bounds

if TYPE_CHECKING:
    from voxelvault.models import StorageConfig


# ---------------------------------------------------------------------------
# Backend capabilities and metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendCapabilities:
    """Declares what a storage backend supports."""

    supports_lossless: bool
    supports_overviews: bool
    supports_windowed_reads: bool
    supports_complex_dtypes: bool
    supported_dtypes: frozenset[str]
    format_name: str


@dataclass(frozen=True)
class RasterMetadata:
    """Format-agnostic metadata extracted from a raster file."""

    path: Path
    width: int
    height: int
    band_count: int
    dtype: str
    crs_epsg: int | None
    transform: tuple[float, ...]
    bounds: tuple[float, float, float, float]  # (west, south, east, north)
    nodata: float | int | None
    compression: str | None
    is_tiled: bool
    file_size_bytes: int
    storage_format: str = "geotiff"


# Backward-compatible alias
COGMetadata = RasterMetadata


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class RasterStorageBackend(ABC):
    """Abstract base class for raster storage backends.

    Subclasses must implement ``write`` and ``capabilities``.  The shared
    ``read_window`` and ``read_metadata`` implementations use rasterio's
    format-agnostic reader (works for GeoTIFF, JP2K, and any GDAL format).
    """

    @abstractmethod
    def write(
        self,
        data: np.ndarray,
        path: Path | str,
        crs: CRS | int,
        transform: Affine,
        nodata: float | int | None = None,
        config: StorageConfig | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write a numpy array to a raster file on disk."""

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension including the leading dot (e.g. '.tif', '.jp2')."""

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return the capability profile for this backend."""

    # -- Shared validation ---------------------------------------------------

    def validate_write(self, data: np.ndarray) -> None:
        """Raise ``ValueError`` if *data* cannot be written by this backend."""
        caps = self.capabilities()
        dtype_name = data.dtype.name

        if np.issubdtype(data.dtype, np.complexfloating) and not caps.supports_complex_dtypes:
            raise ValueError(
                f"Complex dtype {dtype_name!r} is not supported by the "
                f"{caps.format_name} backend"
            )

        if dtype_name not in caps.supported_dtypes:
            raise ValueError(
                f"dtype {dtype_name!r} is not supported by the {caps.format_name} backend. "
                f"Supported dtypes: {sorted(caps.supported_dtypes)}"
            )

    # -- Shared read implementations (rasterio is format-agnostic) -----------

    def read_window(
        self,
        path: Path | str,
        window: Window | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        bands: list[int] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Read a windowed subset of a raster file.

        Parameters
        ----------
        path
            Path to the raster file (any rasterio-supported format).
        window
            Pixel-space window (col_off, row_off, width, height).
        bounds
            Geographic bounds (west, south, east, north).
        bands
            1-based band indices.  ``None`` reads all bands.

        Returns
        -------
        (data, profile) where profile includes crs, transform (adjusted for
        the window), shape, and dtype.
        """
        path = Path(path)

        with rasterio.open(path) as ds:
            if bounds is not None:
                file_bounds = ds.bounds
                clipped_west = max(bounds[0], file_bounds.left)
                clipped_south = max(bounds[1], file_bounds.bottom)
                clipped_east = min(bounds[2], file_bounds.right)
                clipped_north = min(bounds[3], file_bounds.top)

                if clipped_west >= clipped_east or clipped_south >= clipped_north:
                    raise ValueError("Requested bounds do not overlap the file extent")

                window = window_from_bounds(
                    clipped_west, clipped_south, clipped_east, clipped_north,
                    transform=ds.transform,
                )

            if bands is None:
                bands = list(range(1, ds.count + 1))

            data = ds.read(indexes=bands, window=window)

            if window is not None:
                win_transform = ds.window_transform(window)
            else:
                win_transform = ds.transform

            profile = {
                "crs": ds.crs,
                "transform": win_transform,
                "width": data.shape[-1],
                "height": data.shape[-2],
                "count": data.shape[0] if data.ndim == 3 else 1,
                "dtype": str(data.dtype),
                "nodata": ds.nodata,
            }

        return data, profile

    def read_metadata(self, path: Path | str) -> RasterMetadata:
        """Read raster metadata without loading pixel data."""
        path = Path(path)

        with rasterio.open(path) as ds:
            crs_epsg: int | None = None
            if ds.crs is not None:
                crs_epsg = ds.crs.to_epsg()

            b = ds.bounds
            bounds_tuple = (b.left, b.bottom, b.right, b.top)

            compression = ds.compression.name if ds.compression else None

            block_shapes = ds.block_shapes
            is_tiled = (
                all(bh < ds.height or bw < ds.width for bh, bw in block_shapes)
                if block_shapes
                else False
            )

            # Detect storage format from driver
            driver = ds.driver
            if driver == "JP2OpenJPEG":
                storage_format = "jp2k"
            else:
                storage_format = "geotiff"

        file_size = path.stat().st_size

        return RasterMetadata(
            path=path,
            width=ds.width,
            height=ds.height,
            band_count=ds.count,
            dtype=str(ds.dtypes[0]),
            crs_epsg=crs_epsg,
            transform=tuple(ds.transform),
            bounds=bounds_tuple,
            nodata=ds.nodata,
            compression=compression,
            is_tiled=is_tiled,
            file_size_bytes=file_size,
            storage_format=storage_format,
        )


# ---------------------------------------------------------------------------
# GeoTIFF / COG Backend
# ---------------------------------------------------------------------------


class GeoTiffBackend(RasterStorageBackend):
    """Cloud Optimized GeoTIFF backend via rasterio.

    Uses the MemoryFile + copy() pattern for atomic COG creation with
    overviews, tiling, and configurable compression.
    """

    @property
    def file_extension(self) -> str:
        return ".tif"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_lossless=True,
            supports_overviews=True,
            supports_windowed_reads=True,
            supports_complex_dtypes=True,
            supported_dtypes=frozenset({
                "uint8", "uint16", "int16", "uint32", "int32",
                "float32", "float64", "complex64", "complex128",
            }),
            format_name="GeoTIFF/COG",
        )

    def write(
        self,
        data: np.ndarray,
        path: Path | str,
        crs: CRS | int,
        transform: Affine,
        nodata: float | int | None = None,
        config: StorageConfig | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write a numpy array as a Cloud Optimized GeoTIFF."""
        from voxelvault.models import StorageConfig

        if config is None:
            config = StorageConfig()

        path = Path(path)

        if not overwrite and path.exists():
            raise FileExistsError(f"File already exists and overwrite=False: {path}")

        # Normalise 2D → 3D
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        if data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

        bands, height, width = data.shape

        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)

        codec = config.codec.upper()
        if codec == "NONE":
            codec = None

        mem_profile: dict = {
            "driver": "GTiff",
            "dtype": data.dtype.name,
            "width": width,
            "height": height,
            "count": bands,
            "crs": crs,
            "transform": transform,
            "tiled": True,
            "blockxsize": config.tile_size,
            "blockysize": config.tile_size,
        }
        if nodata is not None:
            mem_profile["nodata"] = nodata
        if codec:
            mem_profile["compress"] = codec

        if np.issubdtype(data.dtype, np.floating):
            resampling = Resampling.average
        else:
            resampling = Resampling.nearest

        is_complex = np.issubdtype(data.dtype, np.complexfloating)

        overview_levels = config.overview_levels

        path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.MemoryFile() as memfile:
            with memfile.open(**mem_profile) as mem_ds:
                mem_ds.write(data)

                if overview_levels and not is_complex:
                    mem_ds.build_overviews(overview_levels, resampling)
                    mem_ds.update_tags(ns="rio_overview", resampling=resampling.name)

            copy_options: dict = {
                "driver": "GTiff",
                "tiled": True,
                "blockxsize": config.tile_size,
                "blockysize": config.tile_size,
                "copy_src_overviews": True,
            }
            if codec:
                copy_options["compress"] = codec
                codec_level = config.codec_level
                if codec_level is not None and codec in ("DEFLATE", "ZSTD"):
                    level_key = "ZSTD_LEVEL" if codec == "ZSTD" else "ZLEVEL"
                    copy_options[level_key] = codec_level

            rio_copy(memfile.open(), str(path), **copy_options)

        return path


# ---------------------------------------------------------------------------
# JPEG 2000 Lossless Backend
# ---------------------------------------------------------------------------


class JP2KBackend(RasterStorageBackend):
    """JPEG 2000 lossless backend via GDAL's JP2OpenJPEG driver.

    Uses reversible (5/3) wavelet compression for bit-exact round-trip on
    supported integer dtypes.  Float and complex dtypes are explicitly
    rejected.

    Driver notes:
        - Requires the JP2OpenJPEG GDAL driver (OpenJPEG library).
        - Lossless mode is activated via ``QUALITY='100'`` +
          ``REVERSIBLE='YES'`` creation options.
        - JP2K has built-in multi-resolution via wavelet decomposition
          (controlled by ``RESOLUTIONS``), but does not use GeoTIFF-style
          external overview building.
        - Windowed reads are supported via rasterio/GDAL.
        - int32 / uint32 write successfully but fail on read with some
          OpenJPEG versions — these dtypes are excluded from the supported
          set until upstream stabilises.
    """

    @property
    def file_extension(self) -> str:
        return ".jp2"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_lossless=True,
            supports_overviews=False,
            supports_windowed_reads=True,
            supports_complex_dtypes=False,
            supported_dtypes=frozenset({"uint8", "uint16", "int16"}),
            format_name="JPEG 2000 (lossless)",
        )

    def write(
        self,
        data: np.ndarray,
        path: Path | str,
        crs: CRS | int,
        transform: Affine,
        nodata: float | int | None = None,
        config: StorageConfig | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write a numpy array as a lossless JPEG 2000 file.

        Uses the reversible 5/3 DWT for bit-exact compression.
        """
        from voxelvault.models import StorageConfig

        if config is None:
            config = StorageConfig(format="jp2k", codec="jp2k_lossless")

        path = Path(path)

        if not overwrite and path.exists():
            raise FileExistsError(f"File already exists and overwrite=False: {path}")

        # Normalise 2D → 3D
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        if data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

        self.validate_write(data)

        bands, height, width = data.shape

        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)

        # Compute resolution levels from overview_levels (n overviews → n+1 resolutions)
        resolutions = len(config.overview_levels) + 1 if config.overview_levels else 6

        profile: dict = {
            "driver": "JP2OpenJPEG",
            "dtype": data.dtype.name,
            "width": width,
            "height": height,
            "count": bands,
            "crs": crs,
            "transform": transform,
            "QUALITY": "100",
            "REVERSIBLE": "YES",
            "RESOLUTIONS": str(resolutions),
            "BLOCKXSIZE": config.tile_size,
            "BLOCKYSIZE": config.tile_size,
        }
        if nodata is not None:
            profile["nodata"] = nodata

        path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(str(path), "w", **profile) as ds:
            ds.write(data)

        return path


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


_BACKENDS: dict[str, type[RasterStorageBackend]] = {
    "geotiff": GeoTiffBackend,
    "jp2k": JP2KBackend,
}


def register_backend(format_name: str, backend_cls: type[RasterStorageBackend]) -> None:
    """Register a storage backend for a format name."""
    _BACKENDS[format_name] = backend_cls


def get_backend(format_name: str) -> RasterStorageBackend:
    """Return an instantiated backend for the given format name.

    Raises ``ValueError`` if no backend is registered for *format_name*.
    """
    cls = _BACKENDS.get(format_name)
    if cls is None:
        raise ValueError(
            f"No storage backend registered for format {format_name!r}. "
            f"Available: {sorted(_BACKENDS)}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward compatibility)
# ---------------------------------------------------------------------------


def write_cog(
    data: np.ndarray,
    path: Path | str,
    crs: CRS | int,
    transform: Affine,
    nodata: float | int | None = None,
    compression: str = "deflate",
    compression_level: int = 6,
    tile_size: int = 256,
    overview_levels: list[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a numpy array as a Cloud Optimized GeoTIFF.

    This is a backward-compatible convenience wrapper around
    :class:`GeoTiffBackend`.
    """
    from voxelvault.models import StorageConfig

    config = StorageConfig(
        format="geotiff",
        codec=compression,
        codec_level=compression_level,
        tile_size=tile_size,
        overview_levels=overview_levels if overview_levels is not None else [2, 4, 8, 16],
    )
    return GeoTiffBackend().write(
        data=data, path=path, crs=crs, transform=transform,
        nodata=nodata, config=config, overwrite=overwrite,
    )


def read_window(
    path: Path | str,
    window: Window | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    bands: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Read a windowed subset of a raster file (any rasterio-supported format).

    Backward-compatible convenience wrapper.
    """
    return GeoTiffBackend().read_window(path, window=window, bounds=bounds, bands=bands)


def read_metadata(path: Path | str) -> RasterMetadata:
    """Read raster metadata without loading pixel data.

    Backward-compatible convenience wrapper.
    """
    return GeoTiffBackend().read_metadata(path)
