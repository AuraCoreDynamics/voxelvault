"""GeoTIFF / Cloud Optimized GeoTIFF (COG) I/O via rasterio.

This is the lowest-level storage layer — it converts between numpy arrays and
COG files on disk. It knows nothing about vaults, schemas, or indexes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds as window_from_bounds


# ---------------------------------------------------------------------------
# COG Writer
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

    Uses the MemoryFile + copy() pattern for atomic COG creation:
    1. Write the full dataset to a MemoryFile with tiling options.
    2. Build overviews in the MemoryFile.
    3. Copy from the MemoryFile to the output path with COG creation options.
    """
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

    compress_upper = compression.upper()
    if compress_upper == "NONE":
        compress_upper = None

    # Build rasterio profile for the in-memory dataset
    mem_profile: dict = {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "width": width,
        "height": height,
        "count": bands,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": tile_size,
        "blockysize": tile_size,
    }
    if nodata is not None:
        mem_profile["nodata"] = nodata
    if compress_upper:
        mem_profile["compress"] = compress_upper

    # Choose resampling: nearest for integer/complex, average for float
    if np.issubdtype(data.dtype, np.floating):
        resampling = Resampling.average
    else:
        resampling = Resampling.nearest

    is_complex = np.issubdtype(data.dtype, np.complexfloating)

    if overview_levels is None:
        overview_levels = [2, 4, 8, 16]

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.MemoryFile() as memfile:
        with memfile.open(**mem_profile) as mem_ds:
            mem_ds.write(data)

            # Build overviews (skip for complex dtypes — rasterio can't resample them)
            if overview_levels and not is_complex:
                mem_ds.build_overviews(overview_levels, resampling)
                mem_ds.update_tags(ns="rio_overview", resampling=resampling.name)

        # Copy creation options for the final file
        copy_options: dict = {
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": tile_size,
            "blockysize": tile_size,
            "copy_src_overviews": True,
        }
        if compress_upper:
            copy_options["compress"] = compress_upper
            if compress_upper in ("DEFLATE", "ZSTD"):
                copy_options[f"{compress_upper}_LEVEL" if compress_upper == "ZSTD" else "ZLEVEL"] = compression_level

        rio_copy(memfile.open(), str(path), **copy_options)

    return path


# ---------------------------------------------------------------------------
# Windowed Reader
# ---------------------------------------------------------------------------

def read_window(
    path: Path | str,
    window: Window | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    bands: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Read a windowed subset of a COG.

    Parameters
    ----------
    path : Path | str
        Path to the GeoTIFF file.
    window : Window | None
        Pixel-space window (col_off, row_off, width, height).
    bounds : tuple | None
        Geographic bounds (west, south, east, north). Converted to a Window.
    bands : list[int] | None
        1-based band indices to read.  ``None`` reads all bands.

    Returns
    -------
    (data, profile) where profile includes crs, transform (adjusted for the
    window), shape, and dtype.
    """
    path = Path(path)

    with rasterio.open(path) as ds:
        if bounds is not None:
            # Clip requested bounds to the file extent
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

        # Compute the window transform
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


# ---------------------------------------------------------------------------
# Metadata Reader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class COGMetadata:
    """Metadata extracted from a COG file without loading pixel data."""

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


def read_metadata(path: Path | str) -> COGMetadata:
    """Read COG metadata without loading pixel data."""
    path = Path(path)

    with rasterio.open(path) as ds:
        crs_epsg: int | None = None
        if ds.crs is not None:
            crs_epsg = ds.crs.to_epsg()

        b = ds.bounds
        bounds_tuple = (b.left, b.bottom, b.right, b.top)

        compression = ds.compression.name if ds.compression else None

        # Determine tiling from block shapes
        block_shapes = ds.block_shapes
        is_tiled = all(bh < ds.height or bw < ds.width for bh, bw in block_shapes) if block_shapes else False

    file_size = path.stat().st_size

    return COGMetadata(
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
    )
