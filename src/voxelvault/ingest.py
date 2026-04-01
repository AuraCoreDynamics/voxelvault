"""Ingestion pipeline for VoxelVault.

Handles writing numpy arrays and existing GeoTIFF files into a vault,
including COG creation, checksum computation, catalog registration,
and atomic cleanup on failure.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from voxelvault.models import FileRecord, SpatialExtent, TemporalExtent
from voxelvault.storage import get_backend, read_metadata

if TYPE_CHECKING:
    from voxelvault.vault import Vault


@dataclass(frozen=True)
class IngestResult:
    """Result of a successful file ingestion."""

    file_id: str
    relative_path: str
    file_size_bytes: int
    checksum: str
    elapsed_seconds: float


def _generate_file_path(cube_name: str, temporal_extent: TemporalExtent, ext: str = ".tif") -> str:
    """Generate a relative file path for a new raster file.

    Format: data/{cube_name}/{iso_timestamp}_{uuid8}{ext}
    """
    ts = temporal_extent.start.strftime("%Y%m%dT%H%M%SZ")
    uid = uuid4().hex[:8]
    return f"data/{cube_name}/{ts}_{uid}{ext}"


def _compute_checksum(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file using streaming chunked reads.

    Reads the file in *chunk_size* byte blocks (default 64 KB) to avoid
    loading multi-GB rasters entirely into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ingest_array(
    vault: Vault,
    cube_name: str,
    data: np.ndarray,
    temporal_extent: TemporalExtent,
    spatial_extent: SpatialExtent | None = None,
) -> IngestResult:
    """Ingest a numpy array into a vault as a COG.

    Steps:
        1. Validate cube_name exists and data shape matches band count.
        2. Generate file path: data/{cube_name}/{iso_timestamp}_{uuid8}.tif
        3. Write COG via storage.write_cog() using vault config compression settings.
        4. Compute SHA-256 checksum of the written file.
        5. Create FileRecord and insert into schema engine.
        6. Insert into catalog index.
        7. Return IngestResult.

    Atomic: if any step after COG write fails, delete the COG file.

    Args:
        vault: The target vault.
        cube_name: Name of the cube to ingest into.
        data: Raster data array — (bands, height, width) or (height, width).
        temporal_extent: Time range for this data.
        spatial_extent: Optional spatial extent override. If None, derived from
            the cube's GridDefinition.bounds.

    Returns:
        IngestResult with file metadata.

    Raises:
        ValueError: If cube_name is not registered or band count mismatches.
    """
    t0 = time.monotonic()

    # 1. Validate cube exists
    cube = vault.get_cube(cube_name)
    if cube is None:
        raise ValueError(f"Cube {cube_name!r} not found in vault")

    # Normalise 2D → 3D for band count check
    arr = data
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")

    if arr.shape[0] != cube.band_count:
        raise ValueError(
            f"Data has {arr.shape[0]} bands but cube {cube_name!r} expects {cube.band_count}"
        )

    # 2. Generate path
    cfg = vault.config
    backend = get_backend(cfg.storage.format)
    rel_path = _generate_file_path(cube_name, temporal_extent, ext=backend.file_extension)
    abs_path = vault.path / rel_path

    # Derive spatial extent from grid if not provided
    if spatial_extent is None:
        spatial_extent = cube.grid.bounds

    # 3. Validate and write raster via backend
    backend.validate_write(arr)
    backend.write(
        data=arr,
        path=abs_path,
        crs=CRS.from_epsg(cube.grid.epsg),
        transform=Affine(*cube.grid.transform),
        config=cfg.storage,
    )

    # 4-7: Catalog the file — atomic cleanup on failure
    try:
        checksum = _compute_checksum(abs_path)
        file_size = abs_path.stat().st_size
        file_id = uuid4().hex

        record = FileRecord(
            file_id=file_id,
            cube_name=cube_name,
            relative_path=rel_path,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            band_count=arr.shape[0],
            file_size_bytes=file_size,
            created_at=datetime.now(timezone.utc),
            checksum=checksum,
        )

        # 5. Insert into schema engine
        vault._schema.insert_file(record)

        # 6. Insert into catalog index
        vault._catalog.insert(record)

    except Exception:
        # Atomic cleanup: remove the written COG on failure
        if abs_path.exists():
            abs_path.unlink()
        raise

    elapsed = time.monotonic() - t0

    return IngestResult(
        file_id=file_id,
        relative_path=rel_path,
        file_size_bytes=file_size,
        checksum=checksum,
        elapsed_seconds=elapsed,
    )


def ingest_file(
    vault: Vault,
    cube_name: str,
    source_path: Path | str,
    temporal_extent: TemporalExtent,
) -> IngestResult:
    """Ingest an existing GeoTIFF/COG file into a vault.

    Reads metadata from the source, copies into the vault data directory,
    and catalogs it.

    Args:
        vault: The target vault.
        cube_name: Name of the cube to ingest into.
        source_path: Path to the source GeoTIFF file.
        temporal_extent: Time range for this data.

    Returns:
        IngestResult with file metadata.

    Raises:
        ValueError: If cube_name is not registered or source file metadata
            doesn't match the cube schema.
    """
    t0 = time.monotonic()
    source_path = Path(source_path)

    # Validate cube exists
    cube = vault.get_cube(cube_name)
    if cube is None:
        raise ValueError(f"Cube {cube_name!r} not found in vault")

    # Read metadata from source
    meta = read_metadata(source_path)

    if meta.band_count != cube.band_count:
        raise ValueError(
            f"Source file has {meta.band_count} bands but cube {cube_name!r} expects {cube.band_count}"
        )

    # Generate destination path and copy
    ext = source_path.suffix or ".tif"
    rel_path = _generate_file_path(cube_name, temporal_extent, ext=ext)
    abs_path = vault.path / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_path), str(abs_path))

    # Build spatial extent from source metadata
    w, s, e, n = meta.bounds
    epsg = meta.crs_epsg if meta.crs_epsg is not None else cube.grid.epsg
    spatial_extent = SpatialExtent(west=w, south=s, east=e, north=n, epsg=epsg)

    try:
        checksum = _compute_checksum(abs_path)
        file_size = abs_path.stat().st_size
        file_id = uuid4().hex

        record = FileRecord(
            file_id=file_id,
            cube_name=cube_name,
            relative_path=rel_path,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            band_count=meta.band_count,
            file_size_bytes=file_size,
            created_at=datetime.now(timezone.utc),
            checksum=checksum,
        )

        vault._schema.insert_file(record)
        vault._catalog.insert(record)

    except Exception:
        if abs_path.exists():
            abs_path.unlink()
        raise

    elapsed = time.monotonic() - t0

    return IngestResult(
        file_id=file_id,
        relative_path=rel_path,
        file_size_bytes=file_size,
        checksum=checksum,
        elapsed_seconds=elapsed,
    )
