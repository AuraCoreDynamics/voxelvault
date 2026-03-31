"""Query engine for VoxelVault.

Provides functions to query raster cubes by spatial bounds, temporal range,
and variable names, returning stacked numpy arrays with provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from voxelvault.models import (
    CubeDescriptor,
    FileRecord,
    SpatialExtent,
    TemporalExtent,
    Variable,
)
from voxelvault.storage import read_window

if TYPE_CHECKING:
    from voxelvault.vault import Vault


@dataclass
class QueryResult:
    """Result of a vault query.

    Attributes:
        data: Raster data — (bands, height, width) for single file, or
            (time, bands, height, width) for multiple files.
        cube: The CubeDescriptor for the queried cube.
        spatial_extent: The spatial extent of the query.
        temporal_extent: The temporal extent spanning matched files.
        source_files: FileRecords providing provenance.
    """

    data: np.ndarray
    cube: CubeDescriptor
    spatial_extent: SpatialExtent
    temporal_extent: TemporalExtent
    source_files: list[FileRecord] = field(default_factory=list)

    @property
    def variables(self) -> list[Variable]:
        """Variables present in the result."""
        return self.cube.variables


def query_cube(
    vault: Vault,
    cube_name: str,
    spatial_bounds: tuple[float, float, float, float] | None = None,
    temporal_range: tuple[datetime, datetime] | None = None,
    variables: list[str] | None = None,
) -> QueryResult:
    """Query a raster cube.

    Steps:
        1. Resolve cube descriptor — raise ValueError if not found.
        2. Use catalog index to find matching file_ids (spatial + temporal).
        3. Fetch FileRecords from schema engine.
        4. Map variable names to band indices if variable filter is provided.
        5. For each file, read the requested spatial window and bands via storage.
        6. Stack results along a time axis if multiple files match.
        7. Return QueryResult with provenance.

    Args:
        vault: The vault to query.
        cube_name: Name of the cube.
        spatial_bounds: Optional (west, south, east, north) filter.
        temporal_range: Optional (start, end) datetime filter.
        variables: Optional list of variable names to select specific bands.

    Returns:
        QueryResult with data and provenance.

    Raises:
        ValueError: If cube not found or no files match.
    """
    # 1. Resolve cube
    cube = vault.get_cube(cube_name)
    if cube is None:
        raise ValueError(f"Cube {cube_name!r} not found in vault")

    # 2. Find matching file_ids via catalog index
    matching_ids = vault._catalog.query(
        spatial_bounds=spatial_bounds,
        temporal_range=temporal_range,
    )

    # If no index filters, fall back to schema query for this cube
    if spatial_bounds is None and temporal_range is None:
        records = vault._schema.query_files(cube_name=cube_name)
    else:
        # Fetch full FileRecords, filtered to this cube
        records = []
        for fid in matching_ids:
            rec = vault._schema.get_file(fid)
            if rec is not None and rec.cube_name == cube_name:
                records.append(rec)

    # Sort by temporal start for consistent ordering
    records.sort(key=lambda r: r.temporal_extent.start)

    if not records:
        raise ValueError(f"No files match the query for cube {cube_name!r}")

    # 4. Map variable names to band indices
    band_indices: list[int] | None = None
    if variables is not None:
        band_map = {b.variable.name: b.band_index for b in cube.bands}
        band_indices = []
        for var_name in variables:
            if var_name not in band_map:
                raise ValueError(
                    f"Variable {var_name!r} not found in cube {cube_name!r}. "
                    f"Available: {list(band_map.keys())}"
                )
            band_indices.append(band_map[var_name])

    # 5. Read data from each file
    arrays: list[np.ndarray] = []
    for rec in records:
        file_path = vault.path / rec.relative_path
        data, _profile = read_window(
            path=file_path,
            bounds=spatial_bounds,
            bands=band_indices,
        )
        arrays.append(data)

    # 6. Stack or return single
    if len(arrays) == 1:
        result_data = arrays[0]
    else:
        result_data = np.stack(arrays, axis=0)

    # Determine spatial extent for result
    if spatial_bounds is not None:
        w, s, e, n = spatial_bounds
        result_spatial = SpatialExtent(
            west=w, south=s, east=e, north=n, epsg=cube.grid.epsg,
        )
    else:
        result_spatial = cube.grid.bounds

    # Determine temporal extent spanning all matched files
    all_starts = [r.temporal_extent.start for r in records]
    all_ends = [r.temporal_extent.end for r in records]
    result_temporal = TemporalExtent(start=min(all_starts), end=max(all_ends))

    return QueryResult(
        data=result_data,
        cube=cube,
        spatial_extent=result_spatial,
        temporal_extent=result_temporal,
        source_files=records,
    )


def query_single(
    vault: Vault,
    cube_name: str,
    spatial_bounds: tuple[float, float, float, float] | None = None,
    temporal_range: tuple[datetime, datetime] | None = None,
    variables: list[str] | None = None,
) -> QueryResult:
    """Query a raster cube expecting exactly one matching file.

    Like query_cube but raises ValueError if zero or multiple files match.

    Args:
        vault: The vault to query.
        cube_name: Name of the cube.
        spatial_bounds: Optional (west, south, east, north) filter.
        temporal_range: Optional (start, end) datetime filter.
        variables: Optional list of variable names to select specific bands.

    Returns:
        QueryResult with data from exactly one file.

    Raises:
        ValueError: If zero or multiple files match.
    """
    # Resolve cube
    cube = vault.get_cube(cube_name)
    if cube is None:
        raise ValueError(f"Cube {cube_name!r} not found in vault")

    # Find matching file_ids via catalog index
    matching_ids = vault._catalog.query(
        spatial_bounds=spatial_bounds,
        temporal_range=temporal_range,
    )

    # Filter to this cube
    if spatial_bounds is None and temporal_range is None:
        records = vault._schema.query_files(cube_name=cube_name)
    else:
        records = []
        for fid in matching_ids:
            rec = vault._schema.get_file(fid)
            if rec is not None and rec.cube_name == cube_name:
                records.append(rec)

    if len(records) == 0:
        raise ValueError(f"No files match the query for cube {cube_name!r}")
    if len(records) > 1:
        raise ValueError(
            f"Expected exactly one file but found {len(records)} for cube {cube_name!r}"
        )

    rec = records[0]

    # Map variable names to band indices
    band_indices: list[int] | None = None
    if variables is not None:
        band_map = {b.variable.name: b.band_index for b in cube.bands}
        band_indices = []
        for var_name in variables:
            if var_name not in band_map:
                raise ValueError(
                    f"Variable {var_name!r} not found in cube {cube_name!r}. "
                    f"Available: {list(band_map.keys())}"
                )
            band_indices.append(band_map[var_name])

    file_path = vault.path / rec.relative_path
    data, _profile = read_window(
        path=file_path,
        bounds=spatial_bounds,
        bands=band_indices,
    )

    # Determine spatial extent for result
    if spatial_bounds is not None:
        w, s, e, n = spatial_bounds
        result_spatial = SpatialExtent(
            west=w, south=s, east=e, north=n, epsg=cube.grid.epsg,
        )
    else:
        result_spatial = cube.grid.bounds

    result_temporal = rec.temporal_extent

    return QueryResult(
        data=data,
        cube=cube,
        spatial_extent=result_spatial,
        temporal_extent=result_temporal,
        source_files=[rec],
    )
