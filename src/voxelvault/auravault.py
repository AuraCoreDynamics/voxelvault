"""AuraVault Integration Module for VoxelVault.

Bridges discrete spatiotemporal tracks from AuraVault with continuous 4D/5D
raster data cubes (Cloud-Optimized GeoTIFFs, EO/IR, SAR, multispectral bands) in VoxelVault.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from voxelvault.models import SpatialExtent, TemporalExtent

if TYPE_CHECKING:
    from voxelvault.query import QueryResult
    from voxelvault.vault import Vault


@dataclass(frozen=True)
class TrackRasterChipRequest:
    """Request to extract a VoxelVault raster chip around an AuraVault discrete track."""

    track_id: str
    latitude: float
    longitude: float
    timestamp: datetime
    cube_name: str
    chip_half_size_deg: float = 0.02
    temporal_tolerance_seconds: float = 300.0
    epsg: int = 4326


def extract_track_raster_chip(
    vault: Vault,
    request: TrackRasterChipRequest,
) -> QueryResult | None:
    """Extract a spatiotemporal raster chip from VoxelVault corresponding to an AuraVault track.

    Args:
        vault: The VoxelVault instance to query.
        request: The TrackRasterChipRequest specifying track coordinates and cube.

    Returns:
        QueryResult containing numpy raster data if intersecting files exist, else None.
    """
    spatial = SpatialExtent(
        west=request.longitude - request.chip_half_size_deg,
        south=request.latitude - request.chip_half_size_deg,
        east=request.longitude + request.chip_half_size_deg,
        north=request.latitude + request.chip_half_size_deg,
        epsg=request.epsg,
    )

    temporal = TemporalExtent(
        start=request.timestamp - timedelta(seconds=request.temporal_tolerance_seconds),
        end=request.timestamp + timedelta(seconds=request.temporal_tolerance_seconds),
    )

    try:
        return vault.query(
            cube_name=request.cube_name,
            spatial_extent=spatial,
            temporal_extent=temporal,
        )
    except KeyError:
        return None
