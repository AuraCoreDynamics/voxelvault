"""Tests for AuraVault-VoxelVault integration module."""

from datetime import datetime, timezone
import pytest

from voxelvault.auravault import TrackRasterChipRequest, extract_track_raster_chip


def test_track_raster_chip_request_instantiation():
    req = TrackRasterChipRequest(
        track_id="TRACK-E3-SENTRY",
        latitude=36.5,
        longitude=-115.5,
        timestamp=datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc),
        cube_name="sentinel2_l2a",
        chip_half_size_deg=0.03,
    )
    assert req.track_id == "TRACK-E3-SENTRY"
    assert req.latitude == 36.5
    assert req.longitude == -115.5
    assert req.chip_half_size_deg == 0.03
