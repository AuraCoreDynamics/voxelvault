"""Tests for AuraVault-VoxelVault integration module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

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


def test_extract_track_raster_chip_queries_vault():
    req = TrackRasterChipRequest(
        track_id="TRACK-E3-SENTRY",
        latitude=36.5,
        longitude=-115.5,
        timestamp=datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc),
        cube_name="sentinel2_l2a",
        chip_half_size_deg=0.02,
        temporal_tolerance_seconds=120.0,
    )
    mock_vault = MagicMock()
    mock_vault.query.return_value = MagicMock()

    res = extract_track_raster_chip(mock_vault, req)
    assert res is not None
    assert mock_vault.query.called
    kwargs = mock_vault.query.call_args.kwargs
    assert kwargs["cube_name"] == "sentinel2_l2a"
    assert round(kwargs["spatial_extent"].west, 4) == -115.52
    assert round(kwargs["spatial_extent"].east, 4) == -115.48


def test_extract_track_raster_chip_missing_cube():
    req = TrackRasterChipRequest(
        track_id="TRACK-E3-SENTRY",
        latitude=36.5,
        longitude=-115.5,
        timestamp=datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc),
        cube_name="nonexistent_cube",
    )
    mock_vault = MagicMock()
    mock_vault.query.side_effect = KeyError("Cube not found")

    res = extract_track_raster_chip(mock_vault, req)
    assert res is None
