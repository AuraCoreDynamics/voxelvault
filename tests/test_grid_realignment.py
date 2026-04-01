"""Tests for multi-INT fusion / grid realignment (target_grid reprojection)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from voxelvault import (
    BandDefinition,
    CubeDescriptor,
    GridDefinition,
    TemporalExtent,
    Variable,
    Vault,
)


@pytest.fixture()
def fusion_vault(tmp_path):
    """Create a vault with a cube and two ingested time slices."""
    grid = GridDefinition(
        width=64, height=64, epsg=4326,
        transform=(0.1, 0, -10.0, 0, -0.1, 10.0),
    )
    cube = CubeDescriptor(
        name="sensor_a",
        bands=[
            BandDefinition(
                band_index=1,
                variable=Variable(name="radiance", unit="W/m2/sr", dtype="float32"),
            ),
        ],
        grid=grid,
    )
    with Vault.create(tmp_path / "v") as vault:
        vault.register_cube(cube)
        data = np.ones((1, 64, 64), dtype=np.float32) * 42.0
        te = TemporalExtent(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, 23, 59, 59, tzinfo=timezone.utc),
        )
        vault.ingest("sensor_a", data, te)
    return tmp_path / "v"


class TestGridRealignment:
    """Verify query-time reprojection via target_grid."""

    def test_reproject_to_coarser_grid(self, fusion_vault):
        """Querying with a coarser target_grid returns the target dimensions."""
        target = GridDefinition(
            width=32, height=32, epsg=4326,
            transform=(0.2, 0, -10.0, 0, -0.2, 10.0),
        )
        with Vault.open(fusion_vault) as vault:
            result = vault.query("sensor_a", target_grid=target)
        assert result.data.shape == (1, 32, 32)
        assert result.spatial_extent == target.bounds

    def test_reproject_to_finer_grid(self, fusion_vault):
        """Querying with a finer target_grid upsamples correctly."""
        target = GridDefinition(
            width=128, height=128, epsg=4326,
            transform=(0.05, 0, -10.0, 0, -0.05, 10.0),
        )
        with Vault.open(fusion_vault) as vault:
            result = vault.query("sensor_a", target_grid=target)
        assert result.data.shape == (1, 128, 128)

    def test_reproject_preserves_values(self, fusion_vault):
        """Constant-value raster should stay constant after reprojection."""
        target = GridDefinition(
            width=32, height=32, epsg=4326,
            transform=(0.2, 0, -10.0, 0, -0.2, 10.0),
        )
        with Vault.open(fusion_vault) as vault:
            result = vault.query("sensor_a", target_grid=target)
        # All source pixels are 42.0; bilinear resampling of a constant
        # field should produce ~42.0 everywhere (edge effects may differ).
        interior = result.data[0, 2:-2, 2:-2]
        np.testing.assert_allclose(interior, 42.0, atol=1e-3)

    def test_query_single_with_target_grid(self, fusion_vault):
        """query_single also supports target_grid."""
        target = GridDefinition(
            width=32, height=32, epsg=4326,
            transform=(0.2, 0, -10.0, 0, -0.2, 10.0),
        )
        with Vault.open(fusion_vault) as vault:
            result = vault.query_single("sensor_a", target_grid=target)
        assert result.data.shape == (1, 32, 32)

    def test_no_target_grid_returns_native(self, fusion_vault):
        """Without target_grid, native grid dimensions are returned (backward compat)."""
        with Vault.open(fusion_vault) as vault:
            result = vault.query("sensor_a")
        assert result.data.shape == (1, 64, 64)
