"""Tests for the ingestion pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest

from voxelvault.ingest import IngestResult
from voxelvault.models import (
    BandDefinition,
    CubeDescriptor,
    GridDefinition,
    SpatialExtent,
    TemporalExtent,
    Variable,
)
from voxelvault.vault import Vault


def _make_cube(name: str = "test_cube", width: int = 64, height: int = 64) -> CubeDescriptor:
    grid = GridDefinition(
        width=width,
        height=height,
        epsg=4326,
        transform=(0.01, 0.0, -180.0, 0.0, -0.01, 90.0),
    )
    bands = [
        BandDefinition(band_index=1, variable=Variable(name="red", unit="dn", dtype="float32")),
        BandDefinition(band_index=2, variable=Variable(name="green", unit="dn", dtype="float32")),
        BandDefinition(band_index=3, variable=Variable(name="blue", unit="dn", dtype="float32")),
    ]
    return CubeDescriptor(name=name, bands=bands, grid=grid, temporal_resolution=timedelta(days=1))


def _make_temporal(offset_days: int = 0) -> TemporalExtent:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset_days)
    return TemporalExtent(start=start, end=start + timedelta(hours=1))


class TestIngestArray:
    """Tests for ingest_array / Vault.ingest()."""

    def test_ingest_writes_cog(self, tmp_path):
        """Ingesting an array creates a COG file under data/{cube}/."""
        rng = np.random.default_rng(42)
        cube = _make_cube()
        data = rng.random((3, 64, 64), dtype=np.float32)
        te = _make_temporal()

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            result = v.ingest("test_cube", data, te)

            assert isinstance(result, IngestResult)
            cog_path = v.path / result.relative_path
            assert cog_path.exists()
            assert cog_path.suffix == ".tif"
            assert result.file_size_bytes > 0
            assert len(result.checksum) == 64  # SHA-256 hex

    def test_ingest_catalogs_in_schema(self, tmp_path):
        """Ingested file is stored in the schema engine."""
        rng = np.random.default_rng(42)
        cube = _make_cube()
        data = rng.random((3, 64, 64), dtype=np.float32)
        te = _make_temporal()

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            result = v.ingest("test_cube", data, te)

            rec = v._schema.get_file(result.file_id)
            assert rec is not None
            assert rec.cube_name == "test_cube"
            assert rec.band_count == 3
            assert rec.checksum == result.checksum

    def test_ingest_indexes_spatially(self, tmp_path):
        """Ingested file can be found via spatial query in the catalog index."""
        rng = np.random.default_rng(42)
        cube = _make_cube()
        data = rng.random((3, 64, 64), dtype=np.float32)
        te = _make_temporal()

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            result = v.ingest("test_cube", data, te)

            bounds = cube.grid.bounds
            hits = v._catalog.query(
                spatial_bounds=(bounds.west, bounds.south, bounds.east, bounds.north),
            )
            assert result.file_id in hits

    def test_ingest_correct_extents(self, tmp_path):
        """FileRecord has correct spatial and temporal extents."""
        rng = np.random.default_rng(42)
        cube = _make_cube()
        data = rng.random((3, 64, 64), dtype=np.float32)
        te = _make_temporal()

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            result = v.ingest("test_cube", data, te)

            rec = v._schema.get_file(result.file_id)
            assert rec is not None
            assert rec.temporal_extent.start == te.start
            assert rec.temporal_extent.end == te.end
            # Spatial extent should match grid bounds
            grid_bounds = cube.grid.bounds
            assert rec.spatial_extent.west == grid_bounds.west
            assert rec.spatial_extent.east == grid_bounds.east

    def test_ingest_wrong_band_count_raises(self, tmp_path):
        """Ingesting data with wrong band count raises ValueError."""
        rng = np.random.default_rng(42)
        cube = _make_cube()
        bad_data = rng.random((5, 64, 64), dtype=np.float32)
        te = _make_temporal()

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            with pytest.raises(ValueError, match="bands"):
                v.ingest("test_cube", bad_data, te)

    def test_ingest_nonexistent_cube_raises(self, tmp_path):
        """Ingesting into an unregistered cube raises ValueError."""
        rng = np.random.default_rng(42)
        data = rng.random((3, 64, 64), dtype=np.float32)
        te = _make_temporal()

        with Vault.create(tmp_path / "vault") as v:
            with pytest.raises(ValueError, match="not found"):
                v.ingest("no_such_cube", data, te)

    def test_ingest_atomicity_cleanup(self, tmp_path):
        """If schema insert fails after COG write, the COG file is cleaned up."""
        rng = np.random.default_rng(42)
        cube = _make_cube()
        data = rng.random((3, 64, 64), dtype=np.float32)
        te = _make_temporal()

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)

            with patch.object(v._schema, "insert_file", side_effect=RuntimeError("db error")):
                with pytest.raises(RuntimeError, match="db error"):
                    v.ingest("test_cube", data, te)

            # No COG files should remain in data dir
            cog_files = list((v.path / "data" / "test_cube").glob("*.tif"))
            assert len(cog_files) == 0

    def test_ingest_file_count_increments(self, tmp_path):
        """file_count increments after each successful ingest."""
        rng = np.random.default_rng(42)
        cube = _make_cube()
        data = rng.random((3, 64, 64), dtype=np.float32)

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            assert v.file_count == 0

            v.ingest("test_cube", data, _make_temporal(0))
            assert v.file_count == 1

            v.ingest("test_cube", data, _make_temporal(1))
            assert v.file_count == 2

    def test_ingest_with_custom_spatial_extent(self, tmp_path):
        """Providing explicit spatial_extent overrides the grid bounds."""
        rng = np.random.default_rng(42)
        cube = _make_cube()
        data = rng.random((3, 64, 64), dtype=np.float32)
        te = _make_temporal()
        custom_spatial = SpatialExtent(west=10.0, south=20.0, east=11.0, north=21.0, epsg=4326)

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            result = v.ingest("test_cube", data, te, spatial_extent=custom_spatial)

            rec = v._schema.get_file(result.file_id)
            assert rec is not None
            assert rec.spatial_extent.west == 10.0
            assert rec.spatial_extent.north == 21.0

    def test_ingest_relative_path_format(self, tmp_path):
        """relative_path starts with data/{cube_name}/ and ends with .tif."""
        rng = np.random.default_rng(42)
        cube = _make_cube()
        data = rng.random((3, 64, 64), dtype=np.float32)
        te = _make_temporal()

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            result = v.ingest("test_cube", data, te)

            assert result.relative_path.startswith("data/test_cube/")
            assert result.relative_path.endswith(".tif")
