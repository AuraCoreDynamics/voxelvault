"""Tests for the query engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from voxelvault.models import (
    BandDefinition,
    CubeDescriptor,
    GridDefinition,
    TemporalExtent,
    Variable,
)
from voxelvault.query import QueryResult
from voxelvault.vault import Vault


def _make_cube(name: str = "test_cube", width: int = 32, height: int = 32) -> CubeDescriptor:
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


def _ingest_sample(vault: Vault, cube_name: str, offset_days: int = 0, seed: int = 42):
    """Ingest a sample array and return (data, temporal_extent, result)."""
    rng = np.random.default_rng(seed)
    data = rng.random((3, 32, 32), dtype=np.float32)
    te = _make_temporal(offset_days)
    result = vault.ingest(cube_name, data, te)
    return data, te, result


class TestQueryCube:
    """Tests for query_cube / Vault.query()."""

    def test_query_no_filters(self, tmp_path):
        """Query with no filters returns all files for the cube."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            d1, _, _ = _ingest_sample(v, "test_cube", offset_days=0, seed=42)
            d2, _, _ = _ingest_sample(v, "test_cube", offset_days=1, seed=43)

            result = v.query("test_cube")
            assert isinstance(result, QueryResult)
            # Two files → stacked along time axis
            assert result.data.ndim == 4
            assert result.data.shape[0] == 2
            assert result.data.shape[1] == 3
            assert len(result.source_files) == 2

    def test_query_single_file_no_time_axis(self, tmp_path):
        """Query returning a single file has shape (bands, h, w) — no time axis."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            result = v.query("test_cube")
            assert result.data.ndim == 3
            assert result.data.shape[0] == 3

    def test_query_temporal_range(self, tmp_path):
        """Query with temporal range returns only matching files."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0, seed=42)
            _, te2, _ = _ingest_sample(v, "test_cube", offset_days=5, seed=43)

            # Query only the second day's range
            result = v.query(
                "test_cube",
                temporal_range=(te2.start, te2.end),
            )
            assert len(result.source_files) == 1
            assert result.data.ndim == 3  # single file

    def test_query_spatial_bounds(self, tmp_path):
        """Query with spatial bounds returns spatially matching files."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            # Use bounds that overlap the cube's grid
            bounds = cube.grid.bounds
            result = v.query(
                "test_cube",
                spatial_bounds=(bounds.west, bounds.south, bounds.east, bounds.north),
            )
            assert len(result.source_files) == 1

    def test_query_spatial_bounds_subset(self, tmp_path):
        """Query with sub-extent spatial bounds returns a windowed read."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            grid_bounds = cube.grid.bounds
            # Request a smaller spatial window (inner quarter-ish)
            mid_w = (grid_bounds.west + grid_bounds.east) / 2
            mid_s = (grid_bounds.south + grid_bounds.north) / 2
            sub_bounds = (grid_bounds.west, mid_s, mid_w, grid_bounds.north)
            result = v.query("test_cube", spatial_bounds=sub_bounds)

            # Result should be smaller than full grid
            assert result.data.shape[-1] < cube.grid.width
            assert result.data.shape[-2] < cube.grid.height

    def test_query_variable_filter(self, tmp_path):
        """Query with variable filter returns only requested bands."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            result = v.query("test_cube", variables=["red"])
            # Single band selected
            assert result.data.shape[0] == 1

    def test_query_multiple_variable_filter(self, tmp_path):
        """Query selecting two variables returns two bands."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            result = v.query("test_cube", variables=["red", "blue"])
            assert result.data.shape[0] == 2

    def test_query_nonexistent_cube_raises(self, tmp_path):
        """Query for a nonexistent cube raises ValueError."""
        with Vault.create(tmp_path / "vault") as v:
            with pytest.raises(ValueError, match="not found"):
                v.query("no_such_cube")

    def test_query_no_matches_raises(self, tmp_path):
        """Query that matches no files raises ValueError."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            far_future = (
                datetime(2099, 1, 1, tzinfo=timezone.utc),
                datetime(2099, 12, 31, tzinfo=timezone.utc),
            )
            with pytest.raises(ValueError, match="No files match"):
                v.query("test_cube", temporal_range=far_future)

    def test_query_invalid_variable_raises(self, tmp_path):
        """Query with unknown variable name raises ValueError."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            with pytest.raises(ValueError, match="Variable"):
                v.query("test_cube", variables=["nonexistent_var"])

    def test_query_result_provenance(self, tmp_path):
        """QueryResult.source_files contains the matching FileRecords."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _, _, ingest_result = _ingest_sample(v, "test_cube", offset_days=0)

            qr = v.query("test_cube")
            assert len(qr.source_files) == 1
            assert qr.source_files[0].file_id == ingest_result.file_id

    def test_query_result_variables(self, tmp_path):
        """QueryResult.variables returns the cube's variables."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            qr = v.query("test_cube")
            var_names = [v.name for v in qr.variables]
            assert var_names == ["red", "green", "blue"]

    def test_query_temporal_extent_spanning(self, tmp_path):
        """QueryResult.temporal_extent spans earliest to latest matched files."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _, te0, _ = _ingest_sample(v, "test_cube", offset_days=0, seed=42)
            _, te5, _ = _ingest_sample(v, "test_cube", offset_days=5, seed=43)

            result = v.query("test_cube")
            assert result.temporal_extent.start == te0.start
            assert result.temporal_extent.end == te5.end


class TestQuerySingle:
    """Tests for query_single / Vault.query_single()."""

    def test_query_single_success(self, tmp_path):
        """query_single with exactly one match returns that file's data."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            orig_data, te, _ = _ingest_sample(v, "test_cube", offset_days=0)

            result = v.query_single("test_cube", temporal_range=(te.start, te.end))
            assert result.data.ndim == 3
            assert result.data.shape == (3, 32, 32)
            np.testing.assert_allclose(result.data, orig_data, rtol=1e-4)

    def test_query_single_multiple_matches_raises(self, tmp_path):
        """query_single with multiple matches raises ValueError."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0, seed=42)
            _ingest_sample(v, "test_cube", offset_days=1, seed=43)

            with pytest.raises(ValueError, match="exactly one"):
                v.query_single("test_cube")

    def test_query_single_no_matches_raises(self, tmp_path):
        """query_single with no matches raises ValueError."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)

            with pytest.raises(ValueError, match="No files match"):
                v.query_single("test_cube")

    def test_query_single_nonexistent_cube_raises(self, tmp_path):
        """query_single for a nonexistent cube raises ValueError."""
        with Vault.create(tmp_path / "vault") as v:
            with pytest.raises(ValueError, match="not found"):
                v.query_single("no_cube")


class TestRoundTrip:
    """End-to-end round-trip: create → register → ingest → query → verify."""

    def test_full_roundtrip(self, tmp_path):
        """Full round-trip test: data in = data out."""
        rng = np.random.default_rng(42)
        cube = _make_cube(width=32, height=32)
        data = rng.random((3, 32, 32), dtype=np.float32)
        te = _make_temporal()

        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            v.ingest("test_cube", data, te)

            result = v.query_single("test_cube")
            np.testing.assert_allclose(result.data, data, rtol=1e-4)
            assert result.cube.name == "test_cube"
            assert len(result.source_files) == 1

    def test_roundtrip_reopen(self, tmp_path):
        """Data survives vault close and re-open."""
        rng = np.random.default_rng(42)
        cube = _make_cube(width=32, height=32)
        data = rng.random((3, 32, 32), dtype=np.float32)
        te = _make_temporal()

        vault_dir = tmp_path / "vault"
        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            v.ingest("test_cube", data, te)

        with Vault.open(vault_dir) as v2:
            assert v2.file_count == 1
            result = v2.query_single("test_cube")
            np.testing.assert_allclose(result.data, data, rtol=1e-4)
