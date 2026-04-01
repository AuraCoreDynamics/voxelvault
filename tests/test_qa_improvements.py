"""Tests for QA-identified improvements: temporal persistence, parallelized queries, lifecycle APIs.

These tests validate the three critical improvements identified during QA review:
A. Temporal index persistence across vault close/reopen
B. Parallelized query engine correctness
C. Vault-level delete_cube() and delete_file() lifecycle APIs
"""

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
    rng = np.random.default_rng(seed)
    data = rng.random((3, 32, 32), dtype=np.float32)
    te = _make_temporal(offset_days)
    result = vault.ingest(cube_name, data, te)
    return data, te, result


# =========================================================================
# A. Temporal Index Persistence — Bug Fix Verification
# =========================================================================


class TestTemporalIndexPersistence:
    """Verify that the temporal index is repopulated from SQLite on Vault.open().

    Before the fix, TemporalIndex was memory-only and not repopulated,
    causing temporal-filtered queries to return zero results after reopen.
    """

    def test_temporal_query_after_reopen(self, tmp_path):
        """Temporal filtering works after vault close and reopen."""
        vault_dir = tmp_path / "vault"
        cube = _make_cube()

        # Phase 1: Create vault, ingest 3 time slices
        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0, seed=10)
            _ingest_sample(v, "test_cube", offset_days=5, seed=20)
            _ingest_sample(v, "test_cube", offset_days=10, seed=30)

        # Phase 2: Reopen and query with temporal filter
        with Vault.open(vault_dir) as v:
            # Query middle time slice only
            te_mid = _make_temporal(offset_days=5)
            result = v.query("test_cube", temporal_range=(te_mid.start, te_mid.end))

            assert len(result.source_files) == 1
            assert result.data.ndim == 3  # single file → no time axis

    def test_temporal_range_query_after_reopen(self, tmp_path):
        """Multi-file temporal range query works after reopen."""
        vault_dir = tmp_path / "vault"
        cube = _make_cube()

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            for day in range(5):
                _ingest_sample(v, "test_cube", offset_days=day, seed=day + 100)

        with Vault.open(vault_dir) as v:
            # Query days 1-3 (should match 3 files)
            start = datetime(2024, 1, 2, tzinfo=timezone.utc)
            end = datetime(2024, 1, 4, 1, 0, tzinfo=timezone.utc)
            result = v.query("test_cube", temporal_range=(start, end))

            assert len(result.source_files) == 3
            assert result.data.ndim == 4  # multiple files → time axis
            assert result.data.shape[0] == 3

    def test_temporal_exclusion_after_reopen(self, tmp_path):
        """Temporal filter correctly excludes non-matching files after reopen."""
        vault_dir = tmp_path / "vault"
        cube = _make_cube()

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0, seed=1)
            _ingest_sample(v, "test_cube", offset_days=30, seed=2)

        with Vault.open(vault_dir) as v:
            # Query far future — should find nothing
            far_start = datetime(2099, 1, 1, tzinfo=timezone.utc)
            far_end = datetime(2099, 12, 31, tzinfo=timezone.utc)
            with pytest.raises(ValueError, match="No files match"):
                v.query("test_cube", temporal_range=(far_start, far_end))

    def test_multiple_reopen_cycles(self, tmp_path):
        """Temporal index survives multiple close/reopen cycles."""
        vault_dir = tmp_path / "vault"
        cube = _make_cube()

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0, seed=1)

        # Reopen cycle 1: ingest more data
        with Vault.open(vault_dir) as v:
            _ingest_sample(v, "test_cube", offset_days=10, seed=2)

        # Reopen cycle 2: verify temporal query spans both inserts
        with Vault.open(vault_dir) as v:
            te0 = _make_temporal(offset_days=0)
            result = v.query("test_cube", temporal_range=(te0.start, te0.end))
            assert len(result.source_files) == 1

            te10 = _make_temporal(offset_days=10)
            result = v.query("test_cube", temporal_range=(te10.start, te10.end))
            assert len(result.source_files) == 1

            # Full range gets both
            result = v.query("test_cube")
            assert len(result.source_files) == 2


# =========================================================================
# B. Parallelized Query Engine — Correctness Verification
# =========================================================================


class TestParallelQueryCorrectness:
    """Verify the ThreadPoolExecutor-based parallel query returns correct results.

    The parallelization must not change data ordering or introduce races.
    """

    def test_multi_file_query_preserves_temporal_order(self, tmp_path):
        """Parallel-read files are stacked in temporal order (not arrival order)."""
        cube = _make_cube()
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            arrays = {}
            for day in range(5):
                data, te, _ = _ingest_sample(v, "test_cube", offset_days=day, seed=day)
                arrays[day] = data

            result = v.query("test_cube")
            assert result.data.shape == (5, 3, 32, 32)

            # Verify temporal ordering matches source data
            for i, day in enumerate(range(5)):
                np.testing.assert_allclose(result.data[i], arrays[day], rtol=1e-4)

    def test_parallel_query_data_integrity(self, tmp_path):
        """Each file's data is correctly read and not mixed with others."""
        cube = _make_cube()
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            rng = np.random.default_rng(42)
            ingested = []
            for day in range(8):
                data = rng.random((3, 32, 32), dtype=np.float32)
                te = _make_temporal(offset_days=day)
                v.ingest("test_cube", data, te)
                ingested.append(data)

            result = v.query("test_cube")
            assert result.data.shape == (8, 3, 32, 32)

            for i, expected in enumerate(ingested):
                np.testing.assert_allclose(result.data[i], expected, rtol=1e-4)

    def test_parallel_query_with_variable_filter(self, tmp_path):
        """Variable filtering works correctly with parallel reads."""
        cube = _make_cube()
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            originals = []
            for day in range(3):
                data, _, _ = _ingest_sample(v, "test_cube", offset_days=day, seed=day + 50)
                originals.append(data)

            result = v.query("test_cube", variables=["red"])
            assert result.data.shape == (3, 1, 32, 32)

            for i, orig in enumerate(originals):
                np.testing.assert_allclose(result.data[i, 0], orig[0], rtol=1e-4)


# =========================================================================
# C. Lifecycle Management APIs — delete_cube() and delete_file()
# =========================================================================


class TestDeleteFile:
    """Tests for Vault.delete_file() — the atomic 3-step delete."""

    def test_delete_file_removes_all_artifacts(self, tmp_path):
        """delete_file removes physical file, schema record, and index entry."""
        cube = _make_cube()
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _, _, ingest_result = _ingest_sample(v, "test_cube", offset_days=0)
            file_id = ingest_result.file_id

            # Verify file exists
            assert v.file_count == 1
            abs_path = v.path / ingest_result.relative_path
            assert abs_path.exists()

            # Delete
            v.delete_file(file_id)

            # Schema: no record
            assert v.file_count == 0
            assert v._schema.get_file(file_id) is None

            # Physical file: gone
            assert not abs_path.exists()

            # Index: temporal query should fail (no files)
            with pytest.raises(ValueError, match="No files match"):
                v.query("test_cube")

    def test_delete_file_nonexistent_is_noop(self, tmp_path):
        """delete_file on a nonexistent file_id is a silent no-op."""
        cube = _make_cube()
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            # Should not raise
            v.delete_file("nonexistent-id-12345")
            assert v.file_count == 1  # original file untouched

    def test_delete_file_preserves_other_files(self, tmp_path):
        """Deleting one file doesn't affect other files in the same cube."""
        cube = _make_cube()
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _, te0, r0 = _ingest_sample(v, "test_cube", offset_days=0, seed=10)
            d1, te1, r1 = _ingest_sample(v, "test_cube", offset_days=5, seed=20)

            assert v.file_count == 2

            # Delete first file
            v.delete_file(r0.file_id)
            assert v.file_count == 1

            # Second file still queryable
            result = v.query("test_cube", temporal_range=(te1.start, te1.end))
            assert len(result.source_files) == 1
            np.testing.assert_allclose(result.data, d1, rtol=1e-4)

    def test_delete_file_survives_reopen(self, tmp_path):
        """Deleted file stays deleted after vault close/reopen."""
        cube = _make_cube()
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _, _, r0 = _ingest_sample(v, "test_cube", offset_days=0, seed=10)
            _ingest_sample(v, "test_cube", offset_days=5, seed=20)
            v.delete_file(r0.file_id)

        with Vault.open(vault_dir) as v:
            assert v.file_count == 1


class TestDeleteCube:
    """Tests for Vault.delete_cube() with and without delete_data flag."""

    def test_delete_cube_no_data_empty_cube(self, tmp_path):
        """delete_cube(delete_data=False) succeeds on an empty cube."""
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(_make_cube("alpha"))
            v.register_cube(_make_cube("beta"))

            v.delete_cube("alpha")
            assert "alpha" not in v.list_cubes()
            assert "beta" in v.list_cubes()

    def test_delete_cube_no_data_with_files_raises(self, tmp_path):
        """delete_cube(delete_data=False) raises ValueError when cube has files."""
        cube = _make_cube()
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)

            with pytest.raises(ValueError):
                v.delete_cube("test_cube", delete_data=False)

    def test_delete_cube_with_data_removes_everything(self, tmp_path):
        """delete_cube(delete_data=True) removes files, indexes, physical data, and directory."""
        cube = _make_cube()
        vault_dir = tmp_path / "vault"

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _, _, r0 = _ingest_sample(v, "test_cube", offset_days=0, seed=10)
            _, _, r1 = _ingest_sample(v, "test_cube", offset_days=5, seed=20)

            path0 = v.path / r0.relative_path
            path1 = v.path / r1.relative_path
            cube_dir = v.path / "data" / "test_cube"

            assert path0.exists()
            assert path1.exists()
            assert v.file_count == 2

            # Delete cube with all data
            v.delete_cube("test_cube", delete_data=True)

            # Cube gone
            assert "test_cube" not in v.list_cubes()
            assert v.get_cube("test_cube") is None

            # Files gone
            assert v.file_count == 0
            assert not path0.exists()
            assert not path1.exists()

            # Data directory gone
            assert not cube_dir.exists()

    def test_delete_cube_with_data_preserves_other_cubes(self, tmp_path):
        """Deleting one cube's data doesn't affect other cubes."""
        vault_dir = tmp_path / "vault"
        cube_a = _make_cube("alpha")
        cube_b = _make_cube("beta")

        with Vault.create(vault_dir) as v:
            v.register_cube(cube_a)
            v.register_cube(cube_b)
            _ingest_sample(v, "alpha", offset_days=0, seed=10)
            data_b, te_b, _ = _ingest_sample(v, "beta", offset_days=0, seed=20)

            v.delete_cube("alpha", delete_data=True)

            assert "alpha" not in v.list_cubes()
            assert "beta" in v.list_cubes()
            assert v.file_count == 1

            result = v.query("beta")
            np.testing.assert_allclose(result.data, data_b, rtol=1e-4)

    def test_delete_cube_with_data_survives_reopen(self, tmp_path):
        """Deleted cube and files stay deleted after vault close/reopen."""
        vault_dir = tmp_path / "vault"
        cube = _make_cube()

        with Vault.create(vault_dir) as v:
            v.register_cube(cube)
            _ingest_sample(v, "test_cube", offset_days=0)
            v.delete_cube("test_cube", delete_data=True)

        with Vault.open(vault_dir) as v:
            assert v.list_cubes() == []
            assert v.file_count == 0
