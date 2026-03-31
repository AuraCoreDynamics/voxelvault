"""End-to-end integration tests for VoxelVault.

These tests exercise the full stack — vault creation, cube registration,
ingestion, querying, and the CLI — using real files on disk (via tmp_path).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from voxelvault.cli import cli
from voxelvault.models import (
    BandDefinition,
    CubeDescriptor,
    GridDefinition,
    TemporalExtent,
    Variable,
)
from voxelvault.storage import write_cog
from voxelvault.vault import Vault

RNG = np.random.default_rng(42)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_cube(
    name: str = "test_cube",
    width: int = 32,
    height: int = 32,
    *,
    variables: list[tuple[str, str, str]] | None = None,
) -> CubeDescriptor:
    """Build a small CubeDescriptor for integration tests."""
    if variables is None:
        variables = [
            ("red", "reflectance", "uint16"),
            ("green", "reflectance", "uint16"),
            ("blue", "reflectance", "uint16"),
        ]
    bands = [
        BandDefinition(
            band_index=i + 1,
            variable=Variable(name=vname, unit=unit, dtype=dtype),
        )
        for i, (vname, unit, dtype) in enumerate(variables)
    ]
    grid = GridDefinition(
        width=width,
        height=height,
        epsg=4326,
        transform=(0.01, 0.0, -180.0, 0.0, -0.01, 90.0),
    )
    return CubeDescriptor(
        name=name,
        bands=bands,
        grid=grid,
        temporal_resolution=timedelta(days=1),
        description=f"Integration test cube: {name}",
    )


def _write_test_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    west: float = -180.0,
    south: float = 89.68,
    east: float = -179.68,
    north: float = 90.0,
    epsg: int = 4326,
) -> Path:
    """Write a test GeoTIFF file using storage.write_cog()."""
    if data.ndim == 2:
        h, w = data.shape
    else:
        _, h, w = data.shape
    transform = from_bounds(west, south, east, north, w, h)
    crs = CRS.from_epsg(epsg)
    return write_cog(data=data, path=path, crs=crs, transform=transform)


# ------------------------------------------------------------------
# T6.2: Integration tests
# ------------------------------------------------------------------


class TestFullLifecycle:
    """Create vault → register cube → ingest array → query → verify data."""

    def test_full_lifecycle(self, tmp_path):
        vault_dir = tmp_path / "vault"
        cube = _make_cube()
        data = RNG.integers(0, 10000, size=(3, 32, 32), dtype=np.uint16)
        t_extent = TemporalExtent(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        with Vault.create(vault_dir) as vault:
            vault.register_cube(cube)
            result = vault.ingest(cube.name, data, t_extent)

            assert result.file_size_bytes > 0
            assert result.checksum

            qr = vault.query(cube.name)

            assert qr.data.shape == (3, 32, 32)
            np.testing.assert_array_equal(qr.data, data)
            assert len(qr.source_files) == 1


class TestMultiTemporalIngestAndQuery:
    """Ingest multiple time slices, query a time range."""

    def test_multi_temporal_ingest_and_query(self, tmp_path):
        vault_dir = tmp_path / "vault"
        cube = _make_cube()

        slices = []
        for day in range(1, 4):
            arr = RNG.integers(0, 10000, size=(3, 32, 32), dtype=np.uint16)
            slices.append((
                arr,
                TemporalExtent(
                    start=datetime(2024, 1, day, tzinfo=timezone.utc),
                    end=datetime(2024, 1, day, 23, 59, 59, tzinfo=timezone.utc),
                ),
            ))

        with Vault.create(vault_dir) as vault:
            vault.register_cube(cube)
            for arr, te in slices:
                vault.ingest(cube.name, arr, te)

            # Query all — should get 3 files
            qr_all = vault.query(cube.name)
            assert len(qr_all.source_files) == 3
            assert qr_all.data.shape == (3, 3, 32, 32)  # (time, bands, h, w)

            # Query just days 1-2
            qr_sub = vault.query(
                cube.name,
                temporal_range=(
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, 23, 59, 59, tzinfo=timezone.utc),
                ),
            )
            assert len(qr_sub.source_files) == 2


class TestSpatialSubsetQuery:
    """Ingest a raster, query a small spatial window."""

    def test_spatial_subset_query(self, tmp_path):
        vault_dir = tmp_path / "vault"
        width, height = 64, 64
        cube = _make_cube(width=width, height=height)
        data = RNG.integers(0, 10000, size=(3, height, width), dtype=np.uint16)
        t_extent = TemporalExtent(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        with Vault.create(vault_dir) as vault:
            vault.register_cube(cube)
            vault.ingest(cube.name, data, t_extent)

            # Query a subset: the cube covers -180...-179.36, 89.36...90.0
            # Request a window in the middle
            sub_bounds = (-179.9, 89.5, -179.7, 89.8)
            qr = vault.query(cube.name, spatial_bounds=sub_bounds)

            # Result should be smaller than the full grid
            assert qr.data.shape[1] < height
            assert qr.data.shape[2] < width
            assert qr.data.shape[0] == 3  # all bands


class TestVariableFilterQuery:
    """Ingest multi-band data, query specific variables."""

    def test_variable_filter_query(self, tmp_path):
        vault_dir = tmp_path / "vault"
        cube = _make_cube()
        data = RNG.integers(0, 10000, size=(3, 32, 32), dtype=np.uint16)
        t_extent = TemporalExtent(
            start=datetime(2024, 3, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        with Vault.create(vault_dir) as vault:
            vault.register_cube(cube)
            vault.ingest(cube.name, data, t_extent)

            # Query only "red" variable
            qr = vault.query(cube.name, variables=["red"])
            assert qr.data.shape == (1, 32, 32)
            np.testing.assert_array_equal(qr.data[0], data[0])

            # Query "red" and "blue"
            qr2 = vault.query(cube.name, variables=["red", "blue"])
            assert qr2.data.shape == (2, 32, 32)
            np.testing.assert_array_equal(qr2.data[0], data[0])
            np.testing.assert_array_equal(qr2.data[1], data[2])


class TestVaultReopen:
    """Create vault, close, reopen, verify data persists."""

    def test_vault_reopen(self, tmp_path):
        vault_dir = tmp_path / "vault"
        cube = _make_cube()
        data = RNG.integers(0, 10000, size=(3, 32, 32), dtype=np.uint16)
        t_extent = TemporalExtent(
            start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            end=datetime(2024, 4, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        # Create, register, ingest, close
        with Vault.create(vault_dir) as vault:
            vault.register_cube(cube)
            vault.ingest(cube.name, data, t_extent)

        # Reopen and verify
        with Vault.open(vault_dir) as vault:
            assert vault.list_cubes() == [cube.name]
            assert vault.file_count == 1

            qr = vault.query(cube.name)
            np.testing.assert_array_equal(qr.data, data)


class TestIngestWrongBandCount:
    """Attempt to ingest data with wrong band count."""

    def test_ingest_wrong_band_count(self, tmp_path):
        vault_dir = tmp_path / "vault"
        cube = _make_cube()  # expects 3 bands
        bad_data = RNG.integers(0, 10000, size=(5, 32, 32), dtype=np.uint16)
        t_extent = TemporalExtent(
            start=datetime(2024, 5, 1, tzinfo=timezone.utc),
            end=datetime(2024, 5, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        with Vault.create(vault_dir) as vault:
            vault.register_cube(cube)
            with pytest.raises(ValueError, match="bands"):
                vault.ingest(cube.name, bad_data, t_extent)


# ------------------------------------------------------------------
# CLI integration tests
# ------------------------------------------------------------------


class TestCLICreateAndInfo:
    """Test CLI create and info commands via CliRunner."""

    def test_cli_create_and_info(self, tmp_path):
        runner = CliRunner()
        vault_dir = str(tmp_path / "vault")

        # Create
        result = runner.invoke(cli, ["create", vault_dir, "--compression", "lzw"])
        assert result.exit_code == 0, result.output
        assert "Created vault" in result.output

        # Info (human-readable)
        result = runner.invoke(cli, ["info", vault_dir])
        assert result.exit_code == 0, result.output
        assert "Compression: lzw" in result.output
        assert "Cubes: (none)" in result.output

        # Info (JSON)
        result = runner.invoke(cli, ["info", vault_dir, "--json"])
        assert result.exit_code == 0, result.output
        data = __import__("json").loads(result.output)
        assert data["config"]["compression"] == "lzw"
        assert data["total_files"] == 0

    def test_cli_create_duplicate_fails(self, tmp_path):
        runner = CliRunner()
        vault_dir = str(tmp_path / "vault")

        runner.invoke(cli, ["create", vault_dir])
        result = runner.invoke(cli, ["create", vault_dir])
        assert result.exit_code == 1

    def test_cli_info_nonexistent_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["info", str(tmp_path / "no_vault")])
        assert result.exit_code == 1


class TestCLIIngestAndQuery:
    """Test CLI ingest and query commands via CliRunner."""

    def test_cli_ingest_and_query(self, tmp_path):
        runner = CliRunner()
        vault_dir = tmp_path / "vault"
        cube = _make_cube()

        # Create vault and register cube via Python API
        with Vault.create(vault_dir) as vault:
            vault.register_cube(cube)

        # Write a test GeoTIFF to ingest via CLI
        data = RNG.integers(0, 10000, size=(3, 32, 32), dtype=np.uint16)
        tif_path = tmp_path / "input.tif"
        _write_test_geotiff(tif_path, data)

        # Ingest via CLI
        result = runner.invoke(cli, [
            "ingest", str(vault_dir), cube.name, str(tif_path),
            "--start", "2024-01-15", "--end", "2024-01-15",
        ])
        assert result.exit_code == 0, result.output
        assert "Ingested" in result.output
        assert "File ID:" in result.output

        # Query (summary)
        result = runner.invoke(cli, [
            "query", str(vault_dir), cube.name,
        ])
        assert result.exit_code == 0, result.output
        assert "Shape:" in result.output
        assert "Source files: 1" in result.output

        # Query with --json
        result = runner.invoke(cli, [
            "query", str(vault_dir), cube.name, "--json",
        ])
        assert result.exit_code == 0, result.output
        meta = __import__("json").loads(result.output)
        assert meta["shape"][0] == 3
        assert len(meta["source_files"]) == 1

        # Query with --output
        out_tif = tmp_path / "output.tif"
        result = runner.invoke(cli, [
            "query", str(vault_dir), cube.name,
            "-o", str(out_tif),
        ])
        assert result.exit_code == 0, result.output
        assert "Wrote" in result.output
        assert out_tif.exists()

    def test_cli_query_nonexistent_cube(self, tmp_path):
        runner = CliRunner()
        vault_dir = tmp_path / "vault"
        Vault.create(vault_dir).close()

        result = runner.invoke(cli, [
            "query", str(vault_dir), "no_such_cube",
        ])
        assert result.exit_code == 1
