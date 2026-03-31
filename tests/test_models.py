"""Unit tests for VoxelVault core domain models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from voxelvault.models import (
    BandDefinition,
    CubeDescriptor,
    FileRecord,
    GridDefinition,
    SpatialExtent,
    TemporalExtent,
    Variable,
    VaultConfig,
)


# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------


class TestVariable:
    def test_valid_creation(self, sample_variable: Variable):
        assert sample_variable.name == "temperature"
        assert sample_variable.unit == "K"
        assert sample_variable.dtype == "float32"
        assert sample_variable.nodata is None
        assert sample_variable.description == ""

    def test_with_nodata(self):
        v = Variable(name="elevation", unit="m", dtype="float64", nodata=-9999.0)
        assert v.nodata == -9999.0

    def test_invalid_name_rejects_spaces(self):
        with pytest.raises(ValidationError, match="valid Python identifier"):
            Variable(name="not valid", unit="K", dtype="float32")

    def test_invalid_name_rejects_leading_digit(self):
        with pytest.raises(ValidationError, match="valid Python identifier"):
            Variable(name="3temp", unit="K", dtype="float32")

    def test_invalid_name_rejects_empty(self):
        with pytest.raises(ValidationError, match="valid Python identifier"):
            Variable(name="", unit="K", dtype="float32")

    def test_valid_dtype_int16(self):
        v = Variable(name="val", unit="dn", dtype="int16")
        assert v.dtype == "int16"

    def test_invalid_dtype(self):
        with pytest.raises(ValidationError, match="Invalid numpy dtype"):
            Variable(name="val", unit="dn", dtype="not_a_dtype")


# ---------------------------------------------------------------------------
# BandDefinition
# ---------------------------------------------------------------------------


class TestBandDefinition:
    def test_valid_band(self, sample_variable: Variable):
        bd = BandDefinition(band_index=1, variable=sample_variable)
        assert bd.band_index == 1
        assert bd.component == "scalar"

    def test_reject_zero_index(self, sample_variable: Variable):
        with pytest.raises(ValidationError, match="band_index must be >= 1"):
            BandDefinition(band_index=0, variable=sample_variable)

    def test_reject_negative_index(self, sample_variable: Variable):
        with pytest.raises(ValidationError, match="band_index must be >= 1"):
            BandDefinition(band_index=-1, variable=sample_variable)

    def test_accept_high_index(self, sample_variable: Variable):
        bd = BandDefinition(band_index=100, variable=sample_variable)
        assert bd.band_index == 100

    def test_complex_component(self, sample_variable: Variable):
        bd = BandDefinition(band_index=1, variable=sample_variable, component="real")
        assert bd.component == "real"


# ---------------------------------------------------------------------------
# SpatialExtent
# ---------------------------------------------------------------------------


class TestSpatialExtent:
    def test_valid_extent(self):
        se = SpatialExtent(west=-180.0, south=-90.0, east=180.0, north=90.0, epsg=4326)
        assert se.west == -180.0
        assert se.epsg == 4326

    def test_reject_inverted_longitude(self):
        with pytest.raises(ValidationError, match="west.*must be less than east"):
            SpatialExtent(west=10.0, south=-90.0, east=-10.0, north=90.0, epsg=4326)

    def test_reject_inverted_latitude(self):
        with pytest.raises(ValidationError, match="south.*must be less than north"):
            SpatialExtent(west=-180.0, south=10.0, east=180.0, north=-10.0, epsg=4326)

    def test_reject_equal_longitude(self):
        with pytest.raises(ValidationError):
            SpatialExtent(west=0.0, south=-90.0, east=0.0, north=90.0, epsg=4326)

    def test_reject_equal_latitude(self):
        with pytest.raises(ValidationError):
            SpatialExtent(west=-180.0, south=0.0, east=180.0, north=0.0, epsg=4326)


# ---------------------------------------------------------------------------
# TemporalExtent
# ---------------------------------------------------------------------------


class TestTemporalExtent:
    def test_valid_range(self):
        te = TemporalExtent(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        assert te.start.year == 2024

    def test_same_start_end(self):
        t = datetime(2024, 6, 15, tzinfo=timezone.utc)
        te = TemporalExtent(start=t, end=t)
        assert te.start == te.end

    def test_reject_inverted_range(self):
        with pytest.raises(ValidationError, match="start.*must be <= end"):
            TemporalExtent(
                start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    def test_reject_naive_start(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            TemporalExtent(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 12, 31, tzinfo=timezone.utc),
            )

    def test_reject_naive_end(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            TemporalExtent(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 12, 31),
            )


# ---------------------------------------------------------------------------
# GridDefinition
# ---------------------------------------------------------------------------


class TestGridDefinition:
    def test_resolution(self, sample_grid: GridDefinition):
        rx, ry = sample_grid.resolution
        assert rx == pytest.approx(0.01)
        assert ry == pytest.approx(-0.01)

    def test_bounds(self, sample_grid: GridDefinition):
        b = sample_grid.bounds
        assert b.epsg == 4326
        assert b.west == pytest.approx(-180.0)
        assert b.east == pytest.approx(-180.0 + 256 * 0.01)
        assert b.north == pytest.approx(90.0)
        assert b.south == pytest.approx(90.0 + 256 * (-0.01))


# ---------------------------------------------------------------------------
# CubeDescriptor
# ---------------------------------------------------------------------------


class TestCubeDescriptor:
    def test_variables_property(self, sample_cube_descriptor: CubeDescriptor):
        names = [v.name for v in sample_cube_descriptor.variables]
        assert names == ["red", "green", "blue"]

    def test_band_count(self, sample_cube_descriptor: CubeDescriptor):
        assert sample_cube_descriptor.band_count == 3

    def test_reject_duplicate_band_indices(self, sample_grid: GridDefinition):
        var = Variable(name="val", unit="dn", dtype="float32")
        with pytest.raises(ValidationError, match="Duplicate band_index"):
            CubeDescriptor(
                name="bad",
                bands=[
                    BandDefinition(band_index=1, variable=var),
                    BandDefinition(band_index=1, variable=var),
                ],
                grid=sample_grid,
            )


# ---------------------------------------------------------------------------
# VaultConfig
# ---------------------------------------------------------------------------


class TestVaultConfig:
    def test_defaults(self):
        vc = VaultConfig()
        assert vc.compression == "deflate"
        assert vc.compression_level == 6
        assert vc.tile_size == 256
        assert vc.overview_levels == [2, 4, 8, 16]
        assert vc.default_epsg == 4326

    def test_custom_values(self):
        vc = VaultConfig(compression="zstd", compression_level=3, tile_size=512)
        assert vc.compression == "zstd"
        assert vc.compression_level == 3
        assert vc.tile_size == 512


# ---------------------------------------------------------------------------
# FileRecord
# ---------------------------------------------------------------------------


class TestCubeDescriptorSerialization:
    def test_json_roundtrip(self, sample_cube_descriptor: CubeDescriptor):
        """CubeDescriptor must survive model_dump_json → model_validate_json,
        including tuple transform, timedelta, and dict metadata."""
        json_str = sample_cube_descriptor.model_dump_json()
        restored = CubeDescriptor.model_validate_json(json_str)
        assert restored == sample_cube_descriptor
        assert restored.grid.transform == sample_cube_descriptor.grid.transform
        assert isinstance(restored.grid.transform, tuple)
        assert restored.temporal_resolution == sample_cube_descriptor.temporal_resolution
        assert restored.metadata == sample_cube_descriptor.metadata

    def test_json_roundtrip_with_metadata(self, sample_grid: GridDefinition):
        cube = CubeDescriptor(
            name="meta_test",
            bands=[BandDefinition(band_index=1, variable=Variable(name="ndvi", unit="ratio", dtype="float32"))],
            grid=sample_grid,
            temporal_resolution=timedelta(hours=6),
            metadata={"source": "sentinel-2", "level": "L2A"},
        )
        restored = CubeDescriptor.model_validate_json(cube.model_dump_json())
        assert restored == cube
        assert restored.metadata == {"source": "sentinel-2", "level": "L2A"}
        assert restored.temporal_resolution == timedelta(hours=6)


class TestFileRecord:
    def test_serialization_roundtrip(self):
        now = datetime.now(tz=timezone.utc)
        fr = FileRecord(
            file_id="abc123",
            cube_name="rgb_cube",
            relative_path="data/rgb_cube/2024/001.tif",
            spatial_extent=SpatialExtent(west=-10.0, south=-10.0, east=10.0, north=10.0, epsg=4326),
            temporal_extent=TemporalExtent(start=now, end=now),
            band_count=3,
            file_size_bytes=1024,
            created_at=now,
            checksum="sha256:abcdef",
        )
        json_str = fr.model_dump_json()
        restored = FileRecord.model_validate_json(json_str)
        assert restored == fr
        assert restored.file_id == "abc123"
        assert restored.spatial_extent.epsg == 4326

    def test_reject_naive_created_at(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            FileRecord(
                file_id="x",
                cube_name="c",
                relative_path="p",
                spatial_extent=SpatialExtent(west=-1.0, south=-1.0, east=1.0, north=1.0, epsg=4326),
                temporal_extent=TemporalExtent(
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 1, tzinfo=timezone.utc),
                ),
                band_count=1,
                file_size_bytes=0,
                created_at=datetime(2024, 1, 1),  # naive — must be rejected
                checksum="sha256:000",
            )


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_variable_frozen(self, sample_variable: Variable):
        with pytest.raises(ValidationError):
            sample_variable.name = "other"

    def test_spatial_extent_frozen(self):
        se = SpatialExtent(west=-1.0, south=-1.0, east=1.0, north=1.0, epsg=4326)
        with pytest.raises(ValidationError):
            se.west = 0.0

    def test_vault_config_frozen(self):
        vc = VaultConfig()
        with pytest.raises(ValidationError):
            vc.compression = "lzw"

    def test_grid_frozen(self, sample_grid: GridDefinition):
        with pytest.raises(ValidationError):
            sample_grid.width = 512

    def test_cube_descriptor_frozen(self, sample_cube_descriptor: CubeDescriptor):
        with pytest.raises(ValidationError):
            sample_cube_descriptor.name = "other"

    def test_file_record_frozen(self):
        now = datetime.now(tz=timezone.utc)
        fr = FileRecord(
            file_id="x",
            cube_name="c",
            relative_path="p",
            spatial_extent=SpatialExtent(west=-1.0, south=-1.0, east=1.0, north=1.0, epsg=4326),
            temporal_extent=TemporalExtent(start=now, end=now),
            band_count=1,
            file_size_bytes=0,
            created_at=now,
            checksum="sha256:000",
        )
        with pytest.raises(ValidationError):
            fr.file_id = "y"
