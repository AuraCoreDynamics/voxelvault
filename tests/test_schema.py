"""Tests for the VoxelVault SQLite schema engine."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from voxelvault.models import (
    BandDefinition,
    CubeDescriptor,
    FileRecord,
    GridDefinition,
    SpatialExtent,
    TemporalExtent,
    Variable,
)
from voxelvault.schema import SchemaEngine


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "v2.db"


@pytest.fixture()
def engine(db_path):
    eng = SchemaEngine(db_path)
    eng.initialize()
    yield eng
    eng.close()


@pytest.fixture()
def sample_file_record(sample_cube_descriptor: CubeDescriptor) -> FileRecord:
    return FileRecord(
        file_id="file-001",
        cube_name=sample_cube_descriptor.name,
        relative_path="rgb_cube/2024/01/tile_001.tif",
        spatial_extent=SpatialExtent(west=-180.0, south=-90.0, east=-170.0, north=-80.0, epsg=4326),
        temporal_extent=TemporalExtent(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        ),
        band_count=3,
        file_size_bytes=1024000,
        created_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        checksum="sha256:abc123",
    )


# ── Table Creation & Initialization ─────────────────────────────────

class TestInitialization:
    def test_creates_all_tables(self, engine: SchemaEngine, db_path):
        conn = sqlite3.connect(str(db_path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert {"_meta", "cubes", "variables", "bands", "files"} <= tables

    def test_schema_version_recorded(self, engine: SchemaEngine, db_path):
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "1"

    def test_wal_journal_mode(self, engine: SchemaEngine, db_path):
        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_context_manager(self, db_path):
        with SchemaEngine(db_path) as eng:
            eng.initialize()
            eng.list_cubes()
        # Connection is closed — operations on the raw connection should fail
        with pytest.raises(Exception):
            eng._conn.execute("SELECT 1")


# ── Cube CRUD ────────────────────────────────────────────────────────

class TestCubeCRUD:
    def test_register_and_get_cube_roundtrip(self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor):
        engine.register_cube(sample_cube_descriptor)
        result = engine.get_cube(sample_cube_descriptor.name)

        assert result is not None
        assert result.name == sample_cube_descriptor.name
        assert result.description == sample_cube_descriptor.description
        assert result.grid.width == sample_cube_descriptor.grid.width
        assert result.grid.height == sample_cube_descriptor.grid.height
        assert result.grid.epsg == sample_cube_descriptor.grid.epsg
        assert result.grid.transform == sample_cube_descriptor.grid.transform
        assert result.temporal_resolution == sample_cube_descriptor.temporal_resolution
        assert result.metadata == sample_cube_descriptor.metadata
        assert len(result.bands) == len(sample_cube_descriptor.bands)
        for orig, loaded in zip(sample_cube_descriptor.bands, result.bands):
            assert loaded.band_index == orig.band_index
            assert loaded.variable.name == orig.variable.name
            assert loaded.variable.unit == orig.variable.unit
            assert loaded.variable.dtype == orig.variable.dtype
            assert loaded.variable.nodata == orig.variable.nodata
            assert loaded.component == orig.component

    def test_register_cube_with_metadata(self, engine: SchemaEngine, sample_grid: GridDefinition):
        cube = CubeDescriptor(
            name="meta_cube",
            bands=[BandDefinition(band_index=1, variable=Variable(name="ndvi", unit="ratio", dtype="float32"))],
            grid=sample_grid,
            metadata={"source": "sentinel-2", "processing_level": "L2A"},
        )
        engine.register_cube(cube)
        result = engine.get_cube("meta_cube")
        assert result is not None
        assert result.metadata == {"source": "sentinel-2", "processing_level": "L2A"}

    def test_register_cube_no_temporal_resolution(self, engine: SchemaEngine, sample_grid: GridDefinition):
        cube = CubeDescriptor(
            name="static_cube",
            bands=[BandDefinition(band_index=1, variable=Variable(name="elevation", unit="m", dtype="float32"))],
            grid=sample_grid,
        )
        engine.register_cube(cube)
        result = engine.get_cube("static_cube")
        assert result is not None
        assert result.temporal_resolution is None

    def test_register_cube_duplicate_raises(self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor):
        engine.register_cube(sample_cube_descriptor)
        with pytest.raises(ValueError, match="already exists"):
            engine.register_cube(sample_cube_descriptor)

    def test_get_cube_not_found(self, engine: SchemaEngine):
        assert engine.get_cube("nonexistent") is None

    def test_list_cubes(self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor, sample_grid: GridDefinition):
        engine.register_cube(sample_cube_descriptor)
        cube2 = CubeDescriptor(
            name="another_cube",
            bands=[BandDefinition(band_index=1, variable=Variable(name="temp", unit="K", dtype="float32"))],
            grid=sample_grid,
        )
        engine.register_cube(cube2)
        names = engine.list_cubes()
        assert sorted(names) == ["another_cube", "rgb_cube"]

    def test_delete_cube_with_files_raises(
        self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor, sample_file_record: FileRecord
    ):
        engine.register_cube(sample_cube_descriptor)
        engine.insert_file(sample_file_record)
        with pytest.raises(ValueError, match="file"):
            engine.delete_cube(sample_cube_descriptor.name)

    def test_delete_cube_succeeds(self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor):
        engine.register_cube(sample_cube_descriptor)
        engine.delete_cube(sample_cube_descriptor.name)
        assert engine.get_cube(sample_cube_descriptor.name) is None
        assert sample_cube_descriptor.name not in engine.list_cubes()

    def test_shared_variable_across_cubes(self, engine: SchemaEngine, sample_grid: GridDefinition):
        shared_var = Variable(name="temperature", unit="K", dtype="float32")
        cube_a = CubeDescriptor(
            name="cube_a",
            bands=[BandDefinition(band_index=1, variable=shared_var)],
            grid=sample_grid,
        )
        cube_b = CubeDescriptor(
            name="cube_b",
            bands=[BandDefinition(band_index=1, variable=shared_var)],
            grid=sample_grid,
        )
        engine.register_cube(cube_a)
        engine.register_cube(cube_b)
        assert engine.get_cube("cube_a") is not None
        assert engine.get_cube("cube_b") is not None


# ── File Record CRUD ─────────────────────────────────────────────────

class TestFileRecordCRUD:
    def test_insert_and_get_file_roundtrip(
        self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor, sample_file_record: FileRecord
    ):
        engine.register_cube(sample_cube_descriptor)
        engine.insert_file(sample_file_record)
        result = engine.get_file(sample_file_record.file_id)

        assert result is not None
        assert result.file_id == sample_file_record.file_id
        assert result.cube_name == sample_file_record.cube_name
        assert result.relative_path == sample_file_record.relative_path
        assert result.spatial_extent == sample_file_record.spatial_extent
        assert result.temporal_extent.start == sample_file_record.temporal_extent.start
        assert result.temporal_extent.end == sample_file_record.temporal_extent.end
        assert result.band_count == sample_file_record.band_count
        assert result.file_size_bytes == sample_file_record.file_size_bytes
        assert result.created_at == sample_file_record.created_at
        assert result.checksum == sample_file_record.checksum

    def test_insert_duplicate_file_id_raises(
        self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor, sample_file_record: FileRecord
    ):
        engine.register_cube(sample_cube_descriptor)
        engine.insert_file(sample_file_record)
        with pytest.raises(ValueError):
            engine.insert_file(sample_file_record)

    def test_insert_duplicate_relative_path_raises(
        self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor, sample_file_record: FileRecord
    ):
        engine.register_cube(sample_cube_descriptor)
        engine.insert_file(sample_file_record)
        dup = FileRecord(
            file_id="file-999",
            cube_name=sample_file_record.cube_name,
            relative_path=sample_file_record.relative_path,
            spatial_extent=sample_file_record.spatial_extent,
            temporal_extent=sample_file_record.temporal_extent,
            band_count=3,
            file_size_bytes=500,
            created_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
            checksum="sha256:dup",
        )
        with pytest.raises(ValueError):
            engine.insert_file(dup)

    def test_get_file_not_found(self, engine: SchemaEngine):
        assert engine.get_file("nope") is None

    def test_delete_file(
        self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor, sample_file_record: FileRecord
    ):
        engine.register_cube(sample_cube_descriptor)
        engine.insert_file(sample_file_record)
        engine.delete_file(sample_file_record.file_id)
        assert engine.get_file(sample_file_record.file_id) is None

    def test_file_count(
        self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor, sample_grid: GridDefinition
    ):
        engine.register_cube(sample_cube_descriptor)

        cube2 = CubeDescriptor(
            name="other",
            bands=[BandDefinition(band_index=1, variable=Variable(name="sar", unit="dB", dtype="float32"))],
            grid=sample_grid,
        )
        engine.register_cube(cube2)

        for i in range(3):
            engine.insert_file(FileRecord(
                file_id=f"rgb-{i}",
                cube_name="rgb_cube",
                relative_path=f"rgb/{i}.tif",
                spatial_extent=SpatialExtent(west=0, south=0, east=1, north=1, epsg=4326),
                temporal_extent=TemporalExtent(
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                ),
                band_count=3,
                file_size_bytes=100,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                checksum=f"sha256:{i}",
            ))

        engine.insert_file(FileRecord(
            file_id="other-0",
            cube_name="other",
            relative_path="other/0.tif",
            spatial_extent=SpatialExtent(west=0, south=0, east=1, north=1, epsg=4326),
            temporal_extent=TemporalExtent(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            band_count=1,
            file_size_bytes=50,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            checksum="sha256:other",
        ))

        assert engine.file_count() == 4
        assert engine.file_count(cube_name="rgb_cube") == 3
        assert engine.file_count(cube_name="other") == 1
        assert engine.file_count(cube_name="nonexistent") == 0


# ── Query Files ──────────────────────────────────────────────────────

class TestQueryFiles:
    """Spatial, temporal, and combined file queries."""

    @pytest.fixture(autouse=True)
    def _seed_files(self, engine: SchemaEngine, sample_cube_descriptor: CubeDescriptor):
        engine.register_cube(sample_cube_descriptor)
        # File A: covers lon [-10, 10], lat [-10, 10], Jan 2024
        engine.insert_file(FileRecord(
            file_id="a", cube_name="rgb_cube", relative_path="a.tif",
            spatial_extent=SpatialExtent(west=-10, south=-10, east=10, north=10, epsg=4326),
            temporal_extent=TemporalExtent(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            ),
            band_count=3, file_size_bytes=100,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc), checksum="a",
        ))
        # File B: covers lon [20, 30], lat [20, 30], Feb 2024
        engine.insert_file(FileRecord(
            file_id="b", cube_name="rgb_cube", relative_path="b.tif",
            spatial_extent=SpatialExtent(west=20, south=20, east=30, north=30, epsg=4326),
            temporal_extent=TemporalExtent(
                start=datetime(2024, 2, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 28, tzinfo=timezone.utc),
            ),
            band_count=3, file_size_bytes=200,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc), checksum="b",
        ))
        # File C: covers lon [50, 60], lat [50, 60], Mar 2024
        engine.insert_file(FileRecord(
            file_id="c", cube_name="rgb_cube", relative_path="c.tif",
            spatial_extent=SpatialExtent(west=50, south=50, east=60, north=60, epsg=4326),
            temporal_extent=TemporalExtent(
                start=datetime(2024, 3, 1, tzinfo=timezone.utc),
                end=datetime(2024, 3, 31, tzinfo=timezone.utc),
            ),
            band_count=3, file_size_bytes=300,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc), checksum="c",
        ))

    def test_query_all(self, engine: SchemaEngine):
        results = engine.query_files()
        assert len(results) == 3

    def test_query_spatial_intersects(self, engine: SchemaEngine):
        # Bounding box that intersects only file A
        results = engine.query_files(spatial_bounds=(-5, -5, 5, 5))
        assert len(results) == 1
        assert results[0].file_id == "a"

    def test_query_spatial_no_match(self, engine: SchemaEngine):
        results = engine.query_files(spatial_bounds=(100, 100, 110, 110))
        assert len(results) == 0

    def test_query_spatial_intersects_multiple(self, engine: SchemaEngine):
        # Wide bounding box that intersects A and B
        results = engine.query_files(spatial_bounds=(-20, -20, 25, 25))
        ids = {r.file_id for r in results}
        assert ids == {"a", "b"}

    def test_query_temporal_overlap(self, engine: SchemaEngine):
        # January only — only file A
        results = engine.query_files(temporal_range=(
            datetime(2024, 1, 10, tzinfo=timezone.utc),
            datetime(2024, 1, 20, tzinfo=timezone.utc),
        ))
        assert len(results) == 1
        assert results[0].file_id == "a"

    def test_query_temporal_no_match(self, engine: SchemaEngine):
        results = engine.query_files(temporal_range=(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 12, 31, tzinfo=timezone.utc),
        ))
        assert len(results) == 0

    def test_query_cube_name_filter(self, engine: SchemaEngine, sample_grid: GridDefinition):
        cube2 = CubeDescriptor(
            name="other_cube",
            bands=[BandDefinition(band_index=1, variable=Variable(name="sar", unit="dB", dtype="float32"))],
            grid=sample_grid,
        )
        engine.register_cube(cube2)
        engine.insert_file(FileRecord(
            file_id="d", cube_name="other_cube", relative_path="d.tif",
            spatial_extent=SpatialExtent(west=-10, south=-10, east=10, north=10, epsg=4326),
            temporal_extent=TemporalExtent(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            ),
            band_count=1, file_size_bytes=50,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc), checksum="d",
        ))
        results = engine.query_files(cube_name="other_cube")
        assert len(results) == 1
        assert results[0].file_id == "d"

    def test_query_combined_filters(self, engine: SchemaEngine):
        # Spatial covers A and B; temporal covers only Feb → only B
        results = engine.query_files(
            cube_name="rgb_cube",
            spatial_bounds=(-20, -20, 25, 25),
            temporal_range=(
                datetime(2024, 2, 1, tzinfo=timezone.utc),
                datetime(2024, 2, 28, tzinfo=timezone.utc),
            ),
        )
        assert len(results) == 1
        assert results[0].file_id == "b"


# ── SQL Injection Safety ─────────────────────────────────────────────

class TestSQLInjection:
    def test_cube_name_injection(self, engine: SchemaEngine):
        """SQL injection in cube name should not cause errors or data leaks."""
        malicious = "'; DROP TABLE cubes; --"
        result = engine.get_cube(malicious)
        assert result is None
        # Tables should still be intact
        assert engine.list_cubes() == []

    def test_file_id_injection(self, engine: SchemaEngine):
        result = engine.get_file("' OR 1=1; --")
        assert result is None
